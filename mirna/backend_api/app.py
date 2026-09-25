
# app.py — backend for SeedScope™ (multi-miRNA, multi-target, multi-competitor)
# - Unified FASTA+PDB cascade per molecule (miRNA, target, competitor):
#     FASTA → PDB nucleic acid → AA→NT back-translation → NT sequence for all analytics
# - Non-blocking 3D: PDB/mmCIF is optional; unusable structures are kept for viz and clearly flagged
# - Seed-scan + IG heatmaps + CSV/PNG exports + 3D contact extraction, with RAM-safe batching
# (premium, future-proof, and strictly provenance-grounded)

import os
import io
import re  # [ADDED]
import json
import time
import uuid
import math
import secrets
import logging
import tempfile
import threading
import hashlib
import zipfile
import shutil  # [ADDED]
import functools  # [ADDED]
import flask  # [ADDED]
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import requests
from flask import Flask, request, jsonify, send_file, Response

from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_limiter.errors import RateLimitExceeded
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

# Optional gzip compression for large JSON/CSV
try:
    from flask_compress import Compress
except Exception:  # graceful fallback if not installed
    Compress = None  # type: ignore

# use non-interactive backend for server PNG export
import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.layers import Layer  # type: ignore
from spektral.layers import GCSConv
import joblib

# Molecule processing (your module)
from molecule_processors import process_molecule_universal


# =========================
# Small JSON-safe serializer
# =========================

def _to_py(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (pd.Series, pd.Index)):
        return o.tolist()
    if isinstance(o, (pd.DataFrame,)):
        return o.to_dict(orient="records")
    return o


# =========================
# Paths & Configuration
# =========================
ROOT_DIR = Path(__file__).resolve().parent
JOBS_DIR = ROOT_DIR / "job_cache"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

NONCE_EXPIRY_SECONDS = 300  # 5 minutes
USE_NONCE = False
MIRNA_MAX = int(os.getenv("MIRNA_MAX", "5000"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "12"))
MATURE_TRIM_ENABLED = True
MATURE_TRIM_WINDOW = int(os.getenv("MATURE_TRIM_WINDOW", "22"))
AA_CONVERT_ALLOWED = True
STRUCTURE_MISMATCH_TOL = 1  # 10%
MAX_CONTENT_MB = 100
AA_TO_NT_DEFAULT_MODE = "human_common"  # badge only

# Persist uploaded 3D artifacts for viewer endpoints (kept until job clean-up)
ARTIFACT_TTL_SECONDS = int(os.getenv("ARTIFACT_TTL_SECONDS", "7200"))  # 2h

# Jobs registry
jobs: Dict[str, Dict] = {}

# Google Analytics (GA4) Measurement Protocol (optional)
GA_MEASUREMENT_ID = os.getenv("GA_MEASUREMENT_ID", "G-XXXXXXX")
GA_API_SECRET = os.getenv("GA_API_SECRET", "your_secret")
GA_URL = f"https://www.google-analytics.com/mp/collect?measurement_id={GA_MEASUREMENT_ID}&api_secret={GA_API_SECRET}"


def send_ga_event(event_name: str, params: Dict):
    try:
        payload = {
            "client_id": "backend_server",
            "events": [{"name": event_name, "params": params}]
        }
        r = requests.post(GA_URL, json=payload, timeout=2)
        if r.status_code != 204:
            logging.warning(f"GA event status {r.status_code}: {r.text}")
    except Exception as e:
        logging.warning(f"Failed to send GA event: {e}")


# =========================
# Logging setup
# =========================
logging.basicConfig(
    filename='backend_usage.log',
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)


# =========================
# Flask app setup
# =========================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_MB * 1024 * 1024  # MB → bytes
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

if Compress:
    Compress(app)

CORS(app,
     origins=[
       "https://marzan.org",
       "https://www.marzan.org",
       "https://mirna.marzan.org",
       "https://aamarzan.com",
       "https://www.aamarzan.com",
       "https://mirna.aamarzan.com",
       re.compile(r"http://localhost(:\d+)?$"),
       re.compile(r"http://127\.0\.0\.1(:\d+)?$")
     ],
     methods=["GET","POST","OPTIONS"],
     allow_headers=["Content-Type","X-Nonce","X-API-KEY"])



def _to_bool(v, default=True):
    if isinstance(v, bool): return v
    if v is None: return default
    return str(v).strip().lower() in {"1","true","yes","y"}


@app.after_request
def _nocache(resp):
    resp.headers['Cache-Control'] = 'no-store'
    return resp


@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    return jsonify({"error": f"Uploaded file is too large. Max size is {app.config['MAX_CONTENT_LENGTH'] // (1024*1024)} MB."}), 413


# =========================
# Security: API Key + Nonce (optional safer flow)
# =========================

# New config wrapper (non-breaking, mirrors existing flags) [ADDED]
APP_CFG = {
    "MIRNA_MAX": MIRNA_MAX,
    "MATURE_TRIM_ENABLED": MATURE_TRIM_ENABLED,
    "MATURE_WINDOW": MATURE_TRIM_WINDOW,
    "AA_CONVERT_ALLOWED": AA_CONVERT_ALLOWED,
    "USE_NONCE": USE_NONCE,
    "API_KEY": os.getenv("MIRNA_API_KEY", "").strip() or None,
    "ENABLE_HEATMAPS": True
}

# Single-use, token-keyed nonce store (no IP collisions)
nonce_store: Dict[str, float] = {}  # {nonce: expiry_ts}

@app.route('/nonce', methods=['GET'])
def issue_nonce():
    if not APP_CFG["USE_NONCE"]:
        return jsonify({"error": "nonce disabled"}), 400
    token = secrets.token_urlsafe(32)
    nonce_store[token] = time.time() + NONCE_EXPIRY_SECONDS
    return jsonify({"nonce": token, "expires_in": NONCE_EXPIRY_SECONDS})

def _check_auth() -> bool:
    """Accept either X-API-KEY (if set) or a valid X-Nonce when USE_NONCE=True.
       If USE_NONCE=False and no API_KEY, allow open access."""
    api_key = APP_CFG["API_KEY"]

    # API key path (takes precedence; not single-use)
    provided_key = request.headers.get("X-API-KEY")
    if api_key and provided_key and secrets.compare_digest(api_key, provided_key):
        return True

    # Nonce path (single-use)
    if APP_CFG["USE_NONCE"]:
        n = request.headers.get("X-Nonce")
        if n and n in nonce_store and time.time() <= nonce_store[n]:
            # nonce_store.pop(n, None)   # <-- REMOVE this line
            return True
        return False


    # Open if neither API key nor nonce required
    return not api_key


def require_auth(fn):
    @functools.wraps(fn)
    def _wrap(*args, **kwargs):
        if not _check_auth():
            return jsonify({"error": "Unauthorized"}), 401
        return fn(*args, **kwargs)
    return _wrap


def _nonce_protected(endpoint_name: Optional[str]) -> bool:
    protected = {
        'start_prediction', 'seed_scan', 'explain', 'explain_fast',
        'download_results',
        'download_all_csv', 'download_single_csv',
        'download_heatmap_png',
        'download_seeds_all', 'download_seeds_one',
        'get_structure_artifact', 'get_structure_mirna',
        'get_contacts',
        'download_all_zip', 'download_bundle_zip',
        'get_manifest'
    }
    return endpoint_name in protected


@app.before_request
def require_nonce_or_key():
    if request.method == "OPTIONS":
        return '', 200
    if _nonce_protected(request.endpoint):
        if not _check_auth():  # [UPDATED] now accepts API key OR nonce according to APP_CFG
            return jsonify({"error": "Invalid or missing authorization"}), 403

@app.before_request
def _sweep_expired_nonces():
    try:
        if nonce_store:
            now = time.time()
            # avoid iterating a changing dict
            for n, exp in list(nonce_store.items()):
                if exp <= now:
                    nonce_store.pop(n, None)
    except Exception:
        pass


@app.errorhandler(Exception)
def handle_unexpected_error(e):
    cid = secrets.token_hex(4)
    logging.exception("CID %s | Unexpected error: %s", cid, e)
    return jsonify({"error": f"Unexpected error occurred (CID {cid}). Please try again or contact support with this ID."}), 500


@app.route('/config', methods=['GET'])
def get_config():
    return jsonify({
        "mirna_max": MIRNA_MAX,
        "mature_trim_enabled": MATURE_TRIM_ENABLED,
        "mature_window": MATURE_TRIM_WINDOW,
        "aa_convert_allowed": AA_CONVERT_ALLOWED,
        "use_nonce": APP_CFG["USE_NONCE"],           # [UPDATED]
        "api_key": bool(APP_CFG["API_KEY"])          # [UPDATED]
    })


# =========================
# TensorFlow custom objects
# =========================
class PositionalEncoding(Layer):
    def __init__(self, max_len, embed_dim, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.pos_encoding = self.positional_encoding(max_len, embed_dim)

    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.max_len, "embed_dim": self.embed_dim})
        return config

    def positional_encoding(self, max_len, embed_dim):
        pos = np.arange(max_len)[:, np.newaxis]
        i = np.arange(embed_dim)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embed_dim))
        angle_rads = pos * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]


def create_weighted_mse(pos_weight=5.0, threshold=0.1):
    def weighted_mse(y_true, y_pred):
        mse_loss = tf.keras.losses.MeanSquaredError()
        mse = mse_loss(y_true, y_pred)
        weights = tf.where(y_true >= threshold, pos_weight, 1.0)
        return mse * weights
    return weighted_mse


# =========================
# Load Model and Scaler + Provenance
# =========================
MODELS_DIR = 'model_files'
model = None
scaler = None


def _sha256_file(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


PROVENANCE = {
    "model_name": "supreme_model.keras",
    "model_path": None,
    "model_sha256": None,
    "scaler_name": "minmax_scaler.pkl",
    "scaler_path": None,
    "scaler_sha256": None
}

try:
    print("--- Loading Model and Scaler ---")
    custom_objects = {
        'PositionalEncoding': PositionalEncoding,
        'weighted_mse': create_weighted_mse(),
        'GCSConv': GCSConv
    }
    model_path = os.path.join(MODELS_DIR, 'supreme_model.keras')
    scaler_path = os.path.join(MODELS_DIR, 'minmax_scaler.pkl')
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    scaler = joblib.load(scaler_path)

    PROVENANCE["model_path"] = os.path.abspath(model_path)
    PROVENANCE["scaler_path"] = os.path.abspath(scaler_path)
    PROVENANCE["model_sha256"] = _sha256_file(model_path)
    PROVENANCE["scaler_sha256"] = _sha256_file(scaler_path)
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"FATAL: Could not load model or scaler on startup. Error: {e}")


# =========================
# Helpers: Encoding & Validation
# =========================

def one_hot_encode_sequence(sequence: str, max_len: int):
    sequence = (sequence or "").upper().replace('T', 'U')
    nucleotide_map = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}
    encoded_seq = np.zeros((max_len, len(nucleotide_map)), dtype=np.float32)
    for i, char in enumerate(sequence[:max_len]):
        encoded_seq[i, nucleotide_map.get(char, 4)] = 1
    return encoded_seq


# AA vs NT detection and optional back-translation
AA_SET = set(list("ACDEFGHIKLMNPQRSTVWYBXZ"))
NT_SET = set(list("AUGCTN"))
CODON_MAP = {
    'A':'GCU', 'C':'UGU', 'D':'GAU', 'E':'GAA', 'F':'UUU', 'G':'GGU',
    'H':'CAU', 'I':'AUU', 'K':'AAA', 'L':'UUA', 'M':'AUG', 'N':'AAU',
    'P':'CCU', 'Q':'CAA', 'R':'CGU', 'S':'UCU', 'T':'ACU', 'V':'GUU',
    'W':'UGG', 'Y':'UAU'
}


def is_aa_like(seq: str) -> bool:
    s = ''.join([c for c in (seq or "").upper() if c.isalpha()])
    if not s:
        return False
    aa_frac = sum(c in AA_SET for c in s) / len(s)
    nt_frac = sum(c in NT_SET for c in s) / len(s)
    return aa_frac > 0.8 and nt_frac < 0.6


def back_translate(aa_seq: str) -> str:
    nt = []
    for a in (aa_seq or "").upper():
        if a in CODON_MAP:
            nt.append(CODON_MAP[a])
        elif a == 'X':
            nt.append('NNN')
        else:
            nt.append('NNN')
    return ''.join(nt)


def choose_mature_window(seq: str, window: int = 22) -> str:
    s = (seq or "").upper().replace("T","U")
    if len(s) <= window:
        return s
    import itertools
    def score(sub: str):
        gc = (sub.count('G')+sub.count('C'))/len(sub)
        groups = [list(g) for _, g in itertools.groupby(sub)]
        homo_pen = max((len(max(groups, key=len)))-4, 0)
        gc_pen = abs(gc - 0.5)
        return -(gc_pen*2 + homo_pen*0.5)
    best = None
    for i in range(0, len(s)-window+1):
        sub = s[i:i+window]
        sc = score(sub)
        if best is None or sc > best[0]:
            best = (sc, sub)
    return best[1] if best else s


def numerical_features_from_processed_json(pdata: Dict) -> List[float]:
    gc  = float(pdata.get('gc_content', 0.5))
    dg  = float(pdata.get('dg', 0.0))
    cons= float(pdata.get('conservation', 0.0))
    return [gc, dg, cons]


# =========================
# Keras input-name normalization (strip ':0')
# =========================

def _keras_inputs_map() -> Dict[str, Tuple[Optional[int], ...]]:
    if model is None:
        return {}
    m: Dict[str, Tuple[Optional[int], ...]] = {}
    for inp in model.inputs:
        nm = inp.name.split(':')[0]
        dims = tuple(int(d) if d is not None else None for d in inp.shape)
        m[nm] = dims
    return m

def _has_input(name: str) -> bool:
    return name in _keras_inputs_map()


# =========================
# FASTA parsing helpers (PATCH: preserve FULL header)
# =========================

def parse_fasta_records(text: str):
    """Return list of (full_header, seq) from FASTA or raw (single) text."""
    try:
        from Bio import SeqIO
    except Exception:
        return _parse_fasta_naive(text)
    records = list(SeqIO.parse(io.StringIO(text or ""), "fasta"))
    if records:
        out = []
        for r in records:
            hdr = (getattr(r, "description", None) or r.id or "").strip()
            if not hdr:
                hdr = f"seq_{len(out)+1}"
            out.append((hdr, str(r.seq)))
        return out
    raw = (text or "").strip()
    if raw:
        return [("primary_1", raw)]
    return []


def _parse_fasta_naive(text: str):
    lines = (text or "").splitlines()
    out = []
    cur_id = None
    cur_seq = []
    for ln in lines:
        if ln.startswith(">"):
            if cur_id is not None:
                out.append((cur_id, "".join(cur_seq)))
            cur_id = ln[1:].rstrip("\r\n")
            if not cur_id:
                cur_id = f"seq_{len(out)+1}"
            cur_seq = []
        else:
            cur_seq.append(ln.strip())
    if cur_id is not None:
        out.append((cur_id, "".join(cur_seq)))
    elif "".join(cur_seq).strip():
        out.append(("primary_1", "".join(cur_seq).strip()))
    return out


def has_any_fasta_header(text: str) -> bool:
    return any(ln.strip().startswith(">") for ln in (text or "").splitlines())

# --- Seed-matrix heatmap helpers (server-side 2D heatmap) ---
from matplotlib.colors import LinearSegmentedColormap

def _norm_base(b: str) -> str:
    b = (b or "").upper()
    return "U" if b == "T" else b


def _seed_pair_score(mi: str, tg: str) -> float:
    """1.0 = Watson–Crick, 0.5 = G:U wobble, 0 = mismatch."""
    mi = _norm_base(mi)
    tg = _norm_base(tg)

    # Watson–Crick pairs
    if (mi == "A" and tg == "U") or (mi == "U" and tg == "A"):
        return 1.0
    if (mi == "C" and tg == "G") or (mi == "G" and tg == "C"):
        return 1.0

    # wobble
    if (mi == "G" and tg == "U") or (mi == "U" and tg == "G"):
        return 0.5

    return 0.0


def compute_seed_profile(mirna_seq: str, target_seq: str, seed_len: int = 7) -> np.ndarray:
    """
    1D profile: for each target position, best seed score of any overlapping 7-mer.
    """
    if not mirna_seq or not target_seq:
        return np.zeros(len(target_seq or ""), dtype=float)

    s = mirna_seq.strip().upper()
    t = target_seq.strip().upper()

    if len(s) < seed_len or len(t) < seed_len:
        return np.zeros(len(t), dtype=float)

    # canonical seed: positions 2–8 (1-based) → index 1..7 (0-based)
    seed_start = 1
    seed = s[seed_start: seed_start + seed_len]
    L_t = len(t)

    pos_scores = np.zeros(L_t, dtype=float)

    for offset in range(0, L_t - seed_len + 1):
        score = 0.0
        # reverse seed for pairing (miRNA 5' vs target 3')
        for s_idx in range(seed_len):
            mi = seed[seed_len - 1 - s_idx]
            tg = t[offset + s_idx]
            score += _seed_pair_score(mi, tg)
        # propagate window score to all covered positions (keep max)
        for s_idx in range(seed_len):
            idx = offset + s_idx
            if score > pos_scores[idx]:
                pos_scores[idx] = score

    return pos_scores


# premium dark-blue → light-blue colormap
PREMIUM_BLUE_CMAP = LinearSegmentedColormap.from_list(
    "premium_blue",
    [
        (0.0, (0.07, 0.13, 0.28)),  # deep navy
        (1.0, (0.76, 0.88, 0.98)),  # very light sky-blue
    ],
)


def make_seed_matrix(mirna_seq: str, target_seq: str) -> np.ndarray:
    """
    Returns a 2D matrix (1 x L_target) with values normalized 0..1.
    """
    prof = compute_seed_profile(mirna_seq, target_seq)
    if not np.any(prof):
        return np.zeros((1, len(target_seq)), dtype=float)
    max_val = float(np.max(np.abs(prof)))
    mat = (prof / max_val)[None, :]  # shape: 1 x L
    return mat


def render_seed_matrix_png(
    mirna_id: str,
    target_id: str,
    mirna_seq: str,
    target_seq: str,
    dpi: int = 150,
) -> io.BytesIO:
    """
    Renders the 1×L matrix as a premium blue heatmap PNG and returns a BytesIO buffer.
    Row header  = miRNA ID
    Column headers = target nucleotides (if L <= 80)
    """
    mat = make_seed_matrix(mirna_seq, target_seq)
    n_cols = mat.shape[1]

    # width scales with target length but capped so huge UTRs still look okay
    base_width = 6.0
    extra = max(0.0, (n_cols - 60) * 0.04)  # small growth after 60 nt
    fig_w = min(base_width + extra, 16.0)
    fig_h = 3.2

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    im = ax.imshow(
        mat,
        aspect="auto",
        cmap=PREMIUM_BLUE_CMAP,
        vmin=0.0,
        vmax=1.0,
        origin="lower",
    )

    # Row label = miRNA ID
    ax.set_yticks([0])
    ax.set_yticklabels([mirna_id or "miRNA"], fontsize=9)

    # Column labels = target bases (only if not insanely long)
    target_seq = (target_seq or "").upper()
    if n_cols <= 80:
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(list(target_seq), fontsize=7, rotation=90)
    else:
        ax.set_xticks([])

    ax.set_xlabel(f"Target: {target_id or ''}", fontsize=10)
    ax.set_title(
        f"Seed-match heatmap — {mirna_id or 'miRNA'} vs {target_id or 'target'}",
        fontsize=11,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Seed match strength", fontsize=9)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

# =========================
# 3D structure parsing and validation
# =========================

def extract_seq_from_structure(file_path: str) -> Tuple[Optional[str], str]:
    """Return (kind, seq) where kind in {'AA','NT',None}."""
    try:
        from Bio.PDB import PDBParser, MMCIFParser, PPBuilder
    except Exception as e:
        logging.warning(f"Bio.PDB not available: {e}")
        return (None, "")
    parser = PDBParser(QUIET=True) if file_path.lower().endswith(".pdb") else MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure("struct", file_path)
    except Exception as e:
        logging.warning(f"Structure parse failed for {file_path}: {e}")
        return (None, "")
    # Protein attempt
    try:
        ppb = PPBuilder()
        aas = []
        for pp in ppb.build_peptides(structure):
            aas.append(str(pp.get_sequence()))
        if aas:
            return ("AA", "".join(aas))
    except Exception as e:
        logging.warning(f"PPBuilder failed: {e}")
    # Nucleic acids by residue name (simple)
    NA3 = {"A":"ADE", "U":"URA", "G":"GUA", "C":"CYT", "T":"THY"}
    nts = []
    try:
        for model in structure:
            for chain in model:
                for res in chain:
                    name = res.get_resname().strip().upper()
                    one = None
                    for base, three in NA3.items():
                        if name == three or name == base:
                            one = base
                            break
                    if one:
                        nts.append(one)
        if nts:
            return ("NT", "".join(nts))
    except Exception as e:
        logging.warning(f"Nucleic parse failed: {e}")
    return (None, "")


def validate_structure_matches_sequence(struct_kind: Optional[str], struct_seq: str,
                                        fasta_seq: str, molecule_label: str,
                                        allow_mismatch_ratio: float = STRUCTURE_MISMATCH_TOL) -> Tuple[bool, str]:
    """Kept for info; now used only for 'NT vs FASTA' soft check. AA kind never blocks."""
    try:
        from Bio import pairwise2
    except Exception:
        return (True, "Skipped (Biopython missing)")
    if struct_kind is None or not struct_seq:
        return (False, f"Could not detect polymer sequence in {molecule_label} 3D file.")
    fasta = (fasta_seq or "").upper().replace("T","U")
    if struct_kind == "AA":
        # Non-blocking: we will use this PDB only for viz or back-translation when needed.
        return (True, f"{molecule_label} 3D is protein; continuing (used for viz/back-translation if applicable).")
    if struct_kind == "NT":
        s = struct_seq.upper().replace("T","U")
        if not fasta or not s:
            return (True, f"{molecule_label} 3D had empty sequence during validation; continuing.")
        try:
            alns = pairwise2.align.globalms(fasta, s, 2, -1, -5, -0.5, one_alignment_only=True)
        except Exception as e:
            return (True, "Alignment unavailable; continuing.")
        if not alns:
            return (True, f"Could not align {molecule_label} 3D to provided FASTA; continuing.")
        a = alns[0]
        s1, s2 = a.seqA, a.seqB
        matches = sum(1 for x,y in zip(s1,s2) if x==y and x!='-' and y!='-')
        nongaps = sum(1 for x,y in zip(s1,s2) if x!='-' and y!='-')
        mismatch_ratio = 1 - (matches / max(1, nongaps))
        if mismatch_ratio > allow_mismatch_ratio:
            return (True, f"{molecule_label} 3D vs FASTA mismatch {mismatch_ratio:.1%}; continuing.")
        return (True, "OK")
    return (True, f"Unknown 3D polymer kind for {molecule_label}; continuing.")


def save_filestorage_to_temp(fs) -> str:
    suffix = os.path.splitext(secure_filename(fs.filename))[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    fs.save(tmp.name)
    tmp_path = tmp.name
    tmp.close()
    return tmp_path


def save_bytes_to_temp(content: bytes, suggested_name: str = "artifact.bin") -> str:
    ext = os.path.splitext(suggested_name)[1] or ".bin"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    with open(tmp.name, "wb") as f:
        f.write(content)
    return tmp.name


# =========================
# Structural feature extraction and fallback helpers
# =========================

def _seq_to_struct_column(seq: str, max_len: int) -> np.ndarray:
    mapping = {'A': 0.0, 'U': 0.25, 'G': 0.50, 'C': 0.75, 'T': 0.25, 'N': 1.0}
    s = (seq or "").upper()
    v = np.zeros((max_len, 1), dtype=np.float32)
    for i, ch in enumerate(s[:max_len]):
        v[i, 0] = mapping.get(ch, 1.0)
    return v


def extract_structure_vector_from_file(file_path: str, max_len: int) -> Optional[np.ndarray]:
    try:
        kind, seq = extract_seq_from_structure(file_path)
        if kind is None or not seq:
            return None
        if kind == "AA":
            seq = back_translate(seq)
        return _seq_to_struct_column(seq, max_len)
    except Exception:
        return None


def structure_vector_from_processed_json(struct_json: str, max_len: int) -> np.ndarray:
    try:
        sv = np.array(json.loads(struct_json or "[]"), dtype=np.float32)
    except Exception:
        sv = np.array([], dtype=np.float32)
    if sv.ndim == 0:
        sv = sv.reshape(0)
    if sv.shape[0] > max_len:
        sv = sv[:max_len]
    out = np.zeros((max_len, 1), dtype=np.float32)
    if sv.size > 0:
        out[:sv.shape[0], 0] = sv
    return out


# =========================
# Seed-scan helpers (deterministic, rule-based)
# =========================

def revcomp_rna(seq: str) -> str:
    comp = str.maketrans({'A':'U','U':'A','G':'C','C':'G','T':'A','N':'N'})
    return (seq or '').upper().translate(comp)[::-1].replace('T','U')


def is_wc(a: str, b: str) -> bool:
    return (a=='A' and b=='U') or (a=='U' and b=='A') or (a=='C' and b=='G') or (a=='G' and b=='C')


def is_gu(a: str, b: str) -> bool:
    return (a=='G' and b=='U') or (a=='U' and b=='G')


def match_seed(seed_rc: str, window: str, allow_gu: bool = True, max_mismatch: int = 0):
    mism = wob = 0
    for x,y in zip(seed_rc, window):
        if is_wc(x,y):
            continue
        elif allow_gu and is_gu(x,y):
            wob += 1
        else:
            mism += 1
            if mism > max_mismatch:
                return None
    return {'mismatches': mism, 'wobble': wob}


def classify_seed(miRNA_seq: str, target_seq: str, i: int, L: int) -> str:
    m = (miRNA_seq or '').upper().replace('T','U')
    t = (target_seq or '').upper().replace('T','U')
    label = f'seed{L}'
    upstream_A = (i-1 >= 0 and t[i-1] == 'A')
    if L == 7:
        label = '7mer-m8'
        if upstream_A:
            label = '8mer'
    elif L == 6:
        label = '7mer-A1' if upstream_A else '6mer'
    return label


def _scan_seeds_for_pair(mirna_seq: str, target_seq: str,
                         allow_gu: bool = True, max_mismatch: int = 0) -> List[Dict]:
    m = (mirna_seq or "").upper().replace('T','U')
    t = (target_seq or "").upper().replace('T','U')
    if not m or not t:
        return []
    hits: List[Dict] = []
    seed_2_8 = m[1:8] if len(m) >= 8 else m[1:]
    seed_2_7 = m[1:7] if len(m) >= 7 else m[1:]
    seeds = [(seed_2_8, 7), (seed_2_7, 6)]
    seeds = [(s, L) for (s, L) in seeds if len(s) == L and L in (6,7)]
    for seed, L in seeds:
        seed_rc = revcomp_rna(seed)
        max_i = max(0, len(t) - len(seed_rc) + 1)
        for i in range(0, max_i):
            w = t[i:i+len(seed_rc)]
            score = match_seed(seed_rc, w, allow_gu=allow_gu, max_mismatch=max_mismatch)
            if score is None:
                continue
            hit = {
                'start': i + 1,
                'end': i + len(seed_rc),
                'seed_len': len(seed_rc),
                'seed_type': classify_seed(m, t, i, L),
                **score
            }
            if i - 1 >= 0:
                hit['upstream_base'] = t[i-1]
            hits.append(hit)
    return hits


# =========================
# IG explain helper
# =========================

def integrated_gradients(model, inputs_dict: Dict[str, np.ndarray], input_key: str, steps: int = 50) -> List[float]:
    x = tf.convert_to_tensor(inputs_dict[input_key], dtype=tf.float32)
    baseline = tf.zeros_like(x)
    grads_accum = tf.zeros_like(x)

    for k in range(1, steps + 1):
        alpha = tf.cast(k / steps, tf.float32)
        x_step = baseline + alpha * (x - baseline)
        with tf.GradientTape() as tape:
            tape.watch(x_step)
            feed = {k2: (tf.convert_to_tensor(v) if not isinstance(v, tf.Tensor) else v)
                    for k2, v in inputs_dict.items()}
            feed[input_key] = x_step
            out = model(feed, training=False)
            out = tf.reduce_mean(out)
        grads = tape.gradient(out, x_step)
        grads_accum += grads

    ig = (x - baseline) * grads_accum / steps
    ig_pos = tf.reduce_sum(tf.abs(ig), axis=-1).numpy()[0].tolist()
    return ig_pos


# =========================
# PATCH: tolerant ID variants & helpers
# =========================

def _id_variants(s: str) -> List[str]:
    if not s:
        return []
    t = s.strip()
    cand = {
        t,
        t.replace(" ", "_"),
        t.replace(" ", ""),
        secure_filename(t),
        secure_filename(t).replace("_", " "),
        t.lower(),
        secure_filename(t).lower(),
        t.replace(" ", "_").lower(),
        t.replace(" ", "").lower(),
    }
    return list(cand)


# NEW: Public variants/range helpers (non-breaking) [ADDED]
_ID_CLEAN_RE = re.compile(r"[^a-z0-9_\-\.]", re.I)

def id_variants(s: str):
    s = (s or "").strip()
    if not s:
        return []
    slug = re.sub(r"\s+", "_", s.lower())
    slug = _ID_CLEAN_RE.sub("", slug)
    out = {s, s.replace(" ", "_"), s.replace(" ", ""), slug, slug.replace("_", " "), s.lower(),
           s.replace(" ", "_").lower(), s.replace(" ", "").lower()}
    return list(out)

_RANGE_RE = re.compile(r"^(.+):(\d+)-(\d+)$")
def parse_range_id(any_id: str):
    m = _RANGE_RE.match(any_id or "")
    if not m:
        return None
    base, a, b = m.group(1), int(m.group(2)), int(m.group(3))
    return {"baseId": base, "start": a, "end": b}

def tolerant_get(pool: Dict[str, str], key: str) -> Optional[str]:
    if key in pool:
        return pool[key]
    for v in id_variants(key):
        if v in pool:
            return pool[v]
    return None

def slice_by_range(pool: Dict[str, str], any_id: str) -> str:
    r = parse_range_id(any_id)
    if not r:
        return tolerant_get(pool, any_id) or ""
    base_seq = tolerant_get(pool, r["baseId"]) or ""
    if not base_seq:
        return ""
    s = max(0, r["start"] - 1)
    e = min(len(base_seq), r["end"])
    return base_seq[s:e]


def _lookup_3d(idx: Dict[str, Tuple[Optional[str], str, str]], key: str):
    for k in _id_variants(key):
        if k in idx:
            return idx[k]
    return None


def _save_optional_multi(fs_key: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    files = request.files.getlist(fs_key)
    for f in files:
        if f and f.filename:
            p = save_filestorage_to_temp(f)
            out.append((f.filename, p))
    return out


def _select_struct_path(art: Dict, kind: str, req_id: Optional[str]) -> Optional[str]:
    if req_id:
        idx_key = f"{kind}_3d_index"
        idx = art.get(idx_key) or {}
        val = None
        for k in _id_variants(req_id):
            if k in idx:
                val = idx[k]
                break
        if val and os.path.exists(val):
            return val
    key = f"{kind}_3d_path"
    p = art.get(key)
    return p if p and os.path.exists(p) else None


def _build_heatmap_bytes_for_row(job: Dict, row: Dict, mode: str = 'ig_target', steps: int = 50) -> bytes:
    shapes = job.get("model_input_shapes", {})
    Lp, Lt, Lc = int(shapes.get('Lp', 120)), int(shapes.get('Lt', 200)), int(shapes.get('Lc', 200))

    pseq = row.get('primary_seq_used','') or ''
    tseq = row.get('target_seq_used','') or ''
    cseq = row.get('competitor_seq_used','') or ''

    model_inputs = _keras_inputs_map()
    include_struct = {
        'primary_structure_input': 'primary_structure_input' in model_inputs,
        'target_structure_input': 'target_structure_input' in model_inputs,
        'competitor_structure_input': 'competitor_structure_input' in model_inputs,
    }

    if mode.startswith('ig'):
        feed = _build_ig_feed(pseq, tseq, cseq, Lp, Lt, Lc, include_struct, model_inputs)
        if mode == 'ig_target':
            values = integrated_gradients(model, feed, 'target_sequence_input', steps=steps)
            data = np.array(values[:Lt], dtype=np.float32)[None, :]
            title = f"IG (target) — {row.get('mirna_id')} vs {row.get('target_id')}"
            ytick = ['IG magnitude']
        elif mode == 'ig_competitor':
            if 'competitor_sequence_input' not in model_inputs or not cseq:
                raise RuntimeError("No competitor channel/sequence available.")
            values = integrated_gradients(model, feed, 'competitor_sequence_input', steps=steps)
            data = np.array(values[:Lc], dtype=np.float32)[None, :]
            title = f"IG (competitor) — {row.get('mirna_id')} vs {row.get('competitor_id','')}"
            ytick = ['IG magnitude']
        else:
            raise RuntimeError("Invalid IG mode")
    elif mode == 'seed_density':
        hits = json.loads(row.get('seed_hits_json') or "[]")
        L = len(tseq)
        vec = np.zeros(L, dtype=np.float32)
        for h in hits:
            s = int(h['start'])-1
            e = int(h['end'])
            vec[s:e] += 1.0
        if L == 0:
            vec = np.zeros(1, dtype=np.float32)
        data = vec[None, :]
        title = f"Seed-hit density — {row.get('mirna_id')} on {row.get('target_id')}"
        ytick = ['hit count']
    else:
        raise RuntimeError("Invalid mode")

    fig, ax = plt.subplots(figsize=(max(6, data.shape[1] / 20.0), 1.8))
    im = ax.imshow(data, aspect='auto')
    ax.set_yticks([0]); ax.set_yticklabels(ytick)
    ax.set_xlabel('Position'); ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png', dpi=200)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# =========================
# Prediction & analysis endpoints
# =========================
def _client_key():
    """Visitor identity for rate limiting. Behind the Cloudflare Worker / tunnel every request arrives from
    the proxy, so use the forwarded client IP when present."""
    h = request.headers
    return (h.get("X-Client-IP") or h.get("CF-Connecting-IP")
            or (h.get("X-Forwarded-For") or "").split(",")[0].strip()
            or get_remote_address())


limiter = Limiter(key_func=_client_key)
limiter.init_app(app)


@app.errorhandler(RateLimitExceeded)
def ratelimit_handler(e):
    return jsonify({
        "error": "rate_limit_exceeded",
        "message": "We limit predictions to keep the service fast for everyone. Please try again later."
    }), 429


def _split_ids(val: str) -> List[str]:
    raw = (val or '').replace(';', ',').replace(' ', ',')
    return [x for x in (y.strip() for y in raw.split(',')) if x]


def _download_pdb_to_tmp(pid: str, tmp_list: List[str]) -> Optional[str]:
    pid = (pid or '').strip().upper()
    if not pid:
        return None
    for ext in ('.cif', '.pdb'):
        url = f"https://files.rcsb.org/download/{pid}{ext}"
        try:
            r = requests.get(url, timeout=12)
            if r.status_code == 200 and len(r.content) > 1000:
                p = save_bytes_to_temp(r.content, suggested_name=f"{pid}{ext}")
                tmp_list.append(p)
                return p
        except Exception:
            continue
    return None


def _derive_nt_from_structure(path: str) -> Tuple[Optional[str], str, bool]:
    """Return (nt_seq, source_kind, aa_to_nt_applied) from a structure file."""
    kind, seq = extract_seq_from_structure(path)
    if not kind or not seq:
        return (None, "unknown", False)
    if kind == "NT":
        return (seq.upper().replace("T","U"), "NT", False)
    if kind == "AA":
        return (back_translate(seq), "AA", True)
    return (None, kind, False)


# --- /precheck: light 3D file/PDB sanity, non-blocking ---------------------
_ALLOWED_3D_EXT = {".pdb", ".cif", ".mmcif"}
_AA_RES = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"}
_NT_RES = {"A","C","G","U","DA","DT","DG","DC","DU","I"}

def _guess_polymer_type(sample: str) -> str:
    if any(tok in sample for tok in (" DA ", " DT ", " DG ", " DC ", " DU ", "  A ", "  U ")):
        return "nucleotide"
    for aa in _AA_RES:
        if f" {aa} " in sample:
            return "protein"
    return "unknown"

@app.post("/precheck")
def precheck():
    out = {"targets": [], "competitors": []}
    def handle(kind: str):
        items = []
        for pid in request.form.getlist(f"{kind}_pdb_id"):
            items.append({"id": pid, "polymer_type": "unknown", "present_for_viz": True, "note": "remote PDB id"})
        for f in request.files.getlist(f"{kind}_3d_file"):
            ext = os.path.splitext(f.filename)[1].lower()
            if ext not in _ALLOWED_3D_EXT:
                continue
            sample = f.stream.read(4096).decode(errors="ignore"); f.stream.seek(0)
            items.append({"id": f.filename, "polymer_type": _guess_polymer_type(sample), "present_for_viz": True, "note": ""})
        return items
    out["targets"] = handle("target")
    out["competitors"] = handle("competitor")
    return jsonify(out)
# ---------------------------------------------------------------------------


@app.route('/predict', methods=['POST'])
@limiter.limit("10 per 15 minutes")
def start_prediction():
    # Strict Content-Type check
    if request.mimetype != 'multipart/form-data':
        return jsonify({"error": "Bad request"}), 400

    # Inputs
    fasta_string = request.form.get('primary_molecules', '')
    target_seq_text = request.form.get('target_molecule', '')
    competitor_seq_text = request.form.get('competitor_molecule', '')

    # Flags
    convert_aa_to_nt_flag = request.form.get('convert_aa_to_nt', 'false').lower() == 'true'
    mature_trim_flag = request.form.get('mature_trim', 'true').lower() == 'true' if MATURE_TRIM_ENABLED else False

    # Parse miRNA FASTA (may be empty — PDB-only runs allowed if 3D is present)
    primary_records = parse_fasta_records(fasta_string)

    # If the user actually supplied FASTA text, enforce headers, max count, and min length
    if primary_records:
        if not has_any_fasta_header(fasta_string):
            return jsonify({
                "error": (
                    "Your miRNA input is missing FASTA headers. Please add >accession lines "
                    "(e.g., >hsa-let-7a-5p)."
                )
            }), 400

        if len(primary_records) > MIRNA_MAX:
            return jsonify({
                "error": f"Your submission exceeds the maximum of {MIRNA_MAX} miRNA sequences."
            }), 400

    # Min miRNA length for FASTA-derived miRNAs
    MIN_MIRNA_LEN = 10
    if primary_records:
        short_mirnas = [
            pid for pid, seq in primary_records
            if len((seq or '').replace('\n', '').strip()) < MIN_MIRNA_LEN
        ]
        if short_mirnas:
            return jsonify({
                "error": (
                    f"One or more miRNAs are shorter than {MIN_MIRNA_LEN} nt: "
                    f"{', '.join(short_mirnas[:10])}{' .' if len(short_mirnas) > 10 else ''}"
                )
            }), 400

    # Parse targets (may be empty — PDB-only supported now)
    targets_list = parse_fasta_records(target_seq_text)

    # Optional target range
    target_start_raw = request.form.get('target_start', '').strip()
    target_end_raw   = request.form.get('target_end', '').strip()
    def _to_int_safe(s):
        try: return int(s)
        except Exception: return None
    ts = _to_int_safe(target_start_raw)
    te = _to_int_safe(target_end_raw)

    def _slice_target(tid: str, tseq: str) -> Tuple[str, str]:
        if ts is None and te is None:
            return (tid, tseq)
        if ts is not None and te is not None:
            if ts <= 0 or te <= 0:
                raise ValueError("Target range must be positive integers (1-based).")
            if te < ts:
                raise ValueError("Target range end must be greater than or equal to start.")
            s_idx = max(0, ts - 1)
            e_idx = min(len(tseq), te)
            if s_idx >= len(tseq):
                raise ValueError("Target range start exceeds the target sequence length.")
            if e_idx - s_idx < 1:
                raise ValueError("Selected target range is empty. Please adjust the indices.")
            return (f"{tid}:{ts}-{te}", tseq[s_idx:e_idx])
        return (tid, tseq)

    try:
        targets_list = [_slice_target(tid, tseq) for (tid, tseq) in targets_list]
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400

    # Validate & fix targets
    MIN_TARGET_LEN = 30
    _fixed_targets: List[Tuple[str, str]] = []
    target_meta_map: Dict[str, Dict] = {}

    for (tid, tseq) in targets_list:
        seq = (tseq or '').replace('\n', '').strip()
        aa_to_nt = False
        if is_aa_like(seq):
            if AA_CONVERT_ALLOWED and convert_aa_to_nt_flag:
                seq = back_translate(seq)
                aa_to_nt = True
            else:
                return jsonify({"error": f"Target '{tid}' appears to be an amino-acid sequence. Enable AA→NT (lossy) in Advanced to proceed."}), 400
        if len(seq) < MIN_TARGET_LEN:
            return jsonify({"error": f"Target '{tid}' must be at least {MIN_TARGET_LEN} nt long (after range)."}), 400
        _fixed_targets.append((tid, seq))
        target_meta_map[tid] = {
            "source": "fasta",
            "aa_to_nt_applied": aa_to_nt,
            "aa_to_nt_mode": AA_TO_NT_DEFAULT_MODE if aa_to_nt else ""
        }
    targets_list = _fixed_targets

    # Competitors
    if competitor_seq_text.strip():
        competitors_list_raw = parse_fasta_records(competitor_seq_text)
    else:
        competitors_list_raw = []

    MIN_COMP_LEN = 15
    _fixed_comps: List[Tuple[str, str]] = []
    competitor_meta_map: Dict[str, Dict] = {}

    for (cid, cseq) in competitors_list_raw:
        s = (cseq or '').replace('\n', '').strip()
        aa_to_nt = False
        if is_aa_like(s):
            if AA_CONVERT_ALLOWED and convert_aa_to_nt_flag:
                s = back_translate(s)
                aa_to_nt = True
            else:
                return jsonify({"error": f"Competitor '{cid}' appears to be an amino-acid sequence. Enable AA→NT (lossy) in Advanced to proceed."}), 400
        if len(s) < MIN_COMP_LEN:
            return jsonify({"error": f"Competitor '{cid}' must be at least {MIN_COMP_LEN} nt long."}), 400
        _fixed_comps.append((cid, s))
        competitor_meta_map[cid] = {
            "source": "fasta",
            "aa_to_nt_applied": aa_to_nt,
            "aa_to_nt_mode": AA_TO_NT_DEFAULT_MODE if aa_to_nt else ""
        }

    # Save uploaded 3D files (multi)
    tmp_paths_to_cleanup: List[str] = []
    target_3d_files = _save_optional_multi('target_3d_file')
    competitor_3d_files = _save_optional_multi('competitor_3d_file')
    for _, p in target_3d_files + competitor_3d_files:
        tmp_paths_to_cleanup.append(p)

    target_3d_path = target_3d_files[0][1] if target_3d_files else None
    competitor_3d_path = competitor_3d_files[0][1] if competitor_3d_files else None

    target_3d_index: Dict[str, str] = {}
    for fname, p in target_3d_files:
        stem = os.path.splitext(secure_filename(fname))[0]
        for k in _id_variants(stem):
            target_3d_index[k] = p

    competitor_3d_index: Dict[str, str] = {}
    for fname, p in competitor_3d_files:
        stem = os.path.splitext(secure_filename(fname))[0]
        for k in _id_variants(stem):
            competitor_3d_index[k] = p

    # Accept PDB IDs (download & index) — support both plural string and repeated fields [UPDATED]
    # Old form (comma/space/semicolon separated):
    target_pdb_ids_str = request.form.get('target_pdb_ids', '')
    competitor_pdb_ids_str = request.form.get('competitor_pdb_ids', '')
    t_ids = _split_ids(target_pdb_ids_str)
    c_ids = _split_ids(competitor_pdb_ids_str)
    # New form (repeated fields):
    t_ids += [x.strip() for x in request.form.getlist("target_pdb_id") if x.strip()]
    c_ids += [x.strip() for x in request.form.getlist("competitor_pdb_id") if x.strip()]

    for pid in t_ids:
        p = _download_pdb_to_tmp(pid, tmp_paths_to_cleanup)
        if p:
            stem = pid
            target_3d_files.append((f"{stem}{os.path.splitext(p)[1]}", p))
            for k in _id_variants(stem):
                target_3d_index[k] = p
            if not target_3d_path:
                target_3d_path = p

    for pid in c_ids:
        p = _download_pdb_to_tmp(pid, tmp_paths_to_cleanup)
        if p:
            stem = pid
            competitor_3d_files.append((f"{stem}{os.path.splitext(p)[1]}", p))
            for k in _id_variants(stem):
                competitor_3d_index[k] = p
            if not competitor_3d_path:
                competitor_3d_path = p

    # ---- NEW: PDB-derived sequences + FASTA/PDB unification ----
    # For each role (target / competitor):
    #   • If a PDB/mmCIF exists whose ID matches an existing FASTA ID, we treat
    #     them as ONE entry: sequence from FASTA, 3D from PDB (handled later via
    #     _select_struct_path + validate_structure_matches_sequence).
    #   • If a PDB/mmCIF has NO matching FASTA ID, we derive a nucleotide
    #     sequence from the structure and register it as an extra molecule entry
    #     (PDB-only cascade).
    #
    # This preserves the old behaviour when no FASTA is present at all, but also
    # lets PDB-only molecules coexist with FASTA-defined ones.

    # Targets
    if target_3d_files or t_ids:
        had_any_targets = bool(targets_list)

        # Normalize existing FASTA target IDs for matching
        existing_t_ids = set()
        for tid, _ in targets_list:
            for k in _id_variants(tid):
                existing_t_ids.add(k.lower())

        derived_from_pdb = 0
        for fname, p in target_3d_files:
            stem = os.path.splitext(secure_filename(fname))[0]

            # Skip if this PDB/mmCIF corresponds to an existing FASTA ID
            is_matched = False
            for k in _id_variants(stem):
                if k.lower() in existing_t_ids:
                    is_matched = True
                    break
            if is_matched:
                continue

            # PDB-only (or extra) target → derive NT sequence
            nt_seq, src_kind, aa_to_nt = _derive_nt_from_structure(p)
            if nt_seq and len(nt_seq) >= MIN_TARGET_LEN:
                tid = stem
                targets_list.append((tid, nt_seq))
                derived_from_pdb += 1
                target_meta_map[tid] = {
                    "source": f"pdb:{src_kind}",
                    "aa_to_nt_applied": aa_to_nt,
                    "aa_to_nt_mode": AA_TO_NT_DEFAULT_MODE if aa_to_nt else ""
                }

        # Preserve old error: no FASTA, PDBs present, but nothing usable derived
        if (not targets_list
                and (target_3d_files or t_ids)
                and not had_any_targets
                and derived_from_pdb == 0):
            return jsonify({"error": "Could not derive any nucleotide targets from provided PDB(s)."}), 400

    # Competitors
    if competitor_3d_files or c_ids:
        had_any_competitors = bool(_fixed_comps)

        # Normalize existing FASTA competitor IDs for matching
        existing_c_ids = set()
        for cid, _ in _fixed_comps:
            for k in _id_variants(cid):
                existing_c_ids.add(k.lower())

        derived_comps_from_pdb = 0
        for fname, p in competitor_3d_files:
            stem = os.path.splitext(secure_filename(fname))[0]

            # Skip if this PDB/mmCIF corresponds to an existing FASTA ID
            is_matched = False
            for k in _id_variants(stem):
                if k.lower() in existing_c_ids:
                    is_matched = True
                    break
            if is_matched:
                continue

            # PDB-only (or extra) competitor → derive NT sequence
            nt_seq, src_kind, aa_to_nt = _derive_nt_from_structure(p)
            if nt_seq and len(nt_seq) >= MIN_COMP_LEN:
                cid = stem
                _fixed_comps.append((cid, nt_seq))
                derived_comps_from_pdb += 1
                competitor_meta_map[cid] = {
                    "source": f"pdb:{src_kind}",
                    "aa_to_nt_applied": aa_to_nt,
                    "aa_to_nt_mode": AA_TO_NT_DEFAULT_MODE if aa_to_nt else ""
                }

        # For competitors we keep the previous behaviour: if there are still no
        # competitors at all after this cascade, we later fall back to the
        # placeholder ("none", "") without raising a hard error.

    # If still no competitors, use placeholder (unchanged behaviour)
    competitors_list = _fixed_comps if _fixed_comps else [("none", "")]

    # ---------------------------------------------

    # miRNA 3D index (multi; always register all structures)
    mirna_3d_files = request.files.getlist('mirna_3d_file')
    mirna_3d_index: Dict[str, Tuple[Optional[str], str, str]] = {}
    for f in mirna_3d_files:
        if f and f.filename:
            p = save_filestorage_to_temp(f)
            tmp_paths_to_cleanup.append(p)
            stem = os.path.splitext(secure_filename(f.filename))[0]
            kind, seq = extract_seq_from_structure(p)
            # Keep raw (kind, seq, path) for validation & viewer
            for k in _id_variants(stem):
                mirna_3d_index[k] = (kind, seq, p)

    # ---- UPDATED: unified nt cascade for miRNA PDB/CIF runs ----
    # Behaviour:
    #   • If a miRNA PDB/mmCIF matches an existing FASTA ID, we treat it as
    #     the 3D partner of that FASTA miRNA (handled later via mirna_3d_index
    #     + _lookup_3d) and do NOT create a second primary entry.
    #   • If a miRNA PDB/mmCIF does NOT match any FASTA miRNA ID, we derive
    #     a nucleotide sequence from the structure and register it as an
    #     extra miRNA (so PDB-only miRNAs also enter the affinity/seed/IG pipeline).
    #
    # This works both when primary_records is empty (PDB-only run) and when
    # FASTA + extra PDB-only miRNAs are mixed.

    if mirna_3d_index:
        # Pre-compute set of known FASTA ID variants for matching
        existing_mirna_ids = set()
        for mid, _ in primary_records:
            for v in _id_variants(mid):
                existing_mirna_ids.add(v.lower())

        seen_paths: List[str] = []
        extra_prim: List[Tuple[str, str]] = []

        for key, (_kind, _seq3d, path3d) in mirna_3d_index.items():
            if path3d in seen_paths:
                continue
            seen_paths.append(path3d)

            # Approximate "stem" similar to targets/competitors; here we use the
            # first key associated with this path as the representative ID.
            stem = key

            # If this structure clearly belongs to an existing FASTA miRNA,
            # just keep it as 3D partner and do not create a new sequence entry.
            is_matched = False
            for v in _id_variants(stem):
                if v.lower() in existing_mirna_ids:
                    is_matched = True
                    break
            if is_matched:
                continue

            # PDB-only miRNA → derive NT sequence (NT direct, or AA→NT)
            nt_seq, src_kind, aa_to_nt = _derive_nt_from_structure(path3d)
            if not nt_seq:
                continue

            seq_clean = (nt_seq or "").replace("\n", "").strip()
            if len(seq_clean) < MIN_MIRNA_LEN:
                # Respect the same lower bound as FASTA-derived miRNAs
                continue

            # Use this ID so mirna_3d_index/_lookup_3d can reliably find the
            # structure later.
            mid = stem
            extra_prim.append((mid, seq_clean))

        # If we have extra miRNAs derived from PDB/CIF, merge them in.
        if extra_prim:
            if primary_records:
                primary_records.extend(extra_prim)
            else:
                # PDB-only run: behave like previous "PDB-only cascade"
                primary_records = extra_prim

    # After FASTA + PDB/CIF cascades, enforce cap and require at least one miRNA
    if len(primary_records) > MIRNA_MAX:
        return jsonify({
            "error": f"Your submission exceeds the maximum of {MIRNA_MAX} miRNA sequences."
        }), 400

    if not primary_records:
        # This is the only place we now *stop* for missing miRNA:
        # no FASTA and no usable PDB/CIF.
        return jsonify({"error": "No valid miRNA sequences found in FASTA or PDB/CIF input."}), 400

    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    _total = len(primary_records) * max(1, len(targets_list)) * max(1, len(competitors_list))

    # NEW: optional chain hints manifest capture [ADDED]
    try:
        target_chain_hints = json.loads(request.form.get("target_chain_hints_json") or "{}")
    except Exception:
        target_chain_hints = {}
    try:
        competitor_chain_hints = json.loads(request.form.get("competitor_chain_hints_json") or "{}")
    except Exception:
        competitor_chain_hints = {}

    # --- NEW: unified, role-aware FASTA vs 3D deduplicated counts ---
    def _canonical_role_id(raw_id: str) -> str:
        """
        Normalize IDs so FASTA and PDB/CIF keys can be matched:
        - strip target ranges (id:start-end → id)
        - collapse whitespace, lower-case
        """
        if not raw_id:
            return ""
        r = parse_range_id(raw_id)
        if r:
            raw_id = r["baseId"]
        raw_id = raw_id.strip()
        raw_id = re.sub(r"\s+", "_", raw_id)
        return raw_id.lower()

    def _struct_canonical_ids(idx, triple: bool = False):
        """
        Collapse structure index to canonical IDs per unique file path.

        For target/competitor indices (key → path) we group by path and
        pick one representative key. For miRNA (key → (kind, seq, path))
        we do the same but unpack the triple.
        """
        by_path = {}
        if triple:
            # mirna_3d_index: key -> (kind, seq, path)
            for key, (_kind, _seq3d, path) in (idx or {}).items():
                by_path.setdefault(path, []).append(key)
        else:
            # target_3d_index / competitor_3d_index: key -> path
            for key, path in (idx or {}).items():
                by_path.setdefault(path, []).append(key)

        out = set()
        for path, keys in by_path.items():
            if not keys:
                continue
            # Choose the shortest key as representative for this file
            best = sorted(keys, key=len)[0]
            out.add(_canonical_role_id(best))
        return out

    # Canonicalized FASTA IDs
    mirna_ids_fasta = {_canonical_role_id(mid) for (mid, _seq) in primary_records}
    target_ids_fasta = {_canonical_role_id(tid) for (tid, _seq) in targets_list}
    competitor_ids_fasta = {_canonical_role_id(cid) for (cid, _seq) in competitors_list if cid != "none"}

    # Canonicalized 3D IDs from the indices
    mirna_ids_struct = _struct_canonical_ids(mirna_3d_index, triple=True)
    target_ids_struct = _struct_canonical_ids(target_3d_index, triple=False)
    competitor_ids_struct = _struct_canonical_ids(competitor_3d_index, triple=False)

    def _dedup_counts(fset, sset):
        matched = sorted(fset & sset)
        fasta_only = sorted(fset - sset)
        struct_only = sorted(sset - fset)
        return {
            "total_fasta": len(fset),
            "total_struct": len(sset),
            "matched_pairs": len(matched),
            "fasta_only": len(fasta_only),
            "struct_only": len(struct_only),
            # Keep a few IDs for easy debugging in the manifest
            "example_matched_ids": matched[:5],
            "example_fasta_only": fasta_only[:5],
            "example_struct_only": struct_only[:5],
        }

    mirna_summary = _dedup_counts(mirna_ids_fasta, mirna_ids_struct)
    target_summary = _dedup_counts(target_ids_fasta, target_ids_struct)
    competitor_summary = _dedup_counts(competitor_ids_fasta, competitor_ids_struct)

    jobs[job_id] = {
        "status": "running",
        "results": [],
        "error": None,
        "warnings": [],
        "total": _total,
        "completed": 0,
        "target_id": "MULTIPLE" if len(targets_list) > 1 else (targets_list[0][0] if targets_list else "NONE"),
        "target_len": -1,
        "artifacts": {
            "target_3d_path": target_3d_path,
            "competitor_3d_path": competitor_3d_path,
            "target_3d_index": target_3d_index,
            "competitor_3d_index": competitor_3d_index,
            "mirna_3d_index": mirna_3d_index,
            "expiry": time.time() + ARTIFACT_TTL_SECONDS
        },
        "job_dir": str(job_dir),
        "results_json_path": None,
        "model_input_shapes": {},
        "target_meta": target_meta_map,
        "competitor_meta": competitor_meta_map,
        # UPDATED: richer manifest for unified, role-aware inspection
        "manifest": {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "client": "mirna.js",
            "target_pdb_ids": t_ids,
            "competitor_pdb_ids": c_ids,
            "target_chain_hints": target_chain_hints,
            "competitor_chain_hints": competitor_chain_hints,
            "role_counts": {
                "mirna": mirna_summary,
                "target": target_summary,
                "competitor": competitor_summary,
                "combos": {
                    "primary": len(primary_records),
                    "targets": len(targets_list),
                    "competitors": len(competitors_list),
                    "eval_triples": _total,
                },
            },
            "fasta_ids": {
                "mirna": sorted(list(mirna_ids_fasta)),
                "target": sorted(list(target_ids_fasta)),
                "competitor": sorted(list(competitor_ids_fasta)),
            },
        }
    }


    send_ga_event("prediction_started", {"total": _total})

    threading.Thread(
        target=_queued_process_job,
        args=(job_id, primary_records, targets_list, competitors_list,
              target_3d_path, competitor_3d_path, {}, tmp_paths_to_cleanup,
              convert_aa_to_nt_flag, mature_trim_flag),
        daemon=True
    ).start()

    return jsonify({"job_id": job_id, "status": "started"})


MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "0"))  # 0 = unlimited (local default)
_job_slots = threading.BoundedSemaphore(MAX_CONCURRENT_JOBS) if MAX_CONCURRENT_JOBS > 0 else None
_queue_lock = threading.Lock()
_queue_order: List[str] = []   # job ids waiting for a free slot, oldest first


def _queue_position(job_id: str) -> int:
    """1 = next to run, 0 = not waiting."""
    with _queue_lock:
        return (_queue_order.index(job_id) + 1) if job_id in _queue_order else 0


def _queued_process_job(job_id, *args):
    if _job_slots is None:
        return process_job(job_id, *args)
    with _queue_lock:
        _queue_order.append(job_id)
    try:
        _job_slots.acquire()          # waits here while other jobs use all slots
    finally:
        with _queue_lock:
            if job_id in _queue_order:
                _queue_order.remove(job_id)
    try:
        process_job(job_id, *args)
    finally:
        _job_slots.release()


def process_job(job_id: str,
                primary_records: List[Tuple[str,str]],
                targets_list: List[Tuple[str,str]],
                competitors_list: List[Tuple[str,str]],
                target_3d_path: Optional[str],
                competitor_3d_path: Optional[str],
                mirna_3d_index_unused: Dict[str, Tuple[Optional[str], str, str]],
                tmp_paths_to_cleanup: List[str],
                convert_aa_to_nt_flag: bool,
                mature_trim_flag: bool):
    try:
        if model is None or scaler is None:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = "Model or scaler not loaded on server."
            return

        def warn(msg: str):
            try:
                jobs[job_id]["warnings"].append(msg)
            except Exception:
                pass

        def ensure_dict(data):
            if isinstance(data, tuple):
                return {
                    "sequence": data[1] if len(data) > 1 else "",
                    "gc_content": 0.5, "dg": 0.0, "conservation": 0.0,
                    "structure_vector": "[]", "adjacency_matrix": "[]"
                }
            return data

        model_inputs = _keras_inputs_map()
        max_primary_len    = int((model_inputs.get('primary_sequence_input')      or (None,120))[1] or 120)
        max_target_len     = int((model_inputs.get('target_sequence_input')       or (None,200))[1] or 200)
        max_competitor_len = int((model_inputs.get('competitor_sequence_input')   or (None,200))[1] or 200)

        empty_comp_enc     = one_hot_encode_sequence('', max_competitor_len)

        has_comp_input = 'competitor_sequence_input' in model_inputs
        has_num_input  = 'numerical_features_input' in model_inputs
        has_p_struct   = 'primary_structure_input' in model_inputs
        has_t_struct   = 'target_structure_input' in model_inputs
        has_c_struct   = 'competitor_structure_input' in model_inputs

        jobs[job_id]["model_input_shapes"] = {"Lp": max_primary_len, "Lt": max_target_len, "Lc": max_competitor_len}

        art = jobs[job_id]["artifacts"]
        t_meta = jobs[job_id].get("target_meta", {}) or {}
        c_meta = jobs[job_id].get("competitor_meta", {}) or {}

        interaction_counter = 0

        for (competitor_id, competitor_str) in competitors_list:
            competitor_processed = {'sequence': ''}
            if competitor_str.strip():
                competitor_processed = ensure_dict(process_molecule_universal(((competitor_id, competitor_str), {}, 'competitor_molecule')))

            # Select competitor 3D path per ID (multi-index aware)
            comp_path = _select_struct_path(art, 'competitor', competitor_id)

            # Soft validation if both present
            if comp_path and competitor_processed.get('sequence','').strip():
                kind, seq = extract_seq_from_structure(comp_path)
                ok, msg = validate_structure_matches_sequence(kind, seq, competitor_processed.get('sequence',''), "Competitor")
                if msg:
                    warn(msg)

            comp_seq_enc = None
            if competitor_processed.get('sequence', '').strip() and has_comp_input:
                comp_seq_enc = one_hot_encode_sequence(competitor_processed.get('sequence', ''), max_competitor_len)

            # Competitor structural input (if expected)
            competitor_struct_input = None
            if has_c_struct:
                vec = None
                if comp_path:
                    vec = extract_structure_vector_from_file(comp_path, max_competitor_len)
                if vec is None:
                    if competitor_processed.get('sequence','').strip():
                        vec = structure_vector_from_processed_json(competitor_processed.get('structure_vector', '[]'), max_competitor_len)
                    else:
                        vec = np.zeros((max_competitor_len,1), dtype=np.float32)
                competitor_struct_input = vec

            for (target_id, target_str) in targets_list:
                target_processed = ensure_dict(process_molecule_universal(((target_id, target_str), {}, 'target_molecule')))
                target_seq_used = target_processed.get('sequence', '')
                target_seq_enc  = one_hot_encode_sequence(target_seq_used, max_target_len)

                # Select target 3D path per ID (multi-index aware)
                t_path = _select_struct_path(art, 'target', target_id)

                # Soft validation if both present
                if t_path and target_seq_used:
                    kind, seq = extract_seq_from_structure(t_path)
                    ok, msg = validate_structure_matches_sequence(kind, seq, target_seq_used, "Target")
                    if msg:
                        warn(msg)

                # Target structural input (if expected)
                target_struct_input = None
                if has_t_struct:
                    vec = None
                    if t_path:
                        vec = extract_structure_vector_from_file(t_path, max_target_len)
                    if vec is None:
                        vec = structure_vector_from_processed_json(target_processed.get('structure_vector', '[]'), max_target_len)
                    target_struct_input = vec

                # -------- Batch over primaries --------
                for start in range(0, len(primary_records), BATCH_SIZE):
                    batch_records = primary_records[start:start + BATCH_SIZE]
                    prim_seq_list, num_feat_list, prim_struct_list = [], [], []
                    trimmed_sequences: List[str] = []

                    for pri_id, pri_seq in batch_records:
                        pdata = ensure_dict(process_molecule_universal(((pri_id, pri_seq), {}, 'primary_molecule')))
                        seq = pdata.get('sequence', '')
                        if MATURE_TRIM_ENABLED and mature_trim_flag and len(seq) > 30:
                            seq = choose_mature_window(seq, window=MATURE_TRIM_WINDOW)
                            pdata['sequence'] = seq

                        trimmed_sequences.append(seq)
                        prim_seq_list.append(one_hot_encode_sequence(seq, max_primary_len))
                        if has_num_input:
                            num_feat_list.append(numerical_features_from_processed_json(pdata))

                        if has_p_struct:
                            sp = structure_vector_from_processed_json(pdata.get('structure_vector','[]'), max_primary_len)
                            prim_struct_list.append(sp)

                        # miRNA 3D validation (soft)
                        idx = art.get('mirna_3d_index') or {}
                        val = _lookup_3d(idx, pri_id)
                        if val is not None:
                            kind, seq3d, path3d = val
                            ok, msg = validate_structure_matches_sequence(kind, seq3d, pdata.get('sequence',''), f"miRNA {pri_id}")
                            if msg:
                                warn(msg)

                    if has_num_input:
                        try:
                            if hasattr(scaler, 'feature_names_in_'):
                                df_features = pd.DataFrame(num_feat_list, columns=scaler.feature_names_in_)
                                scaled_num = scaler.transform(df_features)
                            else:
                                scaled_num = scaler.transform(num_feat_list)
                        except Exception as e:
                            jobs[job_id]["status"] = "error"
                            jobs[job_id]["error"] = f"Numeric feature scaling failed: {e}"
                            return

                    pri_seq_enc  = np.stack(prim_seq_list, axis=0).astype(np.float32)
                    batch_size   = pri_seq_enc.shape[0]

                    common_inputs = {
                        'primary_sequence_input': pri_seq_enc,
                        'target_sequence_input':  np.repeat(target_seq_enc[np.newaxis, ...], batch_size, axis=0),
                    }
                    if has_num_input:
                        common_inputs['numerical_features_input'] = scaled_num
                    if has_p_struct:
                        pri_struct = np.stack(prim_struct_list, axis=0).astype(np.float32) if prim_struct_list else np.zeros((batch_size, max_primary_len, 1), dtype=np.float32)
                        common_inputs['primary_structure_input'] = pri_struct
                    if has_t_struct and target_struct_input is not None:
                        common_inputs['target_structure_input'] = np.repeat(target_struct_input[np.newaxis, ...], batch_size, axis=0)
                    if has_c_struct and competitor_struct_input is not None:
                        common_inputs['competitor_structure_input'] = np.repeat(competitor_struct_input[np.newaxis, ...], batch_size, axis=0)

                    if has_comp_input:
                        with_comp = dict(common_inputs)
                        if comp_seq_enc is not None:
                            with_comp['competitor_sequence_input'] = np.repeat(comp_seq_enc[np.newaxis, ...], batch_size, axis=0)
                        else:
                            with_comp['competitor_sequence_input'] = np.repeat(empty_comp_enc[np.newaxis, ...], batch_size, axis=0)

                        no_comp = dict(common_inputs)
                        no_comp['competitor_sequence_input'] = np.repeat(empty_comp_enc[np.newaxis, ...], batch_size, axis=0)

                        preds_with = model.predict(with_comp, verbose=0).reshape(-1).astype(np.float64)
                        preds_no   = model.predict(no_comp,   verbose=0).reshape(-1).astype(np.float64)
                    else:
                        preds_no = model.predict(common_inputs, verbose=0).reshape(-1).astype(np.float64)
                        preds_with = preds_no.copy()

                    for row_idx, ((pri_id, _), p_base, p_with) in enumerate(zip(batch_records, preds_no, preds_with)):
                        interaction_counter += 1
                        pri_seq_used = trimmed_sequences[row_idx]
                        seed_hits = _scan_seeds_for_pair(pri_seq_used, target_seq_used, allow_gu=True, max_mismatch=0)

                        best = None
                        if seed_hits:
                            seed_hits_sorted = sorted(seed_hits, key=lambda h: (-h['seed_len'], h['mismatches'], h.get('wobble',0), h['start']))
                            best = seed_hits_sorted[0]

                        # Badges / provenance per row
                        t_badge = t_meta.get(target_id, {})
                        c_badge = c_meta.get(competitor_id, {}) if competitor_id in c_meta else {}

                        pdb_used_target = bool(_select_struct_path(art, 'target', target_id))
                        pdb_used_comp   = bool(_select_struct_path(art, 'competitor', competitor_id)) if competitor_id and competitor_id != "none" else False
                        struct_feats_on = bool('target_structure_input' in _keras_inputs_map() or
                                               'competitor_structure_input' in _keras_inputs_map() or
                                               'primary_structure_input' in _keras_inputs_map())

                        row = {
                            'interaction_id': f"I{interaction_counter:07d}",
                            'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                            'mirna_id': pri_id,
                            'primary_molecule_id': pri_id,
                            'target_id': target_id,
                            'competitor_id': competitor_id if competitor_str else '',
                            'predicted_affinity_baseline': format(float(p_base), '.10f'),
                            'predicted_affinity_with_competitor': format(float(p_with), '.10f'),
                            'competitive_effect (higher_is_better)': format(float(p_base - p_with), '.10f'),

                            'primary_seq_used': pri_seq_used,
                            'target_seq_used': target_seq_used,
                            'competitor_seq_used': competitor_processed.get('sequence','') if competitor_str else '',

                            'seed_hits_json': json.dumps(seed_hits, separators=(',',':')),
                            'seed_best_type': (best or {}).get('seed_type', ''),
                            'seed_best_start': (best or {}).get('start', ''),
                            'seed_best_end': (best or {}).get('end', ''),
                            'seed_best_wobble': (best or {}).get('wobble', ''),
                            'seed_best_mismatches': (best or {}).get('mismatches', ''),

                            # NEW badges
                            'pdb_target_used': 'yes' if pdb_used_target else 'no',
                            'pdb_competitor_used': 'yes' if pdb_used_comp else 'no',
                            'aa_to_nt_applied_target': 'yes' if t_badge.get('aa_to_nt_applied') else 'no',
                            'aa_to_nt_mode_target': t_badge.get('aa_to_nt_mode',''),
                            'aa_to_nt_applied_competitor': 'yes' if c_badge.get('aa_to_nt_applied') else 'no',
                            'aa_to_nt_mode_competitor': c_badge.get('aa_to_nt_mode',''),
                            'structure_features': 'on' if struct_feats_on else 'off'
                        }

                        row.update({
                            'prov_model_path': PROVENANCE.get('model_path'),
                            'prov_model_sha256': PROVENANCE.get('model_sha256'),
                            'prov_scaler_path': PROVENANCE.get('scaler_path'),
                            'prov_scaler_sha256': PROVENANCE.get('scaler_sha256'),
                            'prov_explain_method': 'integrated_gradients',
                            'prov_explain_steps': 50,
                            'prov_seed_rules': 'v1 canonical (6/7mer, upstream-A for 7mer-A1/8mer), allow_gu=True, max_mismatch=0'
                        })

                        jobs[job_id]["results"].append(row)
                        jobs[job_id]["completed"] += 1

        try:
            def _safe_float(x):
                try:
                    return float(x)
                except Exception:
                    return -math.inf
            jobs[job_id]["results"].sort(key=lambda r: _safe_float(r.get('predicted_affinity_baseline', -1)), reverse=True)

            job_dir = Path(jobs[job_id]["job_dir"])  # type: ignore
            results_path = job_dir / "results.json"
            payload = {"results": jobs[job_id]["results"]}
            with results_path.open('w', encoding='utf-8') as f:
                json.dump(payload, f, default=_to_py, ensure_ascii=False, separators=(',', ':'))
            jobs[job_id]["results_json_path"] = str(results_path)
        except Exception as werr:
            logging.warning(f"Failed to prewrite results.json: {werr}")

        jobs[job_id]["status"] = "completed"
        send_ga_event("prediction_completed", {"total": jobs[job_id]["total"]})
    except Exception as e:
        logging.exception(f"Prediction error: {e}")
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)[:500]
    finally:
        start_janitor()


@app.route('/progress/<job_id>', methods=['GET'])
def get_progress(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Invalid job ID"}), 404
    return jsonify({
        "status": job["status"],
        "completed": job["completed"],
        "total": job["total"],
        "error": job["error"],
        "warnings": job.get("warnings", []),
        "queue_position": _queue_position(job_id),
        "results": job["results"] if job["status"] == "completed" else []
    })


@app.route('/download/<job_id>', methods=['GET'])
def download_results(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Invalid job ID"}), 404
    if job["status"] != "completed":
        return jsonify({"error": "Job not completed yet"}), 400

    # Pagination + "light" projection (defaults: light=1, limit=500)
    try:
        limit = int(request.args.get('limit') or 500)
        offset = int(request.args.get('offset') or 0)
    except Exception:
        limit, offset = 500, 0
    light = (request.args.get('light', '1') != '0')

    rows = job["results"]
    total = len(rows)
    page = rows[offset:offset+limit]

    if light:
        keep = (
            "interaction_id","mirna_id","target_id","competitor_id",
            "predicted_affinity_baseline","predicted_affinity_with_competitor",
            "competitive_effect (higher_is_better)",
            "seed_best_type","seed_best_start","seed_best_end",
            "pdb_target_used","pdb_competitor_used","structure_features"
        )
        page = [{k: r.get(k) for k in keep} for r in page]

    return jsonify({"results": page, "total": total, "offset": offset, "limit": limit})


@app.post("/explain_fast")
def explain_fast():
    try:
        data = request.get_json(force=True) or {}
        mirna = (data.get("mirna_seq") or "").upper().replace("T","U")
        target = (data.get("target_seq") or "").upper().replace("T","U")
        comp   = (data.get("competitor_seq") or "")
        comp   = comp.upper().replace("T","U") if comp else ""

        if not mirna or not target:
            return jsonify({"target_attrib": [], "competitor_attrib": []}), 200

        def wobble_pair(a,b):
            return (a,b) in {("A","U"),("U","A"),("C","G"),("G","C"),("G","U"),("U","G")}
        def score_pos(seq, probe):
            m = 0
            k = min(len(probe), 7)
            for j in range(k):
                if wobble_pair(probe[j], seq[j]): m += 1
            return m

        compmap = {"A":"U","U":"A","G":"C","C":"G"}
        rc = "".join(compmap.get(ch,"N") for ch in mirna[::-1])

        targ_attr = []
        for i in range(len(target)):
            window = target[i:i+7]
            if len(window) < 7: window = window + "N"*(7-len(window))
            targ_attr.append(score_pos(window, rc))

        comp_attr = []
        if comp:
            for i in range(len(comp)):
                window = comp[i:i+7]
                if len(window) < 7: window = window + "N"*(7-len(window))
                comp_attr.append(score_pos(window, rc))

        def norm(a):
            if not a: return []
            mx = max(1, max(a))
            return [x / mx for x in a]

        return jsonify({
            "target_attrib": norm(targ_attr),
            "competitor_attrib": norm(comp_attr) if comp else []
        }), 200
    except Exception:
        return jsonify({"target_attrib": [], "competitor_attrib": []}), 200


# =========================
# CSV + Heatmap Downloads
# =========================

def _results_df(job_id: str) -> pd.DataFrame:
    job = jobs.get(job_id)
    if not job or job["status"] != "completed":
        return pd.DataFrame()
    df = pd.DataFrame(job["results"])
    preferred = [
        'interaction_id', 'timestamp_utc',
        'mirna_id', 'primary_molecule_id', 'target_id', 'competitor_id',
        'predicted_affinity_baseline', 'predicted_affinity_with_competitor',
        'competitive_effect (higher_is_better)',
        'seed_best_type','seed_best_start','seed_best_end','seed_best_wobble','seed_best_mismatches',
        'seed_hits_json',
        'primary_seq_used','target_seq_used','competitor_seq_used',
        'pdb_target_used','pdb_competitor_used','aa_to_nt_applied_target','aa_to_nt_mode_target','aa_to_nt_applied_competitor','aa_to_nt_mode_competitor','structure_features',
        'prov_model_path','prov_model_sha256','prov_scaler_path','prov_scaler_sha256',
        'prov_explain_method','prov_explain_steps','prov_seed_rules'
    ]
    rest = [c for c in df.columns if c not in preferred]
    return df[preferred + rest]


@app.route('/download/<job_id>/all.csv', methods=['GET'])
def download_all_csv(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Invalid job ID"}), 404
    if job["status"] != "completed":
        return jsonify({"error": "Job not completed yet"}), 400

    import io
    df = pd.DataFrame(job["results"])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=mirna_results_{job_id}.csv"}
    )



@app.route('/download/<job_id>/<interaction_id>.csv', methods=['GET'])
def download_single_csv(job_id, interaction_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Invalid job ID"}), 404
    if job["status"] != "completed":
        return jsonify({"error": "Job not completed yet"}), 400

    row = next((r for r in job["results"] if r.get('interaction_id') == interaction_id), None)
    if not row:
        return jsonify({"error": "Invalid interaction_id"}), 404

    df = pd.DataFrame([row])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    send_ga_event("download_single_csv", {"job_id": job_id})
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=interaction_{interaction_id}.csv"}
    )


def _explode_seed_hits(row: Dict, allow_gu: Optional[bool] = None, max_mm: Optional[int] = None) -> List[Dict]:
    out = []
    hits = []
    if allow_gu is None and max_mm is None:
        try:
            hits = json.loads(row.get('seed_hits_json') or "[]")
        except Exception:
            hits = []
    else:
        mseq = row.get('primary_seq_used') or ''
        tseq = row.get('target_seq_used') or ''
        hits = _scan_seeds_for_pair(mseq, tseq, allow_gu=bool(allow_gu), max_mismatch=int(max_mm or 0))

    for h in hits:
        out.append({
            "interaction_id": row.get("interaction_id"),
            "mirna_id": row.get("mirna_id"),
            "target_id": row.get("target_id"),
            "competitor_id": row.get("competitor_id",""),
            "seed_type": h.get("seed_type",""),
            "start": h.get("start",""),
            "end": h.get("end",""),
            "seed_len": h.get("seed_len",""),
            "mismatches": h.get("mismatches",""),
            "wobble": h.get("wobble",""),
            "upstream_base": h.get("upstream_base","")
        })
    return out


@app.route('/download/<job_id>/seeds_all.csv', methods=['GET'])
def download_seeds_all(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Invalid job ID"}), 404
    if job["status"] != "completed":
        return jsonify({"error": "Job not completed yet"}), 400

    allow_gu_q = request.args.get('allow_gu')
    max_mm_q = request.args.get('max_mm')
    allow_gu = None if allow_gu_q is None else (allow_gu_q.lower() == 'true')
    max_mm = None if max_mm_q is None else int(max_mm_q)

    rows = []
    for r in job["results"]:
        rows.extend(_explode_seed_hits(r, allow_gu, max_mm))
    if not rows:
        return jsonify({"error": "No seed hits available"}), 400

    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    send_ga_event("download_seeds_all", {"job_id": job_id, "rows": len(df)})
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=seed_hits_{job_id}.csv"}
    )


@app.route('/download/<job_id>/<interaction_id>/seeds.csv', methods=['GET'])
def download_seeds_one(job_id, interaction_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Invalid job ID"}), 404
    if job["status"] != "completed":
        return jsonify({"error": "Job not completed yet"}), 400

    row = next((r for r in job["results"] if r.get('interaction_id') == interaction_id), None)
    if not row:
        return jsonify({"error": "Invalid interaction_id"}), 404

    allow_gu_q = request.args.get('allow_gu')
    max_mm_q = request.args.get('max_mm')
    allow_gu = None if allow_gu_q is None else (allow_gu_q.lower() == 'true')
    max_mm = None if max_mm_q is None else int(max_mm_q)

    rows = _explode_seed_hits(row, allow_gu, max_mm)
    if not rows:
        return jsonify({"error": "No seed hits for this interaction"}), 400

    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    send_ga_event("download_seeds_one", {"job_id": job_id, "interaction_id": interaction_id})
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=seed_hits_{interaction_id}.csv"}
    )


def _build_ig_feed(pseq: str, tseq: str, cseq: str,
                   Lp: int, Lt: int, Lc: int,
                   include_struct: Dict[str, bool],
                   model_inputs: Optional[Dict[str, Tuple[Optional[int], ...]]] = None) -> Dict[str, np.ndarray]:
    model_inputs = model_inputs or _keras_inputs_map()
    pri_enc = one_hot_encode_sequence(pseq, Lp)[None, ...].astype(np.float32)
    tgt_enc = one_hot_encode_sequence(tseq, Lt)[None, ...].astype(np.float32)

    feed: Dict[str, np.ndarray] = {
        'primary_sequence_input': pri_enc,
        'target_sequence_input':  tgt_enc,
    }

    if 'competitor_sequence_input' in model_inputs:
        cmp_enc = one_hot_encode_sequence(cseq or '', Lc)[None, ...].astype(np.float32)
        feed['competitor_sequence_input'] = cmp_enc

    if 'numerical_features_input' in model_inputs:
        if hasattr(scaler, 'feature_names_in_'):
            z = np.zeros((1, len(scaler.feature_names_in_)), dtype=np.float32)
            scaled_num = scaler.transform(z).astype(np.float32)
        else:
            scaled_num = scaler.transform([[0.5, 0.0, 0.0]]).astype(np.float32)
        feed['numerical_features_input'] = scaled_num

    if include_struct.get('primary_structure_input', False):
        feed['primary_structure_input'] = np.zeros((1, Lp, 1), dtype=np.float32)
    if include_struct.get('target_structure_input', False):
        feed['target_structure_input'] = np.zeros((1, Lt, 1), dtype=np.float32)
    if include_struct.get('competitor_structure_input', False) and 'competitor_sequence_input' in model_inputs:
        feed['competitor_structure_input'] = np.zeros((1, Lc, 1), dtype=np.float32)

    return feed


@app.route('/download/<job_id>/<interaction_id>/heatmap.png', methods=['GET'])
def download_heatmap_png(job_id, interaction_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Invalid job ID"}), 404
    if job["status"] != "completed":
        return jsonify({"error": "Job not completed yet"}), 400

    row = next((r for r in job["results"] if r.get('interaction_id') == interaction_id), None)
    if not row:
        return jsonify({"error": "Invalid interaction_id"}), 404

    mode = (request.args.get('mode') or 'ig_target').lower().strip()
    steps = int(request.args.get('steps') or 50)

    # sequences used by the model
    pseq = row.get('primary_seq_used', '') or ''
    tseq = row.get('target_seq_used', '') or ''
    cseq = row.get('competitor_seq_used', '') or ''

    # 🔵 NEW: server-side 2D seed-matrix heatmap (premium blue)
    if mode == 'seed_matrix':
        if not pseq or not tseq:
            return jsonify({"error": "Missing miRNA or target sequence for seed_matrix"}), 400

        buf = render_seed_matrix_png(
            mirna_id=row.get('mirna_id') or row.get('primary_molecule_id') or "miRNA",
            target_id=row.get('target_id') or "target",
            mirna_seq=pseq,
            target_seq=tseq,
            dpi=200,
        )
        send_ga_event("download_heatmap_png", {
            "job_id": job_id,
            "interaction_id": interaction_id,
            "mode": mode
        })
        return send_file(
            buf,
            mimetype="image/png",
            as_attachment=True,
            download_name=f"{interaction_id}_seed_matrix.png"
        )

    # ----- existing IG + seed-density logic -----
    shapes = job.get("model_input_shapes", {})
    Lp = int(shapes.get('Lp', 120))
    Lt = int(shapes.get('Lt', 200))
    Lc = int(shapes.get('Lc', 200))

    model_inputs = _keras_inputs_map()
    include_struct = {
        'primary_structure_input': 'primary_structure_input' in model_inputs,
        'target_structure_input': 'target_structure_input' in model_inputs,
        'competitor_structure_input': 'competitor_structure_input' in model_inputs,
    }

    if mode.startswith('ig'):
        feed = _build_ig_feed(pseq, tseq, cseq, Lp, Lt, Lc, include_struct, model_inputs)
        if mode == 'ig_target':
            values = integrated_gradients(model, feed, 'target_sequence_input', steps=steps)
            title = f"IG (target) — {row.get('mirna_id')} vs {row.get('target_id')}"
            L = Lt
        elif mode == 'ig_competitor':
            if 'competitor_sequence_input' not in model_inputs:
                return jsonify({"error": "Model does not include a competitor input."}), 400
            values = integrated_gradients(model, feed, 'competitor_sequence_input', steps=steps)
            title = f"IG (competitor) — {row.get('mirna_id')} vs {row.get('competitor_id','')}"
            L = Lc
        else:
            return jsonify({"error": "Invalid mode. Use ig_target, ig_competitor, seed_density, or seed_matrix."}), 400

        data = np.array(values[:L], dtype=np.float32)[None, :]
        ytick = ['IG magnitude']

    elif mode == 'seed_density':
        hits = json.loads(row.get('seed_hits_json') or "[]")
        L = len(tseq)
        vec = np.zeros(L, dtype=np.float32)
        for h in hits:
            s = int(h['start']) - 1
            e = int(h['end'])
            vec[s:e] += 1.0
        if L == 0:
            vec = np.zeros(1, dtype=np.float32)
        data = vec[None, :]
        title = f"Seed-hit density — {row.get('mirna_id')} on {row.get('target_id')}"
        ytick = ['hit count']

    else:
        return jsonify({"error": "Invalid mode. Use ig_target, ig_competitor, seed_density, or seed_matrix."}), 400

    fig, ax = plt.subplots(figsize=(max(6, data.shape[1] / 20.0), 1.8))
    im = ax.imshow(data, aspect='auto')
    ax.set_yticks([0])
    ax.set_yticklabels(ytick)
    ax.set_xlabel('Position')
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png', dpi=200)
    plt.close(fig)
    buf.seek(0)

    send_ga_event("download_heatmap_png", {
        "job_id": job_id,
        "interaction_id": interaction_id,
        "mode": mode
    })
    return send_file(
        buf,
        mimetype="image/png",
        as_attachment=True,
        download_name=f"{interaction_id}_{mode}.png"
    )


@app.route('/download/<job_id>/<interaction_id>/bundle.zip', methods=['GET'])
def download_bundle_zip(job_id, interaction_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Invalid job ID"}), 404
    if job["status"] != "completed":
        return jsonify({"error": "Job not completed yet"}), 400

    row = next((r for r in job["results"] if r.get('interaction_id') == interaction_id), None)
    if not row:
        return jsonify({"error": "Invalid interaction_id"}), 404

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        df = pd.DataFrame([row])
        csv_io = io.StringIO(); df.to_csv(csv_io, index=False); csv_io.seek(0)
        zf.writestr(f"interaction_{interaction_id}.csv", csv_io.getvalue())

        seeds = _explode_seed_hits(row, allow_gu=None, max_mm=None)
        s_io = io.StringIO()
        (pd.DataFrame(seeds) if seeds else pd.DataFrame(columns=[
            "interaction_id","mirna_id","target_id","competitor_id","seed_type","start","end","seed_len","mismatches","wobble","upstream_base"
        ])).to_csv(s_io, index=False); s_io.seek(0)
        zf.writestr(f"seed_hits_{interaction_id}.csv", s_io.getvalue())

        try:
            zf.writestr(f"{interaction_id}_ig_target.png", _build_heatmap_bytes_for_row(job, row, 'ig_target', steps=50))
        except Exception:
            pass
        try:
            zf.writestr(f"{interaction_id}_ig_competitor.png", _build_heatmap_bytes_for_row(job, row, 'ig_competitor', steps=50))
        except Exception:
            pass
        try:
            zf.writestr(f"{interaction_id}_seed_density.png", _build_heatmap_bytes_for_row(job, row, 'seed_density', steps=50))
        except Exception:
            pass

    buf.seek(0)
    send_ga_event("download_bundle_zip", {"job_id": job_id, "interaction_id": interaction_id})
    return send_file(buf, mimetype='application/zip', as_attachment=True,
                     download_name=f"interaction_{interaction_id}.zip")


@app.route('/download/<job_id>/all.zip', methods=['GET'])
def download_all_zip(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Invalid job ID"}), 404
    if job["status"] != "completed":
        return jsonify({"error": "Job not completed yet"}), 400

    df = _results_df(job_id)
    if df.empty:
        return jsonify({"error": "No results available"}), 400

    rows = []
    for r in job["results"]:
        rows.extend(_explode_seed_hits(r, allow_gu=None, max_mm=None))
    seeds_df = pd.DataFrame(rows)

    results_json_path = job.get("results_json_path")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        csv_all = io.StringIO(); df.to_csv(csv_all, index=False); csv_all.seek(0)
        zf.writestr(f"mirna_results_{job_id}.csv", csv_all.getvalue())

        csv_seeds = io.StringIO(); seeds_df.to_csv(csv_seeds, index=False); csv_seeds.seek(0)
        zf.writestr(f"seed_hits_{job_id}.csv", csv_seeds.getvalue())

        if results_json_path and os.path.exists(results_json_path):
            with open(results_json_path, 'rb') as f:
                zf.writestr("results.json", f.read())
        else:
            zf.writestr("results.json", json.dumps({"results": job["results"]}, default=_to_py, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))

        prov = {
            "provenance": PROVENANCE,
            "config": {
                "mirna_max": MIRNA_MAX,
                "mature_trim_enabled": MATURE_TRIM_ENABLED,
                "mature_window": MATURE_TRIM_WINDOW,
                "aa_convert_allowed": AA_CONVERT_ALLOWED
            },
            "warnings": job.get("warnings", [])
        }
        zf.writestr("provenance.json", json.dumps(prov, indent=2).encode('utf-8'))

    buf.seek(0)
    send_ga_event("download_all_zip", {"job_id": job_id})
    return send_file(buf, mimetype='application/zip', as_attachment=True,
                     download_name=f"mirna_job_{job_id}_all.zip")


# =========================
# Public structure artifacts (3D viewer)
# =========================
@app.route('/structure/<job_id>/<kind>', methods=['GET'])
def get_structure_artifact(job_id, kind):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Invalid job ID"}), 404
    art = job.get("artifacts") or {}
    if time.time() > float(art.get("expiry", 0)):
        return jsonify({"error": "Artifacts expired"}), 410

    if kind not in {'target','competitor'}:
        return jsonify({"error": "Invalid kind"}), 400

    req_id = (request.args.get('id') or '').strip() or None
    path = _select_struct_path(art, kind, req_id)
    if not path:
        return jsonify({"error": "No artifact available"}), 404

    return send_file(path, as_attachment=True, download_name=os.path.basename(path))


@app.route('/structure/<job_id>/miRNA/<mirna_id>', methods=['GET'])
def get_structure_mirna(job_id, mirna_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Invalid job ID"}), 404
    art = job.get("artifacts") or {}
    if time.time() > float(art.get("expiry", 0)):
        return jsonify({"error": "Artifacts expired"}), 410
    idx = art.get('mirna_3d_index') or {}
    val = _lookup_3d(idx, mirna_id)
    if not val:
        return jsonify({"error": "No miRNA 3D file for this ID"}), 404
    _, __, p = val
    if not p or not os.path.exists(p):
        return jsonify({"error": "Artifact not found"}), 404
    return send_file(p, as_attachment=True, download_name=os.path.basename(p))


# =========================
# Contacts (3D overlay)
# =========================
def _compute_contacts(path_a: str, path_b: str, cutoff: float = 4.0) -> Dict:
    try:
        from Bio.PDB import PDBParser, MMCIFParser, NeighborSearch, Selection
    except Exception as e:
        return {"error": "Biopython with Bio.PDB is required on the server", "detail": str(e)}

    def _load_atoms(p):
        parser = PDBParser(QUIET=True) if p.lower().endswith(".pdb") else MMCIFParser(QUIET=True)
        struct = parser.get_structure("s", p)
        atoms = [a for a in Selection.unfold_entities(struct, 'A') if a.element.strip()]
        return struct, atoms

    def _is_base_atom(atom):
        name = atom.get_name().upper()
        return name in {"C2","C4","C5","C6","C8","N1","N3","N7","N9","C5M"}

    def _res_serial(atom):
        res = atom.get_parent()
        ch = res.get_parent()
        r = res.get_id()[1] if isinstance(res.get_id(), tuple) else str(res.get_id())
        return {"chain": getattr(ch, "id", "?"), "resname": res.get_resname(), "resid": int(r) if isinstance(r, int) else r}

    struct_a, atoms_a = _load_atoms(path_a)
    struct_b, atoms_b = _load_atoms(path_b)

    ns = NeighborSearch(atoms_a + atoms_b)
    pairs = ns.search_all(cutoff)

    out = []
    hbonds = 0
    salts = 0
    for x, y in pairs:
        in_a = x in atoms_a
        in_b = y in atoms_b
        if not (in_a and in_b) and not (y in atoms_a and x in atoms_b):
            continue
        a1, a2 = (x, y) if (in_a and in_b) else (y, x)
        d = (a1.coord - a2.coord)
        dist = float(np.sqrt(np.dot(d, d)))

        n1 = a1.element.upper()
        n2 = a2.element.upper()
        nm1 = a1.get_name().upper()
        nm2 = a2.get_name().upper()
        rn1 = a1.get_parent().get_resname().upper()
        rn2 = a2.get_parent().get_resname().upper()

        ctype = "close"
        if (n1 in {"N","O"} and n2 in {"N","O"} and dist <= 3.5):
            ctype = "hbond_like"; hbonds += 1
        basic = (rn1 in {"LYS","ARG","HIS"} and n1 == "N") or (rn2 in {"LYS","ARG","HIS"} and n2 == "N")
        phosphate = (rn1 in {"A","U","G","C","T"} and (nm1.startswith("OP") or nm1 in {"O1P","O2P","O3*","P"})) or \
                    (rn2 in {"A","U","G","C","T"} and (nm2.startswith("OP") or nm2 in {"O1P","O2P","O3*","P"}))
        if basic and phosphate and dist <= 4.0:
            ctype = "salt_bridge_like"; salts += 1

        out.append({
            "a": {**_res_serial(a1), "atom": nm1, "element": n1},
            "b": {**_res_serial(a2), "atom": nm2, "element": n2},
            "distance": round(dist, 3),
            "type": ctype
        })

    def _base_centroids(struct):
        cents = []
        for model in struct:
            for chain in model:
                for res in chain:
                    pts = [atom.coord for atom in res if _is_base_atom(atom)]
                    if len(pts) >= 4:
                        pts = np.array(pts, dtype=np.float32)
                        cents.append(pts.mean(axis=0))
        return cents

    cents_a = _base_centroids(struct_a)
    cents_b = _base_centroids(struct_b)
    pi_pairs = 0
    for ca in cents_a:
        for cb in cents_b:
            dist = float(np.linalg.norm(ca - cb))
            if dist <= 4.5:
                pi_pairs += 1

    return {
        "contacts": out,
        "summary": {
            "total": len(out),
            "hbond_like": hbonds,
            "salt_bridge_like": salts,
            "pi_stacking_like_pairs": pi_pairs
        }
    }


@app.route('/contacts/<job_id>/<mirna_id>', methods=['GET'])
def get_contacts(job_id, mirna_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Invalid job ID"}), 404
    art = job.get("artifacts") or {}
    if time.time() > float(art.get("expiry", 0)):
        return jsonify({"error": "Artifacts expired"}), 410

    idx = art.get('mirna_3d_index') or {}
    mval = _lookup_3d(idx, mirna_id)
    if not mval:
        return jsonify({"error": "No miRNA 3D available for this ID"}), 404
    _, __, mpath = mval
    if not mpath or not os.path.exists(mpath):
        return jsonify({"error": "miRNA artifact not found"}), 404

    target_kind = (request.args.get('with') or 'target').strip().lower()
    if target_kind not in {'target','competitor'}:
        return jsonify({"error": "Parameter 'with' must be target or competitor"}), 400

    if target_kind == 'target':
        tpath = art.get('target_3d_path')
    else:
        tpath = art.get('competitor_3d_path')

    if not tpath or not os.path.exists(tpath):
        return jsonify({"error": f"No {target_kind} 3D artifact available"}), 404

    try:
        cutoff = float(request.args.get('cutoff') or 4.0)
    except Exception:
        cutoff = 4.0

    result = _compute_contacts(mpath, tpath, cutoff=cutoff)
    if isinstance(result, dict) and result.get("error"):
        return jsonify(result), 500

    return jsonify(result)


# =========================
# Manifest (one-click run summary)
# =========================
@app.get("/manifest/<job_id>")
def get_manifest(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Invalid job ID"}), 404
    m = {
        "status": job.get("status"),
        "total": job.get("total"),
        "completed": job.get("completed"),
        "warnings": job.get("warnings", []),
        "model_input_shapes": job.get("model_input_shapes", {}),
        "provenance": PROVENANCE,
        "target_meta": job.get("target_meta", {}),
        "competitor_meta": job.get("competitor_meta", {}),
        "manifest": job.get("manifest", {})  # <-- add this line
    }
    return jsonify(m)



# =========================
# Janitor
# =========================
_janitor_started = False

def start_janitor():
    global _janitor_started
    if _janitor_started:
        return
    _janitor_started = True

    def _run():
        while True:
            try:
                now = time.time()
                for jid, job in list(jobs.items()):
                    art = job.get("artifacts") or {}
                    exp = float(art.get("expiry", 0))
                    if exp and now > exp:
                        for key in ('target_3d_path', 'competitor_3d_path'):
                            p = art.get(key)
                            if p and os.path.exists(p):
                                try: os.unlink(p)
                                except Exception: pass
                        try:
                            tidx = art.get('target_3d_index') or {}
                            for p in set(tidx.values()):
                                if p and os.path.exists(p):
                                    try: os.unlink(p)
                                    except Exception: pass
                        except Exception:
                            pass
                        try:
                            cidx = art.get('competitor_3d_index') or {}
                            for p in set(cidx.values()):
                                if p and os.path.exists(p):
                                    try: os.unlink(p)
                                    except Exception: pass
                        except Exception:
                            pass
                        idx = art.get('mirna_3d_index') or {}
                        for _, (_, __, p) in idx.items():
                            if p and os.path.exists(p):
                                try: os.unlink(p)
                                except Exception: pass
                        rjp = job.get("results_json_path")
                        if rjp and os.path.exists(rjp):
                            try: os.unlink(rjp)
                            except Exception: pass
                        jdir = job.get("job_dir")
                        try:
                            if jdir and os.path.isdir(jdir) and not os.listdir(jdir):
                                os.rmdir(jdir)
                        except Exception:
                            pass
                        job["artifacts"]["expiry"] = 0
                time.sleep(120)
            except Exception:
                time.sleep(120)

    threading.Thread(target=_run, daemon=True).start()


# =========================
# Seed-scan endpoint (public API)
# =========================
@app.route('/seed_scan', methods=['POST'])
@limiter.limit("30 per 15 minutes")
def seed_scan():
    try:
        data = request.get_json(force=True, silent=True) or {}
        mirna = (data.get('mirna_seq') or '').strip()
        targets = data.get('targets') or {}
        competitors = data.get('competitors') or {}
        allow_gu   = _to_bool(data.get('allow_gu'), True)
        max_mism   = int(data.get('max_mismatch') or 0)
        convert_aa = _to_bool(data.get('convert_aa_to_nt'), False)

        if not mirna or not targets:
            return jsonify({'error': 'Provide mirna_seq and at least one target'}), 400

        def _normalize_to_rna(seq: str) -> str:
            s = (seq or '').strip()
            if not s:
                return ''
            if is_aa_like(s):
                if AA_CONVERT_ALLOWED and convert_aa:
                    s = back_translate(s)
                else:
                    raise ValueError("Amino-acid sequence provided; enable AA→NT (lossy) to back-translate for seed scanning.")
            s = s.upper().replace('T','U')
            s = ''.join(ch for ch in s if ch in 'AUGCN')
            return s

        m = (mirna or '').upper().replace('T','U')

        norm_targets: Dict[str, str] = {}
        for sid, sseq in targets.items():
            try:
                norm_targets[sid] = _normalize_to_rna(sseq)
            except ValueError as ve:
                return jsonify({'error': f"Target '{sid}': {ve}"}), 400

        norm_comp: Dict[str, str] = {}
        for cid, cseq in competitors.items():
            try:
                norm_comp[cid] = _normalize_to_rna(cseq)
            except ValueError as ve:
                return jsonify({'error': f"Competitor '{cid}': {ve}"}), 400

        hits: List[Dict] = []

        seed_2_8 = m[1:8] if len(m) >= 8 else m[1:]
        seed_2_7 = m[1:7] if len(m) >= 7 else m[1:]
        seeds = [(seed_2_8, 7), (seed_2_7, 6)]
        seeds = [(s, L) for (s, L) in seeds if len(s) == L and L in (6,7)]

        def scan_one(label: str, seq_map: Dict[str, str]):
            for sid, t in seq_map.items():
                for seed, L in seeds:
                    if len(seed) < L:
                        continue
                    seed_rc = revcomp_rna(seed)
                    max_i = max(0, len(t) - len(seed_rc) + 1)
                    for i in range(0, max_i):
                        w = t[i:i+len(seed_rc)]
                        score = match_seed(seed_rc, w, allow_gu=allow_gu, max_mismatch=max_mism)
                        if score is None:
                            continue
                        stype = classify_seed(m, t, i, L)
                        hit = {
                            'molecule': label,
                            'id': sid,
                            'start': i + 1,
                            'end': i + len(seed_rc),
                            'seed_len': len(seed_rc),
                            'seed_type': stype,
                            **score
                        }
                        if i - 1 >= 0:
                            hit['upstream_base'] = t[i-1]
                        hits.append(hit)

        scan_one('target', norm_targets)
        scan_one('competitor', norm_comp)
        return jsonify({'hits': hits})
    except Exception as e:
        logging.exception(f"/seed_scan error: {e}")
        return jsonify({'error': str(e)}), 500


# =========================
# Explain (Integrated Gradients) endpoint (public API)
# =========================
@app.route('/explain', methods=['POST'])
@limiter.limit("20 per 15 minutes")
def explain():
    try:
        if model is None or scaler is None:
            return jsonify({"error": "Model or scaler not loaded on server."}), 500

        data = request.get_json(force=True, silent=True) or {}
        mirna = (data.get('mirna_seq') or '').strip()
        target = (data.get('target_seq') or '').strip()
        competitor = (data.get('competitor_seq') or '').strip()

        if not mirna or not target:
            return jsonify({'error': 'Provide mirna_seq and target_seq'}), 400

        model_inputs = _keras_inputs_map()
        Lp = int((model_inputs.get('primary_sequence_input') or (None,120))[1] or 120)
        Lt = int((model_inputs.get('target_sequence_input')  or (None,200))[1] or 200)
        Lc = int((model_inputs.get('competitor_sequence_input') or (None,200))[1] or 200)

        has_comp_input = 'competitor_sequence_input' in model_inputs
        has_num_input  = 'numerical_features_input' in model_inputs
        has_p_struct   = 'primary_structure_input' in model_inputs
        has_t_struct   = 'target_structure_input' in model_inputs
        has_c_struct   = 'competitor_structure_input' in model_inputs

        def ensure_dict(data):
            if isinstance(data, tuple):
                return {
                    "sequence": data[1] if len(data) > 1 else "",
                    "gc_content": 0.5, "dg": 0.0, "conservation": 0.0,
                    "structure_vector": "[]", "adjacency_matrix": "[]"
                }
            return data

        pdat = ensure_dict(process_molecule_universal((("miRNA", mirna), {}, 'primary_molecule')))
        tdat = ensure_dict(process_molecule_universal((("target", target), {}, 'target_molecule')))
        if competitor:
            cdat = ensure_dict(process_molecule_universal((("competitor", competitor), {}, 'competitor_molecule')))
        else:
            cdat = {'sequence': ''}

        pseq = pdat.get('sequence','')
        if MATURE_TRIM_ENABLED and len(pseq) > 30:
            pseq = choose_mature_window(pseq, window=MATURE_TRIM_WINDOW)

        pri_enc = one_hot_encode_sequence(pseq, Lp)[None, ...].astype(np.float32)
        tgt_enc = one_hot_encode_sequence(tdat.get('sequence',''), Lt)[None, ...].astype(np.float32)
        if has_comp_input:
            if competitor:
                cmp_enc = one_hot_encode_sequence(cdat.get('sequence',''), Lc)[None, ...].astype(np.float32)
            else:
                cmp_enc = one_hot_encode_sequence('', Lc)[None, ...].astype(np.float32)

        feed: Dict[str, np.ndarray] = {
            'primary_sequence_input': pri_enc,
            'target_sequence_input':  tgt_enc,
        }
        if has_comp_input:
            feed['competitor_sequence_input'] = cmp_enc
        if has_num_input:
            num_list = [numerical_features_from_processed_json(pdat)]
            if hasattr(scaler, 'feature_names_in_'):
                df_features = pd.DataFrame(num_list, columns=scaler.feature_names_in_)
                scaled_num = scaler.transform(df_features)
            else:
                scaled_num = scaler.transform(num_list)
            feed['numerical_features_input'] = scaled_num.astype(np.float32)

        if has_p_struct:
            feed['primary_structure_input'] = structure_vector_from_processed_json(pdat.get('structure_vector','[]'), Lp)[None, ...].astype(np.float32)
        if has_t_struct:
            feed['target_structure_input'] = structure_vector_from_processed_json(tdat.get('structure_vector','[]'), Lt)[None, ...].astype(np.float32)
        if has_c_struct and has_comp_input:
            if competitor:
                feed['competitor_structure_input'] = structure_vector_from_processed_json(cdat.get('structure_vector','[]'), Lc)[None, ...].astype(np.float32)
            else:
                feed['competitor_structure_input'] = np.zeros((1, Lc, 1), dtype=np.float32)

        tgt_attr = integrated_gradients(model, feed, 'target_sequence_input', steps=50)
        cmp_attr = integrated_gradients(model, feed, 'competitor_sequence_input', steps=50) if (has_comp_input and competitor) else []

        return jsonify({
            'target_attrib': tgt_attr,
            'competitor_attrib': cmp_attr
        })
    except Exception as e:
        logging.exception(f"/explain error: {e}")
        return jsonify({'error': str(e)}), 500


# =========================
# Health & Startup
# =========================
@app.get("/healthz")
def healthz():
    ok = (model is not None) and (scaler is not None)
    return jsonify({
        "ok": ok,
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "provenance": PROVENANCE,
        "active_jobs": len(jobs)
    }), (200 if ok else 503)


def main():
    port = int(os.environ.get("PORT", 8080))
    try:
        start_janitor()
    except Exception as e:
        app.logger.warning("Janitor failed to start: %s", e)
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False, threaded=True)


if __name__ == '__main__':
    main()