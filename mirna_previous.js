// mirna.js — upgraded & sync’d with multi-target/competitor backend (+ seed-scan + IG heatmap)
// Supreme edition: tolerant header matching, range-aware coords, premium-sized buttons, seed CSV export,
// persistent analysis controls, sortable table, safe bindings, graceful fallbacks, server CSV/PNG downloads,
// progress stall detector, 3D viewer, and precise UX around optional/visualization-only PDB usage.
// + 2025-11-11: multi-PDB upload (target/competitor), per-row bundle download, global “Download All”,
//   perfectly leveled Heatmap controls, and AA→NT before any analysis (no AA letters in alignments).
// + 2025-11-12: Accept PDB IDs in FASTA headers and pass to backend so jobs start with IDs or files.
// + 2025-11-12 (this build):
//   • PDB is optional/non-blocking; PDB-only and FASTA-only supported; protein→AA→NT auto (mode selectable)
//   • Staging basket for target/competitor structures (pick multiple times; accumulate chips; remove)
//   • Pre-validation “Use/Skip” table (best-effort; graceful if /precheck is unavailable)
//   • Tolerant chain hints in FASTA header (e.g., >TP53_3UTR|chain=A)
//   • Result badges (“PDB used”, “AA→NT mode”, “Structure-features on/off”) and “Run Manifest” download
//   • Progress errors with PDB mismatches try to fetch & render results with warnings instead of hard fail

// =====================================================
// Global state
// =====================================================
let predictionResults = [];
let CURRENT_JOB_ID = null; // track backend job for server CSV & heatmap & structures
let CONFIG = {
  mirna_max: 5000,
  mature_trim_enabled: true,
  mature_window: 22,
  aa_convert_allowed: true,
  use_nonce: false
};

// Store the exact inputs used at submit time so downstream analysis matches predictions
const CURRENT_INPUTS = {
  mirnas: {},      // id -> sequence (as typed)
  targets: {},     // id -> sequence (as typed)
  competitors: {}  // id -> sequence (as typed)
};

// Staged 3D files accumulated across multiple selections
const STAGED = {
  target3dFiles: [],      // File[]
  competitor3dFiles: []   // File[]
};

// Last analysis cache (for exports)
let LAST_SEED_HITS = null;   // Array of hits
let LAST_SEED_META = null;   // { mirnaId, targetId, compId }

// Run manifest (augmented as the run proceeds)
let RUN_MANIFEST = null;

// Guards to prevent duplicate injections/bindings
const GUARDS = {
  advancedInjected: false,
  fastaTipsInjected: false,
  tabWiringDone: false,
  formBindingDone: false,
  analysisControlsInjected: false,
  modalInjected: false,
  styleInjected: false,
  nglLoaded: false,
  threeDToolbarInjected: false,
  stagingDone: false
};

// =====================================================
// API routing and auth
// =====================================================
const LOCAL_BASE = "http://127.0.0.1:8080";
const PROD_BASE  = "https://mirna.marzan.org";
const isLocal =
  window.location.hostname === "localhost" ||
  window.location.hostname === "127.0.0.1";
const BASE_URL = isLocal ? LOCAL_BASE : PROD_BASE;

const API_URL        = `${BASE_URL}/predict`;
const PRECHECK_URL   = `${BASE_URL}/precheck`;
const PROGRESS_URL   = (jobId) => `${BASE_URL}/progress/${jobId}`;
const DOWNLOAD_URL   = (jobId) => `${BASE_URL}/download/${jobId}`;            // JSON (unprotected)
const DOWNLOAD_ALL_CSV_URL = (jobId) => `${BASE_URL}/download/${jobId}/all.csv`;
const DOWNLOAD_ROW_CSV_URL = (jobId, interactionId) => `${BASE_URL}/download/${jobId}/${interactionId}.csv`;
const HEATMAP_PNG_URL = (jobId, interactionId, mode, steps) => `${BASE_URL}/download/${jobId}/${interactionId}/heatmap.png?mode=${encodeURIComponent(mode)}&steps=${encodeURIComponent(steps)}`;
const STRUCTURE_URL   = (jobId, kind) => `${BASE_URL}/structure/${jobId}/${kind}`; // kind: target|competitor

const NONCE_URL      = `${BASE_URL}/nonce`;
const CONFIG_URL     = `${BASE_URL}/config`;
const SEED_SCAN_URL  = `${BASE_URL}/seed_scan`;
const EXPLAIN_URL    = `${BASE_URL}/explain`;

const MAX_FILE_SIZE_MB = 100;

// =====================================================
// Helpers: DOM, UI, utils
// =====================================================
function $(id){ return document.getElementById(id); }
function byQS(sel, scope=document){ return scope.querySelector(sel); }
function byQSA(sel, scope=document){ return Array.from(scope.querySelectorAll(sel)); }
function setHTML(el, html){ if(el) el.innerHTML = html; }
function appendHTML(el, html){ if(el) el.insertAdjacentHTML('beforeend', html); }
function prependHTML(el, html){ if(el) el.insertAdjacentHTML('afterbegin', html); }
function show(el){ if(el) el.classList.remove('hidden'); }
function hide(el){ if(el) el.classList.add('hidden'); }
function text(el, t){ if(el) el.textContent = t; }

function safeParseFloat(x, d=0){
  const v = parseFloat(x);
  return Number.isFinite(v) ? v : d;
}

function escapeHTML(s){
  return String(s || '')
    .replace(/&/g,"&amp;")
    .replace(/</g,"&lt;")
    .replace(/>/g,"&gt;");
}

function formatError(msg){
  return `<p style="color:#c22;margin:8px 0;">${escapeHTML(msg)}</p>`;
}
function formatWarn(msg){
  return `<p style="color:#b36b00;margin:8px 0;">${escapeHTML(msg)}</p>`;
}
// A) Centered info note (CSS styles it)
function formatInfo(msg){
  return `<p class="info-note">${escapeHTML(msg)}</p>`;
}

// Plain-text modal (no HTML tags rendered)
function openModalText(title, textMessage, toolbarHTML=''){
  ensureModal();
  const modal = $('analysis-modal');
  if(!modal) return;
  text($('modal-title'), title || 'Message');
  const mc = $('modal-content');
  mc.textContent = textMessage || '';
  setHTML($('modal-tools'), toolbarHTML || '');
  modal.style.display = 'flex';
}

function validateFileSize(file){
  if(file && file.size > MAX_FILE_SIZE_MB * 1024 * 1024){
    alert(`File "${file.name}" exceeds ${MAX_FILE_SIZE_MB} MB limit.`);
    return false;
  }
  return true;
}

function bindFileToTextarea(fileInputId, textareaId){
  const fileInput = $(fileInputId);
  const textarea  = $(textareaId);
  if(!fileInput || !textarea) return;

  // Replace input to drop old listeners (prevents duplicates)
  const clone = fileInput.cloneNode(true);
  fileInput.parentNode.replaceChild(clone, fileInput);

  clone.addEventListener('change', function(){
    const file = this.files && this.files[0];
    if(!file) return;
    const reader = new FileReader();
    reader.onload = (e) => { textarea.value = e.target.result; };
    reader.readAsText(file);
  });
}

function countFastaRecords(seqText){
  if(!seqText) return 0;
  const lines = seqText.trim().split(/\r?\n/);
  let count = 0;
  for(const line of lines){ if(line.trim().startsWith('>')) count++; }
  if(count === 0 && seqText.trim().length > 0) count = 1;
  return count;
}

function hasFastaHeaders(text){
  if(!text || !text.trim()) return false;
  return text.split(/\r?\n/).some(line => line.trim().startsWith('>'));
}

function ensureSingleton(id, html, parent){
  if(!parent) return null;
  let el = $(id);
  if(el) return el;
  const holder = document.createElement('div');
  holder.innerHTML = html.trim();
  const created = holder.firstElementChild;
  if(created) parent.appendChild(created);
  return created;
}

// Inject premium styles and small utility classes (kept inline to be drop-in)
function injectPremiumStyles(){
  if(GUARDS.styleInjected) return;
  const css = `
    .btn-premium{padding:10px 14px;min-height:42px;min-width:130px;border-radius:12px;border:1px solid #d9d9e3;background:linear-gradient(180deg,#ffffff,#f6f7fb);
      font-weight:600;letter-spacing:.2px;box-shadow:0 1px 1px rgba(0,0,0,.04), 0 8px 20px rgba(17,24,39,.06);transition:.15s transform ease,.2s box-shadow ease;}
    .btn-premium:hover{transform:translateY(-1px);box-shadow:0 10px 24px rgba(17,24,39,.09);}
    .btn-action{min-width:128px;min-height:40px;padding:9px 12px;border-radius:10px;font-weight:600;border:1px solid #d8dee9;background:linear-gradient(180deg,#fff,#f8fafc);}
    .btn-accent{background:#0ea5e9;color:#fff;border:1px solid #0284c7;}
    .chip{display:inline-block;padding:2px 8px;border:1px solid #e5e7eb;border-radius:10px;background:#f8fafc;color:#334155;font-size:12px;margin-left:6px;}
    table#results-table thead th{position:sticky;top:0;background:#fff;z-index:1}
    table#results-table tbody tr:hover{filter:brightness(0.98)}
    .toolbar-btn{min-height:32px;padding:6px 10px;border-radius:8px;border:1px solid #d8dee9;background:#fff;font-weight:600}

    /* Center the friendly info + the no-refresh warning */
    .info-note{ text-align:center; margin:8px 0; }
    .reload-warning{ text-align:center; margin:8px 0; }

    /* B) perfectly leveled analysis controls (grid) */
    .controls-grid{
      display:grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, max-content));
      gap:12px;
      align-items:center;
      justify-content:center;
      margin:8px 0 12px;
    }
    .controls-grid .ctrl{
      display:flex;
      align-items:center;
      gap:8px;
      justify-content:center;
      white-space:nowrap;
    }
    .controls-grid .ctrl span{
      font-weight:700;
      line-height:40px;
    }
    .controls-grid .ctrl input[type="number"],
    .controls-grid .ctrl select{
      padding:6px 10px;
      border:1px solid #d8dee9;
      border-radius:8px;
      min-height:40px;
      height:40px;
      line-height:40px;
    }
    .controls-grid .ctrl input[type="checkbox"]{ transform: translateY(1px); }

    /* D) two-line action cluster so nothing gets cut off */
    .action-grid{
      display:grid;
      grid-template-columns: repeat(3, minmax(120px, 1fr));
      gap:8px;
    }
    .action-grid .btn-action{ width:100%; }
    .action-grid .action-spacer{ display:block; }

    /* Staging baskets */
    .staging-box{border:1px dashed #cbd5e1;border-radius:10px;padding:10px;margin-top:8px;background:#fafbff;}
    .staging-head{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:6px;}
    .chipfile{display:inline-flex;align-items:center;gap:6px;padding:4px 8px;border-radius:9999px;border:1px solid #e2e8f0;background:#fff;margin:4px 6px 0 0;font-size:12px;}
    .chipfile button{border:none;background:transparent;cursor:pointer;font-weight:700;}
    .badge{display:inline-block;border-radius:9999px;padding:2px 8px;font-size:11px;border:1px solid #e5e7eb;background:#f8fafc;margin:2px 4px;color:#334155;}
    .badge.ok{background:#ecfdf5;border-color:#bbf7d0;color:#065f46;}
    .badge.warn{background:#fff7ed;border-color:#fed7aa;color:#7c2d12;}
    .badge.off{background:#f1f5f9;border-color:#cbd5e1;color:#475569;}
    .precheck-table{width:100%;border-collapse:collapse;margin:6px 0;}
    .precheck-table th,.precheck-table td{border-bottom:1px solid #e5e7eb;padding:6px 8px;text-align:left;font-size:13px;}
  `;
  const style = document.createElement('style');
  style.id = 'mirna-js-style';
  style.textContent = css;
  document.head.appendChild(style);
  GUARDS.styleInjected = true;
}

// Small spinner HTML for modal bodies
function smallSpinner(text='Working...'){
  return `<div style="text-align:center;padding:10px 0;">
    <span class="loader-spinner"></span>
    <span style="vertical-align:middle;">${escapeHTML(text)}</span>
  </div>`;
}

// --- fetch with AbortController timeout (works for GET/POST) ---
function fetchWithTimeout(url, options={}, ms=30000){
  const ac = new AbortController();
  const timer = setTimeout(()=>ac.abort(), ms);
  return fetch(url, { ...options, signal: ac.signal })
    .finally(()=>clearTimeout(timer));
}

// Simple FASTA parser → { id: seq, ... } (if no headers: {"<prefix>_1": raw})
// Preserves FULL header text after ">"
function parseFastaToMap(text, defaultPrefix='seq'){
  const map = {};
  if(!text || !text.trim()){
    return map;
  }
  const hasHeader = hasFastaHeaders(text);
  if(!hasHeader){
    map[`${defaultPrefix}_1`] = text.replace(/^>.*$/gm,'').replace(/\s+/g,'').toUpperCase();
    return map;
  }
  let curId = null;
  let curSeq = [];
  const lines = text.split(/\r?\n/);
  for(const ln of lines){
    if(ln.trim().startsWith('>')){
      if(curId){
        map[curId] = (curSeq.join('')).toUpperCase();
      }
      // keep entire header after '>'
      curId = ln.replace(/^>/,'').trim() || `${defaultPrefix}_${Object.keys(map).length+1}`;
      curSeq = [];
    }else{
      curSeq.push(ln.trim());
    }
  }
  if(curId){
    map[curId] = (curSeq.join('')).toUpperCase();
  }
  return map;
}

// ===== Tolerant ID helpers (mirror backend tolerant matching) =====
function idVariants(s){
  if(!s) return [];
  const t = String(s).trim();
  const slug = t.toLowerCase().replace(/\s+/g,'_').replace(/[^a-z0-9_\-\.]/g,'');
  const set = new Set([
    t,
    t.replace(/\s+/g,'_'),
    t.replace(/\s+/g,''),
    slug,
    slug.replace(/_/g,' '),
    t.toLowerCase(),
    t.replace(/\s+/g,'_').toLowerCase(),
    t.replace(/\s+/g,'').toLowerCase()
  ]);
  return Array.from(set);
}

function lookupTolerant(pool, key){
  if(!pool || !key) return undefined;
  if(Object.prototype.hasOwnProperty.call(pool, key)) return pool[key];
  for(const v of idVariants(key)){
    if(Object.prototype.hasOwnProperty.call(pool, v)) return pool[v];
  }
  return undefined;
}

// Parse an ID like "TP53:90-150" → {baseId, start, end} (1-based)
function parseIdRange(id){
  const m = String(id||'').match(/^(.+):(\d+)-(\d+)$/);
  if(!m) return null;
  return { baseId: m[1], start: parseInt(m[2],10), end: parseInt(m[3],10) };
}

// Get (possibly sliced) sequence for ID with tolerant baseId lookup
function tolerantGetAnySeqForId(anyId, pool){
  const r = parseIdRange(anyId);
  if(!r){
    const exact = lookupTolerant(pool, anyId);
    return typeof exact === 'string' ? exact : '';
  }
  const base = lookupTolerant(pool, r.baseId) || '';
  if(!base) return '';
  const sIdx = Math.max(0, r.start - 1);
  const eIdx = Math.min(base.length, r.end);
  return base.slice(sIdx, eIdx);
}

// For displaying global coords when a :start-end slice is used
function globalCoordForId(anyId, localStart, localEnd){
  const r = parseIdRange(anyId);
  if(!r) return null; // no global translation needed
  const offset = (r.start || 1) - 1; // 0-based offset
  return { globalStart: offset + localStart, globalEnd: offset + localEnd };
}

// Quick check: exact key (respecting baseId for ranged IDs)
function exactKeyExists(pool, anyId){
  if(!pool || !anyId) return false;
  const r = parseIdRange(anyId);
  const k = r ? r.baseId : anyId;
  return Object.prototype.hasOwnProperty.call(pool, k);
}

// =====================================================
// NEW: AA ↔ NT helpers
// =====================================================
const NUCLEOTIDE_CHARS = new Set(['A','C','G','U','T','N','R','Y','K','M','S','W','B','D','H','V']);
function isLikelyAA(seq){
  if(!seq) return false;
  const s = String(seq).replace(/[\s\-]/g,'').toUpperCase();
  if(!s) return false;
  // If every char is nucleotide-ish, it's NT; otherwise likely AA
  return /[^ACGTUNRYKMSWBVDH]/.test(s);
}
function toRNA(seq){
  return String(seq||'').toUpperCase().replace(/T/g,'U').replace(/[^ACGU]/g, (ch)=>{
    return NUCLEOTIDE_CHARS.has(ch) ? ch : '';
  });
}
// Canonical RNA codon picks per amino acid (lossy but stable and deterministic)
const AA2RNA_CANON = {
  A:'GCU', R:'CGU', N:'AAU', D:'GAU', C:'UGU',
  Q:'CAA', E:'GAA', G:'GGU', H:'CAU', I:'AUU',
  L:'UUA', K:'AAA', M:'AUG', F:'UUU', P:'CCU',
  S:'UCU', T:'ACU', W:'UGG', Y:'UAU', V:'GUU',
  U:'UGA', O:'UAG', B:'AAN', Z:'CAN', X:'NNN', '*':'NNN'
};
// GC-balanced (rough heuristic)
const AA2RNA_GC = {
  A:'GCC', R:'CGC', N:'AAC', D:'GAC', C:'UGC',
  Q:'CAG', E:'GAG', G:'GGC', H:'CAC', I:'AUC',
  L:'CUG', K:'AAG', M:'AUG', F:'UUC', P:'CCC',
  S:'UCC', T:'ACC', W:'UGG', Y:'UAC', V:'GUG',
  U:'UGA', O:'UAG', B:'AAN', Z:'CAN', X:'NNN', '*':'NNN'
};
// NNK degenerate (keeps codon length; visualization-safe)
const AA2RNA_NNK = {
  A:'NNK', R:'NNK', N:'NNK', D:'NNK', C:'NNK',
  Q:'NNK', E:'NNK', G:'NNK', H:'NNK', I:'NNK',
  L:'NNK', K:'NNK', M:'AUG', F:'NNK', P:'NNK',
  S:'NNK', T:'NNK', W:'UGG', Y:'NNK', V:'NNK',
  U:'UGA', O:'UAG', B:'NNK', Z:'NNK', X:'NNK', '*':'NNN'
};
function aaToRNAWithMode(aaSeq, mode='canonical'){
  const s = String(aaSeq||'').replace(/\s+/g,'').toUpperCase();
  const table = mode === 'gc_balanced' ? AA2RNA_GC : (mode === 'nnk' ? AA2RNA_NNK : AA2RNA_CANON);
  let out = '';
  for(const ch of s){
    if(table[ch]) out += table[ch];
    else if(NUCLEOTIDE_CHARS.has(ch)) out += ch; // if already NT-ish, keep
    else out += 'NNN';
  }
  return out;
}
/**
 * Resolve a sequence by ID from a pool, slicing if :start-end, and
 * converting AA→RNA when:
 *  - the sequence looks like amino acids, AND
 *  - server allows AA conversion, AND
 *  - the user toggled the AA conversion flag ON.
 * Returns {seq, converted:boolean, note:string, mode:string}
 */
function resolveSeqWithAAHandling(anyId, pool){
  const raw = tolerantGetAnySeqForId(anyId, pool);
  if(!raw) return { seq:'', converted:false, note:'', mode:'' };

  const uiFlag = $('aa-convert-flag')?.checked ?? CONFIG.aa_convert_allowed;
  const mode = (byQS('#aa-nt-mode')?.value || 'canonical').toLowerCase();
  const canConvert = CONFIG.aa_convert_allowed && uiFlag;

  if(isLikelyAA(raw)){
    if(!canConvert){
      return { seq:'', converted:false, note:'Target appears to be amino acids; enable AA→NT conversion or supply nucleotides.', mode:'' };
    }
    const nt = aaToRNAWithMode(raw, mode);
    return { seq: toRNA(nt), converted:true, note:`AA→NT conversion applied (${mode}).`, mode };
  }
  return { seq: toRNA(raw), converted:false, note:'', mode:'' };
}

// =====================================================
// Config loader
// =====================================================
async function loadConfig(){
  try{
    const res = await fetch(CONFIG_URL, { method:'GET' });
    if(res.ok){
      const cfg = await res.json();
      CONFIG = { ...CONFIG, ...cfg };
    }
  }catch(_){ /* keep defaults */ }
}

// =====================================================
// Nonce (optional; graceful when disabled on server)
// =====================================================
async function getNonceOrKeyHeaders() {
  const h = {};
  try {
    if (CONFIG && CONFIG.use_nonce) {
      const r = await fetch(NONCE_URL, { method: 'GET', cache: 'no-store' });
      if (r.ok) {
        const j = await r.json();
        if (j && j.nonce) h['X-Nonce'] = j.nonce;
      } else {
        console.warn('Nonce fetch failed:', r.status);
      }
    } else if (CONFIG && CONFIG.api_key) {
      h['X-API-KEY'] = CONFIG.api_key;
    }
  } catch (e) {
    console.warn('Auth header setup warning:', e);
  }
  return h;
}

// =====================================================
// Safe event binding (prevent duplicates)
// =====================================================
function bindOnce(el, event, handler, key){
  if(!el) return;
  const k = key || `${event}__bound`;
  if(el.dataset && el.dataset[k] === '1') return;
  el.addEventListener(event, handler);
  if(el.dataset) el.dataset[k] = '1';
}

// =====================================================
// Initialization
// =====================================================
document.addEventListener('DOMContentLoaded', async () => {
  injectPremiumStyles();
  await loadConfig();
  ensureModal(); // make sure modal exists early
  syncStickyOffset(); // keep sticky headers perfect
  window.addEventListener('resize', syncStickyOffset, { passive:true });

  const loader = $('loader');
  if(loader){
    text(loader, "Please input your sequences to start a prediction.");
    show(loader);
  }

  // Link file pickers → textareas
  bindFileToTextarea('mirna-seq-file', 'primary-seqs');
  bindFileToTextarea('target-seq-file', 'target-seq');
  bindFileToTextarea('competitor-seq-file', 'competitor-seq');

  // Build staging baskets for 3D files
  setupStagingBaskets();

  // Form submit
  const form = $('prediction-form');
  if(form && !GUARDS.formBindingDone){
    bindOnce(form, 'submit', handleSubmit, 'submitGuard');
    GUARDS.formBindingDone = true;
  }

  // Make primary action buttons look premium if present
  ['load-sample-btn','clear-btn','clear-inputs-btn','seed-scan-global-btn','explain-global-btn'].forEach(id=>{
    const el = $(id);
    if(el) el.classList.add('btn-premium');
  });

  // Advanced options + tabs
  injectAdvancedOnce();
  wireTabButtonsOnce();
});

// =====================================================
// Advanced options injection (singleton)
// =====================================================
function injectAdvancedOnce(){
  const advTab = byQS('#advanced-tab');
  if(!advTab || GUARDS.advancedInjected) return;

  // Server configuration badge
  const serverCfgId = 'server-config-note';
  ensureSingleton(
    serverCfgId,
    `
    <div id="${serverCfgId}" style="margin:8px 0;color:#333;">
      <strong>Server configuration:</strong>
      <ul style="margin:6px 0 0 16px;">
        <li>Max miRNAs per request: <code>${CONFIG.mirna_max}</code></li>
        <li>Mature trimming enabled: <code>${CONFIG.mature_trim_enabled ? 'yes' : 'no'}</code> (window: ${CONFIG.mature_window})</li>
        <li>AA→NT conversion allowed: <code>${CONFIG.aa_convert_allowed ? 'yes' : 'no'}</code></li>
        <li>Auth mode: <code>${CONFIG.use_nonce ? 'nonce' : 'open'}</code></li>
      </ul>
    </div>
    `,
    advTab
  );

  // Flags + AA→NT mode
  const flagsWrapperId = 'advanced-flags-wrapper';
  ensureSingleton(
    flagsWrapperId,
    `
    <div id="${flagsWrapperId}" style="margin-top:8px;">
      <div style="display:flex;gap:20px;align-items:center;flex-wrap:wrap;">
        <label style="display:flex;gap:8px;align-items:center;cursor:pointer;">
          <input type="checkbox" id="mature-trim-flag" ${CONFIG.mature_trim_enabled ? 'checked' : ''}/>
          <span>Auto-trim miRNAs &gt; 30nt to mature-like ${CONFIG.mature_window}nt</span>
        </label>
        <label style="display:flex;gap:8px;align-items:center;cursor:pointer;">
          <input type="checkbox" id="aa-convert-flag" ${CONFIG.aa_convert_allowed ? 'checked' : 'disabled'}/>
          <span>Convert protein AA → NT for targets/competitors (lossy)</span>
        </label>
        <label style="display:flex;gap:8px;align-items:center;">
          <span>AA→NT mode</span>
          <select id="aa-nt-mode" ${CONFIG.aa_convert_allowed ? '' : 'disabled'}>
            <option value="canonical" selected>Most-common human codon</option>
            <option value="gc_balanced">GC-balanced</option>
            <option value="nnk">NNK (degenerate)</option>
          </select>
        </label>
      </div>
      <small style="color:#555;">PDB is optional and never blocks scoring. If a PDB is protein-only, we’ll auto back-translate for scanning/visualization.</small>
    </div>
    `,
    advTab
  );

  // Respect server toggle
  const aaFlag = $('aa-convert-flag');
  if(aaFlag && !CONFIG.aa_convert_allowed){
    aaFlag.disabled = true;
    aaFlag.checked = false;
    const modeSel = $('aa-nt-mode');
    if(modeSel){ modeSel.disabled = true; }
  }

  GUARDS.advancedInjected = true;
}

// =====================================================
// Utility: extract PDB IDs and chain hints from FASTA headers
// Accepts patterns like "PDB:7YTW", "rcsb=2ABC", "pdb 1XYZ_A"
// Chain hint: '>TP53_3UTR|chain=A' (also tolerantly reads 'chain = A')
// =====================================================
function extractPdbIdsFromFasta(text){
  const ids = [];
  if(!text) return ids;
  const lines = text.split(/\r?\n/);
  for(const ln of lines){
    if(!ln.trim().startsWith('>')) continue;
    const header = ln.slice(1);
    const re = /(pdb|rcsb)\s*[:=]\s*([0-9][A-Za-z0-9]{3})(?:[_\-\s]*([A-Za-z0-9]))?/gi;
    let m;
    while((m = re.exec(header)) !== null){
      const code = (m[2] || '').toUpperCase();
      const chain = (m[3] || '').toUpperCase();
      ids.push(chain ? `${code}_${chain}` : code);
    }
  }
  return Array.from(new Set(ids));
}
function extractChainHintsFromFasta(text){
  const hints = {};
  if(!text) return hints;
  const lines = text.split(/\r?\n/);
  for(const ln of lines){
    if(!ln.trim().startsWith('>')) continue;
    const header = ln.slice(1).trim();
    const id = header.replace(/\|.*$/,'').trim(); // before first pipe
    const m = header.match(/\bchain\b\s*=\s*([A-Za-z0-9])/i);
    if(m && id){ hints[id] = m[1].toUpperCase(); }
  }
  return hints; // { headerId: 'A' }
}

// =====================================================
// Staging baskets (accumulate 3D files across selections)
// =====================================================
function setupStagingBaskets(){
  if(GUARDS.stagingDone) return;

  setupOneBasket('target');
  setupOneBasket('competitor');

  GUARDS.stagingDone = true;
}

function setupOneBasket(kind){
  const legacyInput = $(`${kind}-file`); // existing <input type="file" multiple>
  if(!legacyInput || !legacyInput.parentElement) return;

  // Hide legacy input but keep it as fallback (we still read its files on submit)
  legacyInput.style.display = 'none';

  // Create staging box
  const boxId = `${kind}-staging-box`;
  const chipsId = `${kind}-staged-chips`;
  const pickId = `${kind}-staged-picker`;

  const html = `
    <div id="${boxId}" class="staging-box">
      <div class="staging-head">
        <div><strong>${kind === 'target' ? 'Target' : 'Competitor'} 3D files</strong>
          <span class="badge off" id="${kind}-staged-count">0 staged</span>
        </div>
        <div style="display:flex;gap:8px;align-items:center;">
          <button id="${kind}-staged-add" class="btn-action">Add PDB/mmCIF</button>
          <button id="${kind}-staged-clear" class="btn-action">Clear</button>
        </div>
      </div>
      <div id="${chipsId}" style="display:flex;flex-wrap:wrap;"></div>
      <input id="${pickId}" type="file" accept=".pdb,.cif,.mmcif" multiple style="display:none"/>
    </div>
  `;
  legacyInput.insertAdjacentHTML('afterend', html);

  const pick = $(pickId);
  const addBtn = $(`${kind}-staged-add`);
  const clrBtn = $(`${kind}-staged-clear`);

  if(addBtn && pick){
    bindOnce(addBtn, 'click', () => pick.click(), `${kind}-stage-add`);
  }
  if(pick){
    bindOnce(pick, 'change', () => {
      const list = Array.from(pick.files || []);
      if(list.length){
        for(const f of list){
          if(!validateFileSize(f)) continue;
          STAGED[`${kind}3dFiles`].push(f);
        }
        renderStagedChips(kind);
      }
      pick.value = ''; // reset
    }, `${kind}-stage-change`);
  }
  if(clrBtn){
    bindOnce(clrBtn, 'click', () => {
      STAGED[`${kind}3dFiles`] = [];
      renderStagedChips(kind);
    }, `${kind}-stage-clear`);
  }
  renderStagedChips(kind);
}

function renderStagedChips(kind){
  const area = $(`${kind}-staged-chips`);
  const countLbl = $(`${kind}-staged-count`);
  if(!area) return;
  area.innerHTML = '';
  const arr = STAGED[`${kind}3dFiles`];
  arr.forEach((f, idx) => {
    const chip = document.createElement('span');
    chip.className = 'chipfile';
    chip.innerHTML = `${escapeHTML(f.name)} <button title="Remove">×</button>`;
    const btn = chip.querySelector('button');
    if(btn){
      bindOnce(btn, 'click', () => {
        STAGED[`${kind}3dFiles`].splice(idx,1);
        renderStagedChips(kind);
      }, `${kind}-rm-${idx}-${Date.now()}`);
    }
    area.appendChild(chip);
  });
  if(countLbl){
    countLbl.textContent = `${arr.length} staged`;
    countLbl.className = `badge ${arr.length ? 'ok' : 'off'}`;
  }
}

// =====================================================
// Submit handler
// =====================================================
async function handleSubmit(event){
  event.preventDefault();

  const loader = $('loader');
  const resultsContainer = $('results-container');

  const primarySeqs   = $('primary-seqs')?.value?.trim() ?? '';
  const targetSeq     = $('target-seq')?.value?.trim() ?? '';
  const competitorSeq = $('competitor-seq')?.value?.trim() ?? '';

  // Snapshot FASTA → maps for downstream analysis
  CURRENT_INPUTS.mirnas      = parseFastaToMap(primarySeqs, 'miRNA');
  CURRENT_INPUTS.targets     = parseFastaToMap(targetSeq, 'target');
  CURRENT_INPUTS.competitors = parseFastaToMap(competitorSeq, 'competitor');

  // Reset run manifest
  RUN_MANIFEST = {
    created_at: new Date().toISOString(),
    client: 'mirna.js',
    config: {
      mirna_max: CONFIG.mirna_max,
      mature_trim_enabled: CONFIG.mature_trim_enabled,
      mature_window: CONFIG.mature_window,
      aa_convert_allowed: CONFIG.aa_convert_allowed
    },
    aa_nt_mode: (byQS('#aa-nt-mode')?.value || 'canonical'),
    flags: {
      mature_trim: $('mature-trim-flag')?.checked ?? CONFIG.mature_trim_enabled,
      aa_convert: $('aa-convert-flag')?.checked ?? CONFIG.aa_convert_allowed
    },
    inputs: {
      mirna_count: Object.keys(CURRENT_INPUTS.mirnas).length,
      target_count: Object.keys(CURRENT_INPUTS.targets).length,
      competitor_count: Object.keys(CURRENT_INPUTS.competitors).length,
      target_pdb_ids: extractPdbIdsFromFasta(targetSeq),
      competitor_pdb_ids: extractPdbIdsFromFasta(competitorSeq),
      target_chain_hints: extractChainHintsFromFasta(targetSeq),
      competitor_chain_hints: extractChainHintsFromFasta(competitorSeq),
      staged_target_files: STAGED.target3dFiles.map(f => f.name),
      staged_competitor_files: STAGED.competitor3dFiles.map(f => f.name)
    }
  };

  // Clear results view
  if(resultsContainer) setHTML(resultsContainer, '');
  predictionResults = [];
  LAST_SEED_HITS = null; LAST_SEED_META = null; CURRENT_JOB_ID = null;

  // Anti-refresh note
  prependHTML(resultsContainer, `<div class="reload-warning">
    Please do not refresh or close this page while your prediction is running — this will cancel the analysis in progress.
  </div>`);

  // Require miRNA FASTA headers
  if(!hasFastaHeaders(primarySeqs)){
    setHTML(resultsContainer, formatError(
      'Your miRNA input is missing FASTA headers. Please add lines starting with ">" (e.g., >hsa-let-7a-5p) so results can be labeled correctly.'
    ));
    return;
  }

  // Count records for ETA hint (targets/competitors may be filled by PDB-only later; we allow 0 here)
  const mirnaCount = countFastaRecords(primarySeqs);
  let tgtCount  = countFastaRecords(targetSeq);      if(!tgtCount && targetSeq)  tgtCount  = 1;
  let compCount = countFastaRecords(competitorSeq);  if(!compCount && competitorSeq) compCount = 1;

  // Friendly info + estimated total pairs (centered)
  const estTotal = (mirnaCount || 0) * (Math.max(tgtCount, 1)) * (Math.max(compCount, 1));
  prependHTML(resultsContainer, formatInfo(
    `Detected ${tgtCount||0} target(s) and ${compCount||0} competitor(s) from FASTA. Staged 3D files: target=${STAGED.target3dFiles.length}, competitor=${STAGED.competitor3dFiles.length}. Estimated evaluations: ${estTotal}.`
  ));

  // Non-blocking tips (backend enforces), FASTA-only/PDB-only allowed
  const MIN_TARGET_LEN = 30;
  const MIN_COMP_LEN   = 15;
  if (targetSeq && (targetSeq.replace(/^>.*$/gm,'').replace(/\s+/g,'')).length < MIN_TARGET_LEN){
    appendHTML(resultsContainer, formatWarn(`Tip: Target should be at least ${MIN_TARGET_LEN} nt if provided. PDB-only runs are also supported.`));
  }
  if(competitorSeq && (competitorSeq.replace(/^>.*$/gm,'').replace(/\s+/g,'')).length < MIN_COMP_LEN){
    appendHTML(resultsContainer, formatWarn(`Tip: Competitor should be at least ${MIN_COMP_LEN} nt or leave it blank. PDB-only runs are supported.`));
  }
  if(mirnaCount > CONFIG.mirna_max){
    setHTML(resultsContainer, formatError(
      `You entered ${mirnaCount} miRNAs, but the maximum allowed is ${CONFIG.mirna_max}. Please reduce your input and try again.`
    ));
    return;
  }
  if(tgtCount >= 1 && !hasFastaHeaders(targetSeq)){
    prependHTML(resultsContainer, formatWarn('Tip: Add FASTA headers to targets (e.g., >target1) for clean labels in results. PDB can still be used for visualization.'));
  }
  if(competitorSeq && !hasFastaHeaders(competitorSeq)){
    prependHTML(resultsContainer, formatWarn('Tip: Add FASTA headers to competitors (e.g., >comp1) for clean labels in results.'));
  }

  // Switch to results tab & scroll to page top
  const resultsTabButton = Array.from(document.querySelectorAll('button.tab-btn'))
    .find(b => /results/i.test(b.textContent || ''));
  if (resultsTabButton) {
    openTab(resultsTabButton, 'results-tab');
    window.scrollTo({ top: 0, left: 0, behavior: 'smooth' });
  }

  // Show loader
  if(loader){
    text(loader, "Running prediction...");
    show(loader);
  }

  // Build FormData
  const formData = new FormData();
  formData.append('primary_molecules', primarySeqs);
  formData.append('target_molecule', targetSeq);
  formData.append('competitor_molecule', competitorSeq);
  formData.append('target_start', $('target-start')?.value ?? '');
  formData.append('target_end',   $('target-end')?.value ?? '');

  // Flags
  const matureTrimFlag = $('mature-trim-flag')?.checked ?? CONFIG.mature_trim_enabled;
  const aaConvertFlag  = $('aa-convert-flag')?.checked ?? false;
  const aaMode         = (byQS('#aa-nt-mode')?.value || 'canonical').toLowerCase();
  formData.append('mature_trim', matureTrimFlag ? 'true' : 'false');
  formData.append('convert_aa_to_nt', aaConvertFlag ? 'true' : 'false');
  formData.append('aa_nt_mode', aaMode);

  // NEW: PDB IDs & chain hints in FASTA headers → pass through
  const targetPdbIds = extractPdbIdsFromFasta(targetSeq);
  const compPdbIds   = extractPdbIdsFromFasta(competitorSeq);
  targetPdbIds.forEach(id => formData.append('target_pdb_id', id));
  compPdbIds.forEach(id   => formData.append('competitor_pdb_id', id));

  const targetChainHints = extractChainHintsFromFasta(targetSeq);
  const compChainHints   = extractChainHintsFromFasta(competitorSeq);
  if(Object.keys(targetChainHints).length){
    formData.append('target_chain_hints_json', JSON.stringify(targetChainHints));
  }
  if(Object.keys(compChainHints).length){
    formData.append('competitor_chain_hints_json', JSON.stringify(compChainHints));
  }

  // --- Optional 3D files (multiple allowed; include STAGED first) ---
  for (const f of STAGED.target3dFiles) { if(!validateFileSize(f)) { continue; } formData.append('target_3d_file', f); }
  for (const f of STAGED.competitor3dFiles) { if(!validateFileSize(f)) { continue; } formData.append('competitor_3d_file', f); }

  // Also include any legacy one-off selections still present (compat)
  const legacyTargetFiles = $('target-file')?.files;
  if (legacyTargetFiles?.length) {
    for (const f of legacyTargetFiles) {
      if (!validateFileSize(f)) { $('target-file').value=''; break; }
      formData.append('target_3d_file', f);
    }
  }
  const legacyCompFiles = $('competitor-file')?.files;
  if (legacyCompFiles?.length) {
    for (const f of legacyCompFiles) {
      if (!validateFileSize(f)) { $('competitor-file').value=''; break; }
      formData.append('competitor_3d_file', f);
    }
  }
  const mirnaFileInput = $('mirna-file');
  if (mirnaFileInput?.files?.length) {
    for (const f of mirnaFileInput.files) {
      if (!validateFileSize(f)) { mirnaFileInput.value=''; break; }
      formData.append('mirna_3d_file', f);
    }
  }

  // Fire a non-blocking precheck (if supported). Show Use/Skip table.
  tryPrecheck(formData).catch(()=>{ /* silent */ });

  try{
    const authHeaders = await getNonceOrKeyHeaders();

    // 1) Start
    if(loader) text(loader, "Job started. Preparing batches...");
    const startRes = await fetch(API_URL, { method:'POST', headers:authHeaders, body:formData });

    if(!startRes.ok){
      let errorMsg;
      try{
        const errorData = await startRes.json();
        errorMsg = errorData.message || errorData.error || null;
      }catch(_){ }
      throw new Error(errorMsg || 'Something went wrong while starting your job.');
    }

    const { job_id } = await startRes.json();
    if(!job_id) throw new Error('No job ID returned from server.');
    CURRENT_JOB_ID = job_id;
    RUN_MANIFEST.job_id = job_id;

    // 2) Poll progress (with stall detection)
    let lastCompleted = -1;
    let lastTick = Date.now();

    const poll = async () => {
      const res = await fetch(PROGRESS_URL(job_id), { method:'GET' });
      if(!res.ok) throw new Error('Failed to check job progress.');
      const data = await res.json();

      if(data.status === 'running'){
        const total     = Number.isFinite(data.total) ? data.total : '?';
        const completed = Number.isFinite(data.completed) ? data.completed : '?';

        if(loader){
          if(!loader.querySelector('.loader-spinner')){
            loader.innerHTML = `<span class="loader-spinner"></span><span id="loader-text"></span>`;
          }
          const lt = loader.querySelector('#loader-text');
          if(lt) lt.textContent = `Processing... ${completed}/${total} completed`;
          show(loader);
        }

        // stall hint if progress hasn't changed for 180s
        if(Number.isFinite(completed) && completed !== lastCompleted){
          lastCompleted = completed; lastTick = Date.now();
        }else if(Date.now() - lastTick > 180000){
          const friendly = [
            'Still working — this is taking longer than usual.',
            'Please keep this page open; closing it will stop the analysis.',
          ].join(' ');

          const details = `
            <details style="margin-top:6px;">
              <summary style="cursor:pointer;color:#1e5a9c;">Technical details (for administrators)</summary>
              <div style="margin-top:6px;font-size:13px;color:#444;">
                On some servers, the Flask <em>debug reloader</em> can start a second process and break live progress (it may show 0/… forever).
                If you manage this server, run it in single-process mode:
                <code>debug=False</code> and <code>use_reloader=False</code>.
              </div>
            </details>
          `;

          prependHTML(resultsContainer, formatWarn(friendly) + details);
          lastTick = Date.now(); // show only occasionally
        }

        setTimeout(poll, 1200);
        return;
      }

      if(data.status === 'error'){
        // Soft-handle PDB type mismatches: try to fetch results anyway
        const rw = resultsContainer.querySelector('.reload-warning'); if(rw) rw.remove();
        const maybePdbWarning = /pdb|structure|polymer|chain|back-translate/i.test(data.error || '');
        try{
          if(loader) text(loader, "Attempting to fetch partial results...");
          const dr = await fetch(DOWNLOAD_URL(job_id), { method:'GET' });
          if(dr.ok){
            const finalDataSoft = await dr.json();
            const rows = finalDataSoft.results || [];
            if(rows.length){
              predictionResults = rows;
              displayResults(predictionResults, finalDataSoft);
              if(maybePdbWarning){
                prependHTML(resultsContainer, formatWarn('Structure warning encountered. PDB files were kept for visualization; scoring continued using nucleotide sequences.'));
              }
              if(loader){ text(loader, "✅ Prediction completed with warnings."); setTimeout(()=>hide(loader), 3000); }
              return;
            }
          }
        }catch(_){}
        // If nothing could be recovered, show error
        throw new Error(data.error || 'We encountered a technical issue while processing your request.');
      }

      if(data.status === 'completed'){
        const rw = resultsContainer.querySelector('.reload-warning'); if(rw) rw.remove();

        if(loader) text(loader, "Fetching final results...");
        const dr = await fetch(DOWNLOAD_URL(job_id), { method:'GET' });
        if(!dr.ok) throw new Error('Failed to download results.');
        const finalData = await dr.json();

        predictionResults = finalData.results || [];
        displayResults(predictionResults, finalData);

        if(loader){
          text(loader, "✅ Prediction completed. Results are shown below.");
          setTimeout(() => hide(loader), 3000);
        }
      }
    };

    await poll();

  }catch(error){
    const rw = resultsContainer?.querySelector('.reload-warning'); if(rw) rw.remove();

    const friendly = error?.message && !/server error/i.test(error.message)
      ? error.message
      : 'Something went wrong while processing your request. Please try again later.';
    setHTML(resultsContainer, formatError(friendly));
    if(loader) hide(loader);
  }
}

// =====================================================
// Precheck (best-effort; shows Use/Skip table if backend supports /precheck)
// =====================================================
async function tryPrecheck(formData){
  // clone FormData (can reuse same files safely)
  const fd = new FormData();
  for (const [k,v] of formData.entries()) { fd.append(k, v); }

  const headers = await getNonceOrKeyHeaders();
  let res;
  try{
    res = await fetchWithTimeout(PRECHECK_URL, { method:'POST', headers, body: fd }, 20000);
  }catch(_){ /* ignore */ }
  if(!res || !res.ok){
    const rc = $('results-container');
    if(rc){
      appendHTML(rc, formatInfo('Pre-validation skipped (not available). We’ll auto-handle PDB-only, FASTA-only, and protein back-translation. PDB never blocks scoring.'));
    }
    return;
  }
  const data = await res.json();
  renderPrecheckPanel(data);
}

function renderPrecheckPanel(data){
  const rc = $('results-container');
  if(!rc) return;

  const rows = [];
  const add = (arr, label) => {
    (arr || []).forEach(o => {
      rows.push({
        kind: label,
        id: o.id || o.header || o.filename || '(unknown)',
        chain: o.chain || o.chain_id || '',
        polymer: o.polymer || o.polymer_type || 'unknown',
        length: o.length || o.seq_len || '',
        used_scoring: !!o.used_for_scoring,
        used_viz: !!o.used_for_viz || !!o.present_for_viz,
        back_tx: !!o.back_translated,
        note: o.note || ''
      });
    });
  };
  add(data.targets, 'Target');
  add(data.competitors, 'Competitor');

  let html = `
    <div class="staging-box" style="background:#f8fffb;border-color:#bbf7d0;">
      <div class="staging-head">
        <div><strong>Pre-validation</strong> <span class="badge ok">non-blocking</span></div>
        <div class="badge off">PDB optional</div>
      </div>
      <table class="precheck-table">
        <thead><tr>
          <th>Role</th><th>ID / File</th><th>Chain</th><th>Polymer</th><th>Len</th>
          <th>Used for scoring?</th><th>Used for viz?</th><th>AA→NT</th><th>Note</th>
        </tr></thead>
        <tbody>
  `;
  if(rows.length === 0){
    html += `<tr><td colspan="9">No structures detected to pre-validate. This is fine — runs can be FASTA-only.</td></tr>`;
  }else{
    rows.forEach(r=>{
      const bScore = r.used_scoring ? `<span class="badge ok">yes</span>` : `<span class="badge off">no</span>`;
      const bViz   = r.used_viz ? `<span class="badge ok">yes</span>` : `<span class="badge off">no</span>`;
      const bBT    = r.back_tx ? `<span class="badge warn">yes</span>` : `<span class="badge off">no</span>`;
      html += `<tr>
        <td>${escapeHTML(r.kind)}</td>
        <td>${escapeHTML(r.id)}</td>
        <td>${escapeHTML(r.chain)}</td>
        <td>${escapeHTML(r.polymer)}</td>
        <td>${escapeHTML(r.length)}</td>
        <td>${bScore}</td>
        <td>${bViz}</td>
        <td>${bBT}</td>
        <td>${escapeHTML(r.note)}</td>
      </tr>`;
    });
  }
  html += `</tbody></table>
      <small style="color:#475569;">If a PDB isn’t nucleotide or doesn’t match the FASTA, it’s kept for visualization with a “not used in scoring” note. Protein chains are auto back-translated for seed/IG scanning when enabled.</small>
    </div>
  `;
  prependHTML(rc, html);

  // merge some info into manifest if provided
  RUN_MANIFEST = RUN_MANIFEST || {};
  RUN_MANIFEST.precheck = { targets: data.targets || [], competitors: data.competitors || [] };
}

// =====================================================
// Display results (sorted by baseline; gradient by baseline)
// + injects analysis controls + filter chips + per-row action buttons + row badges + manifest download
// =====================================================
function displayResults(results, finalData=null){
  const container = $('results-container');
  if(!container) return;

  setHTML(container, '');

  if(!results || results.length === 0){
    setHTML(container, '<p>No results to display.</p>');
    return;
  }

  // Inject analysis controls (singleton)
  injectAnalysisControls(container);

  // Update RUN_MANIFEST with any server-side manifest/meta if present
  if(finalData && finalData.manifest){
    RUN_MANIFEST = { ...(RUN_MANIFEST || {}), server_manifest: finalData.manifest };
  }

  // Top badges about run
  const runBadgesId = 'run-badges';
  const topBadges = `
    <div id="${runBadgesId}" style="display:flex;gap:8px;flex-wrap:wrap;justify-content:center;margin:6px 0 8px;">
      <span class="badge ${hasAnyStructure() ? 'ok':'off'}">PDB present: ${hasAnyStructure() ? 'yes':'no'}</span>
      <span class="badge ${($('aa-convert-flag')?.checked ? 'warn':'off')}">AA→NT: ${$('aa-convert-flag')?.checked ? 'yes':'no'} (mode: ${(byQS('#aa-nt-mode')?.value || 'canonical')})</span>
      <span class="badge off">Seed/IG computed on NT</span>
    </div>
  `;
  appendHTML(container, topBadges);

  // Sort by baseline desc
  results.sort((a,b) =>
    safeParseFloat(b.predicted_affinity_baseline ?? b.baseline_score ?? 0, 0) -
    safeParseFloat(a.predicted_affinity_baseline ?? a.baseline_score ?? 0, 0)
  );

  // Gradient by baseline in [0,1]
  function getGradientColor(score){
    const s = Math.max(0, Math.min(1, parseFloat(score) || 0));
    const viridis = [
      [68, 1, 84],    // #440154
      [59, 82, 139],  // #3b528b
      [33, 144, 141], // #21908d
      [93, 201, 99],  // #5dc963
      [253, 231, 37]  // #fde725
    ];
    const idx = s * (viridis.length - 1);
    const low = Math.floor(idx);
    const high= Math.min(low + 1, viridis.length - 1);
    const t   = idx - low;
    const r = Math.round(viridis[low][0] + t * (viridis[high][0] - viridis[low][0]));
    const g = Math.round(viridis[low][1] + t * (viridis[high][1] - viridis[low][1]));
    const b = Math.round(viridis[low][2] + t * (viridis[high][2] - viridis[low][2]));
    return `rgba(${r},${g},${b},0.3)`;
  }

  // Legend
  const legendId = 'affinity-legend';
  const legendHTML = `
  <div id="${legendId}" class="affinity-legend" style="margin-bottom:10px;text-align:center;">
    <h4 style="margin:6px 0 10px 0;">Affinity Classification Guide</h4>
    <table style="margin:0 auto;">
      <thead><tr><th>Category</th><th>Score Range</th><th>Interpretation</th></tr></thead>
      <tbody>
        <tr style="background-color:rgba(189,223,38,0.3)"><td>High Affinity</td><td>0.76–1.00</td><td>Strong binding; prioritized for validation</td></tr>
        <tr style="background-color:rgba(74,193,109,0.3)"><td>Medium Affinity</td><td>0.51–0.75</td><td>Moderate; candidate for confirmation</td></tr>
        <tr style="background-color:rgba(43,116,142,0.3)"><td>Low Affinity</td><td>0.26–0.50</td><td>Weak prediction</td></tr>
        <tr style="background-color:rgba(72,36,117,0.3)"><td>No Affinity</td><td>0.00–0.25</td><td>No meaningful binding</td></tr>
      </tbody>
    </table>
  </div>
  `;

  // Download + Copy + Manifest buttons
  const buttonsHTML = `<div style="margin-bottom:12px;display:flex;gap:8px;flex-wrap:wrap;justify-content:center;">
    <button id="download-all-server-csv" class="btn-premium">Download Results (CSV)</button>
    <button id="download-all-bundles" class="btn-premium">Download All</button>
    <button id="copy-results-btn" class="btn-premium btn-accent">Copy Results (TSV)</button>
    <button id="download-manifest" class="btn-premium">Run Manifest (JSON)</button>
  </div>`;

  appendHTML(container, legendHTML);
  appendHTML(container, buttonsHTML);

  // Manifest button
  bindOnce($('download-manifest'), 'click', () => {
    const manifest = RUN_MANIFEST || {};
    const blob = new Blob([JSON.stringify(manifest, null, 2)], {type:'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `mirna_run_manifest_${CURRENT_JOB_ID || 'NA'}.json`;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, 'dlManifestOnce');

  bindOnce($('download-all-server-csv'), 'click', async () => {
    if(!CURRENT_JOB_ID){ alert('No active job.'); return; }
    try{
      const allowGU = byQS('#allow-gu')?.checked ?? true;
      const maxMM   = parseInt(byQS('#max-mm')?.value ?? '0', 10);
      const headers = await getNonceOrKeyHeaders();
      const url = DOWNLOAD_ALL_CSV_URL(CURRENT_JOB_ID) +
        `?allow_gu=${allowGU ? 1 : 0}&max_mismatch=${Number.isFinite(maxMM)?maxMM:0}&range_aware=1&tolerant=1`;
      const res = await fetch(url, { method:'GET', headers });
      if(!res.ok) throw new Error('Download failed');
      const blob = await res.blob();
      const dl  = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = dl; a.download = `mirna_results_${CURRENT_JOB_ID}.csv`;
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
      URL.revokeObjectURL(dl);
    }catch(err){ alert('Could not download CSV.'); }
  }, 'dlAllCsvOnce');

  // Download All (zip of all artifacts)
  bindOnce($('download-all-bundles'), 'click', async () => {
    if (!CURRENT_JOB_ID) { alert('No active job.'); return; }
    try {
      const headers = await getNonceOrKeyHeaders();
      const res = await fetch(`${BASE_URL}/download/${CURRENT_JOB_ID}/all.zip`, { method:'GET', headers });
      if (!res.ok) throw new Error('Download failed');
      const blob = await res.blob();
      const url  = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = `mirna_job_${CURRENT_JOB_ID}_all.zip`;
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (e) { alert('Could not download all bundles.'); }
  }, 'dlAllZipOnce');

  bindOnce($('copy-results-btn'), 'click', () => {
    const hasTargetCol = (predictionResults || []).some(r => typeof r.target_id !== 'undefined');
    const hasCompCol   = (predictionResults || []).some(r => (r.competitor_id ?? '') !== '');
    const lines = predictionResults.map(item => {
      const id        = item.primary_molecule_id ?? item.mirna_id ?? 'N/A';
      const tid       = item.target_id ?? '';
      const cid       = item.competitor_id ?? '';
      const baseline  = (item.predicted_affinity_baseline ?? item.baseline_score ?? '').toString();
      const withComp  = (item.predicted_affinity_with_competitor ?? item.score_with_competitor ?? '').toString();
      const compEff   = (item["competitive_effect (higher_is_better)"] ?? item.competitive_effect ?? '').toString();
      return [
        id,
        ...(hasTargetCol ? [tid] : []),
        ...(hasCompCol   ? [cid] : []),
        baseline, withComp, compEff
      ].join('\t');
    });
    navigator.clipboard.writeText(lines.join('\n')).then(() => alert('Results copied to clipboard.'));
  }, 'copyResultsClick');

  // Table with optional Target/Competitor columns + Badges + Analysis col
  const hasTargetCol = (results || []).some(r => typeof r.target_id !== 'undefined');
  const hasCompCol   = (results || []).some(r => (r.competitor_id ?? '') !== '');

  let table = '<table id="results-table" style="margin-bottom:20px;width:100%;border-collapse:collapse;"><thead><tr>' +
    '<th>Primary Molecule ID</th>' +
    (hasTargetCol ? '<th>Target ID</th>' : '') +
    (hasCompCol   ? '<th>Competitor ID</th>' : '') +
    '<th>Predicted Affinity (Baseline)</th>' +
    '<th>Predicted Affinity (With Competitor)</th>' +
    '<th>Competitive Effect (higher is better)</th>' +
    '<th>Notes</th>' +
    '<th>Analysis</th>' +
    '</tr></thead><tbody>';

  results.forEach((item, idx) => {
    const id        = item.primary_molecule_id ?? item.mirna_id ?? 'N/A';
    const tid       = item.target_id ?? '';
    const cid       = item.competitor_id ?? '';
    const baseline  = (item.predicted_affinity_baseline ?? item.baseline_score ?? '').toString();
    const withComp  = (item.predicted_affinity_with_competitor ?? item.score_with_competitor ?? '').toString();
    const compEff   = (item["competitive_effect (higher_is_better)"] ?? item.competitive_effect ?? '').toString();
    const bgColor   = getGradientColor(baseline);

    // filter flags per row
    const isRange = (!!tid && /:\d+-\d+/.test(tid)) || (!!cid && /:\d+-\d+/.test(cid));
    const tolT = tid ? (!exactKeyExists(CURRENT_INPUTS.targets, parseIdRange(tid)?.baseId || tid) && !!lookupTolerant(CURRENT_INPUTS.targets, parseIdRange(tid)?.baseId || tid)) : false;
    const tolC = cid ? (!exactKeyExists(CURRENT_INPUTS.competitors, parseIdRange(cid)?.baseId || cid) && !!lookupTolerant(CURRENT_INPUTS.competitors, parseIdRange(cid)?.baseId || cid)) : false;
    const isTol = tolT || tolC;

    // Row badges (best-effort from server fields if present)
    const badgeBits = [];
    if (typeof item.pdb_used !== 'undefined'){
      badgeBits.push(`<span class="badge ${item.pdb_used ? 'ok':'off'}">PDB used: ${item.pdb_used ? 'yes':'no'}</span>`);
    }else if (hasAnyStructure()){
      badgeBits.push(`<span class="badge off">PDB used: —</span>`);
    }
    if (typeof item.aa_to_nt_mode !== 'undefined' || typeof item.aa_to_nt !== 'undefined'){
      const yn = item.aa_to_nt || !!item.aa_to_nt_mode;
      const mode = (item.aa_to_nt_mode || (byQS('#aa-nt-mode')?.value || 'canonical')).toString();
      badgeBits.push(`<span class="badge ${yn ? 'warn':'off'}">AA→NT: ${yn ? 'yes':'no'}${yn ? ` (${escapeHTML(mode)})` : ''}</span>`);
    }
    if (typeof item.structure_features_on !== 'undefined'){
      badgeBits.push(`<span class="badge ${item.structure_features_on ? 'ok':'off'}">Structure-features: ${item.structure_features_on ? 'on':'off'}</span>`);
    }

    const badgesHTML = badgeBits.length ? badgeBits.join(' ') :
      `<span class="badge off">Notes unavailable</span>`;

    const seedBtn    = `<button class="seed-btn btn-action" data-row="${idx}">Seed Sites</button>`;
    const heatBtn    = `<button class="heatmap-btn btn-action" data-row="${idx}">Heatmap</button>`;
    const csvBtn     = `<button class="rowcsv-btn btn-action" data-row="${idx}">Row CSV</button>`;
    const t3dBtn     = `<button class="t3d-btn btn-action" data-row="${idx}">3D Target</button>`;
    const c3dBtn     = `<button class="c3d-btn btn-action" data-row="${idx}">3D Comp</button>`;
    const bundleBtn  = `<button class="bundle-btn btn-action" data-row="${idx}">Download</button>`;

    // two-line action block
    const actionBlock = `
      <div class="action-grid">
        ${seedBtn}
        ${heatBtn}
        ${csvBtn}
        ${t3dBtn}
        ${c3dBtn}
        ${bundleBtn}
      </div>
    `;

    table += `<tr data-range="${isRange ? '1':'0'}" data-tolerant="${isTol ? '1':'0'}" style="background-color:${bgColor}">
      <td>${escapeHTML(id)}</td>` +
      (hasTargetCol ? `<td>${escapeHTML(tid)}</td>` : '') +
      (hasCompCol   ? `<td>${escapeHTML(cid)}</td>` : '') +
      `<td>${escapeHTML(baseline)}</td>
       <td>${escapeHTML(withComp)}</td>
       <td>${escapeHTML(compEff)}</td>
       <td>${badgesHTML}</td>
       <td>${actionBlock}</td>
    </tr>`;
  });

  table += '</tbody></table>';
  appendHTML(container, table);
  makeTableSortable('results-table');

  // Keep range/tolerant filter chips
  injectResultFilters();

  // Delegate click handlers for action buttons
  const resultsTable = $('results-table');
  if(resultsTable){
    bindOnce(resultsTable, 'click', async (e) => {
      const t = e.target;
      if(!(t instanceof HTMLElement)) return;
      const rowIdx = t.dataset?.row ? parseInt(t.dataset.row, 10) : NaN;
      if(Number.isNaN(rowIdx) || !predictionResults[rowIdx]) return;
      const item = predictionResults[rowIdx];

      if(t.classList.contains('seed-btn')){
        await handleSeedSitesClick(item);
      }else if(t.classList.contains('heatmap-btn')){
        await handleHeatmapClick(item);
      }else if(t.classList.contains('rowcsv-btn')){
        await handleRowCsvClick(item);
      }else if(t.classList.contains('t3d-btn')){
        await open3DOrExplain(item.target_id || '', 'target');
      }else if(t.classList.contains('c3d-btn')){
        await open3DOrExplain(item.competitor_id || '', 'competitor');
      }else if(t.classList.contains('bundle-btn')){
        await handleBundleClick(item);
      }
    }, 'resultsActions');
  }
}

function hasAnyStructure(){
  return STAGED.target3dFiles.length + STAGED.competitor3dFiles.length > 0 ||
         !!$('target-file')?.files?.length || !!$('competitor-file')?.files?.length;
}

// === Inject range/tolerant filter chips (toggle behavior) ===
function injectResultFilters(){
  if($('result-filters')) return;
  const box = document.createElement('div');
  box.id = 'result-filters';
  box.className = 'result-filters';
  box.style.cssText = 'display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:8px 0 12px;justify-content:center;';
  box.innerHTML = `
    <label class="chip-toggle" id="chip-range" style="display:inline-flex;align-items:center;gap:8px;padding:4px 10px;border-radius:999px;background:#eef5ff;border:1px solid #cfe0ff;color:#163b66;cursor:pointer;font-size:12px;font-weight:600;">
      <input type="checkbox" id="filter-range" style="accent-color:#1e5a9c;"> range-aware only
    </label>
    <label class="chip-toggle" id="chip-tol" style="display:inline-flex;align-items:center;gap:8px;padding:4px 10px;border-radius:999px;background:#eef5ff;border:1px solid #cfe0ff;color:#163b66;cursor:pointer;font-size:12px;font-weight:600;">
      <input type="checkbox" id="filter-tolerant" style="accent-color:#1e5a9c;"> tolerant-matched only
    </label>
    <button class="chip-clear" id="clear-filters" style="padding:4px 10px;border-radius:999px;border:1px solid #e5e7eb;background:#fff;cursor:pointer;font-size:12px;">clear</button>
  `;
  byQS('#results-container')?.insertBefore(box, byQS('#results-container').firstChild);

  const apply = () => {
    const rOnly = byQS('#filter-range')?.checked || false;
    const tOnly = byQS('#filter-tolerant')?.checked || false;
    $('chip-range')?.classList.toggle('active', rOnly);
    $('chip-tol')?.classList.toggle('active', tOnly);

    byQSA('#results-table tbody tr').forEach(tr=>{
      const hasRange = tr.getAttribute('data-range') === '1';
      const isTol    = tr.getAttribute('data-tolerant') === '1';
      const hide = (rOnly && !hasRange) || (tOnly && !isTol);
      tr.style.display = hide ? 'none' : '';
    });
  };

  bindOnce($('filter-range'),'change',apply,'rFilter');
  bindOnce($('filter-tolerant'),'change',apply,'tFilter');
  bindOnce($('clear-filters'),'click',()=>{
    const r=$('filter-range'), t=$('filter-tolerant');
    if(r) r.checked=false; if(t) t.checked=false; apply();
  },'cFilter');
}

// =====================================================
// Analysis controls (singleton)
// =====================================================
function injectAnalysisControls(container){
  if(GUARDS.analysisControlsInjected) return;

  const html = `
  <div id="analysis-controls" class="controls-grid">
    <label class="ctrl"><input type="checkbox" id="allow-gu" checked /><span>Allow G:U wobble</span></label>
    <label class="ctrl"><span>Max mismatches</span><input id="max-mm" type="number" value="0" min="0" max="3" step="1"></label>
    <label class="ctrl"><span>Heatmap</span>
      <select id="heatmap-mode">
        <option value="ig_target" selected>IG → Target</option>
        <option value="ig_competitor">IG → Competitor</option>
        <option value="seed_density">Seed density (fast)</option>
      </select>
    </label>
    <label class="ctrl"><span>Steps</span><input id="heatmap-steps" type="number" value="64" min="10" max="200" step="2"></label>

    <button id="seed-scan-global-btn" class="btn-premium">Seed Sites (top row)</button>
    <button id="explain-global-btn"   class="btn-premium btn-accent">Heatmap (top row)</button>
  </div>
  `;

  prependHTML(container, html);

  // Global buttons: operate on the top-ranked row (simple, deterministic)
  const seedBtn = $('seed-scan-global-btn');
  const hmBtn   = $('explain-global-btn');

  bindOnce(seedBtn, 'click', async ()=>{
    if(!predictionResults.length){
      openModal('Seed Sites', formatInfo('Run a prediction first so we can use the top-ranked row.'));
      return;
    }
    await handleSeedSitesClick(predictionResults[0]);
  }, 'seedGlobalOnce');

  bindOnce(hmBtn, 'click', async ()=>{
    if(!predictionResults.length){
      openModal('Heatmap', formatInfo('Run a prediction first so we can use the top-ranked row.'));
      return;
    }
    await handleHeatmapClick(predictionResults[0]); // uses current #heatmap-mode/#heatmap-steps
  }, 'hmGlobalOnce');

  GUARDS.analysisControlsInjected = true;
}

// =====================================================
// CSV & bundles
// =====================================================
async function handleRowCsvClick(item){
  if(!CURRENT_JOB_ID){ alert('No active job.'); return; }
  const interactionId = item.interaction_id || null;
  if(!interactionId){ alert('Row is missing interaction_id.'); return; }
  try{
    const headers = await getNonceOrKeyHeaders();
    const res = await fetch(DOWNLOAD_ROW_CSV_URL(CURRENT_JOB_ID, interactionId), { method:'GET', headers });
    if(!res.ok) throw new Error('Download failed');
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `interaction_${interactionId}.csv`;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }catch(err){ alert('Could not download row CSV.'); }
}

// Per-row bundle (zip) download
async function handleBundleClick(item){
  if(!CURRENT_JOB_ID){ alert('No active job.'); return; }
  const interactionId = item.interaction_id || null;
  if(!interactionId){ alert('Row is missing interaction_id.'); return; }
  try{
    const headers = await getNonceOrKeyHeaders();
    const res = await fetch(`${BASE_URL}/download/${CURRENT_JOB_ID}/${interactionId}/bundle.zip`, { method:'GET', headers });
    if(!res.ok) throw new Error('Download failed');
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `interaction_${interactionId}.zip`;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }catch(err){ alert('Could not download this interaction bundle.'); }
}

// =====================================================
// Heatmap (server PNG first; fallback to client IG; fallback-fallback to seed density)
// =====================================================
async function handleHeatmapClick(item){
  const modeSel  = byQS('#heatmap-mode');
  const stepsInp = byQS('#heatmap-steps');
  const mode  = (modeSel?.value || 'ig_target').toLowerCase();
  const steps = Math.max(10, Math.min(200, parseInt(stepsInp?.value || '64', 10) || 64));

  openModal('Heatmap', smallSpinner('Generating heatmap...'));

  if(!CURRENT_JOB_ID){
    return clientExplainHeatmapFallback(item, mode);
  }

  const interactionId = item.interaction_id || null;
  if(!interactionId){
    return clientExplainHeatmapFallback(item, mode);
  }

  if(mode === 'ig_competitor' && !(item.competitor_id || '').trim()){
    setHTML($('modal-content'), formatWarn('This row has no competitor. Showing IG for target instead.') + smallSpinner());
    return clientExplainHeatmapFallback(item, 'ig_target');
  }

  try{
    const headers = await getNonceOrKeyHeaders();
    const res = await fetchWithTimeout(
      HEATMAP_PNG_URL(CURRENT_JOB_ID, interactionId, mode, steps),
      { method:'GET', headers },
      30000
    );

    if(!res.ok){ throw new Error('PNG fetch failed'); }
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);

    const title = `Heatmap (${mode.replace('_',' → ')}) — ${escapeHTML(item.primary_molecule_id || item.mirna_id || '')}`;
    const toolbar = `
      <button id="hm-open"  class="toolbar-btn">Open in new tab</button>
      <button id="hm-save"  class="toolbar-btn">Download PNG</button>
    `;
    const html = `<img id="hm-img" alt="Heatmap" src="${url}" style="max-width:100%;height:auto;border:1px solid #e5e7eb;border-radius:8px;"/>`;
    openModal(title, html, toolbar);

    const openBtn = $('hm-open');
    const saveBtn = $('hm-save');
    if(openBtn) bindOnce(openBtn, 'click', () => {
      const w = window.open(url, '_blank');
      if(w) w.opener = null;
    }, 'hmOpenOnce');
    if(saveBtn) bindOnce(saveBtn, 'click', () => {
      const a = document.createElement('a');
      a.href = url; a.download = `${interactionId}_${mode}.png`;
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
    }, 'hmSaveOnce');

  }catch(_){
    await clientExplainHeatmapFallback(item, mode);
  }
}

async function clientExplainHeatmapFallback(item, forcedMode){
  try{
    const mirnaId = item.primary_molecule_id ?? item.mirna_id;
    const targetId= item.target_id ?? '';
    const compId  = item.competitor_id ?? '';

    const mirnaSeq = lookupTolerant(CURRENT_INPUTS.mirnas, mirnaId);
    const tRes = resolveSeqWithAAHandling(targetId, CURRENT_INPUTS.targets);
    const cRes = compId ? resolveSeqWithAAHandling(compId, CURRENT_INPUTS.competitors) : {seq:'', converted:false, note:'', mode:''};

    const targetSeq= tRes.seq;
    const compSeq  = cRes.seq;

    if(!mirnaSeq || !targetSeq){
      setHTML($('modal-content'), formatError('Could not resolve miRNA and/or target sequences for this row.'));
      return;
    }

    const uiMode  = (forcedMode || byQS('#heatmap-mode')?.value || 'ig_target').toLowerCase();
    const uiSteps = Math.max(10, Math.min(200, parseInt(byQS('#heatmap-steps')?.value || '64', 10) || 64));

    if(uiMode === 'seed_density'){
      const html = renderSeedDensityFromScan(mirnaSeq, targetId, targetSeq);
      setHTML($('modal-content'), html);
      return;
    }

    setHTML($('modal-content'), smallSpinner('Computing attributions...'));

    const headers = await getNonceOrKeyHeaders();
    const res = await fetchWithTimeout(EXPLAIN_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...headers },
      body: JSON.stringify({
        mirna_seq: toRNA(mirnaSeq),
        target_seq: targetSeq,
        competitor_seq: compSeq || undefined,
        steps: uiSteps,
        mode: uiMode
      })
    }, 30000);

    if(!res.ok){
      const fallback = renderSeedDensityFromScan(mirnaSeq, targetId, targetSeq);
      const aaNote = (tRes.converted || cRes.converted)
        ? `<div style="margin-top:6px;color:#333;"><em>AA→NT conversion was applied before fallback visualization.</em></div>` : '';
      setHTML($('modal-content'), `<div>${formatWarn('Attribution failed. Showing seed density instead.')}${aaNote}${fallback}</div>`);
      return;
    }

    const data = await res.json();
    const targAttr = Array.isArray(data.target_attrib) ? data.target_attrib : (Array.isArray(data.attribution) ? data.attribution : []);
    const compAttr = Array.isArray(data.competitor_attrib) ? data.competitor_attrib : null;

    const targAttrTrim = targAttr.slice(0, targetSeq.length);
    const compAttrTrim = compSeq && compAttr ? compAttr.slice(0, compSeq.length) : null;

    let html = '';

    if(uiMode === 'ig_target'){
      html += renderAttributionPanel('Target', targetSeq, targAttrTrim);
    }
    if((uiMode === 'ig_competitor') && compSeq){
      html += `<div style="height:12px;"></div>`;
      html += renderAttributionPanel('Competitor', compSeq, compAttrTrim || []);
    }
    if(tRes.converted || cRes.converted){
      const modeTxt = byQS('#aa-nt-mode')?.value || 'canonical';
      html += `<div style="margin-top:6px;color:#333;"><small><em>AA→NT conversion applied (${escapeHTML(modeTxt)}) for ${tRes.converted ? 'target' : ''}${tRes.converted && cRes.converted ? ' & ' : ''}${cRes.converted ? 'competitor' : ''}.</em></small></div>`;
    }

    setHTML($('modal-content'), html);

  }catch(err){
    const mirnaId = item.primary_molecule_id ?? item.mirna_id;
    const targetId= item.target_id ?? '';
    const mirnaSeq = lookupTolerant(CURRENT_INPUTS.mirnas, mirnaId);
    const tRes = resolveSeqWithAAHandling(targetId, CURRENT_INPUTS.targets);
    const targetSeq= tRes.seq;
    const html = (mirnaSeq && targetSeq)
      ? `<div>${formatWarn('Attribution failed. Showing seed density instead.')}${renderSeedDensityFromScan(mirnaSeq, targetId, targetSeq)}</div>`
      : formatError(err?.message || 'Unexpected error during explanation.');
    setHTML($('modal-content'), html);
  }
}

// Render a simple seed-density heat strip from LAST_SEED_HITS (client fallback)
function renderSeedDensityFromScan(mirnaSeq, targetId, targetSeq){
  const L = targetSeq.length;
  const density = new Array(L).fill(0);
  if(Array.isArray(LAST_SEED_HITS)){
    LAST_SEED_HITS
      .filter(h => h.molecule === 'target' && h.id === targetId)
      .forEach(h=>{
        for(let i=Math.max(0,h.start-1); i<Math.min(L,h.end); i++) density[i] += 1;
      });
  }
  const max = Math.max(1, ...density);
  const norm = density.map(v => v / max);

  let strip = `<div style="font-family:ui-monospace, Menlo, Consolas;overflow:auto;border:1px solid #eee;border-radius:8px;padding:8px;">`;
  strip += `<div style="white-space:nowrap;">`;
  for(let i=0;i<L;i++){
    const color = viridisColor(norm[i] || 0, 0.85);
    strip += `<span title="pos ${i+1} • ${(targetSeq[i]||'')} • ${(norm[i]||0).toFixed(3)}" style="display:inline-block;min-width:10px;padding:2px 0;text-align:center;background:${color};color:#000;border-radius:2px;margin:0 1px;">${escapeHTML(targetSeq[i] || '')}</span>`;
  }
  strip += `</div></div>`;
  const note = Array.isArray(LAST_SEED_HITS) && LAST_SEED_HITS.length ? '' : '<br/><small style="color:#666;">Tip: run Seed Sites first for a more informative density.</small>';
  return `<div><h4 style="margin:6px 0;">Seed density (client fallback)</h4>${strip}${note}</div>`;
}

// Render one attribution panel (mini heat-strip + top peaks)
function renderAttributionPanel(label, seq, attrib){
  if(!Array.isArray(attrib) || attrib.length === 0){
    return `<div><h4 style="margin:6px 0;">${escapeHTML(label)}</h4><p>No attribution available.</p></div>`;
  }
  const max = Math.max(1e-12, ...attrib.map(v => Math.abs(v)));
  const norm = attrib.map(v => Math.abs(v) / max);

  let strip = `<div style="font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;overflow:auto;border:1px solid #eee;border-radius:8px;padding:8px;">`;
  strip += `<div style="white-space:nowrap;">`;
  for(let i=0;i<seq.length;i++){
    const v = norm[i] || 0;
    const color = viridisColor(v, 0.85);
    strip += `<span title="pos ${i+1} • ${seq[i]} • ${v.toFixed(3)}" style="display:inline-block;min-width:10px;padding:2px 0;text-align:center;background:${color};color:#000;border-radius:2px;margin:0 1px;">${escapeHTML(seq[i] || '')}</span>`;
  }
  strip += `</div></div>`;

  const idxs = norm.map((v,i)=>({i,v})).sort((a,b)=>b.v-a.v).slice(0,5);
  const peaks = idxs.map(o => `pos ${o.i+1} (${escapeHTML(seq[o.i]||'')}) → ${o.v.toFixed(3)}`).join(', ');

  return `
    <div>
      <h4 style="margin:6px 0;">${escapeHTML(label)}</h4>
      ${strip}
      <div style="margin-top:6px;color:#333;"><strong>Top positions:</strong> ${peaks}</div>
      <small style="color:#666;">Color scale is relative within each sequence (min→max saliency, Viridis).</small>
    </div>
  `;
}

// Viridis color helper for heatmaps (0..1 → rgba)
function viridisColor(t, alpha=1.0){
  const lut = [
    [68,1,84],[71,44,122],[59,82,139],[44,113,142],[33,144,141],[39,173,129],[92,200,99],[170,220,50],[253,231,37]
  ];
  const x = Math.max(0, Math.min(1, t)) * (lut.length-1);
  const i = Math.floor(x);
  const j = Math.min(i+1, lut.length-1);
  const f = x - i;
  const r = Math.round(lut[i][0] + f*(lut[j][0]-lut[i][0]));
  const g = Math.round(lut[i][1] + f*(lut[j][1]-lut[i][1]));
  const b = Math.round(lut[i][2] + f*(lut[j][2]-lut[i][2]));
  return `rgba(${r},${g},${b},${alpha})`;
}

// =====================================================
// Seed Sites (exact base-level coordinates) — RANGE-AWARE + tolerant lookup + CSV export
// =====================================================
async function handleSeedSitesClick(item){
  try{
    const allowGU = byQS('#allow-gu')?.checked ?? true;
    const maxMM   = parseInt(byQS('#max-mm')?.value ?? '0', 10);

    const mirnaId = item.primary_molecule_id ?? item.mirna_id;
    const targetId= item.target_id ?? '';
    const compId  = item.competitor_id ?? '';

    const mirnaSeq = toRNA(lookupTolerant(CURRENT_INPUTS.mirnas, mirnaId));
    const tRes = resolveSeqWithAAHandling(targetId, CURRENT_INPUTS.targets);
    const cRes = compId ? resolveSeqWithAAHandling(compId, CURRENT_INPUTS.competitors) : {seq:'', converted:false, note:'', mode:''};

    const targetSeq= tRes.seq;
    const compSeq  = cRes.seq;

    if(!mirnaSeq || !targetSeq){
      const why = (tRes.note || '').trim();
      openModal('Seed Sites', formatError('Could not resolve miRNA and/or target sequences for this row. ' + (why ? ` ${escapeHTML(why)}` : '')));
      return;
    }

    const payload = {
      mirna_seq: mirnaSeq,
      targets: { [targetId]: targetSeq },
      competitors: compSeq ? { [compId]: compSeq } : {},
      allow_gu: !!allowGU,
      max_mismatch: Number.isFinite(maxMM) ? maxMM : 0
    };

    const headers = await getNonceOrKeyHeaders();
    const res = await fetch(SEED_SCAN_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...headers },
      body: JSON.stringify(payload)
    });

    if(!res.ok){
      let msg = 'Seed scanning failed.';
      try{ const j = await res.json(); msg = j.error || msg; }catch(_){ }
      openModal('Seed Sites', formatError(msg));
      return;
    }

    const data = await res.json();
    let hits = Array.isArray(data.hits) ? data.hits : [];

    if(hits.length === 0){
      const convNote = (tRes.converted || cRes.converted)
        ? `<div style="margin-top:6px;color:#333;"><small><em>AA→NT conversion was applied prior to scanning.</em></small></div>` : '';
      openModal('Seed Sites', `<p>No canonical seed matches found under current settings (GU=${allowGU ? 'on':'off'}, max mismatch=${maxMM}).</p>${convNote}`);
      return;
    }

    const tRange = parseIdRange(targetId);
    const cRange = compId ? parseIdRange(compId) : null;

    hits = hits.map(h => {
      if(h.molecule === 'target' && tRange){
        const g = globalCoordForId(targetId, h.start, h.end);
        return { ...h, global_start: g.globalStart, global_end: g.globalEnd };
      }
      if(h.molecule === 'competitor' && cRange){
        const g = globalCoordForId(compId, h.start, h.end);
        return { ...h, global_start: g.globalStart, global_end: g.globalEnd };
      }
      return h;
    });

    LAST_SEED_HITS = hits;
    LAST_SEED_META = { mirnaId, targetId, compId };

    const showGlobalCols = !!(tRange || cRange);

    let html = `<div style="margin-bottom:8px;">Found <b>${hits.length}</b> seed-site hit(s). Coordinates are 1-based on the displayed sequence${showGlobalCols ? ' and global positions are shown when a :start-end range was applied' : ''}.</div>`;
    if(tRes.converted || cRes.converted){
      const modeTxt = byQS('#aa-nt-mode')?.value || 'canonical';
      html += `<div style="margin:6px 0;color:#333;"><small><em>AA→NT conversion applied (${escapeHTML(modeTxt)}) for ${tRes.converted ? 'target' : ''}${tRes.converted && cRes.converted ? ' & ' : ''}${cRes.converted ? 'competitor' : ''}.</em></small></div>`;
    }
    html += `<table style="width:100%;border-collapse:collapse;">
      <thead>
        <tr style="text-align:left;border-bottom:1px solid #ddd;">
          <th>Molecule</th><th>ID</th><th>Start</th><th>End</th>${showGlobalCols ? '<th>Global Start</th><th>Global End</th>' : ''}<th>Seed</th><th>Type</th><th>Wobble</th><th>Mismatches</th><th>Upstream</th>
        </tr>
      </thead>
      <tbody>`;
    hits.forEach(h => {
      html += `<tr style="border-bottom:1px solid #f0f0f0;">
        <td>${escapeHTML(h.molecule)}</td>
        <td>${escapeHTML(h.id)}</td>
        <td>${h.start}</td>
        <td>${h.end}</td>
        ${showGlobalCols ? `<td>${h.global_start ?? ''}</td><td>${h.global_end ?? ''}</td>` : ''}
        <td>${h.seed_len}</td>
        <td>${escapeHTML(h.seed_type || '')}</td>
        <td>${h.wobble ?? 0}</td>
        <td>${h.mismatches ?? 0}</td>
        <td>${escapeHTML(h.upstream_base ?? '')}</td>
      </tr>`;
    });
    html += `</tbody></table>`;

    const toolbar = `<button id="download-seed-csv" class="btn-action">Download CSV</button>`;
    openModal('Seed Sites', html, toolbar);

    const dlBtn = $('download-seed-csv');
    if(dlBtn){
      bindOnce(dlBtn,'click',downloadSeedCSV,'seedCsvOnce');
    }

  }catch(err){
    openModal('Seed Sites', formatError(err?.message || 'Unexpected error during seed scan.'));
  }
}

function downloadSeedCSV(){
  if(!Array.isArray(LAST_SEED_HITS) || LAST_SEED_HITS.length === 0) return;
  const showGlobal = LAST_SEED_HITS.some(h => typeof h.global_start !== 'undefined');
  const headers = [
    'Molecule','ID','Start','End',
    ...(showGlobal ? ['Global_Start','Global_End'] : []),
    'Seed_Len','Seed_Type','Wobble','Mismatches','Upstream_Base'
  ];
  const rows = [headers.join(',')];
  LAST_SEED_HITS.forEach(h => {
    const cells = [
      h.molecule,
      h.id,
      h.start,
      h.end,
      ...(showGlobal ? [h.global_start ?? '', h.global_end ?? ''] : []),
      h.seed_len,
      h.seed_type || '',
      h.wobble ?? 0,
      h.mismatches ?? 0,
      h.upstream_base ?? ''
    ].map(v=>{
      const s = String(v ?? '');
      return /[",\n]/.test(s) ? `"${s.replace(/"/g,'""')}"` : s;
    });
    rows.push(cells.join(','));
  });
  const nameBits = [];
  if(LAST_SEED_META?.mirnaId) nameBits.push(String(LAST_SEED_META.mirnaId).replace(/[^a-z0-9_\-]+/gi,'_'));
  if(LAST_SEED_META?.targetId) nameBits.push(String(LAST_SEED_META.targetId).replace(/[^a-z0-9_\-]+/gi,'_'));
  const fname = `seed_hits_${nameBits.join('__') || 'results'}.csv`;

  const blob = new Blob([rows.join('\n')], {type:'text/csv'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = fname;
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// =====================================================
// 3D Viewer — friendly plain-text message if missing; viewer if present
// =====================================================
async function ensureNGL(){
  if(GUARDS.nglLoaded) return true;
  try{
    await new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = 'https://cdn.jsdelivr.net/npm/ngl@latest/dist/ngl.min.js';
      s.async = true;
      s.onload = () => resolve();
      s.onerror = () => reject(new Error('Failed to load NGL viewer.'));
      document.head.appendChild(s);
    });
    GUARDS.nglLoaded = true;
    return true;
  }catch(_){ return false; }
}

// Entry from table buttons
async function open3DOrExplain(anyId, kind /* 'target' | 'competitor' */){
  if(!CURRENT_JOB_ID){
    openModalText('3D Viewer', 'Run a prediction first.');
    return;
  }
  const baseId = (parseIdRange(anyId)?.baseId || anyId || '').trim();

  let res;
  try{
    const headers = await getNonceOrKeyHeaders();
    res = await fetchWithTimeout(STRUCTURE_URL(CURRENT_JOB_ID, kind), { method:'GET', headers }, 15000);
  }catch(_){}

  if(!res || !res.ok){
    const prettyId = anyId || '(Unknown)';
    const sample   = baseId || (kind === 'target' ? 'TARGET' : 'COMPETITOR');
    openModalText(
      '3D Viewer',
      `No 3D structure found for ${prettyId}. Upload a PDB or mmCIF named exactly after the FASTA header (for example: ${sample}.pdb or ${sample}.cif), or embed a PDB ID in the FASTA header like "PDB:7YTW", then re-run the prediction.`
    );
    return;
  }

  const ok = await ensureNGL();
  if(!ok){
    openModalText('3D Viewer', 'Could not load the 3D engine (NGL). Check your connection.');
    return;
  }

  try{
    const blob = await res.blob();
    await open3DStageFromBlob(kind, blob, anyId);
  }catch(err){
    openModalText('3D Viewer', err?.message || '3D viewer error.');
  }
}

async function open3DStageFromBlob(kind, blob, anyId){
  const url  = URL.createObjectURL(blob);
  const title = `3D Viewer — ${kind === 'target' ? 'Target' : 'Competitor'}${anyId ? ' • ' + anyId : ''}`;
  const toolbar = `
    <button id="ngl-center" class="toolbar-btn">Center on site</button>
    <button id="ngl-snap" class="toolbar-btn">Snapshot PNG</button>
    <button id="ngl-open" class="toolbar-btn">Open File</button>
  `;
  const html = `<div id="ngl-stage" style="width:100%;height:70vh;background:#0b1020;border-radius:10px;"></div>`;
  openModal(title, html, toolbar);

  const stage = new window.NGL.Stage('ngl-stage', { backgroundColor: 'black' });
  window.addEventListener('resize', () => stage.handleResize(), { passive:true });

  const comp = await stage.loadFile(url);
  comp.addRepresentation('cartoon', { colorScheme: 'chainid' });
  comp.addRepresentation('ball+stick', { multipleBond: true });
  stage.autoView();

  const centerBtn = $('ngl-center');
  const snapBtn   = $('ngl-snap');
  const openBtn   = $('ngl-open');

  if(centerBtn) bindOnce(centerBtn, 'click', () => { stage.autoView(); }, 'centerOnce');
  if(snapBtn) bindOnce(snapBtn, 'click', async () => {
    const img = await stage.makeImage({ factor: 2, antialias: true, trim: false, transparent: false });
    const a = document.createElement('a');
    a.href = img.toDataURL('image/png');
    a.download = `structure_${kind}.png`;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
  }, 'snapOnce');
  if(openBtn) bindOnce(openBtn, 'click', () => {
    const w = window.open(url, '_blank'); if(w) w.opener = null;
  }, 'openOnce');
}

// Legacy helper — plain-text on errors
async function open3DViewer(kind){
  if(!CURRENT_JOB_ID){ openModalText('3D Viewer', 'Run a prediction first.'); return; }
  if(!['target','competitor'].includes(kind)){ openModalText('3D Viewer', 'Invalid molecule kind.'); return; }
  const ok = await ensureNGL();
  if(!ok){ openModalText('3D Viewer', 'Could not load 3D engine.'); return; }

  try{
    const headers = await getNonceOrKeyHeaders();
    const res = await fetch(STRUCTURE_URL(CURRENT_JOB_ID, kind), { method:'GET', headers });
    if(!res.ok){
      openModalText('3D Viewer', 'No 3D structure available (maybe not uploaded or expired).');
      return;
    }
    const blob = await res.blob();
    await open3DStageFromBlob(kind, blob, '');
  }catch(err){
    openModalText('3D Viewer', err?.message || '3D viewer error.');
  }
}

// =====================================================
// Tabs / helpers
// =====================================================
function openTab(element, tabId){
  const targetCard = $(tabId);
  if(!targetCard) return;
  byQSA('.card').forEach(card => card.classList.remove('active'));
  byQSA('.tab-btn').forEach(btn => btn.classList.remove('active'));
  targetCard.classList.add('active');
  if(element && element.classList) element.classList.add('active');
}

function makeTableSortable(tableId){
  const table = document.getElementById(tableId);
  if(!table) return;
  table.querySelectorAll('th').forEach((header, idx) => {
    header.style.cursor = 'pointer';
    header.addEventListener('click', () => {
      const rows = Array.from(table.querySelectorAll('tbody tr'));
      const asc  = header.classList.toggle('asc');
      rows.sort((a,b) => {
        const aText = a.children[idx].textContent.trim();
        const bText = b.children[idx].textContent.trim();
        const na = parseFloat(aText), nb = parseFloat(bText);
        if(!Number.isNaN(na) && !Number.isNaN(nb)){
          return asc ? na - nb : nb - na;
        }
        return asc ? aText.localeCompare(bText) : bText.localeCompare(aText);
      });
      const tbody = table.querySelector('tbody');
      rows.forEach(row => tbody.appendChild(row));
    });
  });
}

function wireTabButtonsOnce(){
  if(GUARDS.tabWiringDone) return;
  const tabs = byQSA('.tab-btn');
  const loader = $('loader');

  tabs.forEach(btn => {
    bindOnce(btn, 'click', () => {
      const name = (btn.textContent || '').toLowerCase();
      if(name.includes('inputs')){
        if(loader){
          text(loader, "Please input your sequences to start a prediction.");
          show(loader);
        }
      }
      if(name.includes('results')){
        const rc = $('results-container');
        if(rc && !rc.innerHTML.trim()){
          setHTML(rc, formatInfo('Results will appear here after you run a prediction.'));
        }
      }
      if (name.includes('introduction')) { if (loader) hide(loader); }
    }, 'tabClick');
  });

  GUARDS.tabWiringDone = true;
}

// =====================================================
// Modal (singleton)
// =====================================================
function ensureModal(){
  if(GUARDS.modalInjected) return;
  const body = document.body;
  ensureSingleton(
    'analysis-modal',
    `
    <div id="analysis-modal" style="position:fixed;inset:0;display:none;align-items:center;justify-content:center;z-index:9999;">
      <div data-overlay style="position:absolute;inset:0;background:rgba(0,0,0,0.45);"></div>
      <div data-panel style="position:relative;max-width:980px;width:96%;max-height:86vh;overflow:auto;background:#fff;border-radius:12px;padding:16px;box-shadow:0 10px 30px rgba(0,0,0,0.3);">
        <div style="display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:8px;">
          <h3 id="modal-title" style="margin:0;">Analysis</h3>
          <div id="modal-tools" style="display:flex;gap:8px;align-items:center;"></div>
          <button id="modal-close" aria-label="Close">✕</button>
        </div>
        <div id="modal-content"></div>
      </div>
    </div>
    `,
    body
  );
  const modal = $('analysis-modal');
  const closeBtn = $('modal-close');
  const overlay = modal?.querySelector('[data-overlay]');
  if(closeBtn) bindOnce(closeBtn,'click',closeModal,'mclose');
  if(overlay)  bindOnce(overlay,'click',closeModal,'moverlay');
  GUARDS.modalInjected = true;
}
function openModal(title, html, toolbarHTML=''){
  ensureModal();
  const modal = $('analysis-modal');
  if(!modal) return;
  text($('modal-title'), title || 'Analysis');
  setHTML($('modal-content'), html || '');
  setHTML($('modal-tools'), toolbarHTML || '');
  modal.style.display = 'flex';
}
function closeModal(){
  const modal = $('analysis-modal');
  if(modal) modal.style.display = 'none';
}

// =====================================================
// Sticky header offset sync (matches CSS --sticky-offset-main)
// =====================================================
function syncStickyOffset(){
  // Try common site header; fall back to 96px (CSS default)
  const header = document.querySelector('header.site-header, header#site-header, .site-header');
  const h = header ? Math.max(64, Math.round(header.getBoundingClientRect().height)) : 96;
  document.documentElement.style.setProperty('--sticky-offset-main', `${h}px`);
}

// =====================================================
// Guard for missing elements (console hint, no crash)
// =====================================================
(function guardMissingElements(){
  const form = $('prediction-form');
  if(!form) console.warn("Prediction form not found on page.");
})();
