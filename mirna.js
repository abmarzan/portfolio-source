
// mirna.js — frontend for SeedScope™ miRNA–RNA affinity (multi-miRNA, multi-target, multi-competitor)
// Core features:
//   • Tolerant FASTA parsing, header matching & range-aware coordinates (Header:Start-End)
//   • Multi-role 3D support (miRNA, target, competitor) from PDB/mmCIF files and PDB IDs
//   • Unified FASTA+PDB cascade per molecule:
//       - If FASTA exists → FASTA provides the nucleotide sequence for scoring
//       - If no FASTA → derive NT from PDB nucleic chains; if protein-only → AA→NT back-translation
//   • NT-first analytics: Seed Sites, IG heatmaps, density plots always run on nucleotide strings
//   • Server CSV/PNG downloads, per-row bundles, “Download All”, RAM-safe batching & progress polling
//   • Multi-file 3D viewer (miRNA/Target/Competitor/combined), best-effort seed highlighting in 3D
//   • Non-blocking auth (nonce/API key/open) and guards so missing/odd PDB never silently kills a job
//
// This file only coordinates UI, pre-validation and result rendering; all model decisions live in app.py.

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
  analysisControlsWired: false,
  modalInjected: false,
  styleInjected: false,
  nglLoaded: false,
  analysisControlsInjected: false,
  stickyShimWired: false
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
const DOWNLOAD_URL   = (jobId) => `${BASE_URL}/download/${jobId}`;            // JSON
const DOWNLOAD_ALL_CSV_URL = (jobId) => `${BASE_URL}/download/${jobId}/all.csv`;
const DOWNLOAD_ROW_CSV_URL = (jobId, interactionId) => `${BASE_URL}/download/${jobId}/${interactionId}.csv`;
const HEATMAP_PNG_URL = (jobId, interactionId, mode, steps) => `${BASE_URL}/download/${jobId}/${interactionId}/heatmap.png?mode=${encodeURIComponent(mode)}&steps=${encodeURIComponent(steps)}`;
// NOTE: 'kind' now accepts 'target' | 'competitor' | 'mirna'
const STRUCTURE_URL   = (jobId, kind) => `${BASE_URL}/structure/${jobId}/${kind}`;

const NONCE_URL      = `${BASE_URL}/nonce`;
const CONFIG_URL     = `${BASE_URL}/config`;
const SEED_SCAN_URL  = `${BASE_URL}/seed_scan`;
const EXPLAIN_URL    = `${BASE_URL}/explain`;

const MAX_FILE_SIZE_MB = 100;

// ================= NEW: tiny util to infer extension from headers ==============
function inferExtFromResponse(res){
  const ctype = (res.headers.get('Content-Type') || '').toLowerCase();
  const dispo = (res.headers.get('Content-Disposition') || '').toLowerCase();
  if(/cif|mmcif/.test(ctype) || /\.mm?cif\b/.test(dispo)) return 'cif';
  if(/pdb|x-pdb|ent/.test(ctype) || /\.pdb\b/.test(dispo)) return 'pdb';
  return 'pdb';
}

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

function getTabbarHeight(){
  const t = getTabsBarEl();
  return t ? Math.round(t.getBoundingClientRect().height) : 0;
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

// Inject premium styles and small utility classes (drop-in)
function injectPremiumStyles(){
  if(GUARDS.styleInjected) return;
  const css = `
    :root{ --sticky-gap: 12px; }
    nav.is-sticky-gap, header.is-sticky-gap { top: 0 !important; }
    .btn-premium{
      padding:10px 14px;
      min-height:42px;
      min-width:130px;
      border-radius:12px;
      border:1px solid #d9d9e3;
      background:linear-gradient(180deg,#ffffff,#f6f7fb);
      font-weight:600;
      letter-spacing:.2px;
      box-shadow:0 1px 1px rgba(0,0,0,.04), 0 8px 20px rgba(17,24,39,.06);
      transition:.15s transform ease,.2s box-shadow ease;
    }
    .btn-premium:hover{
      transform:translateY(-1px);
      box-shadow:0 10px 24px rgba(17,24,39,.09);
    }
    .btn-action{
      min-width:128px;
      min-height:40px;
      padding:9px 12px;
      border-radius:10px;
      font-weight:600;
      border:1px solid #d8dee9;
      background:linear-gradient(180deg,#fff,#f8fafc);
    }
    .btn-accent{background:#0ea5e9;color:#fff;border:1px solid #0284c7;}
    .chip{
      display:inline-block;
      padding:2px 8px;
      border:1px solid #e5e7eb;
      border-radius:10px;
      background:#f8fafc;
      color:#334155;
      font-size:12px;
      margin-left:6px;
    }
    #tabs-anchor{ scroll-margin-top: calc(var(--sticky-offset-main) + var(--sticky-gap)); }
    table#results-table thead th{
      position:static;
      top:0;
      background:#fff;
      z-index:1;
      text-align:center;
    }
    table#results-table tbody tr:hover{filter:brightness(0.98)}
    .toolbar-btn{
      min-height:32px;
      padding:6px 10px;
      border-radius:8px;
      border:1px solid #d8dee9;
      background:#fff;
      font-weight:600
    }
    .info-note{ text-align:center; margin:8px 0; }
    .reload-warning{ text-align:center; margin:8px 0; }
    .precheck-table{width:100%;border-collapse:collapse;margin:6px 0;}
    .precheck-table th,.precheck-table td{
      border-bottom:1px solid #e5e7eb;
      padding:6px 8px;
      text-align:left;
      font-size:13px;
    }
    /* When we detect the sticky nav container, we add this class */
    .is-sticky-gap{ top: calc(var(--sticky-offset-main) + var(--sticky-gap)) !important; }

    /* Premium "Choose file" buttons for FASTA & PDB (light sky blue, matches PDB) */
    input[type="file"].file-premium{
      font-size:13px;
      border-radius:999px;
      border:1px solid #c4ddf9;
      padding:3px;
      background:transparent;
      cursor:pointer;
    }
    input[type="file"].file-premium::-webkit-file-upload-button,
    input[type="file"].file-premium::file-selector-button{
      padding:7px 14px;
      border-radius:999px;
      border:1px solid #93c5fd;
      background:linear-gradient(135deg,#e0f2ff,#bae6fd); /* very light sky blue */
      color:#0f172a;
      font-weight:500;
      letter-spacing:.15px;
      box-shadow:0 1px 4px rgba(15,23,42,.18);
      cursor:pointer;
      background-size:100% 100%;
      transition:
        .12s transform ease,
        .18s box-shadow ease,
        .18s background-color ease;
    }
    input[type="file"].file-premium:hover::-webkit-file-upload-button,
    input[type="file"].file-premium:hover::file-selector-button{
      background:linear-gradient(135deg,#dbeafe,#bfdbfe);
      box-shadow:0 3px 8px rgba(15,23,42,.22);
      transform:translateY(-0.5px);
    }
    input[type="file"].file-premium:active::-webkit-file-upload-button,
    input[type="file"].file-premium:active::file-selector-button{
      transform:translateY(0);
      box-shadow:0 1px 4px rgba(15,23,42,.18);
    }

    /* Shell that keeps "Choose File" + "Clear" in ONE pill box */
    .fasta-file-shell{
      display:flex;
      align-items:stretch;        /* let children fill height, we’ll center inside them */
      flex-wrap:nowrap;
      padding:3px 8px;
      border-radius:12px;         /* ⬅️ rectangle with gentle rounded corners */
      border:1px solid #c4ddf9;
      background:#ffffff;
      width:100%;
      box-sizing:border-box;
      min-height:40px;            /* keep a nice consistent height */
    }
    .fasta-file-shell input[type="file"].file-premium{
      border:none;                 /* remove inner border inside the shell only */
      padding:0;
      box-shadow:none;
    }
      .fasta-file-shell .fasta-clear-btn{
      display:flex;
      align-items:center;         /* vertically center label inside each */
      height:100%;
    }

    /* Small clear buttons for each FASTA picker – match light sky blue */
    .fasta-clear-btn{
      display:inline-flex;
      align-items:center;
      min-height:35px;
      padding:3px 12px;
      margin-left:auto;            /* push Clear to the far right */
      border-radius:999px;
      border:1px solid #93c5fd;
      background:linear-gradient(135deg,#e0f2ff,#bae6fd); /* same family as Choose File */
      font-size:12px;
      font-weight:500;
      color:#0f172a;
      cursor:pointer;
      box-shadow:0 1px 4px rgba(15,23,42,.18);
      transition:.15s background-color ease,
                .15s border-color ease,
                .15s color ease,
                .12s transform ease;
    }

    .fasta-clear-btn:hover{
      background:linear-gradient(135deg,#dbeafe,#bfdbfe);
      border-color:#93c5fd;
      color:#0f172a;
      transform:translateY(-0.5px);
    }
    .fasta-clear-btn:active{
      transform:translateY(0);
      box-shadow:0 1px 4px rgba(15,23,42,.18);
    }

    /* ===================================================
       PREMIUM ADVANCED OPTIONS (only inside #advanced-tab)
       =================================================== */
    #advanced-flags-wrapper{
      margin-top:8px;
    }
    #advanced-flags-wrapper > div{
      background:linear-gradient(135deg,#f0fdf4,#ecfeff); /* very light bluish-green */
      border-radius:10px;
      padding:10px 12px;
      box-shadow:0 1px 1px rgba(15,23,42,.04),
                 0 8px 20px rgba(15,23,42,.06);
    }
    #advanced-flags-wrapper label{
      display:flex;
      gap:8px;
      align-items:center;
      cursor:pointer;
      font-size:14px;
      color:#111827;
    }
    #advanced-tab input[type="checkbox"]{
      -webkit-appearance:none;
      appearance:none;
      width:18px;
      height:18px;
      border-radius:6px;
      border:1px solid #a7f3d0;
      background:linear-gradient(135deg,#ecfdf3,#d1fae5);
      position:relative;
      outline:none;
      cursor:pointer;
      box-shadow:0 1px 3px rgba(16,185,129,.25);
      transition:
        background .18s ease,
        border-color .18s ease,
        box-shadow .18s ease,
        transform .12s ease;
    }
    #advanced-tab input[type="checkbox"]:hover{
      transform:translateY(-0.5px);
      box-shadow:0 4px 10px rgba(16,185,129,.35);
      border-color:#6ee7b7;
    }
    #advanced-tab input[type="checkbox"]:checked{
      background:linear-gradient(135deg,#22c55e,#0ea5e9);
      border-color:#22c55e;
      box-shadow:
        0 0 0 1px rgba(255,255,255,.45) inset,
        0 7px 16px rgba(14,165,233,.55);
    }
    #advanced-tab input[type="checkbox"]::after{
      content:'';
      position:absolute;
      inset:3px 4px 4px 4px;
      border-radius:4px;
      border:2px solid #ffffff;
      border-top:none;
      border-left:none;
      opacity:0;
      transform:scale(.5) rotate(10deg);
      transition:opacity .16s ease, transform .16s ease;
    }
    #advanced-tab input[type="checkbox"]:checked::after{
      opacity:1;
      transform:scale(.9) rotate(45deg);
    }
    #advanced-tab input[type="checkbox"]:disabled{
      cursor:not-allowed;
      opacity:.55;
      box-shadow:none;
    }
    #advanced-tab select#aa-nt-mode{
      border-radius:8px;
      border:1px solid #a7f3d0;
      padding:4px 26px 4px 10px;
      font-size:13px;
      font-weight:500;
      background:linear-gradient(135deg,#f0fdf4,#ecfeff);
      box-shadow:0 1px 3px rgba(199, 243, 228, 1);
    }
    #advanced-tab select#aa-nt-mode:focus{
      outline:none;
      border-color:#22c55e;
      box-shadow:0 0 0 1px rgba(202, 236, 214, 1);
    }
    #advanced-tab select#aa-nt-mode:disabled{
      opacity:.6;
      cursor:not-allowed;
      box-shadow:none;
    }
  `;
  const style = document.createElement('style');
  style.id = 'mirna-js-style';
  style.textContent = css;
  document.head.appendChild(style);
  GUARDS.styleInjected = true;
}

window.addEventListener('load', () => {
  document.querySelectorAll('.fasta-clear-btn').forEach(btn => {
    const fileInput = btn.previousElementSibling;
    if (!fileInput || !fileInput.classList.contains('file-premium')) return;
    if (btn.parentElement && btn.parentElement.classList.contains('fasta-file-shell')) return;

    const shell = document.createElement('span');
    shell.className = 'fasta-file-shell';

    fileInput.parentNode.insertBefore(shell, fileInput);
    shell.appendChild(fileInput);
    shell.appendChild(btn);
  });
});

// Map a tab button → target card id (robust: data- attrs, aria-controls, text label, href)
function targetIdFromButton(btn){
  if(!btn) return null;
  const byAttr = btn.getAttribute('data-target') || btn.dataset?.target || btn.getAttribute('aria-controls');
  if (byAttr && document.getElementById(byAttr)) return byAttr;

  const href = btn.getAttribute('href');
  if (href && href.startsWith('#') && document.getElementById(href.slice(1))) return href.slice(1);

  const label = (btn.textContent || '').toLowerCase().trim();
  if (/^work(\s*flow)?$/.test(label)) return 'workflow-tab';
  if (/^inputs?$/.test(label))        return 'inputs-tab';
  if (/advanced/.test(label))         return 'advanced-tab';
  if (/results?/.test(label))         return 'results-tab';
  return null;
}

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

// --- robust fetch with retries & backoff (uses fetchWithTimeout) ---
async function fetchRetry(url, options={}, ms=30000, retries=2, backoffMs=600){
  let lastErr = null;
  for (let attempt = 0; attempt <= retries; attempt++){
    try{
      const res = await fetchWithTimeout(url, options, ms);
      if(!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      return res;
    }catch(e){
      lastErr = e;
      if (attempt < retries){
        await new Promise(r => setTimeout(r, backoffMs * (attempt + 1)));
        continue;
      }
    }
  }
  throw lastErr;
}


// --- JSON helper with timeout + no-store cache ---
async function fetchJSONWithTimeout(url, opts={}, ms=60000){
  const c = new AbortController();
  const t = setTimeout(()=>c.abort(), ms);
  try{
    const r = await fetch(url, { ...opts, signal: c.signal, cache: 'no-store' });
    if(!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return await r.json();
  } finally { clearTimeout(t); }
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

// ===== Tolerant ID helpers =====
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
function parseIdRange(id){
  const m = String(id||'').match(/^(.+):(\d+)-(\d+)$/);
  if(!m) return null;
  return { baseId: m[1], start: parseInt(m[2],10), end: parseInt(m[3],10) };
}
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
function globalCoordForId(anyId, localStart, localEnd){
  const r = parseIdRange(anyId);
  if(!r) return null;
  const offset = (r.start || 1) - 1;
  return { globalStart: offset + localStart, globalEnd: offset + localEnd };
}
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
  return /[^ACGTUNRYKMSWBVDH]/.test(s);
}
function toRNA(seq){
  return String(seq||'').toUpperCase().replace(/T/g,'U').replace(/[^ACGU]/g, (ch)=>{
    return NUCLEOTIDE_CHARS.has(ch) ? ch : '';
  });
}
const AA2RNA_CANON = {
  A:'GCU', R:'CGU', N:'AAU', D:'GAU', C:'UGU',
  Q:'CAA', E:'GAA', G:'GGU', H:'CAU', I:'AUU',
  L:'UUA', K:'AAA', M:'AUG', F:'UUU', P:'CCU',
  S:'UCU', T:'ACU', W:'UGG', Y:'UAU', V:'GUU',
  U:'UGA', O:'UAG', B:'AAN', Z:'CAN', X:'NNN', '*':'NNN'
};
const AA2RNA_GC = {
  A:'GCC', R:'CGC', N:'AAC', D:'GAC', C:'UGC',
  Q:'CAG', E:'GAG', G:'GGC', H:'CAC', I:'AUC',
  L:'CUG', K:'AAG', M:'AUG', F:'UUC', P:'CCC',
  S:'UCC', T:'ACC', W:'UGG', Y:'UAC', V:'GUG',
  U:'UGA', O:'UAG', B:'AAN', Z:'CAN', X:'NNN', '*':'NNN'
};
const AA2RNA_NNK = {
  A:'NNK', R:'NNK', N:'NNK', D:'NNK', C:'NNK',
  Q:'NNK', E:'NNK', G:'NNK', H:'NNK', I:'NNK',
  L:'NNK', K:'NNK', M:'AUG', F:'NNK', P:'NNK',
  S:'NNK', T:'NNK', W:'UGG', Y:'NNK', V:'NNK',
  U:'UGA', O:'UAG', B:'NNK', Z:'NNK', X:'NNN', '*':'NNN'
};
function aaToRNAWithMode(aaSeq, mode='canonical'){
  const s = String(aaSeq||'').replace(/\s+/g,'').toUpperCase();
  const table = mode === 'gc_balanced' ? AA2RNA_GC : (mode === 'nnk' ? AA2RNA_NNK : AA2RNA_CANON);
  let out = '';
  for(const ch of s){
    if(table[ch]) out += table[ch];
    else if(NUCLEOTIDE_CHARS.has(ch)) out += ch;
    else out += 'NNN';
  }
  return out;
}
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

// Handy: get staged 3D files from HTML baskets (fallback to empty)
function getBasketFiles(kind){
  try{
    if(window.__BASKET__ && Array.isArray(window.__BASKET__[kind])) return window.__BASKET__[kind];
  }catch(_){}
  return [];
}

// ================= NEW: gather additional sources for multi-PDB ===============
function getLegacyInputFiles(kind){
  const map = {
    target: $('target-file'),
    competitor: $('competitor-file'),
    mirna: $('mirna-file')
  };
  const el = map[kind];
  if (!el?.files?.length) return [];
  return Array.from(el.files);
}
function collectStructureSources(kind, primaryBlob=null, primaryExt='pdb'){
  const sources = [];
  if (primaryBlob) {
    sources.push({label:`server_${kind}.${primaryExt}`, type:'server', payload: primaryBlob, ext: primaryExt});
  }
  // staged via HTML basket (lives across the run)
  const staged = getBasketFiles(kind);
  staged.forEach((f,i)=> sources.push({label:`staged_${i+1}_${f.name}`, type:'basket', payload: f, ext: (f.name.split('.').pop()||'pdb').toLowerCase()}));
  // legacy inputs (if any still in the inputs)
  const legacy = getLegacyInputFiles(kind);
  legacy.forEach((f,i)=> sources.push({label:`legacy_${i+1}_${f.name}`, type:'legacy', payload: f, ext: (f.name.split('.').pop()||'pdb').toLowerCase()}));
  return sources;
}

function filterSourcesForRowByRowItem(sources, kind, rowItem){
  // Best-effort: if a rowItem is provided and its ID appears in the PDB filename/label,
  // keep only those matches. Otherwise, fall back to all sources so we never break 3D.
  if(!rowItem || !sources || !sources.length) return sources || [];

  let wantedId = '';
  if(kind === 'target'){
    wantedId = (rowItem.target_id || '').trim();
  }else if(kind === 'mirna'){
    wantedId = (rowItem.primary_molecule_id || rowItem.mirna_id || '').trim();
  }else if(kind === 'competitor'){
    wantedId = (rowItem.competitor_id || '').trim();
  }

  if(!wantedId) return sources;

  const needle = wantedId.toLowerCase();
  const filtered = sources.filter(src=>{
    const label = (src && src.label ? String(src.label) : '');
    return label.toLowerCase().includes(needle);
  });

  // If nothing matches by name, keep original list so viewer still works
  return filtered.length ? filtered : sources;
}

// --- ultra-smooth scroll animator (cubic ease, cancelable) ---
let __scrollAnim = null;
function cancelScrollAnim(){
  if (__scrollAnim){ cancelAnimationFrame(__scrollAnim.rafId); __scrollAnim = null; }
}
function animateScrollTo(targetY, duration = 450){
  cancelScrollAnim();
  const startY = window.pageYOffset;
  const dist   = targetY - startY;
  const start  = performance.now();
  const ease = t => (t < 0.5) ? 4*t*t*t : 1 - Math.pow(-2*t + 2, 3) / 2; // easeInOutCubic

  function step(now){
    const p = Math.min(1, (now - start) / duration);
    const y = Math.round(startY + dist * ease(p));
    window.scrollTo(0, y);
    if (p < 1 && __scrollAnim) {
      __scrollAnim.rafId = requestAnimationFrame(step);
    } else {
      __scrollAnim = null;
      // snap exactly to the pixel-perfect final target (no visible jump)
      window.scrollTo(0, targetY);
    }
  }
  __scrollAnim = { rafId: requestAnimationFrame(step) };

  // If the user interacts, stop the animation so it feels native
  ['wheel','touchstart','keydown','mousedown'].forEach(ev =>
    window.addEventListener(ev, cancelScrollAnim, { once:true, passive:true })
  );
}

// Apply premium sticky gap ONLY to the tab bar, and paint the gap white (solid)
function ensureStickyGapForTabs(){
  // Never apply sticky-gap to nav/header
  document.querySelectorAll('nav.is-sticky-gap, header.is-sticky-gap')
    .forEach(el => el.classList.remove('is-sticky-gap'));

  // Find your tab buttons container
  const tabbar = document.querySelector('.tab-bar-sticky, .sticky-tabs, .tabbar, .tabs');
  if (!tabbar) return;

  // Make sure the tabbar is sticky and uses the premium offset + gap
  const cs = window.getComputedStyle(tabbar);
  if (cs.position !== 'sticky') {
    tabbar.style.position = 'sticky';
  }
  tabbar.classList.add('is-sticky-gap');
  tabbar.style.top = `calc(var(--sticky-offset-main) + var(--sticky-gap))`;
  tabbar.style.zIndex = '11';     // above white shim
  if (!tabbar.style.background || tabbar.style.background === 'initial') {
    tabbar.style.background = '#fff'; // solid background under the buttons themselves
  }

  // Create a single fixed white shim (the visible "gap") just below the nav
  let shim = document.getElementById('sticky-gap-shim');
  if (!shim) {
    shim = document.createElement('div');
    shim.id = 'sticky-gap-shim';
    document.body.appendChild(shim);
  }

  // Toggle shim only when the tabbar is actually "stuck" at its top position
  const updateShim = () => {
    const topPx = parseInt(getComputedStyle(tabbar).top || '0', 10) || 0;
    const rectTop = Math.round(tabbar.getBoundingClientRect().top);
    const stuck = rectTop <= topPx + 1; // allow a 1px tolerance
    shim.style.display = stuck ? 'block' : 'none';
  };

  // Wire once
  if (!GUARDS.stickyShimWired){
    window.addEventListener('scroll', updateShim, { passive:true });
    window.addEventListener('resize', () => { updateShim(); }, { passive:true });
    GUARDS.stickyShimWired = true;
  }

  // Initial evaluation
  updateShim();
}

// Smooth scroll so the *first heading inside the section* sits exactly under
// the sticky header + white gap + tabbar height (no extra lines).
function scrollToCardTop(id){
  const el = document.getElementById(id);
  if(!el) return;

  // zero out any element scroll-margin so math is exact
  const prevSM = el.style.scrollMarginTop;
  el.style.scrollMarginTop = '0px';

  // ensure layout is stable before measuring
  requestAnimationFrame(() => {
    const desired = getStickySum(true); // header + white gap + tab bar height
    const absTop  = window.pageYOffset + el.getBoundingClientRect().top;
    const targetY = Math.max(0, Math.round(absTop - desired));

    animateScrollTo(targetY, 480);  // smooth as butter
    // restore
    setTimeout(() => { el.style.scrollMarginTop = prevSM; }, 0);
  });
}


// =====================================================
// Initialization
// =====================================================
document.addEventListener('DOMContentLoaded', async () => {
  injectPremiumStyles();
  await loadConfig();
  ensureModal(); // make sure modal exists early
  syncStickyOffset(); // keep sticky headers perfect
  ensureStickyGapForTabs(); // add premium gap to sticky nav
  window.addEventListener('resize', () => { syncStickyOffset(); ensureStickyGapForTabs(); ensureTabsAnchor();}, { passive:true });

  const loader = $('loader');
  if(loader){
    text(loader, "Please input your sequences to start a prediction.");
    show(loader);
  }

  // --- HARDEN SUBMIT so the page never navigates, changes hash, or adds '?'
  (function hardenSubmit(){
    const form = $('prediction-form');

    // Force safe form config at runtime
    if (form) {
      form.method = 'post';
      // Neutralize any action that is a hash or non-empty URL (prevents #workflow jumps)
      form.setAttribute('action', '');
      form.removeAttribute('action');
      form.setAttribute('novalidate', 'novalidate');
    }

    // Intercept the form submit at capture phase (wins over other listeners)
    window.addEventListener('submit', (ev) => {
      const f = ev.target;
      if (f && f.id === 'prediction-form') {
        ev.preventDefault();
        ev.stopPropagation();
        if (typeof ev.stopImmediatePropagation === 'function') ev.stopImmediatePropagation();
        handleSubmit(ev);

        // Clean any accidental query/hash immediately
        const cleanUrl = window.location.pathname + window.location.search.replace(/^\?$/, '');
        if (window.location.hash) {
          history.replaceState(null, '', cleanUrl); // remove #anything
        } else if (window.location.search === '?') {
          history.replaceState(null, '', cleanUrl);
        }
      }
    }, true);

    // Neutralize every possible "Run" trigger (anchors, buttons with formaction, etc.)
    const runSelectors = [
      '#run-prediction',
      '[data-run="prediction"]',
      'form#prediction-form button[type="submit"]',
      'form#prediction-form input[type="submit"]',
      'form#prediction-form button[formaction]',
      'form#prediction-form [type="submit"][formaction]',
      'form#prediction-form a[href*="#"]'
    ];
    document.querySelectorAll(runSelectors.join(',')).forEach((el) => {
      // Remove any navigation attributes that could hijack the flow
      el.removeAttribute('href');
      el.removeAttribute('formaction');
      el.removeAttribute('formtarget');

      bindOnce(el, 'click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (typeof e.stopImmediatePropagation === 'function') e.stopImmediatePropagation();
        const f = $('prediction-form');
        if (!f) return;
        // Trigger OUR submit path
        (f.requestSubmit ? f.requestSubmit()
                        : f.dispatchEvent(new Event('submit', { cancelable: true })));
      }, 'runClickGuard2');
    });

    // If something else still manages to push #workflow, snap it back instantly
    window.addEventListener('hashchange', (ev) => {
      if (window.location.hash && /workflow/i.test(window.location.hash)) {
        ev.preventDefault?.();
        history.replaceState(null, '', window.location.pathname + window.location.search);
      }
    }, true);
  })();


  // Link file pickers → textareas (sequence FASTA inputs)
  bindFileToTextarea('mirna-seq-file', 'primary-seqs');
  bindFileToTextarea('target-seq-file', 'target-seq');
  bindFileToTextarea('competitor-seq-file', 'competitor-seq');

  // Premium styling + Clear buttons for each FASTA file input
  const fastaFileMap = [
    { fileId: 'mirna-seq-file',      textareaId: 'primary-seqs'   },
    { fileId: 'target-seq-file',     textareaId: 'target-seq'     },
    { fileId: 'competitor-seq-file', textareaId: 'competitor-seq' }
  ];

  fastaFileMap.forEach(({ fileId, textareaId }) => {
    const input = $(fileId);
    if (!input) return;

    // make it match the light sky-blue PDB style
    input.classList.add('file-premium');

    // create a small Clear button next to each FASTA picker (once)
    if (!input.dataset.hasClearBtn) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.textContent = 'Clear';
      btn.className = 'fasta-clear-btn';
      input.insertAdjacentElement('afterend', btn);

      btn.addEventListener('click', () => {
        input.value = '';
        const ta = $(textareaId);
        if (ta) ta.value = '';
      });

      input.dataset.hasClearBtn = '1';
    }
  });

  // (optional) also make PDB inputs use the same light sky-blue style
  ['mirna-file','target-file','competitor-file'].forEach(id => {
    const el = $(id);
    if (el) el.classList.add('file-premium');
  });


  // Form submit
  const form = $('prediction-form');
  if(form && !GUARDS.formBindingDone){
    bindOnce(form, 'submit', handleSubmit, 'submitGuard');
    GUARDS.formBindingDone = true;
  }

  // Premium look for primary buttons if present
  ['load-sample-btn','clear-btn','clear-inputs-btn','seed-scan-global-btn','explain-global-btn'].forEach(id=>{
    const el = $(id);
    if(el) el.classList.add('btn-premium');
  });

  // Advanced options + tabs
  injectAdvancedOnce();
  wireTabButtonsOnce();
  scrollTabsToTop();
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
    const id = header.replace(/\|.*$/,'').trim();
    const m = header.match(/\bchain\b\s*=\s*([A-Za-z0-9])/i);
    if(m && id){ hints[id] = m[1].toUpperCase(); }
  }
  return hints;
}

async function fetchServerHeatmapPng(item, mode = 'ig_target', steps = 64) {
  if (!CURRENT_JOB_ID) {
    throw new Error('No active job id for heatmap.');
  }
  if (!item || !item.interaction_id) {
    throw new Error('Row is missing interaction_id for heatmap.');
  }

  const headers = await getNonceOrKeyHeaders();
  const url = HEATMAP_PNG_URL(
    CURRENT_JOB_ID,
    item.interaction_id,
    mode,
    steps
  );

  const res = await fetchRetry(url, { method: 'GET', headers }, 45000, 2, 700);
  if (!res.ok) {
    throw new Error(`Heatmap PNG request failed: ${res.status}`);
  }
  return await res.blob();
}

// =====================================================
// Submit handler
// =====================================================
async function handleSubmit(event){
  event.preventDefault();
  gotoResultsTab();

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
      staged_target_files: getBasketFiles('target').map(f => f.name),
      staged_competitor_files: getBasketFiles('competitor').map(f => f.name)
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

  // Premium guard: require at least one representation (FASTA or PDB/ID) for BOTH miRNA and target
  const hasMirnaFasta      = !!primarySeqs;
  const hasMirnaPdbFile    = ($('mirna-file')?.files?.length || 0) > 0;
  const hasMirnaPdbStaged  = getBasketFiles('mirna').length > 0;
  const hasMirnaAny        = hasMirnaFasta || hasMirnaPdbFile || hasMirnaPdbStaged;

  const hasTargetFasta        = !!targetSeq;
  const targetPdbIdsGuard     = extractPdbIdsFromFasta(targetSeq); // PDB IDs in target FASTA headers also count
  const hasTargetPdbFile      = ($('target-file')?.files?.length || 0) > 0;
  const hasTargetPdbStaged    = getBasketFiles('target').length > 0;
  const hasTargetAny          = hasTargetFasta || hasTargetPdbFile || hasTargetPdbStaged ||
                                (targetPdbIdsGuard && targetPdbIdsGuard.length > 0);

  // Competitor is completely optional: FASTA, PDB, both, or none — no impact on starting the analysis.

  // If either miRNA or target is missing (no FASTA, no PDB, no PDB ID), do NOT start the analysis
  if (!hasMirnaAny || !hasTargetAny) {
    const rw = resultsContainer?.querySelector('.reload-warning');
    if (rw) rw.remove();

    const missingBits = [];
    if (!hasMirnaAny) {
      missingBits.push(
        '<li><strong>miRNA</strong> — provide a FASTA sequence in “Primary miRNAs” or attach a PDB/mmCIF file.</li>'
      );
    }
    if (!hasTargetAny) {
      missingBits.push(
        '<li><strong>Target</strong> — provide a FASTA sequence (e.g., 3′UTR / CDS fragment), attach a PDB/mmCIF file, or specify a PDB ID in the FASTA header.</li>'
      );
    }

    setHTML(resultsContainer, `
      <div class="staging-box"
           style="background:linear-gradient(135deg,#fff7ed,#e0f2fe);
                  border-radius:12px;
                  border:1px solid #fed7aa;
                  padding:12px;
                  margin:8px 0;">
        <div style="font-weight:600;margin-bottom:4px;">We need your core pair to start</div>
        <p style="margin:0 0 4px;color:#334155;font-size:14px;">
          To launch the analysis, please provide both a <strong>miRNA</strong> and a <strong>target</strong> in any of these forms:
          FASTA, PDB/mmCIF, or (for targets) a PDB ID inside the FASTA header.
        </p>
        <ul style="margin:0 0 4px 18px;padding:0;font-size:13px;color:#1f2937;">
          ${missingBits.join('')}
        </ul>
        <p style="margin:0;font-size:12px;color:#64748b;">
          Competitor remains optional — add it in FASTA and/or PDB if you want us to model competitive displacement.
        </p>
      </div>
    `);

    if (loader) hide(loader);
    return;
  }

  // Soft hint: if one of them is PDB-only, remind that some client-side plots prefer FASTA
  if (!hasMirnaFasta || !hasTargetFasta) {
    prependHTML(resultsContainer, `
      <div class="staging-box"
           style="background:linear-gradient(135deg,#f0f9ff,#eef2ff);
                  border-radius:12px;
                  border:1px solid #bfdbfe;
                  padding:10px;
                  margin:8px 0;">
        <div style="font-weight:600;margin-bottom:2px;">Running with structural-only input</div>
        <p style="margin:0;color:#334155;font-size:13px;">
          Your combination (FASTA and/or PDB) is valid — we will start the analysis. Just note that some client-side views
          (Seed Sites, IG heatmaps) work best when FASTA sequences are also provided.
        </p>
      </div>
    `);
  }

  // Require miRNA FASTA headers only if FASTA text was supplied
  if (primarySeqs && !hasFastaHeaders(primarySeqs)) {
    setHTML(resultsContainer, formatError(
      'Your miRNA input is missing FASTA headers. Please add lines starting with ">" (e.g., >hsa-let-7a-5p) so results can be labeled correctly.'
    ));
    return;
  }

  // Count records for ETA hint (targets/competitors may be filled by PDB-only later; allow 0 here)
  const mirnaCount = countFastaRecords(primarySeqs);
  let tgtCount  = countFastaRecords(targetSeq);      if(!tgtCount && targetSeq)  tgtCount  = 1;
  let compCount = countFastaRecords(competitorSeq);  if(!compCount && competitorSeq) compCount = 1;

  // Friendly info
  //const estTotal = (mirnaCount || 0) * (Math.max(tgtCount, 1)) * (Math.max(compCount, 1));
  //prependHTML(resultsContainer, formatInfo(
  //  `Detected ${tgtCount||0} target(s) and ${compCount||0} competitor(s) from FASTA. Staged 3D files: target=${getBasketFiles('target').length}, competitor=${getBasketFiles('competitor').length}. Estimated evaluations: ${estTotal}.`
  //));

  // Non-blocking tips
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

  // Optional 3D files:
  // The HTML fetch wrapper will append window.__BASKET__ files automatically on /predict.
  // For resilience, also include any still-selected legacy inputs.
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
      if(res.status === 404) throw new Error('The server was restarted while your job was running, so it was lost. Please click Run Prediction again.');
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
          if(lt) lt.textContent = (data.queue_position > 0)
            ? `Queued: you are number ${data.queue_position} in line. Your analysis starts automatically, please keep this page open.`
            : `Processing... ${completed}/${total} completed`;
          show(loader);
        }

        // stall hint if progress hasn't changed for 180s
        if(Number.isFinite(completed) && completed !== lastCompleted){
          lastCompleted = completed; lastTick = Date.now();
        }else if(!(data.queue_position > 0) && Date.now() - lastTick > 180000){
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
          lastTick = Date.now();
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
          const headers = await getNonceOrKeyHeaders();
          let finalDataSoft = null;
          try {
            finalDataSoft = await fetchJSONWithTimeout(
              DOWNLOAD_URL(job_id),
              { method:'GET', headers },
              60000
            );
          } catch(_){}
          if(finalDataSoft){
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
        try {
          const headers = await getNonceOrKeyHeaders();
          const finalData = await fetchJSONWithTimeout(
            DOWNLOAD_URL(job_id),
            { method:'GET', headers },
            60000
          );

          predictionResults = finalData.results || [];
          displayResults(predictionResults, finalData);

          if(loader){
            text(loader, "✅ Prediction completed. Results are shown below.");
            setTimeout(() => hide(loader), 3000);
          }
        } catch (err){
          // Graceful fallback: give a direct link and clear the loader
          prependHTML(resultsContainer, `
            <div class="note error" style="margin:8px 0;">
              Couldn’t fetch results quickly. You can still open them directly:
              <a href="${DOWNLOAD_URL(job_id)}" target="_blank" rel="noopener">Open results JSON</a>
            </div>
          `);
          if(loader) hide(loader);
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
    <div class="staging-box" style="background:#f8fffb;border:1px solid #bbf7d0;border-radius:10px;padding:10px;margin:8px 0;">
      <div class="staging-head" style="display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:6px;">
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
      <small style="color:#475569;">If a PDB isn’t nucleotide and if AA to NT conversion gets too noise or doesn’t match the FASTA, it’s kept for visualization with a “not used in scoring” note. Protein chains are auto back-translated for seed/IG scanning when enabled.</small>
    </div>
  `;
  prependHTML(rc, html);

  RUN_MANIFEST = RUN_MANIFEST || {};
  RUN_MANIFEST.precheck = { targets: data.targets || [], competitors: data.competitors || [] };
}

// =====================================================
// Display results (sorted by baseline; gradient by baseline)
// + wires existing analysis controls + per-row action buttons + row badges + manifest/buttons
// (UPDATED: adds 3D miRNA + 3D All, structure-features badge, rowItem passthrough)
// =====================================================
function displayResults(results, finalData=null){
  const container = $('results-container');
  if(!container) return;

  setHTML(container, '');

  if(!results || results.length === 0){
    setHTML(container, '<p>No results to display.</p>');
    return;
  }

  // Wire up existing analysis controls (in HTML) and show them
  injectAnalysisControls(container);

  // Update RUN_MANIFEST with any server-side manifest/meta if present
  if(finalData && finalData.manifest){
    RUN_MANIFEST = { ...(RUN_MANIFEST || {}), server_manifest: finalData.manifest };
  }

  // Top badges about run
  const runBadgesId = 'run-badges';
  const anyStruct = hasAnyStructure();
  const aaOn = !!$('aa-convert-flag')?.checked;
  const aaMode = (byQS('#aa-nt-mode')?.value || 'canonical');
  const topBadges = `
    <div id="${runBadgesId}" style="display:flex;gap:8px;flex-wrap:wrap;justify-content:center;margin:6px 0 8px;">
      <span class="badge ${anyStruct ? 'ok':'off'}">PDB present: ${anyStruct ? 'yes':'no'}</span>
      <span class="badge ${aaOn ? 'warn':'off'}">AA→NT: ${aaOn ? 'yes':'no'}${aaOn ? ` (${escapeHTML(aaMode)})` : ''}</span>
      <span class="badge off">Seed/IG computed on NT</span>
    </div>
  `;
  appendHTML(container, topBadges);

  // Sort by baseline desc
  results.sort((a,b) =>
    safeParseFloat(b.predicted_affinity_baseline ?? b.baseline_score ?? 0, 0) -
    safeParseFloat(a.predicted_affinity_baseline ?? a.baseline_score ?? 0, 0)
  );

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

  const legendHTML = `
  <div id="affinity-legend" class="affinity-legend" style="margin-bottom:10px;text-align:center;">
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

  // Table
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

    const isRange = (!!tid && /:\d+-\d+/.test(tid)) || (!!cid && /:\d+-\d+/.test(cid));
    const tolT = tid ? (!exactKeyExists(CURRENT_INPUTS.targets, parseIdRange(tid)?.baseId || tid) && !!lookupTolerant(CURRENT_INPUTS.targets, parseIdRange(tid)?.baseId || tid)) : false;
    const tolC = cid ? (!exactKeyExists(CURRENT_INPUTS.competitors, parseIdRange(cid)?.baseId || cid) && !!lookupTolerant(CURRENT_INPUTS.competitors, parseIdRange(cid)?.baseId || cid)) : false;
    const isTol = tolT || tolC;

    const badgeBits = [];
    if (typeof item.pdb_used !== 'undefined'){
      badgeBits.push(`<span class="badge ${item.pdb_used ? 'ok':'off'}">PDB used: ${item.pdb_used ? 'yes':'no'}</span>`);
    }else if (anyStruct){
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
    const badgesHTML = badgeBits.length ? badgeBits.join(' ') : `<span class="badge off">Notes unavailable</span>`;

    const seedBtn    = `<button class="seed-btn btn-action" data-row="${idx}">Seed Sites</button>`;
    const heatBtn    = `<button class="heatmap-btn btn-action" data-row="${idx}">Heatmap</button>`;
    const csvBtn     = `<button class="rowcsv-btn btn-action" data-row="${idx}">Row CSV</button>`;
    const t3dBtn     = `<button class="t3d-btn btn-action" data-row="${idx}">3D Target</button>`;
    const c3dBtn     = `<button class="c3d-btn btn-action" data-row="${idx}">3D Comp</button>`;
    const m3dBtn     = `<button class="m3d-btn btn-action" data-row="${idx}">3D miRNA</button>`;
    const all3dBtn   = `<button class="x3d-btn btn-action" data-row="${idx}">3D All</button>`;
    const bundleBtn  = `<button class="bundle-btn btn-action" data-row="${idx}">Download</button>`;

    const actionBlock = `
      <div class="action-grid" style="display:grid;grid-template-columns: repeat(3, minmax(120px, 1fr));gap:8px;">
        ${seedBtn}
        ${heatBtn}
        ${csvBtn}
        ${t3dBtn}
        ${c3dBtn}
        ${m3dBtn}
        ${all3dBtn}
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

  // Range / tolerant filter chips
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
        await open3DOrExplain(item.target_id || '', 'target', item);
      }else if(t.classList.contains('c3d-btn')){
        await open3DOrExplain(item.competitor_id || '', 'competitor', item);
      }else if(t.classList.contains('m3d-btn')){
        await open3DOrExplain(item.primary_molecule_id || item.mirna_id || '', 'mirna', item);
      }else if(t.classList.contains('x3d-btn')){
        await open3DCombined(item);
      }else if(t.classList.contains('bundle-btn')){
        await handleBundleClick(item);
      }
    }, 'resultsActions');
  }
}

function hasAnyStructure(){
  const tgt = getBasketFiles('target').length;
  const cmp = getBasketFiles('competitor').length;
  const mir = getBasketFiles('mirna').length || 0;
  const legacy =
    ($('target-file')?.files?.length || 0) +
    ($('competitor-file')?.files?.length || 0) +
    ($('mirna-file')?.files?.length || 0);
  return (tgt + cmp + mir + legacy) > 0;
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
// Wire existing analysis controls (HTML block) once
// Adds Allow G:U, Max mismatches, and two global buttons.
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
        <option value="seed_density">Seed-hit density (fast)</option>
      </select>
    </label>
    <label class="ctrl"><span>Steps</span><input id="heatmap-steps" type="number" value="64" min="10" max="200" step="2"></label>

    <button id="seed-scan-global-btn" class="btn-premium">Seed Sites (top row)</button>
    <button id="explain-global-btn"   class="btn-premium btn-accent">Heatmap (top row)</button>
  </div>
  `;

  prependHTML(container, html);

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
    await handleHeatmapClick(predictionResults[0]);
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
// Heatmap (server PNG → retry; then /explain → client PNG;
//          plus pure client modes: seed_density & seed_matrix)
// =====================================================

async function handleHeatmapClick(item){
  const modeSel  = byQS('#heatmap-mode');
  const stepsInp = byQS('#heatmap-steps');

  const rawMode  = (modeSel?.value || 'ig_target').toLowerCase();

  const steps = Math.max(
    10,
    Math.min(200, parseInt(stepsInp?.value || '64', 10) || 64)
  );

  openModal('Heatmap', smallSpinner('Generating heatmap...'));

  const effMode = (rawMode === 'ig_competitor' && !(item.competitor_id || '').trim())
    ? 'ig_target'
    : rawMode;

  // For IG target / IG competitor we try server PNG first.
  // Only seed_density is pure client in this function now.
  const isPureClient = effMode === 'seed_density';

  if (!isPureClient && CURRENT_JOB_ID && item.interaction_id) {
    try {
      const headers = await getNonceOrKeyHeaders();
      const res = await fetchRetry(
        HEATMAP_PNG_URL(CURRENT_JOB_ID, item.interaction_id, effMode, steps),
        { method: 'GET', headers },
        45000,
        2,
        700
      );

      if (res.ok) {
        const blob = await res.blob();
        const url  = URL.createObjectURL(blob);

        const html = `
          <div class="hm-wrapper">
            <div class="hm-toolbar">
              <button class="btn btn-sm btn-outline" id="hm-download-btn">
                Download PNG
              </button>
            </div>
            <img id="hm-img" src="${url}" alt="Heatmap"
                 style="max-width:100%;height:auto;border-radius:8px;border:1px solid #e5e7eb;"/>
          </div>
        `;
        openModal('Heatmap', html);

        setTimeout(() => {
          const btn = document.getElementById('hm-download-btn');
          if (btn) {
            btn.addEventListener('click', () => {
              const a = document.createElement('a');
              a.href = url;
              a.download = `${item.interaction_id || 'heatmap'}.png`;
              a.click();
            });
          }
        }, 0);

        return; // ⬅️ IMPORTANT: do not fall through to client fallback when PNG works
      }
    } catch (err) {
      console.warn('Server heatmap PNG failed, falling back to client IG/seed.', err);
    }
  }

  // Fallback: client IG / 1D seed-density (for ig_target / ig_competitor / seed_density)
  await clientExplainHeatmapFallback(item, effMode, true);
}


async function clientExplainHeatmapFallback(item, forcedMode, forceCanvasPNG = false){
  const mirnaId  = item.primary_molecule_id ?? item.mirna_id;
  const targetId = item.target_id ?? '';
  const compId   = item.competitor_id ?? '';

  const mirnaSeqRaw = lookupTolerant(CURRENT_INPUTS.mirnas, mirnaId);
  const tRes = resolveSeqWithAAHandling(targetId, CURRENT_INPUTS.targets);
  const cRes = compId
    ? resolveSeqWithAAHandling(compId, CURRENT_INPUTS.competitors)
    : { seq: '', converted: false, note: '', mode: '' };

  const targetSeq = tRes.seq;
  const compSeq   = cRes.seq;
  const mirnaSeq  = mirnaSeqRaw ? toRNA(mirnaSeqRaw) : '';

  if (!mirnaSeq || !targetSeq) {
    const mc = $('modal-content');
    if (mc) {
      setHTML(mc, formatError('Could not resolve miRNA and/or target sequences for this row.'));
    }
    return;
  }

  const uiMode  = (forcedMode || byQS('#heatmap-mode')?.value || 'ig_target').toLowerCase();
  const uiSteps = Math.max(
    10,
    Math.min(200, parseInt(byQS('#heatmap-steps')?.value || '64', 10) || 64)
  );

  // ========= 1) PURE CLIENT MODE: seed_density =========

  // --- Seed-density mode (pure client, no /explain) ---
  if (uiMode === 'seed_density') {
    const density = computeSeedDensityArray(targetId, targetSeq, mirnaSeq);
    const vals    = normalizeArray(density);
    const canvas  = makeHeatCanvas(targetSeq, vals, `Seed density — ${targetId}`);

    showCanvasAsModalPNG(
      canvas,
      'Heatmap — Seed density',
      `${(item.interaction_id || 'local')}_seed_density.png`
    );

    if (tRes.converted || cRes.converted) {
      const modeTxt = byQS('#aa-nt-mode')?.value || 'canonical';
      const mc = $('modal-content');
      if (mc) {
        appendHTML(
          mc,
          `<div style="margin-top:6px;color:#333;">
             <small><em>AA→NT conversion applied (${escapeHTML(modeTxt)}) before visualization.</em></small>
           </div>`
        );
      }
    }
    return;
  }

  // --- Seed-density mode (pure client, no /explain) ---
  if (uiMode === 'seed_density') {
    const density = computeSeedDensityArray(targetId, targetSeq, mirnaSeq);
    const vals    = normalizeArray(density);
    const canvas  = makeHeatCanvas(targetSeq, vals, `Seed density — ${targetId}`);

    showCanvasAsModalPNG(
      canvas,
      'Heatmap — Seed density',
      `${(item.interaction_id || 'local')}_seed_density.png`
    );

    if (tRes.converted || cRes.converted) {
      const modeTxt = byQS('#aa-nt-mode')?.value || 'canonical';
      const mc = $('modal-content');
      if (mc) {
        appendHTML(
          mc,
          `<div style="margin-top:6px;color:#333;">
             <small><em>AA→NT conversion applied (${escapeHTML(modeTxt)}) before visualization.</em></small>
           </div>`
        );
      }
    }
    return;
  }

  // =================== 2) IG MODES: use /explain with fallback ===================

  try {
    const mc = $('modal-content');
    if (mc) {
      setHTML(mc, smallSpinner('Computing attributions.'));
    }

    const headers = await getNonceOrKeyHeaders();
    const body = JSON.stringify({
      mirna_seq: mirnaSeq,
      target_seq: targetSeq,
      competitor_seq: compSeq || undefined,
      steps: uiSteps,
      mode: uiMode
    });

    let data = null;
    try {
      const res = await fetchRetry(
        EXPLAIN_URL,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', ...headers },
          body
        },
        45000,
        1,
        700
      );
      if (res.ok) {
        data = await res.json();
      } else {
        data = null;
      }
    } catch (_) {
      data = null;
    }

    if (data && (Array.isArray(data.target_attrib) || Array.isArray(data.attribution))) {
      const targAttr = Array.isArray(data.target_attrib) ? data.target_attrib : data.attribution;
      const compAttr = Array.isArray(data.competitor_attrib) ? data.competitor_attrib : null;

      const targVals = normalizeArray(targAttr.slice(0, targetSeq.length));
      const titleT   = `IG → Target — ${targetId}`;
      const cnvT     = makeHeatCanvas(targetSeq, targVals, titleT);

      if (uiMode === 'ig_competitor' && compSeq && Array.isArray(compAttr)) {
        const cVals  = normalizeArray(compAttr.slice(0, compSeq.length));
        const titleC = `IG → Competitor — ${compId}`;
        const cnvC   = makeHeatCanvas(compSeq, cVals, titleC);

        const stack = document.createElement('canvas');
        const pad   = 12;
        const W     = Math.max(cnvT.width, cnvC.width);
        const H     = cnvT.height + cnvC.height + pad;
        stack.width = W;
        stack.height = H;

        const g = stack.getContext('2d');
        g.fillStyle = '#fff';
        g.fillRect(0, 0, W, H);
        g.drawImage(cnvT, 0, 0);
        g.drawImage(cnvC, 0, cnvT.height + pad / 2);

        showCanvasAsModalPNG(
          stack,
          'Heatmap — IG (target + competitor)',
          `${(item.interaction_id || 'local')}_ig_both.png`
        );
      } else {
        showCanvasAsModalPNG(
          cnvT,
          'Heatmap — IG → Target',
          `${(item.interaction_id || 'local')}_ig_target.png`
        );
      }

      if (tRes.converted || cRes.converted) {
        const modeTxt = byQS('#aa-nt-mode')?.value || 'canonical';
        const mc2 = $('modal-content');
        if (mc2) {
          appendHTML(
            mc2,
            `<div style="margin-top:6px;color:#333;">
               <small><em>AA→NT conversion applied (${escapeHTML(modeTxt)}) for
               ${tRes.converted ? 'target' : ''}${tRes.converted && cRes.converted ? ' & ' : ''}${cRes.converted ? 'competitor' : ''}.</em></small>
             </div>`
          );
        }
      }
      return;
    }

    // --- No usable attribution returned → seed-based fallback, not hard failure ---
    const density = computeSeedDensityArray(targetId, targetSeq, mirnaSeq);
    const vals    = normalizeArray(density);
    const canvas  = makeHeatCanvas(targetSeq, vals, `Seed density — ${targetId}`);

    showCanvasAsModalPNG(
      canvas,
      'Heatmap — Seed-based fallback',
      `${(item.interaction_id || 'local')}_seed_profile.png`
    );

    const mc3 = $('modal-content');
    if (mc3) {
      appendHTML(
        mc3,
        `<div style="margin-top:6px;color:#333;">
           <small><em>Attribution service unavailable; showing a seed-based profile instead.</em></small>
         </div>`
      );
    }

  } catch (err) {
    // --- Hard error in IG path → still show a seed-based profile, but with message ---
    const density = computeSeedDensityArray(targetId, targetSeq, mirnaSeq);
    const vals    = normalizeArray(density);
    const canvas  = makeHeatCanvas(targetSeq, vals, `Seed density — ${targetId}`);

    showCanvasAsModalPNG(
      canvas,
      'Heatmap — Seed-based fallback',
      `${(item.interaction_id || 'local')}_seed_profile.png`
    );

    const mc = $('modal-content');
    if (mc) {
      appendHTML(
        mc,
        `<div style="margin-top:6px;color:#333;">
           <small><em>${escapeHTML(
             err?.message || 'Attribution failed; showing a seed-based profile instead.'
           )}</em></small>
         </div>`
      );
    }
  }
}

// Smooth Viridis-style colour map used by heatmaps
function viridisColor(value, alpha = 1) {
  const palette = [
    [68, 1, 84],    // #440154
    [59, 82, 139],  // #3b528b
    [33, 144, 141], // #21908d
    [93, 201, 99],  // #5dc963
    [253, 231, 37]  // #fde725
  ];

  const v = Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
  const pos = v * (palette.length - 1);
  const i0 = Math.floor(pos);
  const i1 = Math.min(i0 + 1, palette.length - 1);
  const t  = pos - i0;

  const r = Math.round(palette[i0][0] + t * (palette[i1][0] - palette[i0][0]));
  const g = Math.round(palette[i0][1] + t * (palette[i1][1] - palette[i0][1]));
  const b = Math.round(palette[i0][2] + t * (palette[i1][2] - palette[i0][2]));

  return `rgba(${r},${g},${b},${alpha})`;
}

// -----------------------------------------------------
// 1D letter-strip canvas, normalization, seed density
// -----------------------------------------------------

// Draw a letter-strip heatmap to canvas (Viridis blocks + base letters)
function makeHeatCanvas(seq, values01, titleText){
  const pad = 14, bw = 12, gap = 2, bh = 22;
  const W = pad*2 + (bw + gap) * seq.length;
  const H = pad*2 + bh + 26;

  const c = document.createElement('canvas');
  c.width = Math.max(200, W);
  c.height = H;
  const g = c.getContext('2d');

  // background
  g.fillStyle = '#ffffff';
  g.fillRect(0, 0, c.width, c.height);

  // title
  if (titleText){
    g.fillStyle = '#111';
    g.font = '600 14px ui-monospace, Menlo, Consolas, monospace';
    g.textAlign = 'left';
    g.textBaseline = 'top';
    g.fillText(titleText, pad, 6);
  }

  // strip
  const y = pad + 14;
  g.font = 'bold 12px ui-monospace, Menlo, Consolas, monospace';
  g.textAlign = 'center';
  g.textBaseline = 'middle';

  for (let i = 0; i < seq.length; i++){
    const x = pad + i * (bw + gap);
    const v = Math.max(0, Math.min(1, values01[i] || 0));
    g.fillStyle = viridisColor(v, 0.95);
    g.fillRect(x, y, bw, bh);
    g.fillStyle = '#000';
    g.fillText(seq[i] || '', x + bw/2, y + bh/2);
  }

  // border
  g.strokeStyle = '#e5e7eb';
  g.strokeRect(0.5, 0.5, c.width - 1, c.height - 1);

  return c;
}

// Normalize an array of numbers to [0,1] (abs)
function normalizeArray(arr){
  if (!Array.isArray(arr) || !arr.length) return [];
  const max = arr.reduce((m, v) => {
    const a = Math.abs(v || 0);
    return a > m ? a : m;
  }, 1e-12);
  return arr.map(v => Math.abs(v || 0) / max);
}

// Compute seed-density / seed-profile array (0..N) for a target.
// 1) If we have LAST_SEED_HITS for this target, use them.
// 2) Otherwise, fall back to a simple 7-mer seed-match profile vs mirnaSeq (if provided).
function computeSeedDensityArray(targetId, targetSeq, mirnaSeq){
  const L = targetSeq.length || 0;
  let density = new Array(L).fill(0);

  // 1) Use explicit seed hits when available (from Seed Sites)
  if (Array.isArray(LAST_SEED_HITS) && LAST_SEED_HITS.length) {
    LAST_SEED_HITS
      .filter(h => h.molecule === 'target' && h.id === targetId)
      .forEach(h => {
        for (let i = Math.max(0, h.start - 1); i < Math.min(L, h.end); i++) {
          density[i] += 1;
        }
      });

    // If we actually filled something, keep it
    if (density.some(v => v > 0)) {
      return density;
    }
  }

  // 2) Fallback: derive a seed-based profile directly from sequences (no Seed Sites required)
  if (mirnaSeq) {
    return computeSeedMatchProfile(mirnaSeq, targetSeq);
  }

  // 3) Last resort: flat zero profile (should rarely be hit now)
  return density;
}

// ------------------------------
// Seed-match profile & 2D matrix
// ------------------------------

// Simple base normalizer: T -> U, uppercase all
function normBase(b){
  if (!b) return '';
  b = b.toUpperCase();
  return (b === 'T') ? 'U' : b;
}

// Complementarity score: 1 for Watson–Crick, 0.5 for G-U wobble, 0 otherwise
function seedPairScore(mi, tg){
  mi = normBase(mi);
  tg = normBase(tg);
  if ((mi === 'A' && tg === 'U') || (mi === 'U' && tg === 'A')) return 1;
  if ((mi === 'C' && tg === 'G') || (mi === 'G' && tg === 'C')) return 1;
  if ((mi === 'G' && tg === 'U') || (mi === 'U' && tg === 'G')) return 0.5; // wobble
  return 0;
}

/**
 * Compute a 1D seed-match profile along the target:
 * for each target position we assign the best 7-mer seed score
 * overlapping that nucleotide.
 */
function computeSeedMatchProfile(mirnaSeq, targetSeq){
  if (!mirnaSeq || !targetSeq) return [];

  mirnaSeq  = String(mirnaSeq).trim().toUpperCase();
  targetSeq = String(targetSeq).trim().toUpperCase();

  const Lm = mirnaSeq.length;
  const Lt = targetSeq.length;
  if (Lm < 7 || Lt < 7) {
    // too short for a proper 7-mer; just return zeros
    return new Array(Lt).fill(0);
  }

  // canonical 7-mer seed: positions 2–8 (1-based) → index 1 (0-based)
  const seedStart = 1;
  const seedLen   = Math.min(7, Lm - seedStart);
  const seed      = mirnaSeq.slice(seedStart, seedStart + seedLen);

  const posScores = new Array(Lt).fill(0);

  // slide seed along target
  for (let offset = 0; offset <= Lt - seedLen; offset++){
    let score = 0;
    // miRNA 5' seed binds target 3' → reverse seed for pairing
    for (let s = 0; s < seedLen; s++){
      const mi = seed[seedLen - 1 - s];    // reverse
      const tg = targetSeq[offset + s];
      score += seedPairScore(mi, tg);
    }
    // propagate this window score to all covered positions, keep the max
    for (let s = 0; s < seedLen; s++){
      const idx = offset + s;
      if (score > posScores[idx]) posScores[idx] = score;
    }
  }

  return posScores;
}


// Build premium seed-site summary chips from LAST_SEED_HITS
function buildSeedChipSummaryHTML(){
  if (!Array.isArray(LAST_SEED_HITS) || !LAST_SEED_HITS.length) return '';

  const groups = new Map();
  for (const h of LAST_SEED_HITS){
    const key = `${h.molecule}:${h.id}`;
    if (!groups.has(key)){
      groups.set(key, {
        molecule: h.molecule,
        id: h.id,
        count: 0,
        bestSeed: '',
        minMM: Infinity,
        wobbleHits: 0
      });
    }
    const g = groups.get(key);
    g.count++;
    if (!g.bestSeed && h.seed_type) g.bestSeed = h.seed_type;
    const mm = typeof h.mismatches === 'number' ? h.mismatches : 0;
    if (mm < g.minMM) g.minMM = mm;
    if (h.wobble && h.wobble > 0) g.wobbleHits++;
  }

  const chips = [];
  for (const g of groups.values()){
    const seedTxt = g.bestSeed ? ` • ${g.bestSeed}` : '';
    const mmTxt   = Number.isFinite(g.minMM) ? ` • min mm: ${g.minMM}` : '';
    const wobTxt  = g.wobbleHits ? ` • wobble hits: ${g.wobbleHits}` : '';
    chips.push(
      `<span class="chip">${escapeHTML(g.molecule)}:${escapeHTML(g.id)} • ${g.count} hit(s)${seedTxt}${mmTxt}${wobTxt}</span>`
    );
  }

  if (!chips.length) return '';

  return `
    <div style="margin-top:8px;text-align:center;">
      <div style="font-size:12px;color:#475569;margin-bottom:4px;">
        Seed-site summary (cached from “Seed Sites” scan)
      </div>
      ${chips.join(' ')}
    </div>
  `;
}

// Show a canvas as PNG in the modal with “Open” & “Download” + seed chips
function showCanvasAsModalPNG(canvas, title, filename){
  const data = canvas.toDataURL('image/png');
  const img  = `<img id="hm-img" src="${data}" alt="Heatmap"
                    style="max-width:100%;height:auto;border:1px solid #e5e7eb;border-radius:8px;"/>`;
  const tools = `
    <button id="hm-open" class="toolbar-btn">Open in new tab</button>
    <button id="hm-save" class="toolbar-btn">Download PNG</button>
  `;
  openModal(title, img, tools);

  // Attach seed summary chips (if any cached hits)
  const mc = $('modal-content');
  if (mc) {
    const chipsHtml = buildSeedChipSummaryHTML();
    if (chipsHtml) {
      appendHTML(mc, chipsHtml);
    }
  }

  bindOnce(
    $('hm-open'),
    'click',
    () => {
      const w = window.open(data, '_blank');
      if (w) w.opener = null;
    },
    'hmOpenCanvas'
  );
  bindOnce(
    $('hm-save'),
    'click',
    () => {
      const a = document.createElement('a');
      a.href = data;
      a.download = filename || 'heatmap.png';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    },
    'hmSaveCanvas'
  );
}

// Render a simple seed-density heat strip from LAST_SEED_HITS (client fallback)
function renderSeedDensityFromScan(mirnaSeq, targetId, targetSeq){
  const L = targetSeq.length;
  const density = new Array(L).fill(0);
  if (Array.isArray(LAST_SEED_HITS)){
    LAST_SEED_HITS
      .filter(h => h.molecule === 'target' && h.id === targetId)
      .forEach(h => {
        for (let i = Math.max(0, h.start - 1); i < Math.min(L, h.end); i++) density[i] += 1;
      });
  }
  const max = density.reduce((m, v) => v > m ? v : m, 1);
  const norm = density.map(v => v / max);

  let strip = `<div style="font-family:ui-monospace, Menlo, Consolas;overflow:auto;border:1px solid #eee;border-radius:8px;padding:8px;">`;
  strip += `<div style="white-space:nowrap;">`;
  for (let i = 0; i < L; i++){
    const color = viridisColor(norm[i] || 0, 0.85);
    strip += `<span title="pos ${i+1} • ${(targetSeq[i]||'')} • ${(norm[i]||0).toFixed(3)}"
                    style="display:inline-block;min-width:10px;padding:2px 0;text-align:center;background:${color};color:#000;border-radius:2px;margin:0 1px;">${escapeHTML(targetSeq[i] || '')}</span>`;
  }
  strip += `</div></div>`;
  const note = Array.isArray(LAST_SEED_HITS) && LAST_SEED_HITS.length
    ? ''
    : '<br/><small style="color:#666;">Tip: run Seed Sites first for a more informative density.</small>';
  return `<div><h4 style="margin:6px 0;">Seed density (client fallback)</h4>${strip}${note}</div>`;
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
// 3D Viewer — Multi-file support + Chain/Role panel + Seed highlights
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

async function fetchStructureBlob(kind){
  if(!CURRENT_JOB_ID) return null;
  try{
    const headers = await getNonceOrKeyHeaders();
    const res = await fetchWithTimeout(STRUCTURE_URL(CURRENT_JOB_ID, kind), { method:'GET', headers }, 20000);
    if(!res.ok) return null;
    const ext = inferExtFromResponse(res);
    const blob = await res.blob();
    return { blob, ext };
  }catch(_){ return null; }
}

async function open3DCombined(rowItem){
  // Try to fetch each kind from server; add staged+legacy files for all kinds
  const ok = await ensureNGL();
  if(!ok){ openModalText('3D Viewer', 'Could not load the 3D engine (NGL).'); return; }

    const title = `3D Viewer — Combined (miRNA + Target + Competitor)`;
    const html = `
    <div style="display:grid;grid-template-columns:280px 1fr;gap:10px;max-height:86vh;">
        <div id="ngl-side" style="overflow:auto;border:1px solid #e5e7eb;border-radius:10px;padding:10px;">
        <div style="font-weight:700;margin-bottom:6px;">Chains & Roles</div>
        <div id="chain-list"></div>
        <hr/>
        <div style="display:grid;gap:6px;">
            <label>miRNA chain<select id="role-mirna"></select></label>
            <label>Target chain<select id="role-target"></select></label>
            <label>Competitor chain<select id="role-competitor"></select></label>
            <label>Residue offset<input id="role-offset" type="number" value="0" step="1" style="width:100%;"></label>
            <button id="role-focus-m" class="toolbar-btn">Focus miRNA</button>
            <button id="role-focus-t" class="toolbar-btn">Focus Target</button>
            <button id="role-focus-c" class="toolbar-btn">Focus Competitor</button>
            <hr/>
            <button id="seed-highlight" class="toolbar-btn">Highlight Seed Sites (best-effort)</button>
            <button id="clear-highlights" class="toolbar-btn">Clear Highlights</button>
            <!-- NEW: per-hit focus -->
            <button id="focus-hit" class="toolbar-btn">Focus Selected Hit</button>
            <select id="hit-select"></select>
        </div>
        </div>
        <div id="ngl-stage" style="width:100%;height:70vh;background:#0b1020;border-radius:10px;"></div>
    </div>
    `;
    const tools = `
    <button id="ngl-center" class="toolbar-btn">Center</button>
    <button id="ngl-snap" class="toolbar-btn">Snapshot PNG</button>
    `;
    openModal(title, html, tools);


  const stage = new window.NGL.Stage('ngl-stage', { backgroundColor: 'black' });
  window.addEventListener('resize', () => stage.handleResize(), { passive:true });

  const compInfos = [];   // {comp, id, label, kind, chains:[{name, polymerType, length, rep}]}
  const highlightReps = [];

  // Helper: load a single source (File or Blob)
  async function loadSource(src, kind){
    const comp = await stage.loadFile(src.payload, { ext: src.ext || 'pdb' });
    const info = { comp, id: `${kind}:${src.label}`, label: src.label, kind, chains: [] };

    // create per-chain cartoon reps so they can be toggled individually
    comp.structure.eachChain((cp)=>{
      const cname = cp.chainname || cp.modelIndex + ':' + cp.index;
      const poly  = cp.polymerType || 'unknown';
      const len   = cp.residueCount;
      const rep = comp.addRepresentation('cartoon', { sele: `:${cname}`, colorScheme: 'chainname' });
      info.chains.push({ name: cname, polymerType: poly, length: len, rep });
    });

    // hetero / ligands visibility
    comp.addRepresentation('ball+stick', { sele: 'hetero', multipleBond: true });

    compInfos.push(info);
  }

  // Build all sources (server + staged + legacy) for the three kinds
  const kinds = ['mirna','target','competitor'];
  for (const k of kinds){
    const s = await fetchStructureBlob(k);
    let extra = collectStructureSources(k, s?.blob || null, s?.ext || 'pdb');
    extra = filterSourcesForRowByRowItem(extra, k, rowItem);
    for (const src of extra){
      await loadSource(src, k);
    }
  }

  if (compInfos.length === 0){
    setHTML($('modal-content'), formatError('No 3D structures were found for miRNA/Target/Competitor. Upload PDB/mmCIF or add PDB IDs in FASTA and re-run.'));
    return;
  }

  stage.autoView();

  // UI: chain list + role selects
  function rebuildUI(){
    const chainList = $('chain-list');
    if(!chainList) return;
    const opts = [];
    let html = '';
    compInfos.forEach((ci, ciIdx)=>{
      html += `<div style="margin:6px 0 2px 0;font-weight:600;">${escapeHTML(ci.kind.toUpperCase())} — ${escapeHTML(ci.label)}</div>`;
      ci.chains.forEach((ch, chIdx)=>{
        html += `
          <label style="display:flex;align-items:center;gap:6px;margin:2px 0;">
            <input type="checkbox" data-ci="${ciIdx}" data-ch="${chIdx}" checked />
            <span style="font-family:ui-monospace;">${escapeHTML(ch.name)}</span>
            <small class="chip">${escapeHTML(ch.polymerType||'')}</small>
            <small class="chip">${ch.length || ''}</small>
          </label>`;
        opts.push({label:`${ci.kind}:${ch.name}`, ciIdx, chIdx});
      });
    });
    chainList.innerHTML = html;

    const sM = $('role-mirna'), sT = $('role-target'), sC = $('role-competitor');
    [sM,sT,sC].forEach(sel=>{
      if(!sel) return;
      sel.innerHTML = '<option value="">— select —</option>' + opts.map(o=>`<option value="${o.ciIdx}:${o.chIdx}">${escapeHTML(o.label)}</option>`).join('');
    });

    // naive auto-guess: shortest nucleic as miRNA; longer nucleic as target; another as competitor
    const nucChains = opts.filter(o=>{
      const ch = compInfos[o.ciIdx].chains[o.chIdx];
      return (ch.polymerType || '').toLowerCase().includes('nuc');
    }).sort((a,b)=>{
      const la = compInfos[a.ciIdx].chains[a.chIdx].length||0;
      const lb = compInfos[b.ciIdx].chains[b.chIdx].length||0;
      return la - lb;
    });
    if (nucChains.length){
      const short = nucChains[0];
      sM.value = `${short.ciIdx}:${short.chIdx}`;
      if (nucChains.length > 1){
        sT.value = `${nucChains[nucChains.length-1].ciIdx}:${nucChains[nucChains.length-1].chIdx}`;
      }
      if (nucChains.length > 2){
        sC.value = `${nucChains[1].ciIdx}:${nucChains[1].chIdx}`;
      }
    }

    // wire chain toggles
    byQSA('#chain-list input[type="checkbox"]').forEach(cb=>{
      cb.addEventListener('change', ()=>{
        const ci = parseInt(cb.dataset.ci,10);
        const ch = parseInt(cb.dataset.ch,10);
        const rep = compInfos[ci].chains[ch].rep;
        rep.setVisibility(cb.checked);
      });
    });
  }
  rebuildUI();

    function rebuildHitSelect(rowScoped=false){
    const sel = $('hit-select'); if(!sel) return;
    let hits = Array.isArray(LAST_SEED_HITS) ? [...LAST_SEED_HITS] : [];

    if (rowScoped && rowItem) {
        hits = hits.filter(h =>
        (rowItem.target_id && h.molecule === 'target' && h.id === rowItem.target_id) ||
        (rowItem.competitor_id && h.molecule === 'competitor' && h.id === rowItem.competitor_id)
        );
    }

    if (!hits.length){
        sel.innerHTML = '<option value="">— no cached hits —</option>';
        return;
    }
    sel.innerHTML = hits.map((h, i) =>
        `<option value="${i}">${h.molecule}:${h.id} • ${h.start}-${h.end}${h.global_start?` (g${h.global_start}-${h.global_end})`:''} • seed${h.seed_len}</option>`
    ).join('');
    }

    rebuildHitSelect(true); // combined view is opened from a specific row

    bindOnce($('focus-hit'), 'click', () => {
    const selBox = $('hit-select');
    if (!selBox) return;
    const idx = parseInt(selBox.value||'-1',10);
    if (isNaN(idx) || !Array.isArray(LAST_SEED_HITS) || !LAST_SEED_HITS[idx]) {
        alert('No hit selected. Run Seed Sites first.'); return;
    }
    const h = LAST_SEED_HITS[idx];
    const which = (h.molecule === 'competitor') ? 'competitor' : 'target';
    const selChain = getSelChain(which);
    if (!selChain){ alert(`Pick a ${which} chain first.`); return; }

    const offset = parseInt(($('role-offset')?.value||'0'),10) || 0;
    const start  = (h.global_start || h.start) + offset;
    const end    = (h.global_end   || h.end)   + offset;

    stage.setFocus(
        selChain.comp.structure.getView(
        new window.NGL.Selection(`:${selChain.name} and resno ${start}-${end}`)
        )
    );
    }, 'focusHitOnce_combined');


  function getSelChain(which){ // 'mirna'|'target'|'competitor'
    const map = { mirna:'role-mirna', target:'role-target', competitor:'role-competitor' };
    const el = $(map[which]); if(!el || !el.value) return null;
    const [ciIdx, chIdx] = el.value.split(':').map(x=>parseInt(x,10));
    const ci = compInfos[ciIdx]; if(!ci) return null;
    const ch = ci.chains[chIdx]; if(!ch) return null;
    return { ciIdx, chIdx, comp: ci.comp, name: ch.name, rep: ch.rep };
  }

  function focusChain(which){
    const sel = getSelChain(which); if(!sel) return;
    stage.setFocus( sel.comp.structure.getView(new window.NGL.Selection(`:${sel.name}`)) );
  }

  // Seed highlight (best-effort: assumes residue numbering ~ 1..N)
  function clearHighlights(){
    while (highlightReps.length){
      const r = highlightReps.pop();
      try{ r.component.removeRepresentation(r.rep); }catch(_){}
    }
  }
  function highlightSeedSites(){
    clearHighlights();
    const targetSel = getSelChain('target');
    if(!targetSel){
      alert('Pick a Target chain first.');
      return;
    }
    const offset = parseInt(($('role-offset')?.value||'0'),10) || 0;
    const hits = Array.isArray(LAST_SEED_HITS) ? LAST_SEED_HITS.filter(h => h.molecule === 'target' && (!LAST_SEED_META || !LAST_SEED_META.targetId || h.id === LAST_SEED_META.targetId)) : [];
    if(!hits.length){
      alert('No cached seed hits for this row. Run “Seed Sites” first.');
      return;
    }
    hits.forEach(h=>{
      const start = (h.global_start || h.start) + offset;
      const end   = (h.global_end   || h.end)   + offset;
      const rep = targetSel.comp.addRepresentation('spacefill', { sele: `:${targetSel.name} and resno ${start}-${end}` });
      highlightReps.push({ component: targetSel.comp, rep });
    });
    stage.autoView();
  }

  bindOnce($('role-focus-m'), 'click', ()=>focusChain('mirna'),'rfm');
  bindOnce($('role-focus-t'), 'click', ()=>focusChain('target'),'rft');
  bindOnce($('role-focus-c'), 'click', ()=>focusChain('competitor'),'rfc');
  bindOnce($('seed-highlight'), 'click', highlightSeedSites, 'seedhl');
  bindOnce($('clear-highlights'), 'click', clearHighlights, 'seedcl');

  bindOnce($('ngl-center'), 'click', () => stage.autoView(), 'nglcenter2');
    bindOnce($('ngl-snap'), 'click', async () => {
    try{
        const img = await stage.makeImage({ factor: 2, antialias: true, trim: false, transparent: false });

        if (img instanceof Blob) {
        const url = URL.createObjectURL(img);
        const a = document.createElement('a');
        a.href = url;
        a.download = `structure_combined.png`;
        document.body.appendChild(a); a.click(); document.body.removeChild(a);
        URL.revokeObjectURL(url);
        return;
        }

        if (img && typeof img.toDataURL === 'function') {
        const url = img.toDataURL('image/png');
        const a = document.createElement('a');
        a.href = url; a.download = `structure_combined.png`;
        if (typeof a.download === 'string'){
            document.body.appendChild(a); a.click(); document.body.removeChild(a);
        } else {
            const w = window.open(url, '_blank'); if(w) w.opener = null;
        }
        return;
        }

        if (stage.viewer && typeof stage.viewer.makeImage === 'function') {
        stage.viewer.makeImage({ factor: 2, antialias: true, trim: false, transparent: false })
            .then((blob) => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url; a.download = `structure_combined.png`;
            document.body.appendChild(a); a.click(); document.body.removeChild(a);
            URL.revokeObjectURL(url);
            });
        return;
        }

        alert('Snapshot failed: unsupported return type.');
    }catch(e){
        alert('Snapshot failed.');
    }
    }, 'nglsnap2');
}

async function open3DOrExplain(anyId, kind /* 'target'|'competitor'|'mirna' */, rowItem=null){
  if(!CURRENT_JOB_ID){
    openModalText('3D Viewer', 'Run a prediction first.');
    return;
  }

  // Try server structure for the kind
  let primaryBlob = null, primaryExt = 'pdb';
  try{
    const headers = await getNonceOrKeyHeaders();
    const res = await fetchWithTimeout(STRUCTURE_URL(CURRENT_JOB_ID, kind), { method:'GET', headers }, 20000);
    if(res.ok){
      primaryExt = inferExtFromResponse(res);
      primaryBlob = await res.blob();
    }
  }catch(_){ /* ignore, we’ll fallback to staged/legacy */ }

  await open3DStageManager(kind, anyId, { blob: primaryBlob, ext: primaryExt }, rowItem);
}

async function open3DStageManager(kind, anyId, primary, rowItem){
  const ok = await ensureNGL();
  if(!ok){ openModalText('3D Viewer', 'Could not load the 3D engine (NGL).'); return; }

  const prettyKind = kind === 'mirna' ? 'miRNA' : (kind === 'target' ? 'Target' : 'Competitor');
    const title = `3D Viewer — ${prettyKind}${anyId ? ' • ' + anyId : ''}`;
    const html = `
    <div style="display:grid;grid-template-columns:280px 1fr;gap:10px;max-height:86vh;">
        <div id="ngl-side" style="overflow:auto;border:1px solid #e5e7eb;border-radius:10px;padding:10px;">
        <div style="font-weight:700;margin-bottom:6px;">Chains & Roles</div>
        <div id="chain-list"></div>
        <hr/>
        <div style="display:grid;gap:6px;">
            <label>miRNA chain<select id="role-mirna"></select></label>
            <label>Target chain<select id="role-target"></select></label>
            <label>Competitor chain<select id="role-competitor"></select></label>
            <label>Residue offset<input id="role-offset" type="number" value="0" step="1" style="width:100%;"></label>
            <button id="role-focus-m" class="toolbar-btn">Focus miRNA</button>
            <button id="role-focus-t" class="toolbar-btn">Focus Target</button>
            <button id="role-focus-c" class="toolbar-btn">Focus Competitor</button>
            <hr/>
            <button id="seed-highlight" class="toolbar-btn">Highlight Seed Sites (best-effort)</button>
            <button id="clear-highlights" class="toolbar-btn">Clear Highlights</button>
            <!-- NEW: per-hit focus -->
            <button id="focus-hit" class="toolbar-btn">Focus Selected Hit</button>
            <select id="hit-select"></select>
        </div>
        </div>
        <div id="ngl-stage" style="width:100%;height:70vh;background:#0b1020;border-radius:10px;"></div>
    </div>
    `;
    const tools = `
    <button id="ngl-center" class="toolbar-btn">Center</button>
    <button id="ngl-snap" class="toolbar-btn">Snapshot PNG</button>
    `;
    openModal(title, html, tools);

  const stage = new window.NGL.Stage('ngl-stage', { backgroundColor: 'black' });
  window.addEventListener('resize', () => stage.handleResize(), { passive:true });

  let sources = collectStructureSources(kind, primary?.blob || null, primary?.ext || 'pdb');
  sources = filterSourcesForRowByRowItem(sources, kind, rowItem);

  if (sources.length === 0){
    setHTML($('modal-content'), formatError(`No 3D structures found for ${prettyKind}. Upload PDB/mmCIF or add a PDB ID in FASTA and re-run.`));
    return;
  }

  const compInfos = []; // same structure as in combined
  const highlightReps = [];

  async function loadSource(src){
    const comp = await stage.loadFile(src.payload, { ext: src.ext || 'pdb' });
    const info = { comp, id: `${kind}:${src.label}`, label: src.label, kind, chains: [] };

    comp.structure.eachChain((cp)=>{
      const cname = cp.chainname || cp.modelIndex + ':' + cp.index;
      const poly  = cp.polymerType || 'unknown';
      const len   = cp.residueCount;
      const rep = comp.addRepresentation('cartoon', { sele: `:${cname}`, colorScheme: 'chainname' });
      info.chains.push({ name: cname, polymerType: poly, length: len, rep });
    });

    comp.addRepresentation('ball+stick', { sele: 'hetero', multipleBond: true });

    compInfos.push(info);
  }

  for (const src of sources){ await loadSource(src); }
  stage.autoView();

  function rebuildUI(){
    const chainList = $('chain-list');
    if(!chainList) return;
    const opts = [];
    let html = '';
    compInfos.forEach((ci, ciIdx)=>{
      html += `<div style="margin:6px 0 2px 0;font-weight:600;">${escapeHTML(ci.label)}</div>`;
      ci.chains.forEach((ch, chIdx)=>{
        html += `
          <label style="display:flex;align-items:center;gap:6px;margin:2px 0;">
            <input type="checkbox" data-ci="${ciIdx}" data-ch="${chIdx}" checked />
            <span style="font-family:ui-monospace;">${escapeHTML(ch.name)}</span>
            <small class="chip">${escapeHTML(ch.polymerType||'')}</small>
            <small class="chip">${ch.length || ''}</small>
          </label>`;
        opts.push({label:`${ci.kind||prettyKind}:${ch.name}`, ciIdx, chIdx});
      });
    });
    chainList.innerHTML = html;

    const sM = $('role-mirna'), sT = $('role-target'), sC = $('role-competitor');
    [sM,sT,sC].forEach(sel=>{
      if(!sel) return;
      sel.innerHTML = '<option value="">— select —</option>' + opts.map(o=>`<option value="${o.ciIdx}:${o.chIdx}">${escapeHTML(o.label)}</option>`).join('');
    });

    // simple guess for roles
    const nucChains = opts.filter(o=>{
      const ch = compInfos[o.ciIdx].chains[o.chIdx];
      return (ch.polymerType || '').toLowerCase().includes('nuc');
    }).sort((a,b)=>{
      const la = compInfos[a.ciIdx].chains[a.chIdx].length||0;
      const lb = compInfos[b.ciIdx].chains[b.chIdx].length||0;
      return la - lb;
    });
    if (nucChains.length){
      sM.value = `${nucChains[0].ciIdx}:${nucChains[0].chIdx}`;
      if (nucChains.length > 1){
        sT.value = `${nucChains[nucChains.length-1].ciIdx}:${nucChains[nucChains.length-1].chIdx}`;
      }
      if (nucChains.length > 2){
        sC.value = `${nucChains[1].ciIdx}:${nucChains[1].chIdx}`;
      }
    }

    byQSA('#chain-list input[type="checkbox"]').forEach(cb=>{
      cb.addEventListener('change', ()=>{
        const ci = parseInt(cb.dataset.ci,10);
        const ch = parseInt(cb.dataset.ch,10);
        const rep = compInfos[ci].chains[ch].rep;
        rep.setVisibility(cb.checked);
      });
    });
  }
  rebuildUI();

  function rebuildHitSelect(rowScoped=false){
    const sel = $('hit-select'); if(!sel) return;
    let hits = Array.isArray(LAST_SEED_HITS) ? [...LAST_SEED_HITS] : [];

    if (rowScoped && rowItem) {
        hits = hits.filter(h =>
        (rowItem.target_id && h.molecule === 'target' && h.id === rowItem.target_id) ||
        (rowItem.competitor_id && h.molecule === 'competitor' && h.id === rowItem.competitor_id)
        );
    }

    if (!hits.length){
        sel.innerHTML = '<option value="">— no cached hits —</option>';
        return;
    }
    sel.innerHTML = hits.map((h, i) =>
        `<option value="${i}">${h.molecule}:${h.id} • ${h.start}-${h.end}${h.global_start?` (g${h.global_start}-${h.global_end})`:''} • seed${h.seed_len}</option>`
    ).join('');
    }

    rebuildHitSelect(!!rowItem); // single viewer also knows the row

    bindOnce($('focus-hit'), 'click', () => {
    const selBox = $('hit-select');
    if (!selBox) return;
    const idx = parseInt(selBox.value||'-1',10);
    if (isNaN(idx) || !Array.isArray(LAST_SEED_HITS) || !LAST_SEED_HITS[idx]) {
        alert('No hit selected. Run Seed Sites first.'); return;
    }
    const h = LAST_SEED_HITS[idx];
    const which = (h.molecule === 'competitor') ? 'competitor' : 'target';
    const selChain = getSelChain(which);
    if (!selChain){ alert(`Pick a ${which} chain first.`); return; }

    const offset = parseInt(($('role-offset')?.value||'0'),10) || 0;
    const start  = (h.global_start || h.start) + offset;
    const end    = (h.global_end   || h.end)   + offset;

    stage.setFocus(
        selChain.comp.structure.getView(
        new window.NGL.Selection(`:${selChain.name} and resno ${start}-${end}`)
        )
    );
    }, 'focusHitOnce_single');
    


  function getSelChain(which){
    const map = { mirna:'role-mirna', target:'role-target', competitor:'role-competitor' };
    const el = $(map[which]); if(!el || !el.value) return null;
    const [ciIdx, chIdx] = el.value.split(':').map(x=>parseInt(x,10));
    const ci = compInfos[ciIdx]; if(!ci) return null;
    const ch = ci.chains[chIdx]; if(!ch) return null;
    return { ciIdx, chIdx, comp: ci.comp, name: ch.name, rep: ch.rep };
  }
  function focusChain(which){
    const sel = getSelChain(which); if(!sel) return;
    stage.setFocus( sel.comp.structure.getView(new window.NGL.Selection(`:${sel.name}`)) );
  }
  function clearHighlights(){
    while (highlightReps.length){
      const r = highlightReps.pop();
      try{ r.component.removeRepresentation(r.rep); }catch(_){}
    }
  }
  function highlightSeedSites(){
    clearHighlights();
    const targetSel = getSelChain('target');
    if(!targetSel){
      alert('Pick a Target chain first.');
      return;
    }
    const offset = parseInt(($('role-offset')?.value||'0'),10) || 0;

    // Only highlight seeds relevant to this row when available; else fall back to any cached target seeds.
    let hits = Array.isArray(LAST_SEED_HITS) ? LAST_SEED_HITS.filter(h => h.molecule === 'target') : [];
    if (rowItem?.target_id){
      hits = hits.filter(h => h.id === rowItem.target_id);
    }
    if(!hits.length){
      alert('No cached seed hits for this row. Run “Seed Sites” first.');
      return;
    }
    hits.forEach(h=>{
      const start = (h.global_start || h.start) + offset;
      const end   = (h.global_end   || h.end)   + offset;
      const rep = targetSel.comp.addRepresentation('spacefill', { sele: `:${targetSel.name} and resno ${start}-${end}` });
      highlightReps.push({ component: targetSel.comp, rep });
    });
    stage.autoView();
  }

  bindOnce($('role-focus-m'), 'click', ()=>focusChain('mirna'),'rfm_s');
  bindOnce($('role-focus-t'), 'click', ()=>focusChain('target'),'rft_s');
  bindOnce($('role-focus-c'), 'click', ()=>focusChain('competitor'),'rfc_s');
  bindOnce($('seed-highlight'), 'click', highlightSeedSites, 'seedhl_s');
  bindOnce($('clear-highlights'), 'click', clearHighlights, 'seedcl_s');

  bindOnce($('ngl-center'), 'click', () => stage.autoView(), 'nglcenter_s');
  bindOnce($('ngl-snap'), 'click', async () => {
    try{
      const img = await stage.makeImage({ factor: 2, antialias: true, trim: false, transparent: false });

      const fname = `structure_${kind}.png`;

      if (img instanceof Blob) {
        const url = URL.createObjectURL(img);
        const a = document.createElement('a');
        a.href = url; a.download = fname;
        document.body.appendChild(a); a.click(); document.body.removeChild(a);
        URL.revokeObjectURL(url);
        return;
      }

      if (img && typeof img.toDataURL === 'function') {
        const url = img.toDataURL('image/png');
        const a = document.createElement('a');
        a.href = url; a.download = fname;
        if (typeof a.download === 'string'){
          document.body.appendChild(a); a.click(); document.body.removeChild(a);
        } else {
          const w = window.open(url, '_blank'); if(w) w.opener = null;
        }
        return;
      }

      if (stage.viewer && typeof stage.viewer.makeImage === 'function') {
        stage.viewer.makeImage({ factor: 2, antialias: true, trim: false, transparent: false })
          .then((blob) => {
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url; a.download = fname;
            document.body.appendChild(a); a.click(); document.body.removeChild(a);
            URL.revokeObjectURL(url);
          });
        return;
      }

      alert('Snapshot failed: unsupported return type.');
    }catch(e){
      alert('Snapshot failed.');
    }
  }, 'nglsnap_s'); // (optional) distinct bind key for single-viewer
}

// Legacy minimal viewer (kept for compatibility; unused by new flows)
async function open3DViewer(kind){
  if(!CURRENT_JOB_ID){ openModalText('3D Viewer', 'Run a prediction first.'); return; }
  if(!['target','competitor','mirna'].includes(kind)){ openModalText('3D Viewer', 'Invalid molecule kind.'); return; }
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
    await open3DStageManager(kind, '', { blob, ext: inferExtFromResponse(res) }, null);
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
  byQSA('.tab-btn').forEach(btn  => btn.classList.remove('active'));
  targetCard.classList.add('active');
  if(element && element.classList) element.classList.add('active');

  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      scrollToCardTop(tabId); // now truly snaps to the very top
    });
  });
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

// ---- Find the tabs bar (your 4 buttons) & create a scroll anchor just before it
function getTabsBarEl(){
  return document.querySelector('#tabs-bar, .tabs-bar, .tabbar, .tabs, .sticky-tabs, .tab-bar-sticky');
}

function getTabsBarHeight(){
  const el = getTabsBarEl();
  return el ? Math.round(el.getBoundingClientRect().height) || 0 : 0;
}

function ensureTabsAnchor(){
  const tabsBar = getTabsBarEl();
  if(!tabsBar) return;
  if(document.getElementById('tabs-anchor')) return;

  const a = document.createElement('div');
  a.id = 'tabs-anchor';
  // This makes the browser stop with the tabs exactly under the sticky header + white gap.
  a.style.cssText = 'height:0;margin:0;padding:0;scroll-margin-top:calc(var(--sticky-offset-main) + var(--sticky-gap));';
  tabsBar.parentNode.insertBefore(a, tabsBar);
}

// Smooth, “exact top” snap (double-tap to defeat tiny layout jitters)
function scrollTabsToTop(){
  ensureTabsAnchor();
  const a = document.getElementById('tabs-anchor');
  if(!a) return;
  a.scrollIntoView({ block: 'start', behavior: 'smooth' });
  requestAnimationFrame(() => a.scrollIntoView({ block: 'start', behavior: 'auto' }));
}

function gotoResultsTab(){
  const resultsTabButton = Array.from(document.querySelectorAll('button.tab-btn'))
    .find(b => /results/i.test(b.textContent || ''));
  if (resultsTabButton) {
    openTab(resultsTabButton, 'results-tab');
    scrollTabsToTop();
  }
}

function wireTabButtonsOnce(){
  if(GUARDS.tabWiringDone) return;
  const tabs = byQSA('.tab-btn');
  const loader = $('loader');

  tabs.forEach(btn => {
    bindOnce(btn, 'click', (ev) => {
      // Prevent default anchor jumps — we do a precise sticky-aware snap instead
      ev.preventDefault();

      const targetId = targetIdFromButton(btn);
      if (targetId) {
        openTab(btn, targetId);               // switches + snaps to top
        ensureStickyGapForTabs();             // keeps the white gap logic in sync
      }

      const name = (btn.textContent || '').toLowerCase();
      if(name.includes('inputs')){
        if(loader){ text(loader, "Please input your sequences to start a prediction."); show(loader); }
      }
      if(name.includes('results')){
        const rc = $('results-container');
        if(rc && !rc.innerHTML.trim()){
          setHTML(rc, formatInfo('Results will appear here after you run a prediction.'));
        }
      }
      if(name.includes('workflow')){ if(loader) hide(loader); }
    }, 'tabClickSnap');
  });

  GUARDS.tabWiringDone = true;
}

function getStickySum(includeTabs = true){
  const root = getComputedStyle(document.documentElement);
  const off  = parseInt(root.getPropertyValue('--sticky-offset-main') || '96', 10) || 96;
  const gap  = parseInt(root.getPropertyValue('--sticky-gap') || '12', 10) || 12;
  const tabs = includeTabs ? getTabsBarHeight() : 0;
  return off + gap + tabs;
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
  const header = document.querySelector('header.site-header, header#site-header, .site-header, header#main-header');
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
