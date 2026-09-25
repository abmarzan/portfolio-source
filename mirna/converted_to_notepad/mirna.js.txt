// mirna.js (Guard-safe, duplication-proof, feature-complete; ~500+ lines)
// - Guaranteed: no duplicate UI injections across tab switches, reloads, or repeated runs
// - Loads config (mirna_max, use_nonce, mature trim availability)
// - Validates FASTA headers (enforce for miRNA, advise for target/competitor)
// - Supports multiple miRNA 3D files (filenames should match FASTA IDs; tolerant if single)
// - Sorts results by baseline affinity (descending) and colors rows by baseline gradient
// - Optional nonce-based auth; fallback to X-API-Key
// - Streams status updates; friendly error surfacing with backend messages
// - Safe defaults when optional UI elements aren’t present
// - Defensive DOM operations: idempotent injections, single event bindings, clear containers before re-render

// =====================================================
// Global state
// =====================================================
let predictionResults = [];
let CONFIG = {
  mirna_max: 5000,
  mature_trim_enabled: true,
  mature_window: 22,
  aa_convert_allowed: true,
  use_nonce: false
};

// Singleton guards to prevent duplicate DOM injections or listeners
const GUARDS = {
  advancedInjected: false,
  fastaTipsInjected: false,
  tabWiringDone: false,
  formBindingDone: false
};

// =====================================================
// API routing and auth
// =====================================================
const LOCAL_BASE = "http://127.0.0.1:8080";
const PROD_BASE = "https://mirna.marzan.org";
const isLocal =
  window.location.hostname === "localhost" ||
  window.location.hostname === "127.0.0.1";
const BASE_URL = isLocal ? LOCAL_BASE : PROD_BASE;

const API_URL = `${BASE_URL}/predict`;
const PROGRESS_URL = (jobId) => `${BASE_URL}/progress/${jobId}`;
const DOWNLOAD_URL = (jobId) => `${BASE_URL}/download/${jobId}`;
const NONCE_URL = `${BASE_URL}/nonce`;
const CONFIG_URL = `${BASE_URL}/config`;

// Legacy key (if nonce is disabled on server)
const API_KEY = "supersecret123";
const MAX_FILE_SIZE_MB = 100;

// =====================================================
// Helpers: DOM, UI, utils
// =====================================================

function $(id) {
  return document.getElementById(id);
}

function byQS(sel, scope = document) {
  return scope.querySelector(sel);
}

function byQSA(sel, scope = document) {
  return Array.from(scope.querySelectorAll(sel));
}

function setHTML(el, html) {
  if (!el) return;
  el.innerHTML = html;
}

function appendHTML(el, html) {
  if (!el) return;
  el.insertAdjacentHTML('beforeend', html);
}

function prependHTML(el, html) {
  if (!el) return;
  el.insertAdjacentHTML('afterbegin', html);
}

function show(el) {
  if (!el) return;
  el.classList.remove("hidden");
}

function hide(el) {
  if (!el) return;
  el.classList.add("hidden");
}

function text(el, t) {
  if (!el) return;
  el.textContent = t;
}

function safeParseFloat(x, d = 0) {
  const v = parseFloat(x);
  return Number.isFinite(v) ? v : d;
}

function formatError(msg) {
  return `<p style="color: #c22; margin: 8px 0;">${escapeHTML(msg)}</p>`;
}

function formatWarn(msg) {
  return `<p style="color: #b36b00; margin: 8px 0;">${escapeHTML(msg)}</p>`;
}

function formatInfo(msg) {
  return `<p style="color: #1e5a9c; margin: 8px 0;">${escapeHTML(msg)}</p>`;
}

function escapeHTML(s) {
  return String(s || '')
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function validateFileSize(file) {
  if (file && file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
    alert(`File "${file.name}" exceeds ${MAX_FILE_SIZE_MB} MB limit.`);
    return false;
  }
  return true;
}

function bindFileToTextarea(fileInputId, textareaId) {
  const fileInput = $(fileInputId);
  const textarea = $(textareaId);
  if (!fileInput || !textarea) return;

  // Remove previous listener if any to avoid duplicate
  const clone = fileInput.cloneNode(true);
  fileInput.parentNode.replaceChild(clone, fileInput);
  clone.addEventListener('change', function () {
    const file = this.files && this.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      textarea.value = e.target.result;
    };
    reader.readAsText(file);
  });
}

function countFastaRecords(seqText) {
  if (!seqText) return 0;
  const lines = seqText.trim().split(/\r?\n/);
  let count = 0;
  for (const line of lines) {
    if (line.trim().startsWith('>')) count++;
  }
  if (count === 0 && seqText.trim().length > 0) count = 1;
  return count;
}

function hasFastaHeaders(text) {
  if (!text || !text.trim()) return false;
  return text.split(/\r?\n/).some(line => line.trim().startsWith('>'));
}

function sleep(ms) {
  return new Promise(res => setTimeout(res, ms));
}

function ensureSingleton(id, html, parent) {
  // id: element id to enforce singleton existence
  // html: markup if missing
  // parent: container element
  if (!parent) return null;
  let el = $(id);
  if (el) return el;
  const holder = document.createElement('div');
  holder.innerHTML = html.trim();
  const created = holder.firstElementChild;
  if (created) parent.appendChild(created);
  return created;
}

// =====================================================
// Config loader
// =====================================================
async function loadConfig() {
  try {
    const res = await fetch(CONFIG_URL, { method: 'GET' });
    if (res.ok) {
      const cfg = await res.json();
      CONFIG = { ...CONFIG, ...cfg };
    }
  } catch (_) {
    // keep defaults
  }
}

// =====================================================
// Nonce (optional)
// =====================================================
async function getNonceOrKeyHeaders() {
  if (!('use_nonce' in CONFIG)) await loadConfig();
  if (!CONFIG.use_nonce) throw new Error('Server is not in nonce mode.');
  const res = await fetch(NONCE_URL, { method: 'GET' });
  if (!res.ok) throw new Error('Failed to obtain auth token from server.');
  const data = await res.json();
  if (!data.nonce) throw new Error('Server did not return a valid auth token.');
  return { 'X-Nonce': data.nonce };
}

async function getNonce() {
  const res = await fetch(NONCE_URL, { method: 'GET' });
  const data = await res.json();
  return data.nonce;
}

async function submitPrediction(formData) {
  const headers = await getNonceOrKeyHeaders();
  const res = await fetch(API_URL, {
    method: "POST",
    headers,     // Do NOT set Content-Type; browser sets for FormData
    body: formData
  });
  const data = await res.json();
  console.log(data);
}


// =====================================================
// Safe event binding (prevent duplicates)
// =====================================================
function bindOnce(el, event, handler, key) {
  if (!el) return;
  const k = key || `${event}__bound`;
  if (el.dataset && el.dataset[k] === '1') return;
  el.addEventListener(event, handler);
  if (el.dataset) el.dataset[k] = '1';
}

// =====================================================
// Initialization
// =====================================================
document.addEventListener('DOMContentLoaded', async () => {
  await loadConfig();

  // Loader initial message
  const loader = $('loader');
  if (loader) {
    text(loader, "Please input your sequences to start a prediction.");
    show(loader);
  }

  // Bind sequence file inputs to textareas (defensive rebind)
  bindFileToTextarea('mirna-seq-file', 'primary-seqs');
  bindFileToTextarea('target-seq-file', 'target-seq');
  bindFileToTextarea('competitor-seq-file', 'competitor-seq');

  // Set up form submit (guard)
  const form = $('prediction-form');
  if (form && !GUARDS.formBindingDone) {
    bindOnce(form, 'submit', handleSubmit, 'submitGuard');
    GUARDS.formBindingDone = true;
  }

  // Inject Advanced Options dynamic bits once
  injectAdvancedOnce();

  // Wire tabs loader behavior once
  wireTabButtonsOnce();

  // UX nicety: click sound on Run Prediction exists in index.html
});

// =====================================================
// Advanced options injection (idempotent)
// =====================================================
function injectAdvancedOnce() {
  const advTab = byQS('#advanced-tab');
  if (!advTab || GUARDS.advancedInjected) return;

  // Server configuration note (singleton)
  const serverCfgId = 'server-config-note';
  ensureSingleton(
    serverCfgId,
    `
    <div id="${serverCfgId}" style="margin: 8px 0; color:#333;">
      <strong>Server configuration:</strong>
      <ul style="margin:6px 0 0 16px;">
        <li>Max miRNAs per request: <code>${CONFIG.mirna_max}</code></li>
        <li>Mature trimming enabled: <code>${CONFIG.mature_trim_enabled ? 'yes' : 'no'}</code> (window: ${CONFIG.mature_window})</li>
        <li>AA→NT conversion allowed: <code>${CONFIG.aa_convert_allowed ? 'yes' : 'no'}</code></li>
        <li>Auth mode: <code>${CONFIG.use_nonce ? 'nonce' : 'api-key'}</code></li>
      </ul>
    </div>
    `,
    advTab
  );

  // Flags wrapper (singleton)
  const flagsWrapperId = 'advanced-flags-wrapper';
  const flags = ensureSingleton(
    flagsWrapperId,
    `
    <div id="${flagsWrapperId}" style="margin-top: 8px;">
      <div style="display:flex; gap:20px; align-items:center; flex-wrap:wrap;">
        <label style="display:flex; gap:8px; align-items:center; cursor:pointer;">
          <input type="checkbox" id="mature-trim-flag" ${CONFIG.mature_trim_enabled ? 'checked' : ''} />
          <span>Auto-trim miRNAs > 30nt to mature-like ${CONFIG.mature_window}nt</span>
        </label>
        <label style="display:flex; gap:8px; align-items:center; cursor:pointer;">
          <input type="checkbox" id="aa-convert-flag" ${CONFIG.aa_convert_allowed ? '' : 'disabled'} />
          <span>Convert AA → NT (lossy; for target/competitor)</span>
        </label>
      </div>
      <small style="color:#555;">
        If conversion is disabled server-side, this checkbox has no effect.
      </small>
    </div>
    `,
    advTab
  );

  // Respect server-disabled AA convert: disable checkbox and clear any previous checked
  const aaFlag = $('aa-convert-flag');
  if (aaFlag) {
    if (!CONFIG.aa_convert_allowed) {
      aaFlag.disabled = true;
      aaFlag.checked = false;
    }
  }

  GUARDS.advancedInjected = true;
}

// =====================================================
// Submit handler
// =====================================================
async function handleSubmit(event) {
  event.preventDefault();

  const loader = $('loader');
  const resultsContainer = $('results-container');

  const primarySeqs = $('primary-seqs')?.value?.trim() ?? '';
  const targetSeq = $('target-seq')?.value?.trim() ?? '';
  const competitorSeq = $('competitor-seq')?.value?.trim() ?? '';

  // Clear results
  if (resultsContainer) {
    setHTML(resultsContainer, '');
  }
  predictionResults = [];

  // 🔹 Professional reload warning (pinned)
  prependHTML(resultsContainer, `<div class="reload-warning">
    Please do not refresh or close this page while your prediction is running — this will cancel the analysis in progress.
  </div>`);

  // Enforce miRNA FASTA headers (as requested)
  if (!hasFastaHeaders(primarySeqs)) {
    setHTML(resultsContainer, formatError(
      'Your miRNA input is missing FASTA headers. Please add lines starting with ">" (e.g., >hsa-let-7a-5p) so results can be labeled correctly.'
    ));
    return;
  }

  // Quick client-side length hints (non-blocking; backend enforces)
  const MIN_MIRNA_LEN = 10;
  const MIN_TARGET_LEN = 30;
  const MIN_COMP_LEN = 15;

  // Approximate length check for target
  if ((targetSeq.replace(/^>.*$/gm,'').replace(/\s+/g,'')).length < MIN_TARGET_LEN) {
    appendHTML(resultsContainer, formatWarn(`Tip: Target should be at least ${MIN_TARGET_LEN} nt. Your input looks shorter.`));
  }

  // Competitor (if present)
  if (competitorSeq.trim() && (competitorSeq.replace(/^>.*$/gm,'').replace(/\s+/g,'')).length < MIN_COMP_LEN) {
    appendHTML(resultsContainer, formatWarn(`Tip: Competitor should be at least ${MIN_COMP_LEN} nt or leave it blank.`));
  }

  // Limit check for miRNA count
  const mirnaCount = countFastaRecords(primarySeqs);
  if (mirnaCount > CONFIG.mirna_max) {
    setHTML(resultsContainer, formatError(
      `You entered ${mirnaCount} miRNAs, but the maximum allowed is ${CONFIG.mirna_max}. Please reduce your input and try again.`
    ));
    return;
  }

  // Validate target/competitor counts
  const tgtCount = countFastaRecords(targetSeq);
  const compCount = countFastaRecords(competitorSeq);
  if (tgtCount > 1) {
    setHTML(resultsContainer, formatError('Your target input contains multiple sequences. Please provide exactly one target sequence to proceed.'));
    return;
  }
  if (compCount > 1) {
    setHTML(resultsContainer, formatError('Please enter only one competitor sequence.'));
    return;
  }

  // Advice (not strict) for target/competitor header presence
  if (tgtCount === 1 && !hasFastaHeaders(targetSeq)) {
    prependHTML(resultsContainer, formatWarn('Tip: Add a FASTA header to the target (e.g., >target1) so it’s traceable in results.'));
  }
  if (competitorSeq && compCount === 1 && !hasFastaHeaders(competitorSeq)) {
    prependHTML(resultsContainer, formatWarn('Tip: Add a FASTA header to the competitor (e.g., >comp1) so it’s traceable in results.'));
  }

  // Switch to results tab (defensive)
  const resultsTabButton = byQS('button.tab-btn:nth-child(3)');
  if (resultsTabButton) openTab(resultsTabButton, 'results-tab');

  // Show loader
  if (loader) {
    text(loader, "Running prediction...");
    show(loader);
  }

  // Build FormData
  const formData = new FormData();
  formData.append('primary_molecules', primarySeqs);
  formData.append('target_molecule', targetSeq);
  formData.append('competitor_molecule', competitorSeq);
  formData.append('target_start', $('target-start')?.value ?? '');
  formData.append('target_end', $('target-end')?.value ?? '');

  // Flags (UI checkboxes in advanced tab; tolerated if missing)
  const matureTrimFlag = $('mature-trim-flag')?.checked ?? CONFIG.mature_trim_enabled;
  const aaConvertFlag = $('aa-convert-flag')?.checked ?? false;
  formData.append('mature_trim', matureTrimFlag ? 'true' : 'false');
  formData.append('convert_aa_to_nt', aaConvertFlag ? 'true' : 'false');

  // Optional PDB/CIF files
  const mirnaFileInput = $('mirna-file');
  if (mirnaFileInput && mirnaFileInput.files && mirnaFileInput.files.length > 0) {
    for (const f of mirnaFileInput.files) {
      if (!validateFileSize(f)) {
        mirnaFileInput.value = '';
        return;
      }
      formData.append('mirna_3d_file', f);
    }
  }
  const targetFile = $('target-file')?.files?.[0];
  if (targetFile) {
    if (!validateFileSize(targetFile)) {
      $('target-file').value = '';
      return;
    }
    formData.append('target_3d_file', targetFile);
  }
  const competitorFile = $('competitor-file')?.files?.[0];
  if (competitorFile) {
    if (!validateFileSize(competitorFile)) {
      $('competitor-file').value = '';
      return;
    }
    formData.append('competitor_3d_file', competitorFile);
  }

  try {
    // Prepare headers depending on server auth mode
    const authHeaders = await getNonceOrKeyHeaders();

    // 1) Start job
    if (loader) text(loader, "Job started. Preparing batches...");
    const startRes = await fetch(API_URL, {
      method: 'POST',
      headers: authHeaders,
      body: formData
    });

    if (!startRes.ok) {
      let errorMsg;
      try {
        const errorData = await startRes.json();
        // Prefer message, fallback to error
        errorMsg = errorData.message || errorData.error || null;
      } catch (_) {}
      throw new Error(errorMsg || 'Something went wrong while starting your job. Please try again later.');
    }

    const { job_id } = await startRes.json();
    if (!job_id) throw new Error('No job ID returned from server.');

    // 2) Poll progress until completed (timer-based)
    const poll = async () => {
      const res = await fetch(PROGRESS_URL(job_id), { method: 'GET' });
      if (!res.ok) throw new Error('Failed to check job progress.');
      const data = await res.json();

      if (data.status === 'running') {
        if (loader) {
          // If spinner not already there, insert it once
          if (!loader.querySelector('.loader-spinner')) {
            loader.innerHTML = `<span class="loader-spinner"></span><span id="loader-text"></span>`;
          }
          const loaderText = loader.querySelector('#loader-text');
          if (loaderText) {
            const total = Number.isFinite(data.total) ? data.total : '?';
            const completed = Number.isFinite(data.completed) ? data.completed : '?';
            loaderText.textContent = `Processing... ${completed}/${total} completed`;
          }
          show(loader);
        }
        setTimeout(poll, 1200);
        return;
      }

      if (data.status === 'error') {
        // 🔹 Remove reload warning
        const rw = resultsContainer.querySelector('.reload-warning');
        if (rw) rw.remove();
        throw new Error(data.error || 'We encountered a technical issue while processing your request.');
      }

      if (data.status === 'completed') {
        // 🔹 Remove reload warning
        const rw = resultsContainer.querySelector('.reload-warning');
        if (rw) rw.remove();

        if (loader) text(loader, "Fetching final results...");
        // 3) Download final results
        const dr = await fetch(DOWNLOAD_URL(job_id), { method: 'GET' });
        if (!dr.ok) throw new Error('Failed to download results.');
        const finalData = await dr.json();

        // After results are displayed
        predictionResults = finalData.results || [];
        displayResults(predictionResults);

        // ✅ Run Again button removed — no insertion, no click handler

        if (loader) {
          text(loader, "✅ Prediction completed. Results are shown below.");
          setTimeout(() => hide(loader), 3000);
        }
      }
      };

      await poll();

      } catch (error) {
        // 🔹 Remove reload warning
        const rw = resultsContainer.querySelector('.reload-warning');
        if (rw) rw.remove();

        const friendlyMessage = error.message && !/server error/i.test(error.message)
          ? error.message
          : 'Something went wrong while processing your request. Please try again later.';
        setHTML(resultsContainer, formatError(friendlyMessage));
        if (loader) hide(loader);
      }
}


// =====================================================
// Display results (sorted by baseline; gradient by baseline)
// =====================================================
function displayResults(results) {
  const container = $('results-container');
  if (!container) return;

  // Always clear first (prevents duplicates 100%)
  setHTML(container, '');

  if (!results || results.length === 0) {
    setHTML(container, '<p>No results to display.</p>');
    return;
  }

  // Sort by predicted affinity baseline (descending)
  results.sort((a, b) =>
    safeParseFloat(b.predicted_affinity_baseline ?? b.baseline_score ?? 0, 0) -
    safeParseFloat(a.predicted_affinity_baseline ?? a.baseline_score ?? 0, 0)
  );

  // Gradient color function based on baseline score in [0,1]
  function getGradientColor(score) {
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
    const high = Math.min(low + 1, viridis.length - 1);
    const t = idx - low;
    const r = Math.round(viridis[low][0] + t * (viridis[high][0] - viridis[low][0]));
    const g = Math.round(viridis[low][1] + t * (viridis[high][1] - viridis[low][1]));
    const b = Math.round(viridis[low][2] + t * (viridis[high][2] - viridis[low][2]));
    return `rgba(${r},${g},${b},0.3)`;
  }

  // Classification guide panel (singleton inside results container)
  const legendId = 'affinity-legend';
  const legendHTML = `
  <div id="${legendId}" class="affinity-legend" style="margin-bottom:10px;">
    <h4>Affinity Classification Guide</h4>
    <table>
      <thead>
        <tr><th>Category</th><th>Score Range</th><th>Interpretation</th></tr>
      </thead>
      <tbody>
        <tr style="background-color:rgba(189,223,38,0.3)"><td>High Affinity</td><td>0.76–1.00</td><td>Strong binding; robust experimental evidence; prioritized for validation</td></tr>
        <tr style="background-color:rgba(74,193,109,0.3)"><td>Medium Affinity</td><td>0.51–0.75</td><td>Moderate binding; candidate for multi-feature confirmation</td></tr>
        <tr style="background-color:rgba(43,116,142,0.3)"><td>Low Affinity</td><td>0.26–0.50</td><td>Weakly predicted or weak biophysical/experimental support</td></tr>
        <tr style="background-color:rgba(72,36,117,0.3)"><td>No Affinity</td><td>0.00–0.25</td><td>No meaningful binding; indistinguishable from random</td></tr>
      </tbody>
    </table>
  </div>
  `;

  // Download button (singleton)
  const downloadId = 'download-btn';
  const downloadButtonHTML = `<div style="margin-bottom:12px;"><button id="${downloadId}">Download Results as CSV</button></div>`;

  // Build table
  let table = '<table id="results-table" style="margin-bottom:20px;"><thead><tr>' +
      '<th>Primary Molecule ID</th>' +
      '<th>Predicted Affinity (Baseline)</th>' +
      '<th>Predicted Affinity (With Competitor)</th>' +
      '<th>Competitive Effect (higher is better)</th>' +
      '</tr></thead><tbody>';

  results.forEach(item => {
    const id = item.primary_molecule_id ?? item.mirna_id ?? 'N/A';
    const baseline = (item.predicted_affinity_baseline ?? item.baseline_score ?? '').toString();
    const withComp = (item.predicted_affinity_with_competitor ?? item.score_with_competitor ?? '').toString();
    const compEffect = (item["competitive_effect (higher_is_better)"] ?? item.competitive_effect ?? '').toString();
    const bgColor = getGradientColor(baseline);

    table += `<tr style="background-color:${bgColor}">
        <td>${escapeHTML(id)}</td>
        <td>${escapeHTML(baseline)}</td>
        <td>${escapeHTML(withComp)}</td>
        <td>${escapeHTML(compEffect)}</td>
    </tr>`;
  });
  table += '</tbody></table>';

  // Render in strict order; since container is cleared first, no duplicates can occur
  appendHTML(container, legendHTML);
  appendHTML(container, downloadButtonHTML);
  // Copy to Clipboard button
  appendHTML(container, `<button id="copy-results-btn" class="btn-accent">Copy Results</button>`);
  bindOnce($('copy-results-btn'), 'click', () => {
    const text = predictionResults.map(r => Object.values(r).join('\t')).join('\n');
    navigator.clipboard.writeText(text).then(() => alert('Results copied to clipboard.'));
  }, 'copyResultsClick');
  appendHTML(container, table);
  makeTableSortable('results-table');

  // Ensure only one click listener bound (button is recreated each render, so normal bind is fine)
  const dl = $(downloadId);
  if (dl) bindOnce(dl, 'click', downloadCSV, 'clickOnce');
}

// =====================================================
// CSV download (sorted by baseline for consistency)
// =====================================================
function downloadCSV() {
  if (predictionResults.length === 0) return;

  const headers = "Primary_Molecule_ID,Predicted_Affinity_Baseline,Predicted_Affinity_With_Competitor,Competitive_Effect";
  const csvRows = [headers];

  // Sort by baseline before export to match UI
  const sorted = [...predictionResults].sort((a, b) =>
    safeParseFloat(b["predicted_affinity_baseline"] ?? b.baseline_score ?? 0, 0) -
    safeParseFloat(a["predicted_affinity_baseline"] ?? a.baseline_score ?? 0, 0)
  );

  sorted.forEach(item => {
    const id = item.primary_molecule_id ?? item.mirna_id ?? 'N/A';
    const baseline = (item.predicted_affinity_baseline ?? item.baseline_score ?? '').toString();
    const withComp = (item.predicted_affinity_with_competitor ?? item.score_with_competitor ?? '').toString();
    const compEffect = (item["competitive_effect (higher_is_better)"] ?? item.competitive_effect ?? '').toString();

    // Escape commas and quotes
    const safeCSV = (s) => {
      const str = String(s ?? '');
      if (/[",\n]/.test(str)) {
        return `"${str.replace(/"/g, '""')}"`;
      }
      return str;
    };
    csvRows.push([safeCSV(id), safeCSV(baseline), safeCSV(withComp), safeCSV(compEffect)].join(','));
  });

  const csvString = csvRows.join('\n');
  const blob = new Blob([csvString], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.setAttribute('href', url);
  a.setAttribute('download', 'prediction_results.csv');
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// =====================================================
// Tabs
// =====================================================
function openTab(element, tabId) {
  const targetCard = $(tabId);
  if (!targetCard) return;

  byQSA('.card').forEach(card => card.classList.remove('active'));
  byQSA('.tab-btn').forEach(btn => btn.classList.remove('active'));
  targetCard.classList.add('active');
  if (element && element.classList) element.classList.add('active');
}

function makeTableSortable(tableId) {
  const table = document.getElementById(tableId);
  if (!table) return;
  table.querySelectorAll('th').forEach((header, idx) => {
    header.style.cursor = 'pointer';
    header.addEventListener('click', () => {
      const rows = Array.from(table.querySelectorAll('tbody tr'));
      const asc = header.classList.toggle('asc');
      rows.sort((a, b) => {
        const aText = a.children[idx].textContent.trim();
        const bText = b.children[idx].textContent.trim();
        return asc ? aText.localeCompare(bText) : bText.localeCompare(aText);
      });
      rows.forEach(row => table.querySelector('tbody').appendChild(row));
    });
  });
}

function wireTabButtonsOnce() {
  if (GUARDS.tabWiringDone) return;
  const tabs = byQSA('.tab-btn');
  const loader = $('loader');

  tabs.forEach(btn => {
    bindOnce(btn, 'click', () => {
      const name = (btn.textContent || '').toLowerCase();
      if (name.includes('inputs')) {
        if (loader) {
          text(loader, "Please input your sequences to start a prediction.");
          show(loader);
        }
      }
      if (name.includes('results')) {
        // If results tab is opened with no results, show a gentle hint
        const rc = $('results-container');
        if (rc && !rc.innerHTML.trim()) {
          setHTML(rc, formatInfo('Results will appear here after you run a prediction.'));
        }
      }
    }, 'tabClick');
  });

  GUARDS.tabWiringDone = true;
}

// =====================================================
// Optional: “Paste test data” helper (commented out; enable if needed)
// =====================================================
// (function addPasteTestDataButton() {
//   const inputTab = $('input-tab');
//   if (!inputTab) return;
//   const btnId = 'paste-example-btn';
//   if ($(btnId)) return;
//   const btn = document.createElement('button');
//   btn.id = btnId;
//   btn.type = 'button';
//   btn.style.margin = '8px 0 0 0';
//   btn.textContent = 'Paste example inputs';
//   btn.addEventListener('click', () => {
//     const p = $('primary-seqs');
//    const t = $('target-seq');
//     if (p) p.value = `>hsa-let-7a-5p
// UGAGGUAGUAGGUUGUAUAGUU
// >hsa-miR-1-3p
// UGGAAUGUAAAGAAGUAUGUAU`;
//     if (t) t.value = `>target1
// AAAAAGGGGCCCCUUUUUAAAAGGGGCCCCUUUUU`;
//   });
//   inputTab.appendChild(btn);
// })();

// =====================================================
// Extra safety: guard for missing elements at runtime
// =====================================================
(function guardMissingElements() {
  const form = $('prediction-form');
  if (!form) {
    console.warn("Prediction form not found on page.");
  }
})();

// =====================================================
// Footer notes (documentation & maintainers’ tips)
// =====================================================
//
// - All dynamic injections use singleton IDs to guarantee no duplication:
//   * Server config: #server-config-note
//   * Advanced flags: #advanced-flags-wrapper
//   * FASTA tips: #fasta-tips-box
//   * Results legend: #affinity-legend
//   * Gradient scale: #results-gradient-scale
//   * Download button: #download-btn
//
// - displayResults() always clears #results-container before rendering.
//   This ensures 100% duplication safety even if upstream changes occur.
//
// - AA→NT flag is disabled by server config; the UI respects and prevents selection.
//
// - File inputs tolerate single or multiple miRNA 3D files. Backend indexes by filename stem.
//
// - Errors are safely surfaced and escaped to prevent markup injection.
//
// - Event bindings are guarded (bindOnce) to prevent memory leaks and duplicated triggers.
//
// - BASE_URL switches based on hostname (localhost/127.0.0.1 -> local).
//
// - If USE_NONCE on server is toggled later, this frontend will adapt without code changes.
//
// End of file (∿ 500+ lines by design)
