// mirna.js (Upgraded & Backend-Ready)

// Global variable to hold results for the download function
let predictionResults = [];

// === CONFIGURE API BASE URL ===
const LOCAL_API = "http://127.0.0.1:8080/predict";
const PROD_API = "https://mirna.marzan.org/predict";
const API_URL = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
    ? LOCAL_API
    : PROD_API;
const API_KEY = "supersecret123";

const MAX_FILE_SIZE_MB = 100;

let MAX_MIRNAS = 5000; // default fallback
async function loadConfig() {
  try {
    const base = API_URL.replace(/\/predict$/, '');
    const res = await fetch(`${base}/config`, { method: 'GET' });
    if (res.ok) {
      const cfg = await res.json();
      if (typeof cfg.mirna_max === "number") {
        MAX_MIRNAS = cfg.mirna_max;
      }
    }
  } catch (_) {}
}

document.addEventListener('DOMContentLoaded', async () => {
  await loadConfig();
  // existing bindings...
});

document.addEventListener('DOMContentLoaded', () => {
    const loader = document.getElementById('loader');
    if (loader) {
        loader.textContent = "Please input your sequences to start a prediction.";
        loader.classList.remove('hidden');
    }

    // Bind sequence file inputs to textareas
    bindFileToTextarea('mirna-seq-file', 'primary-seqs');
    bindFileToTextarea('target-seq-file', 'target-seq');
    bindFileToTextarea('competitor-seq-file', 'competitor-seq');
});

function validateFileSize(file) {
    if (file && file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
        alert(`File "${file.name}" exceeds ${MAX_FILE_SIZE_MB} MB limit.`);
        return false;
    }
    return true;
}

function bindFileToTextarea(fileInputId, textareaId) {
    const fileInput = document.getElementById(fileInputId);
    const textarea = document.getElementById(textareaId);
    if (!fileInput || !textarea) return;

    fileInput.addEventListener('change', function () {
        const file = this.files && this.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (e) => { textarea.value = e.target.result; };
        reader.readAsText(file);
    });
}

function countFastaRecords(seqText) {
    if (!seqText) return 0;
    const lines = seqText.trim().split(/\r?\n/);
    let count = 0;
    for (const line of lines) {
        if (line.startsWith('>')) count++;
    }
    if (count === 0 && seqText.trim().length > 0) count = 1;
    return count;
}

document.getElementById('prediction-form').addEventListener('submit', async function(event) {
    event.preventDefault();

    const loader = document.getElementById('loader');
    const resultsContainer = document.getElementById('results-container');

    const primarySeqs = document.getElementById('primary-seqs').value.trim();
    const targetSeq = document.getElementById('target-seq').value.trim();
    const competitorSeq = document.getElementById('competitor-seq').value.trim();

    // Limit check for miRNA count
    const mirnaCount = countFastaRecords(primarySeqs);
    if (mirnaCount > MAX_MIRNAS) {
        resultsContainer.innerHTML = `<p style="color: red;">
            You entered ${mirnaCount} miRNAs, but the maximum allowed is ${MAX_MIRNAS}.
            Please reduce your input and try again.
        </p>`;
        return;
    }

    // Validate target/competitor counts
    if (countFastaRecords(targetSeq) > 1) {
        resultsContainer.innerHTML = '<p style="color: red;">Your target input contains multiple sequences. Please provide exactly one target sequence to proceed.</p>';
        return;
    }
    if (countFastaRecords(competitorSeq) > 1) {
        resultsContainer.innerHTML = '<p style="color: red;">Please enter only one competitor sequence.</p>';
        return;
    }

    // Switch to results tab
    const resultsTabButton = document.querySelector('button[onclick*="results-tab"]');
    if (resultsTabButton) openTab(resultsTabButton, 'results-tab');

    // Show loader
    if (loader) {
        loader.textContent = "Running prediction...";
        loader.classList.remove('hidden');
    }
    resultsContainer.innerHTML = '';
    predictionResults = [];

    const formData = new FormData();
    formData.append('primary_molecules', primarySeqs);
    formData.append('target_molecule', targetSeq);
    formData.append('competitor_molecule', competitorSeq);
    formData.append('target_start', document.getElementById('target-start')?.value ?? '');
    formData.append('target_end', document.getElementById('target-end')?.value ?? '');

    // Get optional PDB/CIF files
    const mirnaFile = document.getElementById('mirna-file')?.files?.[0];
    const targetFile = document.getElementById('target-file')?.files?.[0];
    const competitorFile = document.getElementById('competitor-file')?.files?.[0];

    // Validate sizes
    if (mirnaFile && !validateFileSize(mirnaFile)) {
        document.getElementById('mirna-file').value = '';
        return;
    }
    if (targetFile && !validateFileSize(targetFile)) {
        document.getElementById('target-file').value = '';
        return;
    }
    if (competitorFile && !validateFileSize(competitorFile)) {
        document.getElementById('competitor-file').value = '';
        return;
    }

    if (mirnaFile) formData.append('mirna_3d_file', mirnaFile);
    if (targetFile) formData.append('target_3d_file', targetFile);
    if (competitorFile) formData.append('competitor_3d_file', competitorFile);

    try {
        // 1) Start job
        const startRes = await fetch(API_URL, {
            method: 'POST',
            headers: { "X-API-Key": API_KEY },
            body: formData
        });

        if (!startRes.ok) {
            let errorMsg;
            try {
                const errorData = await startRes.json();
                errorMsg = errorData.error || null;
            } catch (_) {}
            throw new Error(errorMsg || 'Something went wrong while starting your job. Please try again later.');
        }

        const { job_id } = await startRes.json();
        if (!job_id) throw new Error('No job ID returned from server.');

        // 2) Poll progress until completed
        const base = API_URL.replace(/\/predict$/, '');
        const progressUrl = `${base}/progress/${job_id}`;
        const downloadUrl = `${base}/download/${job_id}`;

        const poll = async () => {
            const res = await fetch(progressUrl, { method: 'GET' });
            if (!res.ok) throw new Error('Failed to check job progress.');
            const data = await res.json();

            if (data.status === 'running') {
                if (loader) loader.textContent = `Processing... ${data.completed}/${data.total} completed`;
                setTimeout(poll, 1500);
                return;
            }

            if (data.status === 'error') {
                throw new Error(data.error || 'We encountered a technical issue while processing your request.');
            }

            if (data.status === 'completed') {
                if (loader) loader.textContent = "Fetching final results...";
                // 3) Download final results
                const dr = await fetch(downloadUrl, { method: 'GET' });
                if (!dr.ok) throw new Error('Failed to download results.');
                const finalData = await dr.json();

                predictionResults = finalData.results || [];
                displayResults(predictionResults);

                if (loader) {
                    loader.textContent = "✅ Prediction completed. Results are shown below.";
                    setTimeout(() => loader.classList.add('hidden'), 3000);
                }
            }
        };

        // initial progress message then start polling
        if (loader) loader.textContent = "Job started. Preparing batches...";
        poll();

    } catch (error) {
        const friendlyMessage = error.message && !error.message.includes('server error')
            ? error.message
            : 'Something went wrong while processing your request. Please try again later.';
        resultsContainer.innerHTML = `<p style="color: red;">${friendlyMessage}</p>`;
        if (loader) loader.classList.add('hidden');
    }

});

function displayResults(results) {
    const container = document.getElementById('results-container');
    container.innerHTML = '';
    
    if (!results || results.length === 0) {
        container.innerHTML = '<p>No results to display.</p>';
        return;
    }

    // 1) Sort by baseline (descending)
    results.sort((a, b) =>
    parseFloat(b.predicted_affinity_baseline ?? b.baseline_score ?? 0) -
    parseFloat(a.predicted_affinity_baseline ?? a.baseline_score ?? 0)
    );

    // Gradient color function (0 = red, 1 = green)
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

    // Classification guide panel
    const legendHTML = `
    <div class="affinity-legend">
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

    // Gradient scale bar
    const gradientScaleHTML = `
    <div class="gradient-scale" style="display:flex;align-items:center;gap:8px;margin-bottom:20px;">
      <span>0</span>
      <div style="flex:1;height:20px;background:linear-gradient(to right,
        #440154, /* deep purple */
        #3b528b, /* blue */
        #21908d, /* teal */
        #5dc963, /* green */
        #fde725  /* yellow */);
        border:1px solid #ccc;"></div>
      <span>1</span>
    </div>
    `;

    // Download button right after guide
    const downloadButton = '<div style="margin-bottom:20px;"><button id="download-btn">Download Results as CSV</button></div>';

    // Build table
    let table = '<table style="margin-bottom:20px;"><thead><tr>' +
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
            <td>${id}</td>
            <td>${baseline}</td>
            <td>${withComp}</td>
            <td>${compEffect}</td>
        </tr>`;
    });
    table += '</tbody></table>';

    // Render
    container.innerHTML = legendHTML + downloadButton + table;
    document.getElementById('download-btn').addEventListener('click', downloadCSV);
}

function downloadCSV() {
    if (predictionResults.length === 0) return;

    const headers = "Primary_Molecule_ID,Predicted_Affinity_Baseline,Predicted_Affinity_With_Competitor,Competitive_Effect";
    const csvRows = [headers];
    
    predictionResults.sort((a, b) => parseFloat(b["predicted_affinity_baseline"] ?? b.baseline_score ?? 0) -
                                     parseFloat(a["predicted_affinity_baseline"] ?? a.baseline_score ?? 0));

    predictionResults.forEach(item => {
        const id = item.primary_molecule_id ?? item.mirna_id ?? 'N/A';
        const baseline = (item.predicted_affinity_baseline ?? item.baseline_score ?? '').toString();
        const withComp = (item.predicted_affinity_with_competitor ?? item.score_with_competitor ?? '').toString();
        const compEffect = (item["competitive_effect (higher_is_better)"] ?? item.competitive_effect ?? '').toString();

        csvRows.push([id, baseline, withComp, compEffect].join(','));
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
}

// Tabs function (kept from your original)
function openTab(element, tabId) {
    document.querySelectorAll('.card').forEach(card => card.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    element.classList.add('active');
}

function hasFastaHeaders(text) {
  if (!text || !text.trim()) return false;
  // At least one header line is required for miRNAs when multiple are allowed
  return text.split(/\r?\n/).some(line => line.trim().startsWith('>'));
}

// In submit handler, before building formData:
const primaryHasHeaders = hasFastaHeaders(primarySeqs);
if (!primaryHasHeaders) {
  resultsContainer.innerHTML = `<p style="color:red;">
    Your miRNA input is missing FASTA headers.
    Please add a line starting with ">" (e.g., >hsa-let-7a-5p) for each sequence so we can label results by accession/ID.
  </p>`;
  return;
}

// For target/competitor we accept one sequence; we encourage providing an ID:
const targetHasHeader = hasFastaHeaders(targetSeq);
if (!targetHasHeader) {
  // Not a hard block, but warn inline:
  resultsContainer.innerHTML = `<p style="color: #b36b00;">
    Tip: Add a FASTA header to the target (e.g., >target1) so it’s traceable in results.
  </p>` + resultsContainer.innerHTML;
}
const competitorHasHeader = hasFastaHeaders(competitorSeq);
if (competitorSeq && !competitorHasHeader) {
  resultsContainer.innerHTML = `<p style="color: #b36b00;">
    Tip: Add a FASTA header to the competitor (e.g., >comp1) so it’s traceable in results.
  </p>` + resultsContainer.innerHTML;
}
