const $ = (id) => document.getElementById(id);
const LANGS = [
  ['en', 'English'], ['tr', 'Turkish'], ['de', 'German'], ['fr', 'French'], ['es', 'Spanish'],
  ['it', 'Italian'], ['pt', 'Portuguese'], ['nl', 'Dutch'], ['ru', 'Russian'], ['ar', 'Arabic'],
  ['zh', 'Chinese'], ['ja', 'Japanese'], ['ko', 'Korean'],
];
const STAGES = ['transcribe', 'compose', 'pdf'];
let system = null;
let pollTimer = null;

function el(tag, props = {}, ...children) {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
}
function option(value, label) { return el('option', { value, textContent: label }); }
function clock(s) { return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`; }

async function api(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch {}
    throw new Error(detail);
  }
  return res.json();
}

// ---------- system status ----------
function chip(cls, label, value) {
  return el('span', { className: `chip ${cls}` }, label + ' ', el('b', { textContent: value }));
}
async function loadSystem() {
  system = await api('/api/system');
  const chips = $('chips');
  chips.replaceChildren();
  if (system.gpu) {
    const gb = (system.gpu.vram_total_mb / 1024).toFixed(0);
    chips.append(chip(system.cuda ? 'ok' : 'warn', 'GPU', `${system.gpu.name.replace('NVIDIA ', '')} · ${gb} GB`));
  }
  chips.append(chip(system.cuda ? 'ok' : 'warn', 'Transcription', system.cuda ? 'CUDA' : 'CPU only'));
  chips.append(chip(system.ollama ? (system.llm_models.length ? 'ok' : 'warn') : 'bad', 'Ollama',
    system.ollama ? `${system.llm_models.length} model${system.llm_models.length === 1 ? '' : 's'}` : 'not running'));

  const llm = $('llm');
  const previous = llm.value;
  llm.replaceChildren(...system.llm_models.map((m) => option(m, m)), option('', 'None — cleaned transcript only'));
  // keep the user's choice across refreshes; on first load default to the first installed model
  if (llm.dataset.ready && [...llm.options].some((o) => o.value === previous)) llm.value = previous;
  llm.dataset.ready = '1';
  $('outPath').textContent = system.output_dir;
}
function initForm() {
  $('whisper').append(...system.whisper_models.map((m) => option(m, m === 'large-v3-turbo' ? `${m} (fast, recommended)` : m)));
  $('language').append(option('', 'Auto-detect'), ...LANGS.map(([c, n]) => option(c, n)));
  $('outLanguage').append(option('', 'Same as the lecture'), ...LANGS.map(([c, n]) => option(c, n)));
}

// ---------- drop zones ----------
function wireDrop(zone, input, onFile) {
  ['dragenter', 'dragover'].forEach((t) => zone.addEventListener(t, (e) => { e.preventDefault(); zone.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((t) => zone.addEventListener(t, (e) => { e.preventDefault(); zone.classList.remove('over'); }));
  zone.addEventListener('drop', (e) => { if (e.dataTransfer.files[0]) onFile(e.dataTransfer.files[0]); });
  input.addEventListener('change', () => { if (input.files[0]) onFile(input.files[0]); input.value = ''; });
}

let lectureFile = null;
function setLecture(file) {
  lectureFile = file;
  $('drop').classList.toggle('has-file', !!file);
  $('dropMain').textContent = file ? file.name : 'Drop a lecture recording here, or click to choose';
  $('dropSub').textContent = file ? `${(file.size / 1048576).toFixed(1)} MB — click to choose a different file`
    : 'Audio or video — or a .txt transcript to skip transcription';
  updateStart();
}
function updateStart(busy = false) {
  $('startBtn').disabled = !lectureFile || busy;
}

// ---------- job ----------
$('jobForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!lectureFile) return;
  const body = new FormData();
  body.append('file', lectureFile);
  body.append('title', $('title').value);
  body.append('language', $('language').value);
  body.append('out_language', $('outLanguage').value);
  body.append('whisper_model', $('whisper').value);
  body.append('llm_model', $('llm').value);
  body.append('make_pdf', $('makePdf').checked);
  updateStart(true);
  $('formHint').textContent = 'Uploading…';
  try {
    const job = await api('/api/jobs', { method: 'POST', body });
    $('formHint').textContent = '';
    watch(job.id);
  } catch (err) {
    $('formHint').textContent = err.message;
    updateStart();
  }
});

function watch(id) {
  clearInterval(pollTimer);
  const tick = async () => {
    let job;
    try { job = await api(`/api/jobs/${id}`); } catch { return; }
    renderJob(job);
    if (!['queued', 'running'].includes(job.status)) {
      clearInterval(pollTimer);
      updateStart();
      loadFiles();
    }
  };
  pollTimer = setInterval(tick, 1000);
  tick();
}

function renderJob(job) {
  $('jobCard').hidden = false;
  $('jobName').textContent = job.name;
  const running = ['queued', 'running'].includes(job.status);
  const idx = STAGES.indexOf(job.stage);
  document.querySelectorAll('#stages li').forEach((li) => {
    const i = STAGES.indexOf(li.dataset.stage);
    li.className = '';
    if (li.dataset.stage === 'pdf' && !job.make_pdf) li.classList.add('skipped');
    else if (job.status === 'done' || i < idx) li.classList.add('done');
    else if (i === idx && running) li.classList.add('active');
  });
  const fill = $('barFill');
  fill.style.width = `${Math.round(job.progress * 100)}%`;
  fill.className = 'bar-fill' + (job.status === 'done' ? ' done' : job.status === 'error' ? ' failed' : '');
  const device = job.device ? ` · Whisper on ${job.device === 'cuda' ? 'GPU' : 'CPU'}` : '';
  $('jobMsg').textContent = job.message + (running ? ` · ${Math.round(job.progress * 100)}%` : '') + device;
  $('jobTime').textContent = clock(job.elapsed);
  $('jobError').hidden = !job.error;
  $('jobError').textContent = job.error;
  $('cancelBtn').hidden = !running;
  $('cancelBtn').onclick = () => api(`/api/jobs/${job.id}/cancel`, { method: 'POST' });
  const log = $('jobLog');
  const text = job.log.join('\n');
  if (log.textContent !== text) { log.textContent = text; log.scrollTop = log.scrollHeight; }
}

// ---------- output files ----------
async function loadFiles() {
  const files = await api('/api/files');
  $('filesEmpty').hidden = files.length > 0;
  $('files').replaceChildren(...files.map((f) => {
    const when = new Date(f.modified * 1000).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
    const actions = el('div', { className: 'file-actions' });
    const view = el('button', { className: 'ghost', type: 'button', textContent: 'Preview' });
    view.onclick = () => openPreview(f);
    actions.append(view, el('a', { className: 'ghost', href: `/api/files/${encodeURIComponent(f.md)}?download=true`, textContent: 'Markdown' }));
    if (f.pdf) actions.append(el('a', { className: 'ghost', href: `/api/files/${encodeURIComponent(f.pdf)}`, target: '_blank', textContent: 'PDF' }));
    if (!f.pdf || f.pdf_stale) {
      const make = el('button', { className: 'ghost', type: 'button', textContent: f.pdf ? 'Update PDF' : 'Convert to PDF' });
      make.onclick = async () => {
        make.disabled = true; make.textContent = 'Converting…';
        try { await api(`/api/convert/${encodeURIComponent(f.md)}`, { method: 'POST' }); }
        catch (err) { alert(err.message); }
        loadFiles();
      };
      actions.append(make);
    }
    return el('li', {},
      el('div', { className: 'file-info' },
        el('div', { className: 'file-title', textContent: f.title }),
        el('div', { className: 'file-meta', textContent: `${f.md} · ${f.words.toLocaleString()} words · ${when}` })),
      actions);
  }));
}

function openPreview(f) {
  $('previewTitle').textContent = f.title;
  $('previewFrame').src = `/api/preview/${encodeURIComponent(f.md)}`;
  $('previewDlg').showModal();
}
$('previewClose').onclick = () => $('previewDlg').close();
$('previewDlg').addEventListener('close', () => { $('previewFrame').src = 'about:blank'; });
$('openFolder').onclick = () => api('/api/open-output', { method: 'POST' });

// ---------- markdown -> pdf ----------
async function convertMarkdown(file) {
  const main = $('mdDropMain');
  if (!/\.(md|markdown)$/i.test(file.name)) { main.textContent = 'Please choose a .md file'; return; }
  main.textContent = `Converting ${file.name}…`;
  const body = new FormData();
  body.append('file', file);
  try {
    const out = await api('/api/convert-upload', { method: 'POST', body });
    main.textContent = `Saved ${out.pdf} — drop another .md file`;
    loadFiles();
  } catch (err) {
    main.textContent = `Failed: ${err.message}`;
  }
}

// ---------- start ----------
(async () => {
  await loadSystem();
  initForm();
  wireDrop($('drop'), $('file'), setLecture);
  wireDrop($('mdDrop'), $('mdFile'), convertMarkdown);
  loadFiles();
  if (system.active_job) { updateStart(true); watch(system.active_job); }
  setInterval(() => loadSystem().catch(() => {}), 15000);
})();
