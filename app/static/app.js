const $ = (id) => document.getElementById(id);
const STAGES = ['transcribe', 'compose', 'pdf'];
let system = null;
let pollTimer = null;
let busy = false;

function el(tag, props = {}, ...children) {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
}
function option(value, label) { return el('option', { value, textContent: label }); }
function clock(s) { return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`; }
function megabytes(bytes) { return `${(bytes / 1048576).toFixed(1)} MB`; }
function filePath(folder, name) { return `${encodeURIComponent(folder)}/${encodeURIComponent(name)}`; }

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
  const langs = () => system.languages.map(([code, name]) => option(code, name));
  $('whisper').append(...system.whisper_models.map((m) => option(m, m === 'large-v3-turbo' ? `${m} (fast, recommended)` : m)));
  $('language').append(
    option('', 'Auto-detect (per recording)'),
    option('multi', 'Multilingual — languages mixed within a recording'),
    ...langs());
  $('outLanguage').append(option('', 'Main language of the lecture'), ...langs());
}

// ---------- drop zones ----------
function wireDrop(zone, input, onFiles) {
  ['dragenter', 'dragover'].forEach((t) => zone.addEventListener(t, (e) => { e.preventDefault(); zone.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((t) => zone.addEventListener(t, (e) => { e.preventDefault(); zone.classList.remove('over'); }));
  zone.addEventListener('drop', (e) => { if (e.dataTransfer.files.length) onFiles([...e.dataTransfer.files]); });
  input.addEventListener('change', () => { if (input.files.length) onFiles([...input.files]); input.value = ''; });
}

// ---------- recording queue (drag to reorder) ----------
let recordings = [];   // File objects in processing order
let dragged = null;    // the <li> being dragged

function addRecordings(files) {
  recordings.push(...files);
  renderQueue();
}
function moveRecording(from, to) {
  if (to < 0 || to >= recordings.length) return;
  recordings.splice(to, 0, ...recordings.splice(from, 1));
  renderQueue();
  // keep keyboard focus on the row that moved
  const btn = $('queueList').children[to]?.querySelector(to < from ? '.up' : '.down');
  (btn && !btn.disabled ? btn : $('queueList').children[to]?.querySelector('button:not(:disabled)'))?.focus();
}
function renderQueue() {
  const n = recordings.length;
  $('queue').hidden = n === 0;
  $('drop').classList.toggle('has-file', n > 0);
  $('dropMain').textContent = n ? 'Add more recordings' : 'Drop lecture recordings here, or click to choose';
  $('queueCount').textContent = `${n} recording${n === 1 ? '' : 's'} · ${megabytes(recordings.reduce((s, f) => s + f.size, 0))}`;
  $('queueHint').hidden = n < 2;
  $('queueList').replaceChildren(...recordings.map((file, i) => {
    const up = el('button', { type: 'button', className: 'up', textContent: '↑', title: 'Move up', disabled: i === 0 });
    const down = el('button', { type: 'button', className: 'down', textContent: '↓', title: 'Move down', disabled: i === n - 1 });
    const remove = el('button', { type: 'button', textContent: '✕', title: 'Remove' });
    up.setAttribute('aria-label', `Move ${file.name} up`);
    down.setAttribute('aria-label', `Move ${file.name} down`);
    remove.setAttribute('aria-label', `Remove ${file.name}`);
    up.onclick = () => moveRecording(i, i - 1);
    down.onclick = () => moveRecording(i, i + 1);
    remove.onclick = () => { recordings.splice(i, 1); renderQueue(); };
    const li = el('li', { draggable: true },
      el('span', { className: 'grip', textContent: '⋮⋮', ariaHidden: 'true' }),
      el('div', { className: 'q-info' },
        el('div', { className: 'q-name', textContent: file.name, title: file.name }),
        el('div', { className: 'q-size', textContent: megabytes(file.size) })),
      el('div', { className: 'q-actions' }, up, down, remove));
    li.file = file;
    li.addEventListener('dragstart', (e) => {
      dragged = li;
      e.dataTransfer.effectAllowed = 'move';
      e.dataTransfer.setData('text/plain', file.name);
      requestAnimationFrame(() => li.classList.add('dragging'));
    });
    li.addEventListener('dragend', () => {
      dragged = null;
      recordings = [...$('queueList').children].map((node) => node.file);
      renderQueue();
    });
    return li;
  }));
  updateStart();
}
$('queueList').addEventListener('dragover', (e) => {
  if (!dragged) return;
  e.preventDefault();
  e.dataTransfer.dropEffect = 'move';
  const over = e.target.closest('li');
  if (!over || over === dragged) return;
  const box = over.getBoundingClientRect();
  const after = e.clientY > box.top + box.height / 2;
  over.parentNode.insertBefore(dragged, after ? over.nextSibling : over);
});
$('queueList').addEventListener('drop', (e) => { if (dragged) e.preventDefault(); });
$('queueClear').onclick = () => { recordings = []; renderQueue(); };

function updateStart() {
  const n = recordings.length;
  $('startBtn').disabled = n === 0 || busy;
  $('startBtn').textContent = n > 1 ? `Transcribe & compose ${n} recordings` : 'Transcribe & compose';
}

// ---------- job ----------
$('jobForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!recordings.length) return;
  const body = new FormData();
  recordings.forEach((file) => body.append('files', file));  // order = processing order
  body.append('title', $('title').value);
  body.append('language', $('language').value);
  body.append('out_language', $('outLanguage').value);
  body.append('whisper_model', $('whisper').value);
  body.append('llm_model', $('llm').value);
  body.append('make_pdf', $('makePdf').checked);
  busy = true;
  updateStart();
  $('formHint').textContent = 'Uploading…';
  try {
    const job = await api('/api/jobs', { method: 'POST', body });
    $('formHint').textContent = '';
    recordings = [];
    $('title').value = '';
    renderQueue();
    watch(job.id);
  } catch (err) {
    $('formHint').textContent = err.message;
    busy = false;
    updateStart();
  }
});

function watch(id) {
  clearInterval(pollTimer);
  busy = true;
  updateStart();
  const tick = async () => {
    let job;
    try { job = await api(`/api/jobs/${id}`); } catch { return; }
    renderJob(job);
    if (!['queued', 'running'].includes(job.status)) {
      clearInterval(pollTimer);
      busy = false;
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
  const saved = job.status === 'done' ? ` · saved to output/${job.folder}` : '';
  $('jobMsg').textContent = job.message + (running ? ` · ${Math.round(job.progress * 100)}%` : '') + (running ? device : saved);
  $('jobTime').textContent = clock(job.elapsed);
  $('jobError').hidden = !job.error;
  $('jobError').textContent = job.error;
  $('cancelBtn').hidden = !running;
  $('cancelBtn').onclick = () => api(`/api/jobs/${job.id}/cancel`, { method: 'POST' });
  const log = $('jobLog');
  const text = job.log.join('\n');
  if (log.textContent !== text) { log.textContent = text; log.scrollTop = log.scrollHeight; }
}

// ---------- sessions (one folder each) ----------
async function loadFiles() {
  const files = await api('/api/files');
  $('filesEmpty').hidden = files.length > 0;
  $('files').replaceChildren(...files.map((f) => {
    const when = new Date(f.modified * 1000).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
    const actions = el('div', { className: 'file-actions' });
    const view = el('button', { className: 'ghost', type: 'button', textContent: 'Preview' });
    view.onclick = () => openPreview(f);
    actions.append(view, el('a', { className: 'ghost', href: `/api/files/${filePath(f.folder, f.md)}?download=true`, textContent: 'Markdown' }));
    if (f.pdf) actions.append(el('a', { className: 'ghost', href: `/api/files/${filePath(f.folder, f.pdf)}`, target: '_blank', textContent: 'PDF' }));
    if (!f.pdf || f.pdf_stale) {
      const make = el('button', { className: 'ghost', type: 'button', textContent: f.pdf ? 'Update PDF' : 'Convert to PDF' });
      make.onclick = async () => {
        make.disabled = true; make.textContent = 'Converting…';
        try { await api(`/api/convert/${filePath(f.folder, f.md)}`, { method: 'POST' }); }
        catch (err) { alert(err.message); }
        loadFiles();
      };
      actions.append(make);
    }
    const open = el('button', { className: 'ghost', type: 'button', textContent: 'Folder' });
    open.onclick = () => api(`/api/open-output?folder=${encodeURIComponent(f.folder)}`, { method: 'POST' });
    actions.append(open);
    return el('li', {},
      el('div', { className: 'file-info' },
        el('div', { className: 'file-title', textContent: f.title }),
        el('div', { className: 'file-meta', textContent: `output/${f.folder}/ · ${f.words.toLocaleString()} words · ${when}` })),
      actions);
  }));
}

function openPreview(f) {
  $('previewTitle').textContent = f.title;
  $('previewFrame').src = `/api/preview/${filePath(f.folder, f.md)}`;
  $('previewDlg').showModal();
}
$('previewClose').onclick = () => $('previewDlg').close();
$('previewDlg').addEventListener('close', () => { $('previewFrame').src = 'about:blank'; });
$('openFolder').onclick = () => api('/api/open-output', { method: 'POST' });

// ---------- markdown -> pdf ----------
async function convertMarkdown([file]) {
  const main = $('mdDropMain');
  if (!/\.(md|markdown)$/i.test(file.name)) { main.textContent = 'Please choose a .md file'; return; }
  main.textContent = `Converting ${file.name}…`;
  const body = new FormData();
  body.append('file', file);
  try {
    const out = await api('/api/convert-upload', { method: 'POST', body });
    main.textContent = `Saved output/${out.folder}/${out.pdf} — drop another .md file`;
    loadFiles();
  } catch (err) {
    main.textContent = `Failed: ${err.message}`;
  }
}

// ---------- start ----------
(async () => {
  await loadSystem();
  initForm();
  wireDrop($('drop'), $('file'), addRecordings);
  wireDrop($('mdDrop'), $('mdFile'), convertMarkdown);
  renderQueue();
  loadFiles();
  if (system.active_job) watch(system.active_job);
  setInterval(() => loadSystem().catch(() => {}), 15000);
})();
