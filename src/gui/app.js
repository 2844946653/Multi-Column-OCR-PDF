/* 战锤 PDF 提取工具 - 前端逻辑（pywebview js_api 桥接） */
let API = null;
let pollTimer = null;
let currentBook = null;
let cachedItems = [];
// 待处理 PDF 路径列表：挂到 window 上，与 index.html 内联拖放脚本共享
let inputFiles = window.inputFiles = (window.inputFiles || []);

window.addEventListener('pywebviewready', () => {
  API = window.pywebview.api;
  init();
});

function $(id) { return document.getElementById(id); }

function init() {
  // Tab 切换
  document.querySelectorAll('.tab').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      $('tab-' + btn.dataset.tab).classList.add('active');
      if (btn.dataset.tab === 'books') refreshBooks();
      if (btn.dataset.tab === 'review') refreshReviewBooks();
    });
  });

  $('btn-pick-files').addEventListener('click', async () => {
    const files = await API.pick_input_files();
    addFiles(files);
  });
  $('btn-pick-dir').addEventListener('click', async () => {
    const dir = await API.pick_input_dir();
    if (dir) addFiles([dir]);
  });
  $('btn-clear-files').addEventListener('click', () => {
    inputFiles = [];
    renderFileList();
  });
  $('btn-pick-output').addEventListener('click', async () => {
    $('output-dir').value = await API.pick_output_dir();
  });
  $('btn-extract').addEventListener('click', startExtract);
  $('btn-review-save').addEventListener('click', saveCorrections);
  $('btn-review-export').addEventListener('click', exportCorrected);
  $('review-book').addEventListener('change', loadReviewItems);

  // 拖拽文件进窗口（必须在 window 级 preventDefault，否则浏览器会导航到该文件）
  ['dragover', 'drop'].forEach(ev => {
    document.addEventListener(ev, e => e.preventDefault());
  });
  document.addEventListener('dragenter', () => $('drop-zone').classList.add('dragover'));
  document.addEventListener('dragleave', e => {
    if (e.target === document) $('drop-zone').classList.remove('dragover');
  });
  document.addEventListener('drop', e => {
    $('drop-zone').classList.remove('dragover');
    const files = Array.from(e.dataTransfer.files || [])
      .map(f => f.path || '')
      .filter(p => p.toLowerCase().endsWith('.pdf'));
    if (files.length) addFiles(files);
  });

  $('output-dir').value = '（默认）';
  // 渲染内联拖放脚本在页面加载早期入队的文件（若有）
  if (inputFiles.length) renderFileList();
}

/* ---------- 文件列表 ---------- */
function addFiles(paths) {
  paths.forEach(p => {
    if (p && !inputFiles.includes(p)) inputFiles.push(p);
  });
  renderFileList();
}
// 供 index.html 内联脚本 / Python 侧 on_drop 调用
window.__appAddFiles = addFiles;

function renderFileList() {
  const list = $('file-list');
  if (!inputFiles.length) {
    list.innerHTML = '<div class="drop-hint">也可以把 PDF 文件直接拖进这个窗口</div>';
  } else {
    list.innerHTML = '';
    inputFiles.forEach((p, idx) => {
      const div = document.createElement('div');
      div.className = 'file-chip';
      const name = p.split(/[\\/]/).pop();
      div.innerHTML = `<span class="path" title="${p}">${name}</span>` +
                      `<span class="rm" data-idx="${idx}">✕</span>`;
      div.querySelector('.rm').addEventListener('click', () => {
        inputFiles.splice(idx, 1);
        renderFileList();
      });
      list.appendChild(div);
    });
  }
  const isDir = inputFiles.some(p => !p.toLowerCase().endsWith('.pdf'));
  $('file-count').textContent = inputFiles.length
    ? `${inputFiles.length} 个输入${isDir ? '（文件夹会自动取其中全部 PDF）' : ''}`
    : '';
}

/* ---------- 提取 ---------- */
async function startExtract() {
  if (!inputFiles.length) { showError('请先选择 PDF 文件或文件夹'); return; }
  let output = $('output-dir').value.trim();
  if (output === '（默认）') output = '';
  const from = $('page-from').value ? Number($('page-from').value) : null;
  const to = $('page-to').value ? Number($('page-to').value) : null;
  const fast = document.querySelector('input[name="speed"]:checked').value === 'fast';
  const workers = Number($('workers').value) || 4;
  const force = $('force-reprocess').checked;
  $('extract-error').hidden = true;
  const ok = await API.start_extract(inputFiles, output, from, to, fast, workers, force);
  if (!ok) { showError('已有任务在运行，或输入中没有 PDF 文件'); return; }
  $('progress-wrap').hidden = false;
  $('btn-extract').disabled = true;
  pollTimer = setInterval(pollState, 1000);
}

async function pollState() {
  const s = await API.get_state();
  if (!s.running) {
    clearInterval(pollTimer);
    $('btn-extract').disabled = false;
    if (s.error) showError(s.error);
    else $('progress-label').textContent = '✅ ' + s.message;
    $('progress-fill').style.width = '100%';
    return;
  }
  $('progress-label').textContent = s.message;
  const pct = s.total ? Math.round(s.done / s.total * 100) : 0;
  $('progress-fill').style.width = pct + '%';
}

function showError(msg) {
  const el = $('extract-error');
  el.textContent = '❌ ' + msg;
  el.hidden = false;
}

/* ---------- 结果浏览 ---------- */
async function refreshBooks() {
  const books = await API.list_books('');
  const list = $('book-list');
  list.innerHTML = '';
  books.forEach(b => {
    const div = document.createElement('div');
    div.className = 'book-item';
    div.innerHTML =
      `<div class="name">${b.stem}</div>` +
      `<div class="meta">${b.page_count} 页 · 直取 ${b.text_layer_pages} · ` +
      `OCR ${b.ocr_pages} · <span class="badge">待复核 ${b.needs_human_review}</span>` +
      (b.corrected ? ' · 已导出校正版' : '') + `</div>`;
    div.addEventListener('click', () => showBook(b.stem, div));
    list.appendChild(div);
  });
}

async function showBook(stem, el) {
  currentBook = stem;
  document.querySelectorAll('.book-item').forEach(x => x.classList.remove('active'));
  el.classList.add('active');
  const text = await API.get_book_text('', stem);
  $('book-detail').innerHTML =
    `<h2>${stem}</h2><pre class="book-text">${escapeHtml(text)}</pre>`;
}

function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;')
          .replace(/>/g, '&gt;');
}

/* ---------- 人工复核 ---------- */
async function refreshReviewBooks() {
  const books = await API.list_books('');
  const sel = $('review-book');
  sel.innerHTML = '<option value="">— 选择书 —</option>';
  books.filter(b => b.needs_human_review > 0).forEach(b => {
    const opt = document.createElement('option');
    opt.value = b.stem;
    opt.textContent = `${b.stem}（${b.needs_human_review} 项）`;
    sel.appendChild(opt);
  });
}

async function loadReviewItems() {
  const stem = $('review-book').value;
  $('review-export-result').textContent = '';
  const list = $('review-list');
  list.innerHTML = '';
  if (!stem) { $('review-empty').hidden = false; return; }
  cachedItems = await API.get_review_items('', stem);
  $('review-empty').hidden = cachedItems.length > 0;
  cachedItems.forEach((it, idx) => {
    const div = document.createElement('div');
    div.className = 'review-item';
    const score = it.score == null ? '整块降级' : '置信度 ' + it.score.toFixed(2);
    const bbox = it.bbox
      ? ` 坐标 [${it.bbox.map(v => Math.round(v)).join(', ')}]` : '';
    div.innerHTML =
      `<div class="img-wrap">` +
        (it.image
          ? `<img src="${it.image}" alt="待复核截图">`
          : `<div class="no-img">截图缺失<br>${it.crop}</div>`) +
      `</div>` +
      `<div class="info">` +
        `<div class="meta-line">第 ${it.page} 页 · ${it.category} · ${score}${bbox}` +
        `<br>${it.reason || ''} · ${it.crop || ''}</div>` +
        `<textarea data-idx="${idx}" ` +
        `placeholder="在这里输入正确文本（留空=跳过此项）">` +
        `${escapeHtml(it.correction || '')}</textarea>` +
        (it.correction ? `<div class="done">已保存校正</div>` : '') +
      `</div>`;
    list.appendChild(div);
  });
}

async function saveCorrections() {
  const stem = $('review-book').value;
  if (!stem) return;
  const corrections = {};
  document.querySelectorAll('#review-list textarea').forEach(ta => {
    const item = cachedItems[Number(ta.dataset.idx)];
    if (item && ta.value.trim()) corrections[item.crop] = ta.value;
  });
  const n = await API.save_corrections('', stem, corrections);
  $('review-export-result').textContent = `已保存 ${n} 项校正`;
  loadReviewItems();
}

async function exportCorrected() {
  const stem = $('review-book').value;
  if (!stem) return;
  const r = await API.export_corrected('', stem);
  $('review-export-result').textContent =
    `已导出 corrected_text.md（替换 ${r.replaced} 处占位符）`;
}
