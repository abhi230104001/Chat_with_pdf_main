/**
 * app.js — Chat with PDF frontend application
 *
 * Security:
 * - All model output is rendered via a minimal Markdown parser that escapes HTML.
 * - No untrusted content is inserted via innerHTML without prior sanitization.
 * - Source metadata is validated before rendering.
 * - The Gemini API key is never present in this file.
 */

'use strict';

// ── Constants ──────────────────────────────────────────────────────────────

const API = {
  health:    '/health',
  upload:    '/api/upload',
  documents: '/api/documents',
  chat:      '/api/chat',
  stream:    '/api/chat/stream',
  deleteDoc: (id) => `/api/documents/${id}`,
  clearChat: (sid) => `/api/chat/${sid}`,
};

const SESSION_KEY = 'chatpdf_session';

// ── State ──────────────────────────────────────────────────────────────────

const state = {
  sessionId: loadOrCreateSession(),
  documents: [],          // DocumentMetadata[]
  selectedDocIds: new Set(), // for filtering
  isStreaming: false,
  streamController: null, // AbortController
  pendingDeleteId: null,
  debugOpen: false,
};

// ── DOM References ─────────────────────────────────────────────────────────

const $ = (id) => document.getElementById(id);

const els = {
  dropZone:         $('dropZone'),
  fileInput:        $('fileInput'),
  uploadProgress:   $('uploadProgress'),
  uploadFill:       $('uploadProgressFill'),
  uploadText:       $('uploadProgressText'),
  docsList:         $('docsList'),
  docsCount:        $('docsCount'),
  docsEmpty:        $('docsEmpty'),
  filterSection:    $('filterSection'),
  filterMode:       $('filterMode'),
  healthDot:        $('healthDot'),
  healthLabel:      $('healthLabel'),
  sidebar:          $('sidebar'),
  sidebarToggle:    $('sidebarToggle'),
  chatArea:         $('chatArea'),
  welcomeState:     $('welcomeState'),
  messagesContainer:$('messagesContainer'),
  chatInput:        $('chatInput'),
  sendBtn:          $('sendBtn'),
  stopBtn:          $('stopBtn'),
  clearChatBtn:     $('clearChatBtn'),
  debugToggle:      $('debugToggle'),
  debugDrawer:      $('debugDrawer'),
  debugClose:       $('debugClose'),
  debugContent:     $('debugContent'),
  streamStatus:     $('streamStatus'),
  deleteModal:      $('deleteModal'),
  deleteModalBody:  $('deleteModalBody'),
  deleteCancel:     $('deleteCancel'),
  deleteConfirm:    $('deleteConfirm'),
  toastContainer:   $('toastContainer'),
};

// ── Session Management ─────────────────────────────────────────────────────

function loadOrCreateSession() {
  let sid = sessionStorage.getItem(SESSION_KEY);
  if (!sid) {
    sid = 'session_' + Math.random().toString(36).slice(2, 10);
    sessionStorage.setItem(SESSION_KEY, sid);
  }
  return sid;
}

// ── Minimal Safe Markdown Renderer ────────────────────────────────────────

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function renderMarkdown(text) {
  // Escape HTML first, then apply markdown transformations
  let html = escapeHtml(text);

  // Code blocks (must come before inline code)
  html = html.replace(/```[\w]*\n?([\s\S]*?)```/g, (_, code) =>
    `<pre><code>${code.trim()}</code></pre>`
  );

  // Inline code
  html = html.replace(/`([^`\n]+)`/g, '<code>$1</code>');

  // Headers
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm,  '<h2>$1</h2>');
  html = html.replace(/^# (.+)$/gm,   '<h1>$1</h1>');

  // Bold and italic
  html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
  html = html.replace(/\*\*(.+?)\*\*/g,     '<strong>$1</strong>');
  html = html.replace(/\*(.+?)\*/g,         '<em>$1</em>');

  // Blockquote
  html = html.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');

  // Unordered lists
  html = html.replace(/^[\-\*] (.+)$/gm, '<li>$1</li>');
  html = html.replace(/(<li>.*<\/li>\n?)+/g, (m) => `<ul>${m}</ul>`);

  // Ordered lists
  html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

  // Horizontal rule
  html = html.replace(/^---$/gm, '<hr/>');

  // Paragraphs: wrap double-newline separated blocks
  html = html
    .split(/\n\n+/)
    .map(block => {
      block = block.trim();
      if (!block) return '';
      if (/^<(h[1-6]|ul|ol|pre|blockquote|hr)/.test(block)) return block;
      return `<p>${block.replace(/\n/g, '<br/>')}</p>`;
    })
    .join('\n');

  return html;
}

// ── Toast Notifications ────────────────────────────────────────────────────

function toast(message, type = 'info', duration = 4000) {
  const icons = { success: '✓', error: '✕', info: 'ℹ', warning: '⚠' };
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.innerHTML = `<span class="toast-icon">${icons[type] || 'ℹ'}</span><span>${escapeHtml(message)}</span>`;
  els.toastContainer.appendChild(t);
  setTimeout(() => t.remove(), duration);
}

// ── Health Check ───────────────────────────────────────────────────────────

async function checkHealth() {
  try {
    const res = await fetch(API.health);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const status = data.status || 'error';
    els.healthDot.className = `health-dot ${status}`;
    const apiOk = data.components?.gemini_api?.status === 'ok';
    els.healthLabel.textContent = apiOk ? `API ready · ${data.document_count} docs` : 'API key not set';
  } catch (e) {
    els.healthDot.className = 'health-dot error';
    els.healthLabel.textContent = 'Server unreachable';
  }
}

// ── Document Management ────────────────────────────────────────────────────

async function loadDocuments() {
  try {
    const res = await fetch(API.documents);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    state.documents = data.documents || [];
    renderDocuments();
  } catch (e) {
    console.error('Failed to load documents', e);
  }
}

function renderDocuments() {
  const docs = state.documents;
  els.docsCount.textContent = docs.length;
  els.docsEmpty.style.display = docs.length === 0 ? 'flex' : 'none';
  els.filterSection.hidden = docs.length === 0;

  // Remove existing doc cards (keep empty state)
  const existing = els.docsList.querySelectorAll('.doc-card');
  existing.forEach(el => el.remove());

  docs.forEach(doc => {
    const card = buildDocCard(doc);
    els.docsList.appendChild(card);
  });
}

function buildDocCard(doc) {
  const card = document.createElement('div');
  card.className = 'doc-card';
  card.dataset.docId = doc.document_id;
  card.setAttribute('role', 'listitem');

  if (state.selectedDocIds.has(doc.document_id)) {
    card.classList.add('selected');
  }

  const ext = (doc.extension || '').replace('.', '').toLowerCase();
  const iconColors = { pdf: 'pdf', docx: 'docx', txt: 'txt', md: 'md', csv: 'csv' };
  const iconClass = iconColors[ext] || 'txt';

  const pageInfo = doc.page_count ? `${doc.page_count} pages · ` : '';
  const sizeKb = Math.round((doc.file_size_bytes || 0) / 1024);

  const statusClass = doc.status === 'ready' ? 'ready' : doc.status === 'error' ? 'error' : 'processing';
  const statusLabel = doc.status === 'ready' ? `✓ ${doc.chunk_count} chunks` : doc.status;

  card.innerHTML = `
    <div class="doc-check" aria-hidden="true"></div>
    <div class="doc-icon ${iconClass}" aria-hidden="true">${ext.toUpperCase()}</div>
    <div class="doc-info">
      <div class="doc-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</div>
      <div class="doc-meta">${pageInfo}${sizeKb} KB</div>
      <div class="doc-status ${statusClass}">${escapeHtml(statusLabel)}</div>
    </div>
    <button class="doc-delete" title="Delete document" aria-label="Delete ${escapeHtml(doc.filename)}">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>
      </svg>
    </button>
  `;

  // Toggle selection
  card.addEventListener('click', (e) => {
    if (e.target.closest('.doc-delete')) return;
    toggleDocSelection(doc.document_id, card);
  });

  // Delete button
  card.querySelector('.doc-delete').addEventListener('click', (e) => {
    e.stopPropagation();
    openDeleteModal(doc.document_id, doc.filename);
  });

  return card;
}

function toggleDocSelection(docId, card) {
  if (state.selectedDocIds.has(docId)) {
    state.selectedDocIds.delete(docId);
    card.classList.remove('selected');
  } else {
    state.selectedDocIds.add(docId);
    card.classList.add('selected');
  }
}

// ── File Upload ────────────────────────────────────────────────────────────

async function uploadFiles(files) {
  if (!files || files.length === 0) return;

  const formData = new FormData();
  for (const file of files) {
    formData.append('files', file);
  }

  // Show progress
  els.uploadProgress.hidden = false;
  els.uploadFill.style.width = '30%';
  els.uploadText.textContent = `Uploading ${files.length} file${files.length > 1 ? 's' : ''}…`;

  try {
    els.uploadFill.style.width = '60%';
    const res = await fetch(API.upload, { method: 'POST', body: formData });
    els.uploadFill.style.width = '90%';

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || err.error || `HTTP ${res.status}`);
    }

    const data = await res.json();
    els.uploadFill.style.width = '100%';

    // Process results
    const results = data.results || [];
    results.forEach(r => {
      if (r.duplicate) {
        toast(`"${r.filename}" already uploaded (duplicate)`, 'info');
      } else if (r.status === 'ready') {
        toast(`"${r.filename}" ingested — ${r.chunk_count} chunks`, 'success');
      } else {
        toast(`"${r.filename}": ${r.error_message || 'Failed'}`, 'error');
      }
    });

    if (data.successful > 0) {
      await loadDocuments();
      await checkHealth();
    }

  } catch (err) {
    toast(`Upload failed: ${err.message}`, 'error');
  } finally {
    setTimeout(() => {
      els.uploadProgress.hidden = true;
      els.uploadFill.style.width = '0%';
    }, 600);
  }
}

// ── Delete Modal ───────────────────────────────────────────────────────────

function openDeleteModal(docId, filename) {
  state.pendingDeleteId = docId;
  els.deleteModalBody.textContent = `Delete "${filename}"? This will remove it from the index permanently.`;
  els.deleteModal.hidden = false;
}

function closeDeleteModal() {
  state.pendingDeleteId = null;
  els.deleteModal.hidden = true;
}

async function confirmDelete() {
  const docId = state.pendingDeleteId;
  if (!docId) return;
  closeDeleteModal();

  try {
    const res = await fetch(API.deleteDoc(docId), { method: 'DELETE' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    toast(data.message || 'Document deleted', 'success');
    state.selectedDocIds.delete(docId);
    await loadDocuments();
    await checkHealth();
  } catch (err) {
    toast(`Delete failed: ${err.message}`, 'error');
  }
}

// ── Chat ───────────────────────────────────────────────────────────────────

function getDocumentFilter() {
  if (els.filterMode.value === 'selected' && state.selectedDocIds.size > 0) {
    return Array.from(state.selectedDocIds);
  }
  return null;
}

async function sendMessage() {
  const question = els.chatInput.value.trim();
  if (!question || state.isStreaming) return;

  // Hide welcome, show messages
  els.welcomeState.style.display = 'none';

  // Append user message
  appendMessage('user', question);

  // Clear input
  els.chatInput.value = '';
  els.chatInput.style.height = 'auto';
  updateSendBtn();

  // Start streaming
  await streamChat(question);
}

async function streamChat(question) {
  state.isStreaming = true;
  state.streamController = new AbortController();
  els.sendBtn.hidden = true;
  els.stopBtn.hidden = false;
  els.streamStatus.textContent = 'Retrieving…';

  // Create assistant message placeholder with typing indicator
  const msgEl = appendTypingIndicator();

  let fullText = '';
  let sources = [];
  let debugData = null;
  let receivedDone = false;

  try {
    const payload = {
      question,
      session_id: state.sessionId,
      document_ids: getDocumentFilter(),
      stream: true,
    };

    const res = await fetch(API.stream, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: state.streamController.signal,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || err.error || `HTTP ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // keep incomplete line

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const jsonStr = line.slice(6).trim();
        if (!jsonStr) continue;

        let evt;
        try { evt = JSON.parse(jsonStr); } catch { continue; }

        const { event, data } = evt;

        if (event === 'start') {
          els.streamStatus.textContent = 'Generating…';
          removeTypingIndicator(msgEl);
          msgEl.querySelector('.msg-bubble').innerHTML = '';

        } else if (event === 'token') {
          fullText += data.token || '';
          updateStreamingMessage(msgEl, fullText);

        } else if (event === 'sources') {
          sources = (data.sources || []).filter(validateSource);

        } else if (event === 'done') {
          receivedDone = true;
          debugData = data.debug || null;
          sources = (data.sources || sources).filter(validateSource);

        } else if (event === 'error') {
          throw new Error(data.message || 'Generation error');
        }
      }
    }

  } catch (err) {
    if (err.name === 'AbortError') {
      // User stopped generation — keep whatever was streamed
    } else {
      console.error('Streaming error:', err);
      replaceWithError(msgEl, err.message);
      toast(`Error: ${err.message}`, 'error');
    }
  } finally {
    state.isStreaming = false;
    state.streamController = null;
    els.sendBtn.hidden = false;
    els.stopBtn.hidden = true;
    els.streamStatus.textContent = '';
  }

  // Finalize message
  if (fullText && !msgEl.querySelector('.msg-actions')) {
    finalizeMessage(msgEl, fullText, sources);
  }

  // Update debug panel
  if (debugData) {
    updateDebugPanel(debugData, question);
  }

  scrollToBottom();
}

function validateSource(src) {
  if (!src || typeof src !== 'object') return false;
  if (!src.document_id || !src.filename) return false;
  if (typeof src.chunk_index !== 'number') return false;
  return true;
}

function appendMessage(role, text) {
  const msg = createMessageEl(role, text);
  els.messagesContainer.appendChild(msg);
  scrollToBottom();
  return msg;
}

function createMessageEl(role, text) {
  const msg = document.createElement('div');
  msg.className = `message ${role}`;

  const avatarText = role === 'user' ? 'U' : 'AI';
  const bubbleContent = role === 'assistant' ? renderMarkdown(text) : escapeHtml(text).replace(/\n/g, '<br/>');

  msg.innerHTML = `
    <div class="msg-avatar" aria-hidden="true">${avatarText}</div>
    <div class="msg-content">
      <div class="msg-bubble">${bubbleContent}</div>
    </div>
  `;
  return msg;
}

function appendTypingIndicator() {
  const msg = document.createElement('div');
  msg.className = 'message assistant';
  msg.innerHTML = `
    <div class="msg-avatar" aria-hidden="true">AI</div>
    <div class="msg-content">
      <div class="msg-bubble">
        <div class="typing-indicator" aria-label="Generating response">
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
        </div>
      </div>
    </div>
  `;
  els.messagesContainer.appendChild(msg);
  scrollToBottom();
  return msg;
}

function removeTypingIndicator(msgEl) {
  const indicator = msgEl.querySelector('.typing-indicator');
  if (indicator) indicator.remove();
}

function updateStreamingMessage(msgEl, text) {
  const bubble = msgEl.querySelector('.msg-bubble');
  if (bubble) {
    bubble.innerHTML = renderMarkdown(text);
    scrollToBottom();
  }
}

function finalizeMessage(msgEl, text, sources) {
  const content = msgEl.querySelector('.msg-content');
  if (!content) return;

  // Add action buttons
  const actions = document.createElement('div');
  actions.className = 'msg-actions';
  actions.innerHTML = `
    <button class="msg-action-btn copy-btn" title="Copy answer" aria-label="Copy answer">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
      Copy
    </button>
  `;
  actions.querySelector('.copy-btn').addEventListener('click', () => {
    navigator.clipboard.writeText(text).then(() => toast('Copied!', 'success', 2000));
  });
  content.appendChild(actions);

  // Add sources
  if (sources.length > 0) {
    const srcSection = buildSourcesSection(sources);
    content.appendChild(srcSection);
  }
}

function replaceWithError(msgEl, errMsg) {
  const bubble = msgEl.querySelector('.msg-bubble');
  if (bubble) {
    bubble.innerHTML = `<span style="color:var(--clr-danger)">⚠ ${escapeHtml(errMsg)}</span>`;
  }
}

function buildSourcesSection(sources) {
  const section = document.createElement('div');
  section.className = 'sources-section';

  const label = document.createElement('div');
  label.className = 'sources-label';
  label.textContent = `Sources (${sources.length})`;
  section.appendChild(label);

  const grid = document.createElement('div');
  grid.className = 'sources-grid';

  sources.forEach(src => {
    const card = document.createElement('div');
    card.className = 'source-card';

    const pageStr = src.page_number != null ? `· Page ${src.page_number}` : '';
    const distStr = src.vector_distance != null ? `dist:${src.vector_distance.toFixed(2)}` : '';
    const lexStr  = src.lexical_score  != null ? `lex:${src.lexical_score.toFixed(2)}`  : '';
    const rrStr   = src.rerank_score   != null ? `rr:${src.rerank_score.toFixed(2)}`    : '';

    // Validate snippet before display
    const snippet = typeof src.text_snippet === 'string' ? src.text_snippet : '';

    card.innerHTML = `
      <div class="source-card-header">
        <span class="source-filename">${escapeHtml(src.filename)}</span>
        <span class="source-page">${escapeHtml(pageStr)}</span>
        <div class="source-scores">
          ${distStr ? `<span>${escapeHtml(distStr)}</span>` : ''}
          ${rrStr   ? `<span>${escapeHtml(rrStr)}</span>`   : ''}
        </div>
      </div>
      <div class="source-snippet">${escapeHtml(snippet)}</div>
    `;
    grid.appendChild(card);
  });

  section.appendChild(grid);
  return section;
}

// ── Debug Panel ────────────────────────────────────────────────────────────

function updateDebugPanel(debug, originalQuery) {
  const container = els.debugContent;
  container.innerHTML = '';

  // Latency card
  const latencyCard = makeDebugCard('Latency', [
    ['Retrieval',  `${debug.retrieval_latency_ms?.toFixed(1) ?? '?'} ms`],
    ['Generation', `${debug.generation_latency_ms?.toFixed(1) ?? '?'} ms`],
    ['Total',      `${debug.total_latency_ms?.toFixed(1) ?? '?'} ms`, 'accent'],
  ]);
  container.appendChild(latencyCard);

  // Retrieval card
  const retrievalCard = makeDebugCard('Retrieval', [
    ['Query',      originalQuery?.slice(0, 60) + (originalQuery?.length > 60 ? '…' : ''), 'default'],
    ['Chunks used', `${debug.num_chunks_selected ?? '?'}`, 'success'],
    ['Model',       debug.model_used ?? 'unknown'],
  ]);
  container.appendChild(retrievalCard);

  // Rewritten query card
  if (debug.rewritten_query && debug.rewritten_query !== originalQuery) {
    const qCard = document.createElement('div');
    qCard.className = 'debug-card';
    qCard.innerHTML = `
      <div class="debug-card-title">Rewritten Query</div>
      <div class="debug-query">"${escapeHtml(debug.rewritten_query)}"</div>
    `;
    container.appendChild(qCard);
  }

  // Chunks card
  const chunks = debug.retrieved_chunks || [];
  if (chunks.length > 0) {
    const chunksCard = document.createElement('div');
    chunksCard.className = 'debug-card';
    chunksCard.innerHTML = `<div class="debug-card-title">Retrieved Chunks (${chunks.length})</div>`;

    chunks.forEach((ch, i) => {
      const chunkEl = document.createElement('div');
      chunkEl.className = 'debug-chunk';

      const page = ch.page_number != null ? ` · p.${ch.page_number}` : '';
      const vd  = ch.vector_distance != null ? ch.vector_distance.toFixed(3) : '?';
      const lex = ch.lexical_score   != null ? ch.lexical_score.toFixed(3)   : '?';
      const rr  = ch.rerank_score    != null ? ch.rerank_score.toFixed(3)    : '?';

      chunkEl.innerHTML = `
        <div class="debug-chunk-meta">#${i + 1} · ${escapeHtml(ch.filename || '?')}${escapeHtml(page)} · chunk ${ch.chunk_index ?? '?'}</div>
        <div class="debug-chunk-text">${escapeHtml((ch.text_preview || '').slice(0, 200))}</div>
        <div class="debug-chunk-scores">
          <span class="score-badge vector">dist ${vd}</span>
          <span class="score-badge lex">lex ${lex}</span>
          <span class="score-badge rerank">rr ${rr}</span>
        </div>
      `;
      chunksCard.appendChild(chunkEl);
    });

    container.appendChild(chunksCard);
  }

  // Auto-open debug drawer if closed
  if (!state.debugOpen) {
    // Don't auto-open — let user choose
  }
}

function makeDebugCard(title, metrics) {
  const card = document.createElement('div');
  card.className = 'debug-card';
  let inner = `<div class="debug-card-title">${escapeHtml(title)}</div>`;
  metrics.forEach(([label, value, cls]) => {
    const valueClass = cls ? `debug-metric-value ${cls}` : 'debug-metric-value';
    inner += `
      <div class="debug-metric">
        <span class="debug-metric-label">${escapeHtml(label)}</span>
        <span class="${valueClass}">${escapeHtml(String(value))}</span>
      </div>
    `;
  });
  card.innerHTML = inner;
  return card;
}

// ── UI Utilities ───────────────────────────────────────────────────────────

function scrollToBottom() {
  requestAnimationFrame(() => {
    els.chatArea.scrollTop = els.chatArea.scrollHeight;
  });
}

function updateSendBtn() {
  const hasText = els.chatInput.value.trim().length > 0;
  els.sendBtn.disabled = !hasText || state.isStreaming;
}

function autoResizeTextarea() {
  const ta = els.chatInput;
  ta.style.height = 'auto';
  const maxH = 160;
  ta.style.height = Math.min(ta.scrollHeight, maxH) + 'px';
}

// ── Event Listeners ────────────────────────────────────────────────────────

// Drag and drop
els.dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  els.dropZone.classList.add('drag-over');
});
els.dropZone.addEventListener('dragleave', () => els.dropZone.classList.remove('drag-over'));
els.dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  els.dropZone.classList.remove('drag-over');
  uploadFiles(e.dataTransfer.files);
});
els.dropZone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') els.fileInput.click();
});

els.fileInput.addEventListener('change', (e) => {
  uploadFiles(e.target.files);
  e.target.value = ''; // reset so same file can be re-uploaded
});

// Chat input
els.chatInput.addEventListener('input', () => {
  autoResizeTextarea();
  updateSendBtn();
});
els.chatInput.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault();
    sendMessage();
  }
});

// Send button
els.sendBtn.addEventListener('click', sendMessage);

// Stop button
els.stopBtn.addEventListener('click', () => {
  if (state.streamController) {
    state.streamController.abort();
  }
});

// Clear chat
els.clearChatBtn.addEventListener('click', async () => {
  if (!confirm('Clear conversation history?')) return;
  try {
    await fetch(API.clearChat(state.sessionId), { method: 'DELETE' });
  } catch (_) {}
  els.messagesContainer.innerHTML = '';
  els.welcomeState.style.display = 'flex';
  toast('Conversation cleared', 'info', 2000);
});

// Sidebar toggle
els.sidebarToggle.addEventListener('click', () => {
  const collapsed = els.sidebar.classList.toggle('collapsed');
  els.sidebarToggle.setAttribute('aria-expanded', String(!collapsed));
});

// Debug drawer
els.debugToggle.addEventListener('click', () => {
  state.debugOpen = !state.debugOpen;
  els.debugDrawer.classList.toggle('open', state.debugOpen);
  els.debugDrawer.setAttribute('aria-hidden', String(!state.debugOpen));
  els.debugToggle.classList.toggle('active', state.debugOpen);
  els.debugToggle.setAttribute('aria-pressed', String(state.debugOpen));
});
els.debugClose.addEventListener('click', () => {
  state.debugOpen = false;
  els.debugDrawer.classList.remove('open');
  els.debugDrawer.setAttribute('aria-hidden', 'true');
  els.debugToggle.classList.remove('active');
  els.debugToggle.setAttribute('aria-pressed', 'false');
});

// Delete modal
els.deleteCancel.addEventListener('click', closeDeleteModal);
els.deleteConfirm.addEventListener('click', confirmDelete);
els.deleteModal.addEventListener('click', (e) => {
  if (e.target === els.deleteModal) closeDeleteModal();
});

// Example chips
document.querySelectorAll('.example-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    els.chatInput.value = chip.dataset.question || '';
    updateSendBtn();
    autoResizeTextarea();
    els.chatInput.focus();
  });
});

// ── Initialization ─────────────────────────────────────────────────────────

async function init() {
  updateSendBtn();
  await Promise.all([checkHealth(), loadDocuments()]);

  // Periodic health check
  setInterval(checkHealth, 30000);
}

init();
