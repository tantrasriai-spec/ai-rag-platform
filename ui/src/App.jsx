import { useEffect, useMemo, useRef, useState } from "react";
import axios from "axios";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const api = axios.create({ baseURL: API_BASE, timeout: 180000 });

function fmtUsd(n) {
  if (typeof n !== "number" || !isFinite(n)) return "$0.00";
  if (n > 0 && n < 0.0001) return `$${n.toFixed(6)}`;
  return `$${n.toFixed(4)}`;
}

export default function App() {
  const [docs, setDocs] = useState([]);
  const [selectedDocId, setSelectedDocId] = useState("");
  const [uploading, setUploading] = useState(false);
  const [asking, setAsking] = useState(false);
  const [statusMsg, setStatusMsg] = useState("");
  const [input, setInput] = useState("");
  const [docActionLoading, setDocActionLoading] = useState(false);
  const [chatByDoc, setChatByDoc] = useState({});

  const messages = chatByDoc[selectedDocId] || [];
  const scrollRef = useRef(null);

  useEffect(() => {
    refreshDocs();
  }, []);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [messages.length, asking, docActionLoading]);

  const selectedDocLabel = useMemo(() => {
    const d = docs.find((x) => x.id === selectedDocId);
    if (!d) return "";
    return `${d.filename} (${d.status}) — ${d.id.slice(0, 8)}`;
  }, [docs, selectedDocId]);

  async function refreshDocs() {
    try {
      const res = await api.get("/documents");
      const items = res.data || [];
      setDocs(items);
      if (!selectedDocId && items.length > 0) {
        setSelectedDocId(items[0].id);
      }
      setStatusMsg("");
    } catch (e) {
      setStatusMsg(`Failed to load documents: ${e?.message}`);
    }
  }

  async function onUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setStatusMsg("");

    try {
      const form = new FormData();
      form.append("file", file);

      const res = await api.post("/upload", form, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const newDocId = res.data?.doc_id;
      setStatusMsg(res.data?.message || "Upload done.");
      await refreshDocs();
      if (newDocId) setSelectedDocId(newDocId);
    } catch (e2) {
      setStatusMsg(`Upload failed: ${e2?.response?.data?.detail || e2?.message}`);
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  }

  function appendMessage(docId, msg) {
    setChatByDoc((prev) => {
      const cur = prev[docId] || [];
      return { ...prev, [docId]: [...cur, msg] };
    });
  }

  async function ask(questionOverride) {
    const q = (questionOverride ?? input).trim();
    if (!q) return;

    if (!selectedDocId) {
      setStatusMsg("Please select a document.");
      return;
    }

    setStatusMsg("");
    if (questionOverride == null) setInput("");
    setAsking(true);

    appendMessage(selectedDocId, {
      role: "user",
      content: q,
      ts: Date.now(),
    });

    try {
      const res = await api.post("/answer", {
        question: q,
        k: 6,
        document_id: selectedDocId,
        include_sources: false,
      });

      appendMessage(selectedDocId, {
        role: "assistant",
        content: res.data?.answer || "",
        usage: res.data?.usage || null,
        cost: res.data?.cost || null,
        cache: res.data?.cache,
        retrieval_cache: res.data?.retrieval_cache,
        ts: Date.now(),
      });
    } catch (e3) {
      appendMessage(selectedDocId, {
        role: "assistant",
        content: `Error: ${e3?.response?.data?.detail || e3?.message}`,
        ts: Date.now(),
      });
    } finally {
      setAsking(false);
    }
  }

  async function runDocAction(docTask, userMessage) {
    if (!selectedDocId) {
      setStatusMsg("Please select a document.");
      return;
    }

    setStatusMsg("");
    setDocActionLoading(true);

    appendMessage(selectedDocId, {
      role: "user",
      content: userMessage,
      ts: Date.now(),
    });

    try {
      const res = await api.post("/answer", {
        question: userMessage,
        k: 6,
        document_id: selectedDocId,
        include_sources: false,
        mode: "doc_task",
        doc_task: docTask,
      });

      appendMessage(selectedDocId, {
        role: "assistant",
        content: res.data?.answer || "",
        usage: res.data?.usage || null,
        cost: res.data?.cost || null,
        cache: res.data?.cache,
        retrieval_cache: res.data?.retrieval_cache,
        ts: Date.now(),
      });
    } catch (e) {
      appendMessage(selectedDocId, {
        role: "assistant",
        content: `Error: ${e?.response?.data?.detail || e?.message}`,
        ts: Date.now(),
      });
    } finally {
      setDocActionLoading(false);
    }
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!asking) ask();
    }
  }

  return (
    <div className="page">
      <header className="topbar">
        <div className="title">
          <div className="h1">RAG Chat</div>
          <div className="sub">Docker • Hybrid Retrieval • Vertex (Gemini)</div>
        </div>

        <div className="controls">
          <div className="control">
            <div className="label">Upload PDF</div>
            <input
              className="file"
              type="file"
              accept=".pdf"
              onChange={onUpload}
              disabled={uploading}
            />
            <div className="hint">
              {uploading ? "Uploading…" : "Upload → ingestion runs in worker"}
            </div>
          </div>

          <div className="control grow">
            <div className="label">Document</div>
            <div className="row">
              <select
                className="select"
                value={selectedDocId}
                onChange={(e) => setSelectedDocId(e.target.value)}
              >
                {docs.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.filename} ({d.status}) — {d.id.slice(0, 8)}
                  </option>
                ))}
              </select>
              <button className="btn" onClick={refreshDocs}>
                Refresh
              </button>
            </div>
            {selectedDocLabel && <div className="hint">{selectedDocLabel}</div>}
          </div>
        </div>
      </header>

      {statusMsg && <div className="status">{statusMsg}</div>}

      <main className="chatShell">
        <div className="chat" ref={scrollRef}>
          {messages.length === 0 ? (
            <div className="empty">
              Ask a question about the selected PDF. Document tasks use a separate pipeline from normal Q&amp;A.
            </div>
          ) : (
            messages.map((m, idx) => (
              <div key={idx} className={`msg ${m.role === "user" ? "user" : "assistant"}`}>
                <div className="bubble">
                  <div className="content">{m.content}</div>

                  {m.role === "assistant" && (m.usage || m.cost || m.cache) && (
                    <div className="meta">
                      {m.usage?.total_tokens ? (
                        <span>
                          tokens: {m.usage.total_tokens} (in {m.usage.prompt_tokens}, out {m.usage.completion_tokens})
                        </span>
                      ) : (
                        <span>tokens: —</span>
                      )}

                      {m.cost?.total_cost !== undefined && (
                        <span>
                          {" "}
                          • cost: {fmtUsd(m.cost.total_cost)} ({fmtUsd(m.cost.input_cost)} in, {fmtUsd(m.cost.output_cost)} out)
                        </span>
                      )}

                      {m.cache && <span> • cache: {m.cache}</span>}
                      {m.retrieval_cache && <span> • retrieval: {m.retrieval_cache}</span>}
                    </div>
                  )}
                </div>
              </div>
            ))
          )}

          {(asking || docActionLoading) && (
            <div className="msg assistant">
              <div className="bubble">
                <div className="content">Thinking…</div>
              </div>
            </div>
          )}
        </div>

        <div className="composer">
          <textarea
            className="input"
            rows={2}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Message… (Enter to send, Shift+Enter for new line)"
          />

          <button
            className="send"
            onClick={() => ask()}
            disabled={asking || !input.trim()}
          >
            Send
          </button>

          <button
            className="btn"
            disabled={!selectedDocId || docActionLoading || asking}
            onClick={() => runDocAction("summary", "Summarize this document")}
          >
            Summarize
          </button>

          <button
            className="btn"
            disabled={!selectedDocId || docActionLoading || asking}
            onClick={() =>
              runDocAction(
                "highlights_lowlights",
                "Give highlights and lowlights for this document"
              )
            }
          >
            Highlights + Lowlights
          </button>

          <button
            className="btn"
            disabled={!selectedDocId || docActionLoading || asking}
            onClick={() =>
              runDocAction(
                "definitions_highlights",
                "Compile all definitions and key highlights from this document"
              )
            }
          >
            Definitions + Highlights
          </button>
        </div>
      </main>
    </div>
  );
}