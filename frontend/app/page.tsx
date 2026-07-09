"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useRef, useState } from "react";
import { python } from "@codemirror/lang-python";
import { oneDark } from "@codemirror/theme-one-dark";
import CodeBlock from "@/components/CodeBlock";
import StrategyViewer from "@/components/StrategyViewer";
import { API, Session, createSession, getSession, listSessions, login, runScript, streamResearch, uploadFile } from "@/lib/api";

const CodeMirror = dynamic(() => import("@uiw/react-codemirror"), { ssr: false });
const TVChart = dynamic(() => import("@/components/TVChart"), { ssr: false });

const AGENTS = ["Researcher", "Synthesist", "Quant Coder", "Backtester", "Visualizer", "Archivist"];

/* ---------------------------------------------------------------- entry */
export default function Page() {
  const [user, setUser] = useState<string | null>(null);
  const [name, setName] = useState("");
  useEffect(() => { const u = localStorage.getItem("as_user"); if (u) setUser(u); }, []);
  async function enter() {
    const n = name.trim(); if (!n) return;
    await login(n); localStorage.setItem("as_user", n); setUser(n);
  }
  if (!user) return (
    <div className="min-h-screen flex items-center justify-center px-6">
      <div className="animate-in text-center">
        <h1 className="text-5xl font-semibold tracking-tight">AlphaSeek</h1>
        <p className="text-muted mt-3 text-sm">Agents that research factors.</p>
        <div className="mt-8 flex items-center gap-2 justify-center">
          <input value={name} onChange={(e) => setName(e.target.value)} onKeyDown={(e) => e.key === "Enter" && enter()}
            placeholder="your name" spellCheck={false}
            className="bg-surface border border-border rounded-md px-4 py-2.5 text-sm outline-none focus:border-border2 w-56 text-center" />
          <button onClick={enter} className="btn px-5 py-2.5 text-sm">Enter</button>
        </div>
      </div>
    </div>
  );
  return <Workspace user={user} onLogout={() => { localStorage.removeItem("as_user"); setUser(null); }} />;
}

/* -------------------------------------------------------------- workspace */
type Tab = { kind: "file" | "artifact"; id: string };

function ResizeHandle({ onDrag }: { onDrag: (delta: number) => void }) {
  const dragging = useRef(false);
  return (
    <div onMouseDown={(e) => {
      e.preventDefault();
      dragging.current = true;
      document.body.style.userSelect = "none";
      const start = e.clientX;
      const handler = (ev: MouseEvent) => { if (dragging.current) onDrag(ev.clientX - start); };
      const stop = () => { dragging.current = false; document.body.style.userSelect = ""; document.removeEventListener("mousemove", handler); document.removeEventListener("mouseup", stop); };
      document.addEventListener("mousemove", handler);
      document.addEventListener("mouseup", stop, { once: true });
    }}
    className="w-2 cursor-col-resize shrink-0 flex items-center justify-center hover:bg-[#38383c] transition-colors group"
    title="drag to resize">
      <div className="w-0.5 h-6 rounded-full bg-[#2a2a2e] group-hover:bg-[#5a5a5e] transition-colors" />
    </div>
  );
}

function Workspace({ user, onLogout }: { user: string; onLogout: () => void }) {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [sessionId, setSessionId] = useState<number | null>(null);
  const [input, setInput] = useState("");
  const [rounds, setRounds] = useState(3);
  const [events, setEvents] = useState<any[]>([]);
  const [running, setRunning] = useState(false);
  const [execRunning, setExecRunning] = useState(false);
  const [best, setBest] = useState<any | null>(null);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [meta, setMeta] = useState<{ engine?: string }>({});
  const [files, setFiles] = useState<Record<string, string>>({});
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [artifacts, setArtifacts] = useState<string[]>([]);
  const [exportData, setExportData] = useState<Record<string, { code: string; bt_data: string | null; lang: string }>>({});
  const [tabs, setTabs] = useState<Tab[]>([]);
  const [active, setActive] = useState<Tab | null>(null);
  const [uploadChips, setUploadChips] = useState<string[]>([]);
  const [replaceDefault, setReplaceDefault] = useState(false);
  const [leftW, setLeftW] = useState(200);
  const [rightW, setRightW] = useState(340);
  const fileRef = useRef<HTMLInputElement | null>(null);
  const feedRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);

  const refresh = () => listSessions(user).then(setSessions);
  useEffect(() => { refresh(); /* eslint-disable-next-line */ }, [user]);
  useEffect(() => { feedRef.current?.scrollTo({ top: 1e9, behavior: "smooth" }); }, [events]);

  const activeAgent = useMemo(() => {
    let idx = -1;
    for (const e of events) {
      if (e.type === "handoff") idx = AGENTS.indexOf(e.agent);
      if (e.type === "memory") idx = 4;
      if (e.type === "done") idx = -1;
    }
    return running ? idx : -1;
  }, [events, running]);

  function openTab(t: Tab) {
    setTabs((prev) => (prev.some((x) => x.kind === t.kind && x.id === t.id) ? prev : [...prev, t]));
    setActive(t);
  }
  function closeTab(t: Tab) {
    setTabs((prev) => prev.filter((x) => !(x.kind === t.kind && x.id === t.id)));
    setActive((a) => (a && a.kind === t.kind && a.id === t.id ? null : a));
  }

  // one ingestion path for live events AND replay — workspace state is derived
  function ingest(ev: any, live: boolean) {
    if (ev.type === "start") setMeta({ engine: ev.engine });
    if (ev.type === "code") {
      setFiles((f) => ({ ...f, [ev.filename]: ev.code }));
      setDrafts((d) => { const { [ev.filename]: _omit, ...rest } = d; return rest; });
      if (live) openTab({ kind: "file", id: ev.filename });
    }
    if (ev.type === "backtest" && ev.result) {
      const arts: string[] = ev.result.artifacts ?? [];
      if (arts.length) {
        setArtifacts((a) => [...a, ...arts.filter((x) => !a.includes(x))]);
        if (live) openTab({ kind: "artifact", id: arts[0] });
      }
      if (!ev.exploration && typeof ev.result.sharpe === "number")
        setBest((b: any) => (!b || ev.result.sharpe > b.sharpe ? { ...ev.result, name: ev.name } : b));
    }
    if (ev.type === "export" && ev.filename) {
      setExportData((d) => ({ ...d, [ev.filename]: { code: ev.code, bt_data: ev.bt_data ?? null, lang: ev.lang } }));
      setArtifacts((a) => (a.includes(ev.filename) ? a : [...a, ev.filename]));
      if (live) openTab({ kind: "artifact", id: ev.filename });
    }
    if (ev.type === "done") {
      setBest(ev.best ?? null);
      setRunning(false); refresh();
      const steps = (ev.next_steps ?? []).length ? ev.next_steps : [ev.suggestion].filter(Boolean);
      setSuggestions(steps.slice(0, 3) as string[]);
      setTimeout(() => inputRef.current?.focus(), 60);
    }
    if (ev.type !== "close") setEvents((p) => [...p, ev]);
  }

  function startStream(id: number, prompt?: string) {
    setRunning(true); setSuggestions([]);
    streamResearch(id, (ev) => ingest(ev, true), prompt);
  }

  async function ensureSession(seedText: string): Promise<number> {
    if (sessionId !== null) return sessionId;
    const id = await createSession(user, seedText, rounds);
    setSessionId(id);
    return id;
  }

  async function send(text?: string) {
    const q = (text ?? input).trim();
    if (!q || running) return;
    setInput("");
    setEvents((p) => [...p, { type: "user", text: q }]);
    const first = sessionId === null;
    const id = await ensureSession(q);
    startStream(id, first ? undefined : q);
  }

  async function attach(file: File) {
    const id = await ensureSession(file.name);
    const names = await uploadFile(id, file, replaceDefault);
    setUploadChips(names);
    setEvents((p) => [...p, { type: "upload", filename: file.name }]);
  }

  async function openSession(id: number) {
    if (running) return;
    const s = await getSession(id);
    if (!s || s.error) return;
    setSessionId(id); setBest(s.best ?? null); setSuggestions([]);
    setFiles({}); setDrafts({}); setArtifacts([]); setTabs([]); setActive(null);
    setEvents([{ type: "user", text: s.seed }]);
    for (const ev of s.events || []) ingest(ev, false);
  }

  function newChat() {
    if (running) return;
    setSessionId(null); setEvents([]); setBest(null); setSuggestions([]);
    setFiles({}); setDrafts({}); setArtifacts([]); setTabs([]); setActive(null); setUploadChips([]); setReplaceDefault(false);
  }

  async function runActive() {
    if (!active || active.kind !== "file" || execRunning) return;
    const code = drafts[active.id] ?? files[active.id] ?? "";
    const id = await ensureSession(active.id);
    setExecRunning(true);
    setEvents((p) => [...p, { type: "handoff", agent: "Backtester", action: `running ${active.id}` }]);
    const ev = await runScript(id, active.id, code);
    setExecRunning(false);
    ingest(ev, true);
  }

  const activeCode = active?.kind === "file" ? (drafts[active.id] ?? files[active.id] ?? "") : "";

  return (
    <div className="h-screen flex flex-col">
      <header className="h-12 shrink-0 border-b hair flex items-center px-4 gap-3">
        <span className="font-semibold tracking-tight">AlphaSeek</span>
        <div className="ml-auto flex items-center gap-3 text-[11px] text-muted">
          {meta.engine && <span className="px-2 py-0.5 border border-border rounded">sandbox {meta.engine}</span>}
          <span className="text-text">{user}</span>
          <button onClick={onLogout} className="text-faint hover:text-text">exit</button>
        </div>
      </header>

      <div className="flex-1 flex min-h-0 overflow-hidden">
        {/* LEFT: workspace + sessions */}
        <aside style={{ width: leftW }} className="shrink-0 border-r hair flex flex-col min-h-0">
          <div className="p-2.5 border-b hair">
            <button onClick={newChat} className="btn-outline w-full py-1.5 text-[12.5px]">New chat</button>
          </div>
          <div className="eyebrow px-3 pt-2.5 pb-1.5">Workspace</div>
          <div className="px-2 space-y-0.5 max-h-[32%] overflow-auto">
            {Object.keys(files).map((f) => (
              <button key={f} onClick={() => openTab({ kind: "file", id: f })}
                className={`w-full text-left px-2 py-1 rounded text-[12px] truncate hover:bg-surface2 ${active?.kind === "file" && active.id === f ? "bg-surface2 text-text" : "text-muted"}`}>
                <span className="text-faint mr-1.5">py</span>{f}
              </button>
            ))}
            {artifacts.map((a) => (
              <button key={a} onClick={() => openTab({ kind: "artifact", id: a })}
                className={`w-full text-left px-2 py-1 rounded text-[12px] truncate hover:bg-surface2 ${active?.kind === "artifact" && active.id === a ? "bg-surface2 text-text" : "text-muted"}`}>
                <span className="text-faint mr-1.5">{a.endsWith(".html") ? "3d" : "img"}</span>{a.replace(/^[0-9a-f]+_/, "")}
              </button>
            ))}
            {Object.keys(files).length === 0 && artifacts.length === 0 && (
              <div className="text-[11px] text-faint px-2">empty</div>
            )}
          </div>
          <div className="eyebrow px-3 pt-3 pb-1.5">Sessions</div>
          <div className="flex-1 overflow-auto px-2 pb-2 space-y-1">
            {sessions.map((s) => (
              <button key={s.id} onClick={() => openSession(s.id)}
                className={`card p-2 text-[11.5px] w-full text-left hover:border-border2 transition-colors ${s.id === sessionId ? "border-border2" : ""}`}>
                <div className="truncate text-text/85">{s.seed}</div>
                <div className="flex justify-between mt-1 text-[10.5px] text-faint">
                  <span className="tnum">#{s.id}</span><span>{s.status}</span>
                </div>
              </button>
            ))}
          </div>
        </aside>
        <ResizeHandle onDrag={(d) => setLeftW(Math.max(140, Math.min(400, leftW + d)))} />

        {/* CENTER: editor / artifact viewer */}
        <main className="flex-1 flex flex-col min-h-0">
          <div className="h-10 shrink-0 border-b hair flex items-stretch overflow-x-auto">
            {tabs.map((t) => (
              <div key={t.kind + t.id}
                className={`flex items-center gap-2 px-3 text-[12px] border-r hair cursor-pointer whitespace-nowrap ${active?.kind === t.kind && active.id === t.id ? "bg-surface text-text" : "text-muted hover:text-text"}`}
                onClick={() => setActive(t)}>
                <span className="text-faint">{t.kind === "file" ? "py" : t.id.endsWith(".html") ? "3d" : t.id.endsWith(".pine") ? "📈" : t.id.endsWith(".mq5") ? "⚙" : "img"}</span>
                {t.kind === "artifact" ? t.id.replace(/^[0-9a-f]+_/, "") : t.id}
                <span className="text-faint hover:text-text" onClick={(e) => { e.stopPropagation(); closeTab(t); }}>×</span>
              </div>
            ))}
            <div className="ml-auto flex items-center pr-2">
              {active?.kind === "file" && (
                <button onClick={runActive} disabled={execRunning || running}
                  className="btn px-3 py-1 text-[12px] disabled:opacity-40">
                  {execRunning ? "running…" : "Run"}
                </button>
              )}
            </div>
          </div>
          <div className="flex-1 min-h-0 overflow-hidden">
            {!active && (
              <div className="h-full grid place-items-center text-faint text-sm">
                Files and charts the agents produce open here.
              </div>
            )}
            {active?.kind === "file" && (
              <CodeMirror
                value={activeCode}
                onChange={(v) => setDrafts((d) => ({ ...d, [active.id]: v }))}
                extensions={[python()]}
                theme={oneDark}
                height="100%"
                style={{ height: "100%", fontSize: "12.5px" }}
              />
            )}
            {active?.kind === "artifact" && (
              active.id.endsWith(".html") ? (
                <iframe src={`${API}/api/artifacts/${active.id}`} title={active.id}
                  sandbox="allow-scripts" className="w-full h-full bg-[#111112]" />
              ) : active.id.endsWith(".pine") || active.id.endsWith(".mq5") ? (
                <StrategyViewer
                  code={exportData[active.id]?.code ?? ""}
                  lang={active.id.endsWith(".pine") ? "pine" : "mql5"}
                  btDataFilename={exportData[active.id]?.bt_data ?? null}
                />
              ) : (
                <div className="h-full overflow-auto grid place-items-center p-4">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={`${API}/api/artifacts/${active.id}`} alt={active.id}
                    className="max-w-full rounded-md border border-border" />
                </div>
              )
            )}
          </div>
        </main>
        <ResizeHandle onDrag={(d) => setRightW(Math.max(200, Math.min(600, rightW + d)))} />

        {/* RIGHT: event feed */}
        <aside style={{ width: rightW }} className="shrink-0 border-l hair flex flex-col min-h-0">
          <div className="shrink-0 border-b hair px-4 py-2 flex items-center gap-1.5 overflow-x-auto">
            {AGENTS.map((a, i) => (
              <div key={a} className="flex items-center gap-1.5 whitespace-nowrap">
                <span className={`text-[10.5px] ${i === activeAgent ? "text-text" : "text-faint"}`}>{a}</span>
                {i === activeAgent && <span className="dot on" />}
                {i < AGENTS.length - 1 && <span className="text-faint text-[9px]">›</span>}
              </div>
            ))}
          </div>
          <div ref={feedRef} className="flex-1 overflow-auto px-3 py-3 space-y-2">
            {events.length === 0 && (
              <div className="text-center text-faint mt-24 text-[13px] animate-in">What should the team research?</div>
            )}
            {events.map((ev, i) => <Row key={i} ev={ev} onOpen={openTab} />)}
          </div>
          <div className="shrink-0 border-t hair p-3">
            {suggestions.length > 0 && (
              <div className="card p-2.5 mb-2">
                <div className="eyebrow mb-1.5">Next step — what would you like to do?</div>
                <div className="space-y-1">
                  {suggestions.map((sg, i) => (
                    <button key={i} onClick={() => send(sg)}
                      className="w-full text-left flex items-start gap-2 px-2 py-1.5 rounded border border-border hover:border-border2 hover:bg-surface2 transition-colors">
                      <span className="inline-grid place-items-center w-4 h-4 mt-0.5 rounded border border-border2 text-[9.5px] font-semibold shrink-0">{String.fromCharCode(65 + i)}</span>
                      <span className="text-[12px] text-text/90 leading-snug">{sg}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}
            {uploadChips.length > 0 && (
              <div className="flex gap-1.5 mb-2 flex-wrap">
                {uploadChips.map((f, i) => (
                  <span key={i} className="flex items-center gap-1 text-[10.5px] text-muted border border-border rounded px-1.5 py-0.5">
                    {f}
                    <button onClick={() => setUploadChips((p) => p.filter((_, j) => j !== i))}
                      className="text-faint hover:text-text">×</button>
                  </span>
                ))}
              </div>
            )}
            <div className="flex items-end gap-1.5">
              <input ref={fileRef} type="file" className="hidden" accept=".csv,.parquet,.xlsx,.xls,.json,.npz"
                onChange={(e) => { const f = e.target.files?.[0]; if (f) attach(f); e.target.value = ""; }} />
              <div className="flex flex-col gap-1">
                <button onClick={() => fileRef.current?.click()} disabled={running}
                  className="btn-outline px-2 py-1.5 text-[11px] disabled:opacity-40 whitespace-nowrap"
                  title="Upload a dataset (CSV, Parquet, Excel, JSON, NPZ)">+ upload data</button>
                <label className="flex items-center gap-1 text-[9px] text-faint cursor-pointer">
                  <input type="checkbox" checked={replaceDefault} onChange={(e) => setReplaceDefault(e.target.checked)}
                    className="accent-[#fafafa]" />
                  replace default dataset
                </label>
              </div>
              <textarea ref={inputRef} value={input} rows={1} spellCheck={false}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
                placeholder={running ? "Team is working…" : "Ask AlphaSeek to research, edit, or explain…"}
                disabled={running}
                className="flex-1 bg-surface border border-border rounded-lg px-3 py-2 text-[12.5px] outline-none focus:border-border2 resize-none disabled:opacity-50" />
              <div className="flex flex-col items-end gap-1.5">
                <div className="flex items-center gap-2 bg-surface border border-border rounded-md px-2.5 py-1.5">
                  <div className="flex items-center gap-1 text-[11px]">
                    <span className="text-faint">rounds</span>
                    <input type="number" min={1} max={12} value={rounds} onChange={(e) => setRounds(+e.target.value)}
                      className="w-8 bg-transparent border border-border rounded px-1 py-0.5 text-[11px] tnum text-text text-center outline-none focus:border-border2" />
                  </div>
                </div>
                <button onClick={() => send()} disabled={running || !input.trim()} className="btn px-3.5 py-1.5 text-[12px]">
                  {running ? "…" : "Send"}
                </button>
              </div>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ rows */
function Row({ ev, onOpen }: { ev: any; onOpen: (t: Tab) => void }) {
  if (ev.type === "user")
    return (
      <div className="flex justify-end animate-in">
        <div className="bg-raised border border-border2 rounded-lg px-3 py-2 text-[12.5px] max-w-[90%]">{ev.text}</div>
      </div>
    );
  if (ev.type === "start")
    return <Rule>{ev.resumed ? "continuing" : "started"} · {ev.iterations} rounds · {ev.engine}</Rule>;
  if (ev.type === "round") return <Rule>round {ev.step} / {ev.total}</Rule>;
  if (ev.type === "handoff")
    return <div className="text-[11.5px] text-muted animate-in flex items-center gap-1.5"><span className="dot on" /><span className="text-text/80">{ev.agent}</span> {ev.action}</div>;
  if (ev.type === "agent_msg")
    return (
      <div className="animate-in card p-2.5">
        <div className="eyebrow mb-1">{ev.agent}{ev.title ? ` · ${ev.title}` : ""}</div>
        <div className="text-[12.5px] text-text/90 leading-relaxed">{ev.content}</div>
        {ev.novelty && <div className="text-[11.5px] text-muted mt-1.5"><span className="text-faint">novelty: </span>{ev.novelty}</div>}
        {ev.detail && <div className="text-[11.5px] text-muted mt-1.5 whitespace-pre-wrap">{ev.detail}</div>}
        {(ev.validation_targets ?? []).length > 0 && (
          <div className="text-[11px] text-faint mt-1.5">validate: {(ev.validation_targets ?? []).slice(0, 3).join(" · ")}</div>
        )}
        {ev.acceptance && <div className="text-[11px] text-faint mt-1.5">accept: {ev.acceptance}</div>}
        {(ev.references ?? []).length > 0 && (
          <div className="text-[11px] text-faint mt-1">refs: {(ev.references ?? []).join(" · ")}</div>
        )}
        {(ev.requirements ?? []).length > 0 && (
          <div className="text-[11px] text-faint mt-1">libs: {(ev.requirements ?? []).join(", ")}</div>
        )}
      </div>
    );
  if (ev.type === "provision")
    return (
      <div className="animate-in card p-2.5">
        <div className="filecard-head mb-1">
          <span className="text-faint">provision</span>
          <span className="text-text/90">
            {ev.error ? "install failed — using base image" :
              (ev.installed ?? []).length ? `installed ${(ev.installed ?? []).join(", ")}` : "base image"}
          </span>
          <span className="text-faint ml-auto">{ev.cached ? "cached" : "built"}</span>
        </div>
        {(ev.skipped ?? []).length > 0 && (
          <div className="text-[11px] text-faint">skipped: {(ev.skipped ?? []).join(" · ")}</div>
        )}
        {ev.error && <div className="text-[11px] text-faint">{String(ev.error).slice(0, 160)}</div>}
      </div>
    );
  if (ev.type === "search")
    return (
      <div className="animate-in card p-2.5">
        <div className="filecard-head mb-1.5">
          <span className="text-faint">web search</span>
          <span className="text-text/90 truncate">{ev.query}</span>
          <span className="text-faint ml-auto">{(ev.results ?? []).length} results</span>
        </div>
        <div className="space-y-1">
          {(ev.results ?? []).slice(0, 4).map((r: any, i: number) => (
            <div key={i} className="text-[11.5px] text-muted leading-snug">
              <span className="text-text/85">{r.title}</span>
              <span className="text-faint"> · {r.source}{r.year ? ` ${r.year}` : ""}{r.citations != null ? ` · ${r.citations} cites` : ""}</span>
            </div>
          ))}
        </div>
      </div>
    );
  if (ev.type === "profile")
    return (
      <div className="animate-in card p-2.5">
        <div className="filecard-head mb-1">
          <span className="text-faint">data profile</span>
          <span className="text-text/90 truncate">{(ev.report ? Object.keys(ev.report).length : 0)} keys</span>
        </div>
        <div className="text-[11.5px] text-muted leading-snug">
          {ev.report && typeof ev.report === "object" ? JSON.stringify(ev.report, null, 2).slice(0, 600) : String(ev.stdout ?? "").slice(0, 300)}
        </div>
      </div>
    );
  if (ev.type === "reading")
    return (
      <div className="animate-in card p-2.5">
        <div className="filecard-head mb-1.5">
          <span className="text-faint">paper read</span>
          <span className="text-text/90 truncate">{ev.title}{ev.year ? ` (${ev.year})` : ""}</span>
          <span className="text-faint ml-auto">
            {ev.basis}{ev.citations != null ? ` · ${ev.citations} cites` : ""}
          </span>
        </div>
        {ev.claim && <div className="text-[11.5px] text-muted leading-snug">{ev.claim}</div>}
        {ev.relevant && (ev.method_steps ?? []).length > 0 && (
          <div className="text-[11px] text-faint mt-1 leading-snug">
            method: {(ev.method_steps ?? []).slice(0, 3).join(" → ")}
          </div>
        )}
        {ev.relevant && (ev.reported_numbers ?? []).length > 0 && (
          <div className="text-[11px] text-faint mt-0.5">
            reported: {(ev.reported_numbers ?? []).slice(0, 3).join(" · ")}
          </div>
        )}
        {!ev.relevant && <div className="text-[11px] text-faint mt-1">judged not relevant to this goal — discarded</div>}
      </div>
    );
  if (ev.type === "code")
    return (
      <button onClick={() => onOpen({ kind: "file", id: ev.filename })}
        className="animate-in card p-2.5 w-full text-left hover:border-border2 transition-colors">
        <div className="filecard-head">
          <span className="text-faint">›</span>
          <span className="text-text/90">{ev.filename}</span>
          {ev.revised && <span className="text-faint">revised</span>}
          {ev.model && <span className="text-faint truncate max-w-[160px]">{String(ev.model).split("/").pop()}</span>}
          <span className="text-faint ml-auto">+{ev.code.split("\n").length} · open</span>
        </div>
      </button>
    );
  if (ev.type === "export")
    return (
      <div className="animate-in card p-2.5 w-full text-left">
        <div className="filecard-head">
          <span className="text-faint">export</span>
          <span className="text-text/90">{ev.lang === "pine" ? "TradingView · Pine Script" : "MetaTrader · MQL5"}</span>
          <span className="text-faint ml-auto">{(ev.code?.split("\n").length) ?? 0} lines</span>
        </div>
        {ev.lang === "pine" && (
          <span className="text-[10px] text-[#8b949e] mt-1 mb-1.5 block">
            Open TradingView → Charts → Pine Editor tab (bottom) → New → paste → Add to Chart
          </span>
        )}
        {ev.lang === "mql5" && (
          <div className="text-[10px] text-[#8b949e] mt-1 mb-1.5">
            Paste into MetaEditor (MetaTrader → Tools → MetaQuotes Language Editor → New → Expert Advisor → paste → Compile → Attach to Chart)
          </div>
        )}
        {ev.code && <CodeBlock code={ev.code} lang={ev.lang} className="text-[11px] mt-1.5 max-h-40 overflow-auto p-2" />}
        <div className="flex gap-2 mt-1.5">
          <button onClick={() => onOpen({ kind: "artifact", id: ev.filename })}
            className="text-[10px] text-[#58a6ff] hover:underline">view chart</button>
          <button onClick={() => navigator.clipboard.writeText(ev.code)}
            className="text-[10px] text-[#58a6ff] hover:underline">copy</button>
          <a href={`${API}/api/artifacts/${ev.filename}`} target="_blank" rel="noreferrer"
            className="text-[10px] text-[#58a6ff] hover:underline">download {ev.filename}</a>
        </div>
      </div>
    );
  if (ev.type === "run_error")
    return <div className="text-[11.5px] text-muted animate-in card p-2.5 border-border2 whitespace-pre-wrap">run failed · {ev.message}</div>;
  if (ev.type === "backtest") {
    const r = ev.result;
    const graded = typeof r.sharpe === "number";
    return (
      <div className="animate-in card-raised p-2.5">
        <div className="filecard-head mb-2">
          <span className="text-text/90">run{ev.exploration ? " · exploration" : ""}</span>
          <span className="text-faint">exit 0{ev.result.elapsed_s ? ` · ${ev.result.elapsed_s}s` : ""} · {(ev.result.artifacts ?? []).length} artifact{(ev.result.artifacts ?? []).length === 1 ? "" : "s"} · {ev.engine}</span>
        </div>
        {graded && (
          <div className="grid grid-cols-4 gap-1">
            <Stat label="Sharpe" value={r.sharpe.toFixed(2)} big />
            <Stat label="Net" value={(r.sharpe_net ?? r.sharpe).toFixed(2)} />
            <Stat label="IC" value={r.mean_ic.toFixed(4)} />
            <Stat label="MaxDD" value={`${(r.max_drawdown * 100).toFixed(0)}%`} />
          </div>
        )}
        {r.stdout && r.stdout.trim() && (
          <pre className="code p-2 mt-2 text-[10.5px] max-h-28 overflow-auto whitespace-pre-wrap">{r.stdout.trim()}</pre>
        )}
        {(r.artifacts ?? []).length > 0 && (
          <div className="flex gap-1.5 mt-2 flex-wrap">
            {(r.artifacts ?? []).map((a: string, i: number) => (
              <button key={i} onClick={() => onOpen({ kind: "artifact", id: a })}
                className="btn-outline px-2 py-1 text-[10.5px] text-muted hover:text-text">
                {a.endsWith(".html") ? "interactive · " : "chart · "}{a.replace(/^[0-9a-f]+_/, "")}
              </button>
            ))}
          </div>
        )}
        {graded && <div className="mt-2 -mx-1"><TVChart data={r.equity_curve} height={80} /></div>}
      </div>
    );
  }
  if (ev.type === "verdict") {
    const v = ev.verdict;
    return (
      <div className="animate-in card p-2.5">
        <div className="flex items-center gap-2 mb-1">
          <div className="eyebrow">Verdict</div><Badge g={v.grade} />
          <span className="text-[10.5px] text-muted">{v.keep ? "keep" : "reject"}</span>
          {v.overfit && <span className="text-[10.5px] text-muted">· overfit</span>}
        </div>
        <div className="text-[12.5px] text-text/90">{ev.review?.assessment}</div>
        <div className="text-[11.5px] text-muted mt-1">next: {ev.review?.suggestion}</div>
      </div>
    );
  }
  if (ev.type === "memory")
    return <div className="text-[10.5px] text-faint animate-in">memory · {ev.kept ? "kept" : "discarded"} · {ev.keepers} keepers</div>;
  if (ev.type === "upload") return <Rule>attached {ev.filename}{ev.replaced_default ? " · replaces default dataset" : ""}</Rule>;
  if (ev.type === "done")
    return (
      <div className="animate-in card p-3 border-border2">
        <div className="eyebrow mb-1.5">Answer</div>
        <div className="text-[13px] text-text leading-relaxed">{ev.answer}</div>
        <div className="text-[10.5px] text-muted mt-2">tested {ev.tested} · {ev.keepers?.length ?? 0} keepers{ev.best ? ` · best ${ev.best.name}` : ""}</div>
      </div>
    );
  if (ev.type === "error")
    return (
      <div className="animate-in card p-2.5 border-border2">
        <div className="eyebrow mb-1">{ev.fatal ? "Run stopped" : "Error"}</div>
        <div className="text-[12px] text-text/90 whitespace-pre-wrap">{ev.message}</div>
      </div>
    );
  return null;
}

const Rule = ({ children }: { children: React.ReactNode }) => (
  <div className="flex items-center gap-2.5 py-1 animate-in">
    <div className="flex-1 h-px bg-border" />
    <span className="eyebrow whitespace-nowrap">{children}</span>
    <div className="flex-1 h-px bg-border" />
  </div>
);

const Stat = ({ label, value, big }: { label: string; value: any; big?: boolean }) => (
  <div className="bg-surface border border-border rounded-md px-1.5 py-1 text-center">
    <div className={`mono tnum text-text ${big ? "text-[13.5px] font-semibold" : "text-[12px]"}`}>{value}</div>
    <div className="eyebrow mt-0.5">{label}</div>
  </div>
);

const Badge = ({ g }: { g: string }) => (
  <span className="inline-grid place-items-center rounded border border-border2 font-semibold text-text w-5 h-5 text-[10.5px]">{g}</span>
);
