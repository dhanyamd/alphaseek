// API client for the AlphaSeek backend (FastAPI @ :8000).
export const API = process.env.NEXT_PUBLIC_API ?? "http://localhost:8000";

export type Session = {
  id: number;
  seed: string;
  iterations: number;
  mode: string;
  status: string;
  created: number;
  best: any | null;
};

export async function login(name: string) {
  const r = await fetch(`${API}/api/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  return r.json();
}

export async function listSessions(user: string): Promise<Session[]> {
  const r = await fetch(`${API}/api/sessions?user=${encodeURIComponent(user)}`);
  const j = await r.json();
  return j.sessions ?? [];
}

export async function createSession(user: string, seed: string, iterations: number, mode: string): Promise<number> {
  const r = await fetch(`${API}/api/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user, seed, iterations, mode }),
  });
  const j = await r.json();
  return j.id;
}

export async function getSession(id: number) {
  const r = await fetch(`${API}/api/sessions/${id}`);
  return r.json();
}

export async function uploadFile(id: number, file: File, replaceDefault = false): Promise<string[]> {
  const fd = new FormData();
  fd.append("file", file);
  const params = replaceDefault ? "?replace_default=true" : "";
  const r = await fetch(`${API}/api/sessions/${id}/upload${params}`, { method: "POST", body: fd });
  const j = await r.json();
  return j.files ?? [];
}

// Open the live research stream. Calls onEvent for every agent event.
// Pass `prompt` for follow-up turns in an existing session (continuous chat).
export function streamResearch(id: number, onEvent: (ev: any) => void, prompt?: string): EventSource {
  const url = `${API}/api/sessions/${id}/stream` + (prompt ? `?prompt=${encodeURIComponent(prompt)}` : "");
  const es = new EventSource(url);
  es.onmessage = (e) => {
    try {
      const ev = JSON.parse(e.data);
      onEvent(ev);
      if (ev.type === "close") es.close();
    } catch {}
  };
  es.onerror = () => es.close();
  return es;
}

export async function runScript(id: number, filename: string, code: string) {
  const r = await fetch(`${API}/api/sessions/${id}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename, code }),
  });
  return r.json();
}
