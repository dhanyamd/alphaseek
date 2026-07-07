"use client";

import { useEffect, useState } from "react";
import { API } from "@/lib/api";
import CodeBlock, { type Lang } from "./CodeBlock";
import dynamic from "next/dynamic";

const TVChart = dynamic(() => import("./TVChart"), { ssr: false });

interface BacktestData {
  equity_curve?: number[];
  [key: string]: unknown; // dynamic metrics — never hardcoded
}

/** Render whatever metrics the backend sent — no hardcoded key list. */
function MetricsGrid({ bt }: { bt: BacktestData }) {
  const skip = new Set(["equity_curve", "name", "hypothesis"]);
  const entries = Object.entries(bt).filter(
    ([k, v]) => !skip.has(k) && v != null && typeof v !== "object"
  );
  if (!entries.length) return null;

  const fmt = (k: string, v: unknown): string => {
    if (typeof v === "boolean") return v ? "Yes" : "No";
    if (typeof v === "string") return v;
    if (typeof v === "number") {
      if (k.includes("drawdown")) return `${(v * 100).toFixed(0)}%`;
      if (Math.abs(v) < 0.01 && v !== 0) return v.toFixed(4);
      return v.toFixed(2);
    }
    return String(v);
  };

  const color = (k: string, v: unknown): string => {
    if (k === "grade" && typeof v === "string") {
      return v === "A" ? "#22c55e" : v === "B" ? "#60a5fa" : "#f59e0b";
    }
    if (typeof v === "number" && k.includes("drawdown")) return "#f87171";
    return "inherit";
  };

  return (
    <div className="card p-3">
      <div className="eyebrow mb-2">Metrics</div>
      <div className="grid grid-cols-4 gap-3">
        {entries.map(([k, v]) => (
          <div key={k}>
            <div className="text-[10px] text-faint">{k.replace(/_/g, " ")}</div>
            <div className="text-[15px] tnum" style={{ color: color(k, v) }}>
              {fmt(k, v)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function StrategyViewer({
  code,
  lang,
  btDataFilename,
}: {
  code: string;
  lang: Lang;
  btDataFilename: string | null;
}) {
  const [bt, setBt] = useState<BacktestData | null>(null);
  const [btError, setBtError] = useState(false);
  const [pineStatus, setPineStatus] = useState<"idle" | "running" | "ok" | "error">("idle");
  const [pineResult, setPineResult] = useState<Record<string, number[]>>({});
  const [pineError, setPineError] = useState("");

  useEffect(() => {
    if (!btDataFilename) return;
    fetch(`${API}/api/artifacts/${btDataFilename}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (d) setBt(d);
        else setBtError(true);
      })
      .catch(() => setBtError(true));
  }, [btDataFilename]);

  const equity = bt?.equity_curve;
  const hasEq = Array.isArray(equity) && equity.length > 1;

  // ── PineTS in-browser execution ──────────────────────────────────
  async function runPine() {
    setPineStatus("running");
    setPineError("");
    setPineResult({});
    try {
      const { PineTS } = await import("pinets");

      // Build OHLCV candles from the equity curve if available,
      // otherwise generate a simple synthetic series
      const src = hasEq ? equity! : Array.from({ length: 200 }, (_, i) => 100 + i * 0.1);
      const candles = src.map((v, i) => ({
        openTime: Date.now() - (src.length - i) * 86400000,
        open: v,
        high: v * 1.005,
        low: v * 0.995,
        close: v,
        volume: 1e6,
        closeTime: Date.now() - (src.length - i) * 86400000 + 86399999,
        quoteAssetVolume: 0,
        numberOfTrades: 0,
        takerBuyBaseAssetVolume: 0,
        takerBuyQuoteAssetVolume: 0,
        ignore: 0,
      }));

      const engine = new PineTS(candles);
      const ctx = await engine.run(code);

      // Extract plot data from the PineTS Context
      const plots: Record<string, number[]> = {};
      if (ctx && typeof ctx === "object") {
        // Context stores plot series as properties — iterate and collect arrays
        for (const [k, v] of Object.entries(ctx as unknown as Record<string, unknown>)) {
          if (Array.isArray(v) && v.length > 0 && typeof v[0] === "number") {
            plots[k] = v as number[];
          }
        }
      }
      setPineResult(plots);
      setPineStatus("ok");
    } catch (e: any) {
      setPineError(e?.message ?? String(e));
      setPineStatus("error");
    }
  }

  return (
    <div className="h-full flex flex-col overflow-auto p-4 gap-4">
      {/* Strategy name & hypothesis */}
      {bt && (bt as any).name && (
        <div>
          <div className="text-[13px] font-semibold text-text">{(bt as any).name}</div>
          {(bt as any).hypothesis && (
            <div className="text-[11px] text-muted mt-0.5">{(bt as any).hypothesis}</div>
          )}
        </div>
      )}

      {/* Equity curve */}
      {hasEq && (
        <div className="card p-3">
          <div className="eyebrow mb-2">Equity Curve</div>
          <div className="h-[200px]">
            <TVChart data={equity!} height={200} color="#22c55e" />
          </div>
        </div>
      )}
      {btError && (
        <div className="text-[11px] text-muted">backtest data not available</div>
      )}

      {/* Metrics — rendered dynamically from whatever the backend sent */}
      {bt && <MetricsGrid bt={bt} />}

      {/* PineTS in-browser execution (Pine Script only) */}
      {lang === "pine" && (
        <div className="card p-3">
          <div className="eyebrow mb-2">In-Browser Execution (PineTS)</div>
          <p className="text-[11px] text-muted mb-2">
            Transpile and run this Pine Script in the browser.
            {hasEq
              ? " Uses the backtest equity curve as input data."
              : " Uses synthetic OHLCV data."}
          </p>
          <button
            onClick={runPine}
            disabled={pineStatus === "running"}
            className="btn px-3 py-1.5 text-[12px] disabled:opacity-40"
          >
            {pineStatus === "running" ? "running..." : "Run in Browser"}
          </button>

          {pineStatus === "ok" && (
            <div className="mt-2">
              <div className="text-[11px] text-green-400 mb-1">Transpiled and executed</div>
              {Object.keys(pineResult).length > 0 ? (
                <div className="grid grid-cols-2 gap-2">
                  {Object.entries(pineResult).map(([name, data]) => (
                    <div key={name} className="border border-border rounded p-2">
                      <div className="text-[10px] text-faint mb-1">{name}</div>
                      <div className="h-[60px]">
                        <TVChart data={data} height={60} color="#60a5fa" />
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-[10px] text-muted">
                  Script executed. Indicator plots will appear here when available.
                </div>
              )}
            </div>
          )}

          {pineStatus === "error" && (
            <div className="mt-2">
              <div className="text-[11px] text-red-400">Execution failed</div>
              <pre className="text-[10px] text-muted mt-1 whitespace-pre-wrap max-h-[80px] overflow-auto">
                {pineError}
              </pre>
              <p className="text-[10px] text-muted mt-1">
                PineTS may not support all strategy() directives. The equity curve above
                shows the validated backtest performance.
              </p>
            </div>
          )}
        </div>
      )}

      {/* MQL5 — no in-browser runtime exists, show code + download */}
      {lang === "mql5" && (
        <div className="card p-3">
          <div className="eyebrow mb-2">MQL5 Expert Advisor</div>
          <p className="text-[11px] text-muted mb-2">
            MQL5 runs natively in MetaTrader 5. Copy or download this EA to test it in the
            Strategy Tester.
          </p>
        </div>
      )}

      {/* Source code */}
      <div className="card p-3">
        <div className="eyebrow mb-2">
          {lang === "pine" ? "Pine Script" : "MQL5"} — {code.split("\n").length} lines
        </div>
        <CodeBlock code={code} lang={lang} className="text-[11px] max-h-[400px] overflow-auto" />
        <div className="flex gap-2 mt-2">
          <button
            onClick={() => navigator.clipboard.writeText(code)}
            className="text-[10px] text-[#58a6ff] hover:underline"
          >
            copy code
          </button>
          <a
            href={`data:text/plain;charset=utf-8,${encodeURIComponent(code)}`}
            download={lang === "pine" ? "strategy.pine" : "strategy.mq5"}
            className="text-[10px] text-[#58a6ff] hover:underline"
          >
            download
          </a>
        </div>
      </div>
    </div>
  );
}
