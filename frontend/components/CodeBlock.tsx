"use client";

// Syntax highlighter — GitHub-dark palette, zero dependencies.
// Supports Python, Pine Script v5, and MQL5.
// keywords pink · strings green · numbers blue · comments grey · calls violet

export type Lang = "python" | "pine" | "mql5" | "auto";

const KEYWORDS: Record<Lang, Set<string>> = {
  python: new Set([
    "def", "return", "import", "from", "as", "if", "elif", "else", "for", "while",
    "in", "not", "and", "or", "lambda", "with", "try", "except", "raise", "pass",
    "None", "True", "False",
  ]),
  pine: new Set([
    "indicator", "strategy", "overlay", "title", "shorttitle", "format", "precision",
    "if", "else", "for", "while", "break", "continue", "var", "varip", "int", "float",
    "bool", "color", "string", "line", "label", "shape", "plot", "plotshape", "plotchar",
    "hline", "fill", "bgcolor", "barcolor", "alertcondition", "strategy.entry",
    "strategy.exit", "strategy.close", "strategy.order", "strategy.risk",
    "input", "simple", "series", "true", "false", "na", "return",
  ]),
  mql5: new Set([
    "input", "extern", "enum", "class", "struct", "void", "int", "double", "string",
    "bool", "datetime", "color", "long", "uint", "uchar", "ulong",
    "if", "else", "for", "while", "do", "switch", "case", "break", "continue", "return",
    "true", "false", "NULL", "new", "delete", "this", "virtual", "override",
    "const", "static", "private", "protected", "public",
    "int OnInit", "void OnDeinit", "void OnTick", "void OnTimer", "void OnChartEvent",
    "MqlTick", "MqlTradeRequest", "MqlTradeResult", "MqlRates",
    "OrderSend", "OrderSelect", "OrderDelete", "OrderModify",
    "PositionSelect", "PositionClose", "PositionGetInteger", "PositionGetDouble",
    "SymbolInfoTick", "SymbolInfoDouble", "SymbolInfoInteger",
    "CopyRates", "CopyClose", "CopyHigh", "CopyLow", "CopyOpen",
    "iClose", "iOpen", "iHigh", "iLow", "iVolume",
    "EMA", "SMA", "RSI", "MACD", "BollingerBands", "ATR", "ADX", "Stochastic",
    "ChartOpen", "ChartClose", "ChartSetSymbolPeriod",
    "Print", "Comment", "Alert", "MessageBox",
    "ObjectCreate", "ObjectSetInteger", "ObjectSetDouble", "ObjectSetString",
    "ChartIndicatorAdd", "IndicatorCreate", "IndicatorRelease",
  ]),
  auto: new Set(),
};
const BUILTINS: Record<Lang, Set<string>> = {
  python: new Set(["np", "pd", "plt", "go", "rank", "zscore", "abs", "min", "max", "len", "sum", "range"]),
  pine: new Set(["close", "open", "high", "low", "volume", "time", "bar_index", "barstate"]),
  mql5: new Set(["Close", "Open", "High", "Low", "Volume", "Time", "Bid", "Ask", "Point", "Digits"]),
  auto: new Set(),
};

function detectLang(code: string): Lang {
  if (/^(strategy|indicator|study)\s*\(/im.test(code)) return "pine";
  if (/^\s*#property\s+/im.test(code)) return "mql5";
  if (/^(input\s+int\s+|void\s+OnTick\s*\(|MqlTick\s+)/im.test(code)) return "mql5";
  return "python";
}

const COMMENT_PAT: Record<Lang, string> = {
  python: "^#.*",
  pine: "^//.*",
  mql5: "^//.*",
  auto: "^//.*|^#.*",
};

type Tok = { text: string; cls?: string };

function tokenizeLine(line: string, lang: Lang): Tok[] {
  const toks: Tok[] = [];
  let i = 0;
  const keywords = KEYWORDS[lang];
  const builtins = BUILTINS[lang];
  while (i < line.length) {
    const rest = line.slice(i);
    let m: RegExpMatchArray | null;
    if ((m = rest.match(new RegExp(COMMENT_PAT[lang])))) {
      toks.push({ text: m[0], cls: "text-[#8b949e] italic" }); i += m[0].length;
    } else if ((m = rest.match(/^('([^'\\]|\\.)*'|"([^"\\]|\\.)*")/))) {
      toks.push({ text: m[0], cls: "text-[#7ee787]" }); i += m[0].length;
    } else if ((m = rest.match(/^\d+\.?\d*(e-?\d+)?/))) {
      toks.push({ text: m[0], cls: "text-[#79c0ff]" }); i += m[0].length;
    } else if ((m = rest.match(/^[A-Za-z_][A-Za-z0-9_]*/))) {
      const w = m[0];
      const next = line[i + w.length];
      const cls = keywords.has(w) ? "text-[#ff7b72]"
        : builtins.has(w) ? "text-[#d2a8ff]"
        : next === "(" ? "text-[#d2a8ff]"
        : undefined;
      toks.push({ text: w, cls }); i += w.length;
    } else {
      toks.push({ text: line[i] }); i += 1;
    }
  }
  return toks;
}

export default function CodeBlock({ code, lang: langProp, className = "" }: { code: string; lang?: Lang; className?: string }) {
  const lang = langProp || detectLang(code);
  const lines = code.split("\n");
  return (
    <pre className={`code p-3 text-[12px] leading-[1.65] overflow-x-auto ${className}`}>
      {lines.map((line, li) => (
        <div key={li}>
          {tokenizeLine(line, lang).map((t, ti) =>
            t.cls ? <span key={ti} className={t.cls}>{t.text}</span> : <span key={ti}>{t.text}</span>
          )}
          {line === "" ? " " : ""}
        </div>
      ))}
    </pre>
  );
}
