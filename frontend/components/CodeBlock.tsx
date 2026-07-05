"use client";

// Lightweight Python syntax highlighter — GitHub-dark palette, zero dependencies.
// keywords pink · strings green · numbers blue · comments grey · calls violet

const KEYWORDS = new Set([
  "def", "return", "import", "from", "as", "if", "elif", "else", "for", "while",
  "in", "not", "and", "or", "lambda", "with", "try", "except", "raise", "pass",
  "None", "True", "False",
]);
const BUILTINS = new Set(["np", "rank", "zscore", "abs", "min", "max", "len", "sum", "range"]);

type Tok = { text: string; cls?: string };

function tokenizeLine(line: string): Tok[] {
  const toks: Tok[] = [];
  let i = 0;
  while (i < line.length) {
    const rest = line.slice(i);
    let m: RegExpMatchArray | null;
    if ((m = rest.match(/^#.*/))) {
      toks.push({ text: m[0], cls: "text-[#8b949e] italic" }); i += m[0].length;
    } else if ((m = rest.match(/^('([^'\\]|\\.)*'|"([^"\\]|\\.)*")/))) {
      toks.push({ text: m[0], cls: "text-[#7ee787]" }); i += m[0].length;
    } else if ((m = rest.match(/^\d+\.?\d*(e-?\d+)?/))) {
      toks.push({ text: m[0], cls: "text-[#79c0ff]" }); i += m[0].length;
    } else if ((m = rest.match(/^[A-Za-z_][A-Za-z0-9_]*/))) {
      const w = m[0];
      const next = line[i + w.length];
      const cls = KEYWORDS.has(w) ? "text-[#ff7b72]"
        : BUILTINS.has(w) ? "text-[#d2a8ff]"
        : next === "(" ? "text-[#d2a8ff]"
        : undefined;
      toks.push({ text: w, cls }); i += w.length;
    } else {
      toks.push({ text: line[i] }); i += 1;
    }
  }
  return toks;
}

export default function CodeBlock({ code, className = "" }: { code: string; className?: string }) {
  const lines = code.split("\n");
  return (
    <pre className={`code p-3 text-[12px] leading-[1.65] overflow-x-auto ${className}`}>
      {lines.map((line, li) => (
        <div key={li}>
          {tokenizeLine(line).map((t, ti) =>
            t.cls ? <span key={ti} className={t.cls}>{t.text}</span> : <span key={ti}>{t.text}</span>
          )}
          {line === "" ? " " : ""}
        </div>
      ))}
    </pre>
  );
}
