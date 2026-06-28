import { useEffect, useState } from "react";
import useCountUp from "../useCountUp";

export default function DetectResults({ result }) {
  if (!result) return null;
  return <DetectCard key={result.processing_time_ms + result.label} result={result} />;
}

function DetectCard({ result }) {
  const synthetic = result.label === "synthetic";
  const target = result.confidence * 100;
  const pct = useCountUp(target, { duration: 1000, decimals: 1 });
  const ms = useCountUp(result.processing_time_ms, { duration: 900 });

  // animate the bar from 0 to its target width after mount
  const [w, setW] = useState(0);
  useEffect(() => {
    const id = requestAnimationFrame(() => setW(target));
    return () => cancelAnimationFrame(id);
  }, [target]);

  return (
    <section className="results">
      <div className="results-head">
        <span className={`verdict ${synthetic ? "synthetic" : "authentic"}`}>
          {synthetic ? <CrossGlyph /> : <CheckGlyph />}
          {result.label.toUpperCase()}
        </span>
        <span className="results-label">Forensic verdict</span>
      </div>

      <div className="conf">
        <div className="conf-top">
          <span>Confidence</span>
          <span className="conf-pct">{pct.toFixed(1)}%</span>
        </div>
        <div className="bar">
          <div
            className={`bar-fill ${synthetic ? "synthetic" : "authentic"}`}
            style={{ width: `${w}%` }}
          />
        </div>
      </div>

      <div className="meta">
        <span className="chip">model&nbsp;<b>{result.model}</b></span>
        <span className="chip">latency&nbsp;<b>{ms} ms</b></span>
      </div>
    </section>
  );
}

function CheckGlyph() {
  return (
    <svg className="glyph" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
      <path d="M20 6 9 17l-5-5" />
    </svg>
  );
}

function CrossGlyph() {
  return (
    <svg className="glyph" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
      <path d="M18 6 6 18M6 6l12 12" />
    </svg>
  );
}
