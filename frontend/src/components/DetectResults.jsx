export default function DetectResults({ result }) {
  if (!result) return null;
  const synthetic = result.label === "synthetic";
  const pct = (result.confidence * 100).toFixed(1);
  return (
    <div className="results-card">
      <div className={`badge ${synthetic ? "badge-synthetic" : "badge-authentic"}`}>
        {result.label.toUpperCase()}
      </div>
      <div className="conf-row">
        <span>Confidence</span>
        <div className="bar">
          <div
            className={`bar-fill ${synthetic ? "fill-synthetic" : "fill-authentic"}`}
            style={{ width: `${pct}%` }}
          />
        </div>
        <span>{pct}%</span>
      </div>
      <div className="meta">
        Model: {result.model} · {result.processing_time_ms} ms
      </div>
    </div>
  );
}
