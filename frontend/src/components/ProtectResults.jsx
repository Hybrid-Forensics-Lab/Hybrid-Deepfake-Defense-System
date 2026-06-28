import useCountUp from "../useCountUp";
import ConflictWarning from "./ConflictWarning";

export default function ProtectResults({ result, originalUrl }) {
  if (!result) return null;
  return (
    <ProtectCard
      key={result.processing_time_ms}
      result={result}
      originalUrl={originalUrl}
    />
  );
}

function ProtectCard({ result, originalUrl }) {
  const synthetic = result.forensic_label === "synthetic";
  const ssim = useCountUp(result.ssim, { duration: 1000, decimals: 3 });
  const psnr = useCountUp(result.psnr, { duration: 1000, decimals: 1 });
  const ms = useCountUp(result.processing_time_ms, { duration: 900 });

  return (
    <section className="results">
      <div className="compare">
        <figure>
          <div className="frame">
            <span className="tag">ORIGINAL</span>
            {originalUrl && <img src={originalUrl} alt="original" />}
          </div>
          <figcaption>Original</figcaption>
        </figure>
        <figure>
          <div className="frame">
            <span className="tag">CLOAKED</span>
            <img src={result.cloaked_image_b64} alt="cloaked" />
          </div>
          <figcaption>Cloaked</figcaption>
        </figure>
      </div>

      <ConflictWarning show={result.conflict_warning} />

      <div className="results-head" style={{ marginTop: 18 }}>
        <span className={`verdict ${synthetic ? "synthetic" : "authentic"}`}>
          {synthetic ? <CrossGlyph /> : <CheckGlyph />}
          Forensic: {result.forensic_label.toUpperCase()}
        </span>
        <span className="results-label">Cloaked image still reads as real</span>
      </div>

      <div className="meta">
        <span className="chip">SSIM&nbsp;<b>{ssim.toFixed(3)}</b></span>
        <span className="chip">PSNR&nbsp;<b>{psnr.toFixed(1)} dB</b></span>
        <span className="chip">latency&nbsp;<b>{ms} ms</b></span>
      </div>

      <a className="download" href={result.cloaked_image_b64} download="cloaked.png">
        Download cloaked image
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor"
          strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M5 12h14M13 6l6 6-6 6" />
        </svg>
      </a>
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
