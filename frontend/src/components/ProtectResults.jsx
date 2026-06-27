import ConflictWarning from "./ConflictWarning";

export default function ProtectResults({ result, originalUrl }) {
  if (!result) return null;
  const synthetic = result.forensic_label === "synthetic";
  return (
    <div className="results-card">
      <div className="compare">
        <figure>
          {originalUrl && <img src={originalUrl} alt="original" />}
          <figcaption>Original</figcaption>
        </figure>
        <figure>
          <img src={result.cloaked_image_b64} alt="cloaked" />
          <figcaption>Cloaked</figcaption>
        </figure>
      </div>

      <ConflictWarning show={result.conflict_warning} />

      <div className="status-row">
        <span className={`badge ${synthetic ? "badge-synthetic" : "badge-authentic"}`}>
          Forensic: {result.forensic_label.toUpperCase()}
        </span>
        <span className="metric">SSIM {result.ssim}</span>
        <span className="metric">PSNR {result.psnr} dB</span>
        <span className="metric">{result.processing_time_ms} ms</span>
      </div>

      <a className="download" href={result.cloaked_image_b64} download="cloaked.png">
        Download cloaked image
      </a>
    </div>
  );
}
