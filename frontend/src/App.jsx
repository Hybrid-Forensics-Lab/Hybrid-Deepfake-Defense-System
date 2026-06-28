import { useEffect, useState } from "react";
import UploadPanel from "./components/UploadPanel";
import DetectResults from "./components/DetectResults";
import ProtectResults from "./components/ProtectResults";
import { detect, protect, health } from "./api";

export default function App() {
  const [mode, setMode] = useState("detect");
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [epsilon, setEpsilon] = useState(0.03);
  const [useFacenet, setUseFacenet] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [detectResult, setDetectResult] = useState(null);
  const [protectResult, setProtectResult] = useState(null);
  const [hp, setHp] = useState(null);

  useEffect(() => {
    health().then(setHp).catch(() => setHp(null));
  }, []);

  function onSelect(f) {
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
    setDetectResult(null);
    setProtectResult(null);
    setError("");
  }

  function switchMode(m) {
    setMode(m);
    setDetectResult(null);
    setProtectResult(null);
    setError("");
  }

  async function run() {
    if (!file) return;
    setLoading(true);
    setError("");
    setDetectResult(null);
    setProtectResult(null);
    try {
      if (mode === "detect") {
        setDetectResult(await detect(file));
      } else {
        setProtectResult(await protect(file, epsilon, useFacenet));
      }
    } catch (e) {
      setError(e?.response?.data?.detail || "Request failed — is the API running?");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="shell">
      <header className="masthead">
        <div className="brand">
          <img className="mark" src="/favicon.svg" alt="" />
          <span className="wordmark">Deepfake Defense</span>
        </div>
        <h1>
          See through the fake.<br />
          <em>Protect the real.</em>
        </h1>
        <p className="lede">
          A dual-layer system that detects synthetic media and cloaks real faces
          against recognition — built on forensic CLIP features and adversarial perturbation.
        </p>

        {hp && (
          <div className="stats">
            <span className="stat">
              <span className={`dot${hp.gpu ? "" : " off"}`} />
              API {hp.status}
            </span>
            <span className="stat">GPU&nbsp;<b>{hp.gpu ? "on" : "off"}</b></span>
            <span className="stat">FF++ AUC&nbsp;<b>{hp.ff_auc}</b></span>
            <span className="stat">ProGAN AUC&nbsp;<b>{hp.progan_auc}</b></span>
          </div>
        )}
      </header>

      <main className="panel">
        <div className="seg" data-mode={mode}>
          <span className="seg-thumb" />
          <button
            className={mode === "detect" ? "active" : ""}
            onClick={() => switchMode("detect")}
          >
            Detect
          </button>
          <button
            className={mode === "protect" ? "active" : ""}
            onClick={() => switchMode("protect")}
          >
            Protect
          </button>
        </div>

        <p className="mode-help">
          {mode === "detect"
            ? "Classify an image as authentic or synthetic (deepfake)."
            : "Cloak a face so recognizers fail — while it stays visually unchanged."}
        </p>

        <UploadPanel onSelect={onSelect} previewUrl={previewUrl} />

        {mode === "protect" && (
          <div className="controls">
            <div>
              <div className="field-label">
                <span>Perturbation strength (ε)</span>
                <span className="val">{epsilon.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.01"
                max="0.1"
                step="0.01"
                value={epsilon}
                onChange={(e) => setEpsilon(parseFloat(e.target.value))}
              />
              <div className="range-scale">
                <span>subtle</span>
                <span>stronger</span>
              </div>
            </div>
            <label className="switch">
              <input
                type="checkbox"
                checked={useFacenet}
                onChange={(e) => setUseFacenet(e.target.checked)}
              />
              <span className="track" />
              Also cloak against FaceNet <span style={{ color: "var(--faint)" }}>(slower)</span>
            </label>
          </div>
        )}

        <button className="run-btn" onClick={run} disabled={!file || loading}>
          {loading && <span className="spinner" />}
          {loading
            ? "Processing…"
            : mode === "detect"
            ? "Analyze image"
            : "Protect image"}
        </button>

        {error && <div className="error">{error}</div>}
      </main>

      {mode === "detect" ? (
        <DetectResults result={detectResult} />
      ) : (
        <ProtectResults result={protectResult} originalUrl={previewUrl} />
      )}

      <footer className="foot">
        FT-UnivFD forensic probe · untargeted PGD identity cloaking · running on NVIDIA L4
      </footer>
    </div>
  );
}
