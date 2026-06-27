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
    <div className="app">
      <header>
        <h1>Hybrid Deepfake Defense System</h1>
        <p className="subtitle">Forensic detection &amp; adversarial identity cloaking</p>
        {hp && (
          <p className="health">
            ● API {hp.status} · GPU {hp.gpu ? "on" : "off"} · FF++ AUC {hp.ff_auc} · ProGAN
            AUC {hp.progan_auc}
          </p>
        )}
      </header>

      <div className="mode-toggle">
        <button className={mode === "detect" ? "active" : ""} onClick={() => switchMode("detect")}>
          Detect
        </button>
        <button className={mode === "protect" ? "active" : ""} onClick={() => switchMode("protect")}>
          Protect
        </button>
      </div>

      <p className="mode-help">
        {mode === "detect"
          ? "Classify an image as authentic or synthetic (deepfake)."
          : "Cloak a face so recognizers fail, while staying visually unchanged."}
      </p>

      <UploadPanel onSelect={onSelect} previewUrl={previewUrl} />

      {mode === "protect" && (
        <div className="controls">
          <label className="slider">
            Perturbation ε: <strong>{epsilon.toFixed(2)}</strong>
            <input
              type="range"
              min="0.01"
              max="0.1"
              step="0.01"
              value={epsilon}
              onChange={(e) => setEpsilon(parseFloat(e.target.value))}
            />
          </label>
          <label className="checkbox">
            <input
              type="checkbox"
              checked={useFacenet}
              onChange={(e) => setUseFacenet(e.target.checked)}
            />
            Also cloak against FaceNet (slower)
          </label>
        </div>
      )}

      <button className="run-btn" onClick={run} disabled={!file || loading}>
        {loading ? "Processing…" : mode === "detect" ? "Analyze image" : "Protect image"}
      </button>

      {error && <div className="error">{error}</div>}

      {mode === "detect" ? (
        <DetectResults result={detectResult} />
      ) : (
        <ProtectResults result={protectResult} originalUrl={previewUrl} />
      )}
    </div>
  );
}
