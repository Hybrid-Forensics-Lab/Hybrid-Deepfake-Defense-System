import { useRef, useState } from "react";

const MAX_BYTES = 10 * 1024 * 1024;
const ALLOWED = ["image/png", "image/jpeg"];

export default function UploadPanel({ onSelect, previewUrl }) {
  const inputRef = useRef();
  const [drag, setDrag] = useState(false);
  const [err, setErr] = useState("");

  function handle(file) {
    if (!file) return;
    if (!ALLOWED.includes(file.type)) {
      setErr("PNG or JPEG only");
      return;
    }
    if (file.size > MAX_BYTES) {
      setErr("Image exceeds the 10 MB limit");
      return;
    }
    setErr("");
    onSelect(file);
  }

  return (
    <div>
      <div
        className={`dropzone${drag ? " drag" : ""}${previewUrl ? " has-image" : ""}`}
        onClick={() => inputRef.current.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          handle(e.dataTransfer.files[0]);
        }}
      >
        {previewUrl ? (
          <>
            <img className="preview" src={previewUrl} alt="selected" />
            <p className="dz-replace">Click or drop to replace</p>
          </>
        ) : (
          <>
            <svg className="dz-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
              strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 16V4" />
              <path d="m7 9 5-5 5 5" />
              <path d="M5 16v2a3 3 0 0 0 3 3h8a3 3 0 0 0 3-3v-2" />
            </svg>
            <p className="dz-title">Drop an image here, or click to browse</p>
            <p className="dz-sub">PNG or JPEG · up to 10 MB</p>
          </>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg"
          hidden
          onChange={(e) => handle(e.target.files[0])}
        />
      </div>
      {err && <div className="error">{err}</div>}
    </div>
  );
}
