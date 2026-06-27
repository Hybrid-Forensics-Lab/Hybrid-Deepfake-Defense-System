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
        className={`dropzone${drag ? " drag" : ""}`}
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
          <img className="preview" src={previewUrl} alt="preview" />
        ) : (
          <p>
            Drag &amp; drop an image here, or click to choose
            <br />
            <small>PNG / JPEG · max 10 MB</small>
          </p>
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
