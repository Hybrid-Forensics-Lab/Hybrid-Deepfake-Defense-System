export default function ConflictWarning({ show }) {
  if (!show) return null;
  return (
    <div className="conflict-banner">
      ⚠ Conflict: the cloaked image is still flagged as synthetic by the forensic
      detector (residual false-positive case).
    </div>
  );
}
