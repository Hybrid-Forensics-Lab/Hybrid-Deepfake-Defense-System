export default function ConflictWarning({ show }) {
  if (!show) return null;
  return (
    <div className="conflict">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
        strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z" />
        <path d="M12 9v4M12 17h.01" />
      </svg>
      <span>
        <b>Conflict:</b> the cloaked image is still flagged as synthetic by the forensic
        detector (residual false-positive case).
      </span>
    </div>
  );
}
