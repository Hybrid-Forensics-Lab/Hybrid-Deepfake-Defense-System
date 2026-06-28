import { useEffect, useRef, useState } from "react";

const reduceMotion =
  typeof window !== "undefined" &&
  window.matchMedia &&
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Animates a number from 0 up to `target` with an ease-out curve.
// Returns the current value rounded to `decimals` places.
export default function useCountUp(target, { duration = 900, decimals = 0 } = {}) {
  const to = Number(target) || 0;
  const [val, setVal] = useState(reduceMotion ? to : 0);
  const raf = useRef();

  useEffect(() => {
    if (reduceMotion) {
      setVal(to);
      return;
    }
    const start = performance.now();
    function tick(now) {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3); // easeOutCubic
      setVal(to * eased);
      if (t < 1) raf.current = requestAnimationFrame(tick);
    }
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [to, duration]);

  const p = Math.pow(10, decimals);
  return Math.round(val * p) / p;
}
