import { useEffect, useMemo, useState } from "react";

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function mixColor(a, b, t) {
  const n = clamp(t, 0, 1);
  const ar = (a >> 16) & 255;
  const ag = (a >> 8) & 255;
  const ab = a & 255;
  const br = (b >> 16) & 255;
  const bg = (b >> 8) & 255;
  const bb = b & 255;
  const r = Math.round(ar + (br - ar) * n);
  const g = Math.round(ag + (bg - ag) * n);
  const b2 = Math.round(ab + (bb - ab) * n);
  return `rgb(${r}, ${g}, ${b2})`;
}

export default function VariableProximityText({
  text,
  className = "",
  radius = 180,
  minScale = 1,
  maxScale = 1.12,
}) {
  const [mouse, setMouse] = useState({ x: -9999, y: -9999 });
  const chars = useMemo(() => text.split(""), [text]);

  useEffect(() => {
    const onMove = (event) => {
      setMouse({ x: event.clientX, y: event.clientY });
    };
    const onLeave = () => {
      setMouse({ x: -9999, y: -9999 });
    };

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseout", onLeave);

    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseout", onLeave);
    };
  }, []);

  return (
    <span className={`vp-text ${className}`}>
      {chars.map((char, idx) => (
        <span
          key={`${char}-${idx}`}
          className="vp-char"
          ref={(el) => {
            if (!el) return;
            const rect = el.getBoundingClientRect();
            const cx = rect.left + rect.width / 2;
            const cy = rect.top + rect.height / 2;
            const dx = mouse.x - cx;
            const dy = mouse.y - cy;
            const distance = Math.sqrt(dx * dx + dy * dy);
            const t = clamp(1 - distance / radius, 0, 1);
            const scale = minScale + (maxScale - minScale) * t;
            const color = mixColor(0xb7c3db, 0xffffff, t);
            el.style.transform = `translateZ(0) scale(${scale})`;
            el.style.color = color;
            el.style.opacity = String(0.72 + t * 0.28);
            el.style.textShadow =
              t > 0.02 ? `0 0 ${Math.round(16 * t)}px rgba(120, 205, 255, ${0.55 * t})` : "none";
          }}
        >
          {char === " " ? "\u00A0" : char}
        </span>
      ))}
    </span>
  );
}
