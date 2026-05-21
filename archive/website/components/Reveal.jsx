"use client";

import { useEffect, useRef, useState } from "react";

/**
 * Reveal — wraps content in a div that flips `data-revealed="1"` once it
 * intersects the viewport. The actual animation lives in globals.css under
 * `[data-reveal]` selectors so transitions stay CSS native.
 *
 * Props:
 *   variant    "fade-up" (default) | "fade" | "slide-left" | "slide-right" | "scale"
 *   delay      ms (number)
 *   stagger    when true, the immediate children get sequential delays via CSS
 *   threshold  viewport intersection ratio to trigger (default 0.12)
 *   once       only trigger once (default true)
 *   as         tag to render (default "div")
 */
export default function Reveal({
  children,
  variant = "fade-up",
  delay = 0,
  stagger = false,
  threshold = 0.12,
  once = true,
  as: Tag = "div",
  className = "",
  ...rest
}) {
  const ref = useRef(null);
  const [revealed, setRevealed] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const node = ref.current;
    if (!node) return;

    // Reduced motion or no observer support: reveal immediately.
    const prefersReduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    if (prefersReduced || typeof IntersectionObserver === "undefined") {
      setRevealed(true);
      return;
    }

    // If the element is already in the viewport at mount time, reveal
    // immediately. This handles top-of-page content and avoids the
    // FOUC pattern where above-the-fold elements stay invisible.
    const rect = node.getBoundingClientRect();
    const vh = window.innerHeight || document.documentElement.clientHeight;
    if (rect.top < vh * 0.92 && rect.bottom > 0) {
      setRevealed(true);
      return;
    }

    // Safety timeout: if observer never fires (e.g. headless render without
    // scroll), reveal after 3 seconds so content is never permanently hidden.
    const fallback = setTimeout(() => setRevealed(true), 3000);

    const obs = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            setRevealed(true);
            clearTimeout(fallback);
            if (once) obs.unobserve(e.target);
          } else if (!once) {
            setRevealed(false);
          }
        });
      },
      { threshold, rootMargin: "0px 0px -8% 0px" }
    );
    obs.observe(node);

    return () => {
      clearTimeout(fallback);
      obs.disconnect();
    };
  }, [threshold, once]);

  const style =
    typeof delay === "number" && delay > 0 ? { transitionDelay: `${delay}ms` } : undefined;

  return (
    <Tag
      ref={ref}
      data-reveal={variant}
      data-revealed={revealed ? 1 : 0}
      data-stagger={stagger ? 1 : undefined}
      style={style}
      className={className}
      {...rest}
    >
      {children}
    </Tag>
  );
}
