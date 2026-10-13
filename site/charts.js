// Small dependency-free SVG charts with hover tooltips.
const NS = "http://www.w3.org/2000/svg";
const tip = () => document.getElementById("tooltip");

export function svgEl(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

export function showTip(html, evt) {
  const t = tip();
  t.innerHTML = html;
  t.hidden = false;
  const pad = 14, w = t.offsetWidth, h = t.offsetHeight;
  let x = evt.clientX + pad, y = evt.clientY + pad;
  if (x + w > window.innerWidth - 8) x = evt.clientX - w - pad;
  if (y + h > window.innerHeight - 8) y = evt.clientY - h - pad;
  t.style.left = `${Math.max(8, x)}px`;
  t.style.top = `${Math.max(8, y)}px`;
}
export function hideTip() { tip().hidden = true; }

const scale = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);

function niceTicks(min, max, count = 5) {
  const span = max - min, step0 = span / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count) || step0;
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

function frame(container, margin) {
  container.innerHTML = "";
  const W = container.clientWidth || 600, H = container.clientHeight || 260;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, "aria-hidden": "true" }, container);
  return { svg, W, H, x0: margin.l, x1: W - margin.r, y0: H - margin.b, y1: margin.t };  // W: viewBox width = CSS px
}

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function yAxis(svg, f, ticks, fmt) {
  const g = svgEl("g", { class: "grid" }, svg);
  for (const t of ticks) {
    const y = f.y(t);
    svgEl("line", { x1: f.x0, x2: f.x1, y1: y, y2: y }, g);
    svgEl("text", { x: f.x0 - 8, y: y + 4, "text-anchor": "end" }, svg).textContent = fmt(t);
  }
}

// Re-render on resize, debounced.
export function responsive(container, draw) {
  draw();
  let raf;
  new ResizeObserver(() => { cancelAnimationFrame(raf); raf = requestAnimationFrame(draw); }).observe(container);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
}

// ------------------------------------------------------------ vertical bars
// bars: [{label, value, lo, hi, tip, emphasis}]
export function barChart(container, { bars, yMax, yFormat, onFocus }) {
  responsive(container, () => {
    const f = frame(container, { l: 44, r: 8, t: 14, b: 26 });
    f.y = scale(0, yMax, f.y0, f.y1);
    yAxis(f.svg, f, niceTicks(0, yMax, 4), yFormat);
    const n = bars.length, slot = (f.x1 - f.x0) / n, bw = Math.min(46, slot * 0.56);
    bars.forEach((b, i) => {
      const cx = f.x0 + slot * (i + 0.5), y = f.y(b.value);
      const color = b.emphasis === false ? css("--neutral") : css("--series-1");
      // bar with rounded data-end anchored to the baseline
      const h = Math.max(f.y0 - y, 1), r = Math.min(4, h / 2, bw / 2);
      svgEl("path", { d: `M${cx - bw / 2},${f.y0} V${y + r} q0,-${r} ${r},-${r} H${cx + bw / 2 - r} q${r},0 ${r},${r} V${f.y0} Z`, fill: color }, f.svg);
      if (b.lo !== undefined) {
        const g = svgEl("g", { stroke: css("--ink-2"), "stroke-width": 1.5 }, f.svg);
        svgEl("line", { x1: cx, x2: cx, y1: f.y(b.lo), y2: f.y(b.hi) }, g);
        svgEl("line", { x1: cx - 5, x2: cx + 5, y1: f.y(b.lo), y2: f.y(b.lo) }, g);
        svgEl("line", { x1: cx - 5, x2: cx + 5, y1: f.y(b.hi), y2: f.y(b.hi) }, g);
      }
      svgEl("text", { x: cx, y: f.y0 + 17, "text-anchor": "middle", class: "axis-label" }, f.svg).textContent = b.label;
      const hit = svgEl("rect", { x: cx - slot / 2, y: f.y1, width: slot, height: f.y0 - f.y1, class: "hit", tabindex: 0,
        role: "button", "aria-label": b.aria || b.label }, f.svg);
      const enter = (e) => { if (b.tip) showTip(b.tip, e); onFocus && onFocus(i); };
      hit.addEventListener("mousemove", enter);
      hit.addEventListener("mouseleave", hideTip);
      hit.addEventListener("focus", () => onFocus && onFocus(i));
      hit.addEventListener("click", () => onFocus && onFocus(i));
    });
    svgEl("line", { x1: f.x0, x2: f.x1, y1: f.y0, y2: f.y0, class: "baseline" }, f.svg);
  });
}

// --------------------------------------------------------- horizontal bars
// rows: [{label, value, emphasis, tip}]
export function hbarChart(container, { rows, xMin, xMax, format, ref }) {
  responsive(container, () => {
    const narrow = container.clientWidth < 560;
    const f = frame(container, { l: narrow ? 8 : 250, r: 56, t: narrow ? 4 : 8, b: 24 });
    f.x = scale(xMin, xMax, f.x0, f.x1);
    const g = svgEl("g", { class: "grid" }, f.svg);
    for (const t of niceTicks(xMin, xMax, 5)) {
      svgEl("line", { x1: f.x(t), x2: f.x(t), y1: f.y1, y2: f.y0 }, g);
      svgEl("text", { x: f.x(t), y: f.y0 + 16, "text-anchor": "middle" }, f.svg).textContent = format(t);
    }
    const slot = (f.y0 - f.y1) / rows.length;
    rows.forEach((row, i) => {
      const top = f.y1 + slot * i;
      const bh = narrow ? Math.min(14, slot * 0.34) : Math.min(22, slot * 0.55);
      const by = narrow ? top + slot * 0.55 : top + (slot - bh) / 2;
      const w = Math.max(f.x(row.value) - f.x0, 1), r = Math.min(4, bh / 2);
      const color = row.emphasis ? css("--series-1") : css("--neutral");
      svgEl("path", { d: `M${f.x0},${by} H${f.x0 + w - r} q${r},0 ${r},${r} V${by + bh - r} q0,${r} -${r},${r} H${f.x0} Z`, fill: color }, f.svg);
      const lab = svgEl("text", narrow ? { x: f.x0, y: top + slot * 0.42, class: "axis-label" }
        : { x: f.x0 - 10, y: by + bh / 2 + 4, "text-anchor": "end", class: "axis-label" }, f.svg);
      lab.textContent = row.label;
      svgEl("text", { x: f.x0 + w + 6, y: by + bh / 2 + 4, class: "value-label" }, f.svg).textContent = format(row.value);
      const hit = svgEl("rect", { x: 0, y: top, width: f.x1 + 56, height: slot, class: "hit" }, f.svg);
      hit.addEventListener("mousemove", (e) => row.tip && showTip(row.tip, e));
      hit.addEventListener("mouseleave", hideTip);
    });
    if (ref) {
      const x = f.x(ref.value);
      svgEl("line", { x1: x, x2: x, y1: f.y1, y2: f.y0, stroke: css("--ink-3"), "stroke-width": 1 }, f.svg);
    }
  });
}

// ------------------------------------------------------------- line chart
// series: [{name, color, points:[{x,y,lo,hi}], band, markers, interp, width}]
// xTicks: [{v,label}], gaps: break lines where consecutive x differ by > maxGap
export function lineChart(container, opts) {
  responsive(container, () => drawLines(container, opts));
}

function drawLines(container, { series, xDomain, yDomain, xTicks, yFormat, tipX, tipY, maxGap = Infinity,
  marker, diagonal, margin }) {
  const f = frame(container, margin || { l: 46, r: 14, t: 12, b: 28 });
  f.x = scale(xDomain[0], xDomain[1], f.x0, f.x1);
  f.y = scale(yDomain[0], yDomain[1], f.y0, f.y1);
  yAxis(f.svg, f, niceTicks(yDomain[0], yDomain[1], 4), yFormat);
  for (const t of xTicks) {
    svgEl("text", { x: f.x(t.v), y: f.y0 + 18, "text-anchor": t.anchor || "middle" }, f.svg).textContent = t.label;
  }
  svgEl("line", { x1: f.x0, x2: f.x1, y1: f.y0, y2: f.y0, class: "baseline" }, f.svg);
  if (diagonal) {
    svgEl("line", { x1: f.x(0), y1: f.y(0), x2: f.x(1), y2: f.y(1), stroke: css("--neutral"), "stroke-width": 1.5 }, f.svg);
  }
  const segs = (pts) => {
    const out = [[]];
    pts.forEach((p, i) => {
      if (p.y === null || p.y === undefined) { out.push([]); return; }
      if (i > 0 && pts[i - 1].y !== null && p.x - pts[i - 1].x > maxGap) out.push([]);
      out[out.length - 1].push(p);
    });
    return out.filter((s) => s.length);
  };
  for (const s of series) {
    const color = css(s.color);
    if (s.band) {
      for (const seg of segs(s.points)) {
        const d = seg.map((p, i) => `${i ? "L" : "M"}${f.x(p.x)},${f.y(p.hi)}`).join("") +
          seg.slice().reverse().map((p) => `L${f.x(p.x)},${f.y(p.lo)}`).join("") + "Z";
        svgEl("path", { d, fill: css("--band"), stroke: "none" }, f.svg);
      }
    }
    for (const seg of segs(s.points)) {
      if (seg.length > 1) {
        svgEl("path", { d: seg.map((p, i) => `${i ? "L" : "M"}${f.x(p.x)},${f.y(p.y)}`).join(""),
          fill: "none", stroke: color, "stroke-width": s.width || 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, f.svg);
      }
      if (s.markers || seg.length === 1) {
        for (const p of seg) svgEl("circle", { cx: f.x(p.x), cy: f.y(p.y), r: 4, fill: color,
          stroke: css("--surface"), "stroke-width": 2 }, f.svg);
      }
    }
  }
  if (marker) {
    svgEl("circle", { cx: f.x(marker.x), cy: f.y(marker.y), r: 9, fill: "none", stroke: css("--ink"), "stroke-width": 2 }, f.svg);
    svgEl("circle", { cx: f.x(marker.x), cy: f.y(marker.y), r: 4, fill: css(marker.color) }, f.svg);
  }
  // hover layer: crosshair + tooltip with every series at that x
  const cross = svgEl("line", { y1: f.y1, y2: f.y0, class: "crosshair", visibility: "hidden" }, f.svg);
  const dots = svgEl("g", {}, f.svg);
  const hit = svgEl("rect", { x: f.x0, y: f.y1, width: f.x1 - f.x0, height: f.y0 - f.y1, class: "hit" }, f.svg);
  const xs = [...new Set(series.flatMap((s) => s.interp ? [] : s.points.filter((p) => p.y !== null).map((p) => p.x)))].sort((a, b) => a - b);
  hit.addEventListener("mousemove", (e) => {
    const rect = container.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * f.W;
    let xv = xDomain[0] + ((px - f.x0) / (f.x1 - f.x0)) * (xDomain[1] - xDomain[0]);
    if (xs.length) xv = xs.reduce((a, b) => (Math.abs(b - xv) < Math.abs(a - xv) ? b : a));
    xv = Math.min(Math.max(xv, xDomain[0]), xDomain[1]);
    cross.setAttribute("x1", f.x(xv)); cross.setAttribute("x2", f.x(xv)); cross.setAttribute("visibility", "visible");
    dots.innerHTML = "";
    const lines = [];
    for (const s of series) {
      if (s.noTip) continue;
      const v = valueAt(s, xv);
      if (v === null) continue;
      svgEl("circle", { cx: f.x(xv), cy: f.y(v.y), r: 4, fill: css(s.color), stroke: css("--surface"), "stroke-width": 2 }, dots);
      lines.push(`<span class="sw" style="background:${css(s.color)}"></span>${s.name}: <b>${tipY(v.y, v, s)}</b>`);
    }
    showTip(`<b>${tipX(xv)}</b><br>${lines.join("<br>")}`, e);
  });
  hit.addEventListener("mouseleave", () => { hideTip(); cross.setAttribute("visibility", "hidden"); dots.innerHTML = ""; });
}

function valueAt(s, x) {
  const pts = s.points.filter((p) => p.y !== null && p.y !== undefined);
  if (!pts.length) return null;
  if (!s.interp) return pts.find((p) => p.x === x) || null;
  if (x < pts[0].x || x > pts[pts.length - 1].x) return null;
  for (let i = 1; i < pts.length; i++) {
    if (pts[i].x >= x) {
      const a = pts[i - 1], b = pts[i], t = b.x === a.x ? 0 : (x - a.x) / (b.x - a.x);
      return { ...b, y: a.y + t * (b.y - a.y) };
    }
  }
  return pts[pts.length - 1];
}

export function legend(el, items) {
  el.innerHTML = items.map((it) =>
    `<span><i class="${it.kind || ""}" style="background:${it.pattern || css(it.color)}"></i>${it.label}</span>`).join("");
}
