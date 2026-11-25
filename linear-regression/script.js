/* Linear Regression Lab
 * y = mx + b, fitted two ways: the closed-form normal equation (instant, exact)
 * and gradient descent (iterative, animated).
 */

// --------------------------------------------------------------- constants
const DOMAIN = { xMin: 0, xMax: 10, yMin: 0, yMax: 10 };
const PAD = { top: 16, right: 16, bottom: 34, left: 40 };
const HIT_RADIUS = 12;          // px, for grabbing a point
const DRAG_THRESHOLD = 3;       // px of movement before a click becomes a drag
const DIVERGE_LIMIT = 1e6;      // |m| or |b| beyond this means the run blew up
const CONVERGE_EPS = 1e-7;      // gradient magnitude below this means "settled"
const MAX_LOSS_HISTORY = 5000;

// ------------------------------------------------------------------- state
const state = {
  points: [],
  gd: { m: 0, b: 0 },
  ols: null,              // {m, b} or null when n < 2
  iter: 0,
  running: false,
  diverged: false,
  converged: false,
  lossHistory: [],
  drag: null,             // {index, moved}
};

// ------------------------------------------------------------------- DOM
const $ = (id) => document.getElementById(id);

const plot = $("plot");
const pctx = plot.getContext("2d");
const lossCanvas = $("loss");
const lctx = lossCanvas.getContext("2d");

const ui = {
  equation: $("equation"),
  statM: $("statM"), statB: $("statB"), statMse: $("statMse"),
  statR2: $("statR2"), statN: $("statN"), statIter: $("statIter"),
  status: $("status"),
  playBtn: $("playBtn"), stepBtn: $("stepBtn"), resetBtn: $("resetBtn"),
  randomBtn: $("randomBtn"), clearBtn: $("clearBtn"),
  themeToggle: $("themeToggle"),
  lr: $("lr"), lrOut: $("lrOut"),
  speed: $("speed"), speedOut: $("speedOut"),
  noise: $("noise"), noiseOut: $("noiseOut"),
  showResiduals: $("showResiduals"),
  showOls: $("showOls"),
  showSquares: $("showSquares"),
  emptyHint: $("emptyHint"),
  lossHint: $("lossHint"),
};

// Learning rate slider is logarithmic: value is log10(lr).
const learningRate = () => Math.pow(10, parseFloat(ui.lr.value));
const stepsPerFrame = () => parseInt(ui.speed.value, 10);

// ------------------------------------------------------------------ colors
// Read from CSS custom properties so light/dark stay defined in one place.
function palette() {
  const cs = getComputedStyle(document.querySelector(".viz-root"));
  const get = (name) => cs.getPropertyValue(name).trim();
  return {
    surface: get("--surface-1"),
    grid: get("--gridline"),
    axis: get("--axis"),
    muted: get("--text-muted"),
    text: get("--text-secondary"),
    points: get("--series-points"),
    gd: get("--series-gd"),
    ols: get("--series-ols"),
    res: get("--series-res"),
  };
}

// -------------------------------------------------------------- geometry
function plotBox(canvas) {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  return {
    left: PAD.left,
    top: PAD.top,
    right: w - PAD.right,
    bottom: h - PAD.bottom,
    width: w - PAD.left - PAD.right,
    height: h - PAD.top - PAD.bottom,
  };
}

function toPx(x, y) {
  const box = plotBox(plot);
  return {
    px: box.left + ((x - DOMAIN.xMin) / (DOMAIN.xMax - DOMAIN.xMin)) * box.width,
    py: box.bottom - ((y - DOMAIN.yMin) / (DOMAIN.yMax - DOMAIN.yMin)) * box.height,
  };
}

function toData(px, py) {
  const box = plotBox(plot);
  return {
    x: DOMAIN.xMin + ((px - box.left) / box.width) * (DOMAIN.xMax - DOMAIN.xMin),
    y: DOMAIN.yMin + ((box.bottom - py) / box.height) * (DOMAIN.yMax - DOMAIN.yMin),
  };
}

function pointerPos(evt, canvas) {
  const r = canvas.getBoundingClientRect();
  return { px: evt.clientX - r.left, py: evt.clientY - r.top };
}

// ------------------------------------------------------------------ maths

/* Closed form (ordinary least squares / normal equation):
 *   m = (n·Σxy − Σx·Σy) / (n·Σx² − (Σx)²)
 *   b = (Σy − m·Σx) / n
 * Exact in one pass — no iteration, no learning rate.
 */
function fitOls(points) {
  const n = points.length;
  if (n < 2) return null;

  let sx = 0, sy = 0, sxy = 0, sxx = 0;
  for (const p of points) {
    sx += p.x; sy += p.y; sxy += p.x * p.y; sxx += p.x * p.x;
  }

  const denom = n * sxx - sx * sx;
  if (Math.abs(denom) < 1e-12) return null;   // all x identical: vertical, no function fits

  const m = (n * sxy - sx * sy) / denom;
  return { m, b: (sy - m * sx) / n };
}

/* Mean squared error: the cost we are minimising.
 *   MSE = (1/n) · Σ (yᵢ − (m·xᵢ + b))²
 */
function mse(points, m, b) {
  if (!points.length) return 0;
  let total = 0;
  for (const p of points) {
    const err = p.y - (m * p.x + b);
    total += err * err;
  }
  return total / points.length;
}

/* Coefficient of determination: 1 − SSres/SStot.
 * 1.0 means the line explains every wiggle; 0 means it is no better than the mean.
 */
function rSquared(points, m, b) {
  const n = points.length;
  if (n < 2) return NaN;

  const mean = points.reduce((s, p) => s + p.y, 0) / n;
  let ssRes = 0, ssTot = 0;
  for (const p of points) {
    const err = p.y - (m * p.x + b);
    ssRes += err * err;
    ssTot += (p.y - mean) * (p.y - mean);
  }
  if (ssTot < 1e-12) return NaN;
  return 1 - ssRes / ssTot;
}

/* One gradient descent step. Partial derivatives of MSE:
 *   ∂/∂m = (−2/n) · Σ xᵢ(yᵢ − ŷᵢ)      ∂/∂b = (−2/n) · Σ (yᵢ − ŷᵢ)
 * Then walk downhill: m ← m − lr·∂/∂m
 * Returns the gradient magnitude so the caller can spot convergence.
 */
function gdStep(lr) {
  const pts = state.points;
  const n = pts.length;
  if (n === 0) return 0;

  let dm = 0, db = 0;
  for (const p of pts) {
    const err = p.y - (state.gd.m * p.x + state.gd.b);
    dm += -2 * p.x * err;
    db += -2 * err;
  }
  dm /= n;
  db /= n;

  state.gd.m -= lr * dm;
  state.gd.b -= lr * db;
  state.iter++;

  return Math.hypot(dm, db);
}

// -------------------------------------------------------------- training
function recordLoss() {
  const cost = mse(state.points, state.gd.m, state.gd.b);
  state.lossHistory.push(cost);
  if (state.lossHistory.length > MAX_LOSS_HISTORY) state.lossHistory.shift();
}

function runSteps(count) {
  if (state.diverged || state.points.length === 0) return;

  const lr = learningRate();
  for (let i = 0; i < count; i++) {
    const grad = gdStep(lr);

    if (!Number.isFinite(state.gd.m) || !Number.isFinite(state.gd.b) ||
        Math.abs(state.gd.m) > DIVERGE_LIMIT || Math.abs(state.gd.b) > DIVERGE_LIMIT) {
      state.diverged = true;
      state.running = false;
      break;
    }
    if (grad < CONVERGE_EPS) {
      state.converged = true;
      state.running = false;
      recordLoss();
      break;
    }
    recordLoss();
  }
}

function resetTraining(randomise = true) {
  state.gd.m = randomise ? (Math.random() * 4 - 2) : 0;
  state.gd.b = randomise ? (Math.random() * 6 + 1) : 0;
  state.iter = 0;
  state.diverged = false;
  state.converged = false;
  state.lossHistory = [];
  state.running = false;
  if (state.points.length) recordLoss();
}

// Any change to the data invalidates the run in progress.
function dataChanged() {
  state.ols = fitOls(state.points);
  state.diverged = false;
  state.converged = false;
}

// -------------------------------------------------------------- rendering
function sizeCanvas(canvas, ctx) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
}

function drawAxes(c) {
  const box = plotBox(plot);

  pctx.save();
  pctx.font = '11px system-ui, -apple-system, "Segoe UI", sans-serif';
  pctx.fillStyle = c.muted;

  for (let v = DOMAIN.xMin; v <= DOMAIN.xMax; v++) {
    const { px } = toPx(v, 0);
    pctx.strokeStyle = c.grid;
    pctx.lineWidth = 1;
    pctx.beginPath();
    pctx.moveTo(px, box.top);
    pctx.lineTo(px, box.bottom);
    pctx.stroke();

    pctx.textAlign = "center";
    pctx.textBaseline = "top";
    pctx.fillText(String(v), px, box.bottom + 8);
  }

  for (let v = DOMAIN.yMin; v <= DOMAIN.yMax; v++) {
    const { py } = toPx(0, v);
    pctx.strokeStyle = c.grid;
    pctx.lineWidth = 1;
    pctx.beginPath();
    pctx.moveTo(box.left, py);
    pctx.lineTo(box.right, py);
    pctx.stroke();

    pctx.textAlign = "right";
    pctx.textBaseline = "middle";
    pctx.fillText(String(v), box.left - 8, py);
  }

  // baseline + left axis, slightly stronger than the grid
  pctx.strokeStyle = c.axis;
  pctx.lineWidth = 1.5;
  pctx.beginPath();
  pctx.moveTo(box.left, box.top);
  pctx.lineTo(box.left, box.bottom);
  pctx.lineTo(box.right, box.bottom);
  pctx.stroke();

  pctx.fillStyle = c.text;
  pctx.textAlign = "center";
  pctx.textBaseline = "bottom";
  pctx.fillText("x", box.right - 4, box.bottom + 30);
  pctx.fillText("y", box.left - 28, box.top + 12);
  pctx.restore();
}

// Clip a line to the plot box so it never paints over the axes.
function drawLine(m, b, color, dashed) {
  if (!Number.isFinite(m) || !Number.isFinite(b)) return;

  const box = plotBox(plot);
  const a = toPx(DOMAIN.xMin, m * DOMAIN.xMin + b);
  const z = toPx(DOMAIN.xMax, m * DOMAIN.xMax + b);

  pctx.save();
  pctx.beginPath();
  pctx.rect(box.left, box.top, box.width, box.height);
  pctx.clip();

  pctx.strokeStyle = color;
  pctx.lineWidth = 2;
  pctx.lineCap = "round";
  if (dashed) pctx.setLineDash([6, 5]);

  pctx.beginPath();
  pctx.moveTo(a.px, a.py);
  pctx.lineTo(z.px, z.py);
  pctx.stroke();
  pctx.restore();
}

function drawResiduals(c) {
  const box = plotBox(plot);

  pctx.save();
  pctx.beginPath();
  pctx.rect(box.left, box.top, box.width, box.height);
  pctx.clip();

  for (const p of state.points) {
    const fitted = state.gd.m * p.x + state.gd.b;
    const a = toPx(p.x, p.y);
    const z = toPx(p.x, fitted);

    if (ui.showSquares.checked) {
      // A true square in data units: side = |error|, drawn to the right of the point.
      const err = p.y - fitted;
      const corner = toPx(p.x + Math.abs(err), Math.min(p.y, fitted));
      pctx.fillStyle = c.res;
      pctx.globalAlpha = 0.12;
      pctx.fillRect(a.px, Math.min(a.py, z.py), corner.px - a.px, Math.abs(z.py - a.py));
      pctx.globalAlpha = 1;
    }

    pctx.strokeStyle = c.res;
    pctx.globalAlpha = 0.85;
    pctx.lineWidth = 1.5;
    pctx.beginPath();
    pctx.moveTo(a.px, a.py);
    pctx.lineTo(z.px, z.py);
    pctx.stroke();
    pctx.globalAlpha = 1;
  }
  pctx.restore();
}

function drawPoints(c) {
  for (let i = 0; i < state.points.length; i++) {
    const { px, py } = toPx(state.points[i].x, state.points[i].y);
    const active = state.drag && state.drag.index === i;

    pctx.beginPath();
    pctx.arc(px, py, active ? 7 : 5, 0, Math.PI * 2);
    pctx.fillStyle = c.points;
    pctx.fill();
    // 2px surface ring keeps overlapping points readable
    pctx.strokeStyle = c.surface;
    pctx.lineWidth = 2;
    pctx.stroke();
  }
}

function drawPlot() {
  const c = palette();
  sizeCanvas(plot, pctx);

  pctx.fillStyle = c.surface;
  pctx.fillRect(0, 0, plot.clientWidth, plot.clientHeight);

  drawAxes(c);

  if (state.points.length) {
    if (ui.showResiduals.checked) drawResiduals(c);
    if (ui.showOls.checked && state.ols) drawLine(state.ols.m, state.ols.b, c.ols, true);
    drawLine(state.gd.m, state.gd.b, c.gd, false);
    drawPoints(c);
  }

  ui.emptyHint.hidden = state.points.length > 0;
}

function drawLossCurve() {
  const c = palette();
  sizeCanvas(lossCanvas, lctx);

  lctx.fillStyle = c.surface;
  lctx.fillRect(0, 0, lossCanvas.clientWidth, lossCanvas.clientHeight);

  const hist = state.lossHistory;
  ui.lossHint.hidden = hist.length > 1;
  if (hist.length < 2) return;

  const w = lossCanvas.clientWidth;
  const h = lossCanvas.clientHeight;
  const box = { left: 44, top: 12, right: w - 12, bottom: h - 22 };
  box.width = box.right - box.left;
  box.height = box.bottom - box.top;

  // Log scale: cost drops by orders of magnitude, so linear hides the endgame.
  const safe = hist.map((v) => Math.log10(Math.max(v, 1e-12)));

  // The OLS cost is the floor GD walks toward — keep it inside the range so the
  // gap between "where we are" and "the best possible" stays visible.
  const floor = state.ols
    ? Math.log10(Math.max(mse(state.points, state.ols.m, state.ols.b), 1e-12))
    : null;

  let lo = Math.min(...safe, floor ?? Infinity);
  let hi = Math.max(...safe, floor ?? -Infinity);
  if (hi - lo < 0.5) { hi += 0.25; lo -= 0.25; }
  lo -= (hi - lo) * 0.08;   // breathing room so the floor never sits on the axis

  const xAt = (i) => box.left + (i / (hist.length - 1)) * box.width;
  const yAt = (v) => box.bottom - ((v - lo) / (hi - lo)) * box.height;

  lctx.save();
  lctx.font = '10px system-ui, -apple-system, "Segoe UI", sans-serif';
  lctx.fillStyle = c.muted;
  lctx.textAlign = "right";
  lctx.textBaseline = "middle";

  for (let k = 0; k <= 3; k++) {
    const v = lo + ((hi - lo) * k) / 3;
    const y = yAt(v);
    lctx.strokeStyle = c.grid;
    lctx.lineWidth = 1;
    lctx.beginPath();
    lctx.moveTo(box.left, y);
    lctx.lineTo(box.right, y);
    lctx.stroke();
    lctx.fillText(formatSci(Math.pow(10, v)), box.left - 6, y);
  }

  lctx.strokeStyle = c.axis;
  lctx.lineWidth = 1.5;
  lctx.beginPath();
  lctx.moveTo(box.left, box.top);
  lctx.lineTo(box.left, box.bottom);
  lctx.lineTo(box.right, box.bottom);
  lctx.stroke();

  lctx.strokeStyle = c.gd;
  lctx.lineWidth = 2;
  lctx.lineJoin = "round";
  lctx.beginPath();
  for (let i = 0; i < safe.length; i++) {
    const x = xAt(i);
    const y = yAt(safe[i]);
    if (i === 0) lctx.moveTo(x, y); else lctx.lineTo(x, y);
  }
  lctx.stroke();

  if (floor !== null) {
    const y = yAt(floor);
    lctx.strokeStyle = c.ols;
    lctx.lineWidth = 1.5;
    lctx.setLineDash([5, 4]);
    lctx.beginPath();
    lctx.moveTo(box.left, y);
    lctx.lineTo(box.right, y);
    lctx.stroke();
    lctx.setLineDash([]);

    lctx.fillStyle = c.muted;
    lctx.textAlign = "right";
    lctx.textBaseline = "bottom";
    lctx.fillText("best possible", box.right - 2, y - 3);
  }

  lctx.fillStyle = c.muted;
  lctx.textAlign = "center";
  lctx.textBaseline = "top";
  lctx.fillText("0", box.left, box.bottom + 6);
  lctx.fillText(String(hist.length - 1), box.right, box.bottom + 6);
  lctx.restore();
}

// ------------------------------------------------------------------ format
function formatSci(v) {
  if (!Number.isFinite(v)) return "—";
  if (v === 0) return "0";
  if (v >= 0.01 && v < 10000) return v.toPrecision(3);
  return v.toExponential(1);
}

function fmt(v, digits = 3) {
  if (!Number.isFinite(v)) return "—";
  // A diverged run produces astronomical values; keep them inside their cell.
  if (Math.abs(v) >= 1e5) return v.toExponential(1);
  return v.toFixed(digits);
}

function updateStats() {
  const { m, b } = state.gd;
  const cost = mse(state.points, m, b);
  const r2 = rSquared(state.points, m, b);

  const sign = b < 0 ? "−" : "+";
  ui.equation.textContent =
    `y = ${fmt(m)}x ${sign} ${fmt(Math.abs(b))}`;

  ui.statM.textContent = fmt(m);
  ui.statB.textContent = fmt(b);
  ui.statMse.textContent = state.points.length ? formatSci(cost) : "—";
  ui.statR2.textContent = fmt(r2);
  ui.statN.textContent = String(state.points.length);
  ui.statIter.textContent = String(state.iter);

  ui.playBtn.textContent = state.running ? "❚❚ Pause" : "▶ Train";
  const noData = state.points.length === 0;
  ui.playBtn.disabled = noData || state.diverged;
  ui.stepBtn.disabled = noData || state.diverged;

  ui.status.className = "status";
  if (state.diverged) {
    ui.status.classList.add("diverged");
    ui.status.textContent =
      `Diverged — learning rate ${learningRate().toFixed(4)} is too large. Reset and lower it.`;
  } else if (state.converged) {
    ui.status.classList.add("converged");
    ui.status.textContent = `Converged in ${state.iter} iterations.`;
  } else if (noData) {
    ui.status.textContent = "Add at least two points to fit a line.";
  } else if (state.points.length < 2) {
    ui.status.textContent = "One more point and the closed-form fit appears.";
  } else {
    ui.status.textContent = "";
  }
}

function render() {
  drawPlot();
  drawLossCurve();
  updateStats();
}

// ------------------------------------------------------------------- loop
function frame() {
  if (state.running) {
    runSteps(stepsPerFrame());
    render();
  }
  requestAnimationFrame(frame);
}

// --------------------------------------------------------------- pointer
function nearestPoint(px, py) {
  let best = -1;
  let bestDist = HIT_RADIUS;
  for (let i = 0; i < state.points.length; i++) {
    const q = toPx(state.points[i].x, state.points[i].y);
    const d = Math.hypot(q.px - px, q.py - py);
    if (d < bestDist) { bestDist = d; best = i; }
  }
  return best;
}

function withinPlot(px, py) {
  const box = plotBox(plot);
  return px >= box.left && px <= box.right && py >= box.top && py <= box.bottom;
}

function clampToDomain(p) {
  return {
    x: Math.min(DOMAIN.xMax, Math.max(DOMAIN.xMin, p.x)),
    y: Math.min(DOMAIN.yMax, Math.max(DOMAIN.yMin, p.y)),
  };
}

plot.addEventListener("pointerdown", (e) => {
  const { px, py } = pointerPos(e, plot);
  if (!withinPlot(px, py)) return;

  const hit = nearestPoint(px, py);

  if (e.shiftKey) {
    if (hit >= 0) {
      state.points.splice(hit, 1);
      dataChanged();
      render();
    }
    return;
  }

  if (hit >= 0) {
    state.drag = { index: hit, moved: false, startX: px, startY: py };
    plot.setPointerCapture(e.pointerId);
    plot.classList.add("grabbing");
    render();
    return;
  }

  state.points.push(clampToDomain(toData(px, py)));
  dataChanged();
  render();
});

plot.addEventListener("pointermove", (e) => {
  if (!state.drag) return;

  const { px, py } = pointerPos(e, plot);
  if (Math.hypot(px - state.drag.startX, py - state.drag.startY) > DRAG_THRESHOLD) {
    state.drag.moved = true;
  }
  state.points[state.drag.index] = clampToDomain(toData(px, py));
  dataChanged();
  render();
});

plot.addEventListener("pointerup", (e) => {
  if (!state.drag) return;
  plot.releasePointerCapture(e.pointerId);
  plot.classList.remove("grabbing");
  state.drag = null;
  render();
});

// ------------------------------------------------------------- controls
ui.playBtn.addEventListener("click", () => {
  if (state.converged) { state.converged = false; }
  state.running = !state.running;
  updateStats();
});

ui.stepBtn.addEventListener("click", () => {
  state.running = false;
  runSteps(1);
  render();
});

ui.resetBtn.addEventListener("click", () => {
  resetTraining(true);
  render();
});

ui.clearBtn.addEventListener("click", () => {
  state.points = [];
  dataChanged();
  resetTraining(true);
  render();
});

ui.randomBtn.addEventListener("click", () => {
  generateData();
  render();
});

ui.lr.addEventListener("input", () => {
  ui.lrOut.textContent = learningRate().toFixed(4);
});

ui.speed.addEventListener("input", () => {
  ui.speedOut.textContent = `${stepsPerFrame()} steps / frame`;
});

ui.noise.addEventListener("input", () => {
  ui.noiseOut.textContent = parseFloat(ui.noise.value).toFixed(1);
});

for (const box of [ui.showResiduals, ui.showOls, ui.showSquares]) {
  box.addEventListener("change", render);
}

ui.themeToggle.addEventListener("click", () => {
  const root = document.documentElement;
  const dark = root.getAttribute("data-theme") === "dark" ||
    (!root.hasAttribute("data-theme") &&
      window.matchMedia("(prefers-color-scheme: dark)").matches);
  root.setAttribute("data-theme", dark ? "light" : "dark");
  render();
});

window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);

// -------------------------------------------------------------- generate
// Box-Muller: uniform randoms in, normally distributed noise out.
function gaussian() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function generateData() {
  const n = 24;
  const trueM = Math.random() * 1.6 - 0.8 + (Math.random() < 0.5 ? -0.4 : 0.6);
  const trueB = Math.random() * 3 + 2.5;
  const noise = parseFloat(ui.noise.value);

  state.points = [];
  for (let i = 0; i < n; i++) {
    const x = DOMAIN.xMin + Math.random() * (DOMAIN.xMax - DOMAIN.xMin);
    const y = trueM * x + trueB + gaussian() * noise;
    state.points.push(clampToDomain({ x, y }));
  }

  dataChanged();
  resetTraining(true);
}

// ------------------------------------------------------------------- init
const ro = new ResizeObserver(() => render());
ro.observe(plot);
ro.observe(lossCanvas);

ui.lrOut.textContent = learningRate().toFixed(4);
ui.speedOut.textContent = `${stepsPerFrame()} steps / frame`;
ui.noiseOut.textContent = parseFloat(ui.noise.value).toFixed(1);

generateData();
render();
requestAnimationFrame(frame);
