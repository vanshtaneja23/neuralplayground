import React, { useState, useRef, useEffect, useCallback } from "react";

/* ============================================================
   NEURAL PLAYGROUND
   A multilayer perceptron implemented from scratch — no ML
   libraries. Forward pass, backpropagation, and minibatch SGD
   all hand-written, trained live in the browser while the
   decision boundary renders in real time.
   ============================================================ */

// ---------- palette ----------
const PAPER = "#F7F8F4";
const INK = "#1C2420";
const GRID = "#DAE2DC";
const ORANGE = [232, 90, 30]; // class 0
const TEAL = [14, 122, 110]; // class 1
const ORANGE_HEX = "#E85A1E";
const TEAL_HEX = "#0E7A6E";

// ---------- datasets ----------
function rnd(a = 1) {
  return (Math.random() * 2 - 1) * a;
}
function makeData(kind, n = 110, noise = 0.06) {
  const pts = [];
  if (kind === "spiral") {
    for (let c = 0; c < 2; c++) {
      for (let i = 0; i < n; i++) {
        const r = (i / n) * 0.85 + 0.05;
        const t = 1.75 * (i / n) * 2 * Math.PI + c * Math.PI;
        pts.push({
          x: r * Math.sin(t) + rnd(noise),
          y: r * Math.cos(t) + rnd(noise),
          label: c,
        });
      }
    }
  } else if (kind === "circles") {
    for (let i = 0; i < n; i++) {
      const t = Math.random() * 2 * Math.PI;
      const r = Math.random() * 0.35;
      pts.push({ x: r * Math.cos(t) + rnd(noise), y: r * Math.sin(t) + rnd(noise), label: 0 });
    }
    for (let i = 0; i < n; i++) {
      const t = Math.random() * 2 * Math.PI;
      const r = 0.6 + Math.random() * 0.3;
      pts.push({ x: r * Math.cos(t) + rnd(noise), y: r * Math.sin(t) + rnd(noise), label: 1 });
    }
  } else if (kind === "xor") {
    for (let i = 0; i < n * 2; i++) {
      let x = rnd(0.9);
      let y = rnd(0.9);
      x += x > 0 ? 0.12 : -0.12;
      y += y > 0 ? 0.12 : -0.12;
      pts.push({ x: x + rnd(noise), y: y + rnd(noise), label: x * y > 0 ? 0 : 1 });
    }
  } else {
    // gaussians
    for (let i = 0; i < n; i++) {
      pts.push({ x: -0.45 + rnd(0.28), y: -0.45 + rnd(0.28), label: 0 });
      pts.push({ x: 0.45 + rnd(0.28), y: 0.45 + rnd(0.28), label: 1 });
    }
  }
  return pts;
}

// ---------- the network (from scratch) ----------
function makeNet(layers) {
  // layers like [2, 6, 6, 1]
  const W = [];
  const B = [];
  for (let l = 0; l < layers.length - 1; l++) {
    const scale = Math.sqrt(2 / layers[l]);
    const w = [];
    for (let j = 0; j < layers[l + 1]; j++) {
      const row = [];
      for (let k = 0; k < layers[l]; k++) row.push(rnd(scale));
      w.push(row);
    }
    W.push(w);
    B.push(new Array(layers[l + 1]).fill(0));
  }
  return { layers, W, B };
}

const sigmoid = (z) => 1 / (1 + Math.exp(-z));

function forward(net, x, act) {
  const A = [x];
  const Z = [];
  const L = net.W.length;
  for (let l = 0; l < L; l++) {
    const inA = A[l];
    const z = new Array(net.layers[l + 1]);
    const a = new Array(net.layers[l + 1]);
    for (let j = 0; j < z.length; j++) {
      let s = net.B[l][j];
      const wj = net.W[l][j];
      for (let k = 0; k < inA.length; k++) s += wj[k] * inA[k];
      z[j] = s;
      if (l === L - 1) a[j] = sigmoid(s);
      else a[j] = act === "relu" ? (s > 0 ? s : 0) : Math.tanh(s);
    }
    Z.push(z);
    A.push(a);
  }
  return { A, Z };
}

function predict(net, x, y, act) {
  const { A } = forward(net, [x, y], act);
  return A[A.length - 1][0];
}

// one minibatch SGD step; returns mean loss on the batch
function trainStep(net, data, lr, act, batchSize = 24) {
  const L = net.W.length;
  // zeroed gradient accumulators
  const gW = net.W.map((w) => w.map((row) => row.map(() => 0)));
  const gB = net.B.map((b) => b.map(() => 0));
  let loss = 0;

  for (let s = 0; s < batchSize; s++) {
    const p = data[(Math.random() * data.length) | 0];
    const { A, Z } = forward(net, [p.x, p.y], act);
    const out = A[L][0];
    const y = p.label;
    const eps = 1e-7;
    loss += -(y * Math.log(out + eps) + (1 - y) * Math.log(1 - out + eps));

    // output delta (BCE + sigmoid simplifies)
    let delta = [out - y];
    for (let l = L - 1; l >= 0; l--) {
      const aPrev = A[l];
      for (let j = 0; j < delta.length; j++) {
        gB[l][j] += delta[j];
        const wj = net.W[l][j];
        const gj = gW[l][j];
        for (let k = 0; k < aPrev.length; k++) gj[k] += delta[j] * aPrev[k];
      }
      if (l > 0) {
        const newDelta = new Array(net.layers[l]).fill(0);
        for (let k = 0; k < newDelta.length; k++) {
          let s2 = 0;
          for (let j = 0; j < delta.length; j++) s2 += net.W[l][j][k] * delta[j];
          const z = Z[l - 1][k];
          const dAct = act === "relu" ? (z > 0 ? 1 : 0) : 1 - Math.tanh(z) * Math.tanh(z);
          newDelta[k] = s2 * dAct;
        }
        delta = newDelta;
      }
    }
  }

  const scale = lr / batchSize;
  for (let l = 0; l < L; l++) {
    for (let j = 0; j < net.W[l].length; j++) {
      net.B[l][j] -= scale * gB[l][j];
      for (let k = 0; k < net.W[l][j].length; k++) net.W[l][j][k] -= scale * gW[l][j][k];
    }
  }
  return loss / batchSize;
}

function accuracy(net, data, act) {
  let c = 0;
  for (const p of data) if ((predict(net, p.x, p.y, act) > 0.5 ? 1 : 0) === p.label) c++;
  return c / data.length;
}

// ---------- component ----------
const DATASETS = [
  { id: "spiral", label: "Spiral" },
  { id: "circles", label: "Circles" },
  { id: "xor", label: "XOR" },
  { id: "gauss", label: "Clusters" },
];
const LRS = [0.03, 0.1, 0.3, 1.0];

export default function NeuralPlayground() {
  const [dataset, setDataset] = useState("spiral");
  const [hidden, setHidden] = useState([6, 6]);
  const [act, setAct] = useState("tanh");
  const [lr, setLr] = useState(0.3);
  const [running, setRunning] = useState(false);
  const [stats, setStats] = useState({ steps: 0, loss: 1, acc: 0.5 });
  const [, bump] = useState(0); // forces SVG re-render of weights

  const netRef = useRef(null);
  const dataRef = useRef(makeData("spiral"));
  const lossHist = useRef([]);
  const stepsRef = useRef(0);
  const canvasRef = useRef(null);
  const lossCanvasRef = useRef(null);
  const rafRef = useRef(0);
  const cfgRef = useRef({ act, lr });
  cfgRef.current = { act, lr };

  const reset = useCallback(
    (ds = dataset, hid = hidden) => {
      netRef.current = makeNet([2, ...hid, 1]);
      dataRef.current = makeData(ds);
      lossHist.current = [];
      stepsRef.current = 0;
      setStats({ steps: 0, loss: 1, acc: accuracy(netRef.current, dataRef.current, cfgRef.current.act) });
      bump((v) => v + 1);
      drawAll();
    },
    [dataset, hidden] // eslint-disable-line
  );

  // ----- drawing -----
  const drawBoundary = useCallback(() => {
    const cv = canvasRef.current;
    if (!cv || !netRef.current) return;
    const ctx = cv.getContext("2d");
    const res = 64;
    const img = ctx.createImageData(res, res);
    const a = cfgRef.current.act;
    for (let iy = 0; iy < res; iy++) {
      for (let ix = 0; ix < res; ix++) {
        const x = (ix / (res - 1)) * 2 - 1;
        const y = 1 - (iy / (res - 1)) * 2;
        const p = predict(netRef.current, x, y, a);
        const i = (iy * res + ix) * 4;
        // blend class colors toward paper for soft wash
        const t = p; // 0 → orange, 1 → teal
        const mix = (c0, c1) => c0 + (c1 - c0) * t;
        const wash = 0.72; // pull toward paper
        img.data[i] = mix(ORANGE[0], TEAL[0]) * (1 - wash) + 247 * wash;
        img.data[i + 1] = mix(ORANGE[1], TEAL[1]) * (1 - wash) + 248 * wash;
        img.data[i + 2] = mix(ORANGE[2], TEAL[2]) * (1 - wash) + 244 * wash;
        img.data[i + 3] = 255;
      }
    }
    // draw heatmap scaled up
    const off = document.createElement("canvas");
    off.width = res;
    off.height = res;
    off.getContext("2d").putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.drawImage(off, 0, 0, cv.width, cv.height);

    // grid
    ctx.strokeStyle = "rgba(28,36,32,0.08)";
    ctx.lineWidth = 1;
    for (let g = 1; g < 8; g++) {
      const q = (g / 8) * cv.width;
      ctx.beginPath(); ctx.moveTo(q, 0); ctx.lineTo(q, cv.height); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, q); ctx.lineTo(cv.width, q); ctx.stroke();
    }

    // points
    for (const p of dataRef.current) {
      const px = ((p.x + 1) / 2) * cv.width;
      const py = ((1 - p.y) / 2) * cv.height;
      ctx.beginPath();
      ctx.arc(px, py, 4, 0, Math.PI * 2);
      ctx.fillStyle = p.label === 0 ? ORANGE_HEX : TEAL_HEX;
      ctx.fill();
      ctx.strokeStyle = PAPER;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }, []);

  const drawLoss = useCallback(() => {
    const cv = lossCanvasRef.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, cv.width, cv.height);
    const h = lossHist.current;
    if (h.length < 2) return;
    const max = Math.max(...h, 0.05);
    ctx.beginPath();
    h.forEach((v, i) => {
      const x = (i / (h.length - 1)) * cv.width;
      const y = cv.height - (v / max) * (cv.height - 6) - 3;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = INK;
    ctx.lineWidth = 2;
    ctx.stroke();
  }, []);

  const drawAll = useCallback(() => {
    drawBoundary();
    drawLoss();
  }, [drawBoundary, drawLoss]);

  // ----- training loop -----
  useEffect(() => {
    if (!running) return;
    let frame = 0;
    const loop = () => {
      const { act: a, lr: l } = cfgRef.current;
      let loss = 0;
      for (let i = 0; i < 6; i++) loss = trainStep(netRef.current, dataRef.current, l, a);
      stepsRef.current += 6;
      lossHist.current.push(loss);
      if (lossHist.current.length > 220) lossHist.current.shift();
      frame++;
      if (frame % 2 === 0) drawBoundary();
      drawLoss();
      if (frame % 8 === 0) {
        setStats({
          steps: stepsRef.current,
          loss,
          acc: accuracy(netRef.current, dataRef.current, a),
        });
        bump((v) => v + 1);
      }
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafRef.current);
  }, [running, drawBoundary, drawLoss]);

  // init
  useEffect(() => {
    reset();
  }, []); // eslint-disable-line

  // ----- handlers -----
  const pickDataset = (id) => {
    setDataset(id);
    setRunning(false);
    setTimeout(() => reset(id, hidden), 0);
  };
  const changeNeurons = (i, d) => {
    const next = hidden.map((n, j) => (j === i ? Math.min(9, Math.max(1, n + d)) : n));
    setHidden(next);
    setRunning(false);
    setTimeout(() => reset(dataset, next), 0);
  };
  const changeLayers = (d) => {
    let next;
    if (d > 0 && hidden.length < 4) next = [...hidden, 4];
    else if (d < 0 && hidden.length > 1) next = hidden.slice(0, -1);
    else return;
    setHidden(next);
    setRunning(false);
    setTimeout(() => reset(dataset, next), 0);
  };

  // ----- network diagram -----
  const renderNetSVG = () => {
    const net = netRef.current;
    if (!net) return null;
    const layers = net.layers;
    const W = 260, H = 190;
    const xs = layers.map((_, i) => 22 + (i / (layers.length - 1)) * (W - 44));
    const ys = layers.map((n) => Array.from({ length: n }, (_, j) => (H / (n + 1)) * (j + 1)));
    const lines = [];
    for (let l = 0; l < net.W.length; l++)
      for (let j = 0; j < net.W[l].length; j++)
        for (let k = 0; k < net.W[l][j].length; k++) {
          const w = net.W[l][j][k];
          lines.push(
            <line
              key={`${l}-${j}-${k}`}
              x1={xs[l]} y1={ys[l][k]} x2={xs[l + 1]} y2={ys[l + 1][j]}
              stroke={w >= 0 ? TEAL_HEX : ORANGE_HEX}
              strokeWidth={Math.min(3, Math.abs(w) * 1.2 + 0.2)}
              strokeOpacity={Math.min(0.85, Math.abs(w) * 0.45 + 0.12)}
            />
          );
        }
    return (
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", display: "block" }} aria-label="Network architecture">
        {lines}
        {xs.map((x, l) =>
          ys[l].map((y, j) => (
            <circle key={`n${l}-${j}`} cx={x} cy={y} r={6} fill={PAPER} stroke={INK} strokeWidth={1.5} />
          ))
        )}
      </svg>
    );
  };

  // ----- styles -----
  const mono = { fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" };
  const label = { ...mono, fontSize: 10, letterSpacing: "0.14em", textTransform: "uppercase", color: "#5A6660" };
  const btn = (active) => ({
    ...mono,
    fontSize: 12,
    padding: "7px 12px",
    border: `1.5px solid ${active ? INK : GRID}`,
    background: active ? INK : "transparent",
    color: active ? PAPER : INK,
    cursor: "pointer",
    borderRadius: 3,
  });
  const card = {
    border: `1.5px solid ${GRID}`,
    borderRadius: 6,
    padding: 14,
    background: "#FFFFFF",
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        background: `${PAPER} linear-gradient(${GRID}55 1px, transparent 1px) 0 0/28px 28px, ${PAPER} linear-gradient(90deg, ${GRID}55 1px, transparent 1px) 0 0/28px 28px`,
        color: INK,
        padding: "24px 16px 40px",
        fontFamily: "'Avenir Next','Helvetica Neue',system-ui,sans-serif",
      }}
    >
      <div style={{ maxWidth: 1060, margin: "0 auto" }}>
        {/* header */}
        <header style={{ display: "flex", alignItems: "baseline", gap: 14, flexWrap: "wrap", marginBottom: 6 }}>
          <h1 style={{ margin: 0, fontSize: 30, fontWeight: 800, letterSpacing: "-0.02em" }}>
            Neural Playground
          </h1>
          <span style={{ ...mono, fontSize: 11, color: "#5A6660" }}>
            backprop written from scratch · zero ML libraries
          </span>
        </header>
        <p style={{ ...mono, fontSize: 12, color: "#5A6660", margin: "0 0 18px" }}>
          a live multilayer perceptron, trained in your browser — watch the decision boundary learn
        </p>

        <div style={{ display: "flex", gap: 16, flexWrap: "wrap", alignItems: "flex-start" }}>
          {/* controls */}
          <div style={{ ...card, flex: "1 1 220px", minWidth: 220, display: "flex", flexDirection: "column", gap: 16 }}>
            <div>
              <div style={{ ...label, marginBottom: 8 }}>Dataset</div>
              <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                {DATASETS.map((d) => (
                  <button key={d.id} style={btn(dataset === d.id)} onClick={() => pickDataset(d.id)}>
                    {d.label}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <div style={{ ...label, marginBottom: 8 }}>
                Hidden layers
                <button style={{ ...btn(false), padding: "2px 8px", marginLeft: 8 }} onClick={() => changeLayers(-1)}>−</button>
                <button style={{ ...btn(false), padding: "2px 8px", marginLeft: 4 }} onClick={() => changeLayers(1)}>+</button>
              </div>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                {hidden.map((n, i) => (
                  <div key={i} style={{ ...mono, fontSize: 12, border: `1.5px solid ${GRID}`, borderRadius: 3, padding: "5px 8px", display: "flex", gap: 8, alignItems: "center" }}>
                    <button style={{ all: "unset", cursor: "pointer", fontWeight: 700 }} onClick={() => changeNeurons(i, -1)}>−</button>
                    <span>{n}</span>
                    <button style={{ all: "unset", cursor: "pointer", fontWeight: 700 }} onClick={() => changeNeurons(i, 1)}>+</button>
                  </div>
                ))}
                <span style={{ ...mono, fontSize: 11, color: "#5A6660", alignSelf: "center" }}>neurons / layer</span>
              </div>
            </div>

            <div>
              <div style={{ ...label, marginBottom: 8 }}>Activation</div>
              <div style={{ display: "flex", gap: 6 }}>
                {["tanh", "relu"].map((a) => (
                  <button key={a} style={btn(act === a)} onClick={() => setAct(a)}>{a}</button>
                ))}
              </div>
            </div>

            <div>
              <div style={{ ...label, marginBottom: 8 }}>Learning rate</div>
              <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                {LRS.map((v) => (
                  <button key={v} style={btn(lr === v)} onClick={() => setLr(v)}>{v}</button>
                ))}
              </div>
            </div>

            <div style={{ display: "flex", gap: 8 }}>
              <button
                style={{ ...btn(true), flex: 1, fontSize: 13, padding: "10px 0", background: running ? ORANGE_HEX : INK, borderColor: running ? ORANGE_HEX : INK }}
                onClick={() => setRunning((r) => !r)}
              >
                {running ? "Pause" : "Train"}
              </button>
              <button style={{ ...btn(false), flex: 1, fontSize: 13 }} onClick={() => { setRunning(false); setTimeout(() => reset(), 0); }}>
                Reset
              </button>
            </div>
          </div>

          {/* boundary canvas */}
          <div style={{ ...card, flex: "2 1 360px", minWidth: 300 }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
              <span style={label}>Decision boundary</span>
              <span style={{ ...mono, fontSize: 11 }}>
                <span style={{ color: ORANGE_HEX }}>● class A</span>{"  "}
                <span style={{ color: TEAL_HEX }}>● class B</span>
              </span>
            </div>
            <canvas
              ref={canvasRef}
              width={480}
              height={480}
              style={{ width: "100%", display: "block", borderRadius: 4, border: `1.5px solid ${GRID}` }}
            />
          </div>

          {/* right column */}
          <div style={{ flex: "1 1 240px", minWidth: 240, display: "flex", flexDirection: "column", gap: 16 }}>
            <div style={card}>
              <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
                {[
                  ["steps", stats.steps],
                  ["loss", stats.loss.toFixed(3)],
                  ["accuracy", (stats.acc * 100).toFixed(1) + "%"],
                ].map(([k, v]) => (
                  <div key={k}>
                    <div style={label}>{k}</div>
                    <div style={{ ...mono, fontSize: 20, fontWeight: 700 }}>{v}</div>
                  </div>
                ))}
              </div>
              <div style={{ ...label, margin: "14px 0 6px" }}>Training loss</div>
              <canvas ref={lossCanvasRef} width={260} height={70} style={{ width: "100%", display: "block" }} />
            </div>

            <div style={card}>
              <div style={{ ...label, marginBottom: 8 }}>
                Architecture · 2 → {hidden.join(" → ")} → 1
              </div>
              {renderNetSVG()}
              <div style={{ ...mono, fontSize: 10, color: "#5A6660", marginTop: 6 }}>
                line weight = |w| · <span style={{ color: TEAL_HEX }}>positive</span> / <span style={{ color: ORANGE_HEX }}>negative</span>
              </div>
            </div>
          </div>
        </div>

        <footer style={{ ...mono, fontSize: 11, color: "#5A6660", marginTop: 22 }}>
          Built by Vansh Taneja · forward pass, backpropagation &amp; minibatch SGD implemented by hand in JavaScript
        </footer>
      </div>
    </div>
  );
}
