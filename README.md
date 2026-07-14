# Neural Playground

### ▶ [**Live Demo — train a network in your browser**](https://vanshtaneja23.github.io/neuralplayground/)

A neural network you can watch learn — MLP with forward pass, backprop, and minibatch SGD written from scratch in pure JavaScript, zero ML libraries, training live in your browser.

> Train a neural network live in the browser — forward pass, backpropagation, and minibatch SGD written entirely from scratch in JavaScript. **Zero ML libraries.**

An interactive visualization of how a multilayer perceptron (MLP) learns to classify non-linearly-separable data. Pick a dataset, design the network architecture, hit **Train**, and watch the decision boundary form in real time as gradients flow.

Inspired by [TensorFlow Playground](https://playground.tensorflow.org/), but reimplemented from the ground up — every piece of the math (activations, loss, gradients, weight updates) is hand-written, with no TensorFlow, no NumPy-style helpers, and no autograd.

---

## Screenshot

![Neural Playground — spiral dataset solved, 99.1% accuracy](neural-playground.png)

*The spiral after ~28,800 steps (3 hidden layers, tanh, lr 0.1) — two interleaved arms fully untangled at 99.1% accuracy. The boundary heatmap forms live while it trains.*

---

## Features

- **Live training visualization** — the decision boundary heatmap re-renders every other animation frame while the network trains, so you can literally watch it learn.
- **Four classic benchmark datasets** — Spiral, Concentric Circles, XOR, and Gaussian Clusters. All except Clusters are non-linearly separable, which is exactly why a hidden layer (or three) is needed.
- **Configurable architecture** — add or remove hidden layers (1–4) and resize each layer (1–9 neurons) on the fly. The network reinitializes with He-style scaled random weights.
- **Activation functions** — switch between `tanh` and `ReLU` mid-training and compare convergence behavior.
- **Adjustable learning rate** — 0.03 / 0.1 / 0.3 / 1.0. Try `lr = 1.0` with ReLU on the spiral and watch training destabilize — a hands-on demo of why learning rate tuning matters.
- **Live network diagram** — an SVG rendering of the architecture where each connection's thickness encodes `|w|` and color encodes sign (teal = positive, orange = negative), updating as weights change.
- **Training telemetry** — step counter, current minibatch loss, full-dataset accuracy, and a rolling loss sparkline (last 220 batches).

---

## How it works (the math)

Everything lives in one file. The core is a plain-JavaScript MLP with the structure `[2, h₁, …, hₙ, 1]` — two inputs (the point's x, y coordinates), configurable hidden layers, and a single sigmoid output representing P(class B).

### Forward pass

For each layer *l*:

```
z⁽ˡ⁾ = W⁽ˡ⁾ a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
a⁽ˡ⁾ = σ(z⁽ˡ⁾)        // tanh or ReLU for hidden layers
a⁽ᴸ⁾ = sigmoid(z⁽ᴸ⁾)   // output layer
```

Weights are initialized with scaled random values, `w ~ U(-√(2/nᵢₙ), √(2/nᵢₙ))`, to keep early activations well-conditioned.

### Loss

Binary cross-entropy:

```
L = −[ y·log(ŷ) + (1−y)·log(1−ŷ) ]
```

### Backpropagation

The output delta uses the well-known simplification of BCE composed with sigmoid:

```
δ⁽ᴸ⁾ = ŷ − y
```

Hidden deltas propagate backward through the transpose of each weight matrix, multiplied elementwise by the activation derivative (`1 − tanh²(z)` or the ReLU step function):

```
δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾)ᵀ δ⁽ˡ⁺¹⁾ ⊙ σ′(z⁽ˡ⁾)
```

Gradients accumulate per-sample over a minibatch:

```
∂L/∂W⁽ˡ⁾ += δ⁽ˡ⁾ (a⁽ˡ⁻¹⁾)ᵀ
∂L/∂b⁽ˡ⁾ += δ⁽ˡ⁾
```

### Optimization

Plain minibatch stochastic gradient descent — batch size 24, sampled uniformly at random from the dataset, with the update `W ← W − (lr / batch) · ∇W`. Six SGD steps run per animation frame, so at 60 fps the network sees roughly **8,600 minibatches per second of wall time**.

### Rendering

- The decision boundary is computed on a 64×64 grid of forward passes, written into an `ImageData` buffer, and upscaled with canvas image smoothing — fast enough to redraw ~30×/second during training.
- Class probabilities blend between the two class colors and wash toward the paper background, so confident regions read saturated and uncertain regions read pale.
- Data points, the loss sparkline, and the weight diagram are drawn with the 2D canvas API and inline SVG. React state is only touched a few times per second; all hot-path mutation happens through refs to avoid re-render thrash.

---

## Datasets

| Dataset | Description | Why it's interesting |
|---|---|---|
| **Spiral** | Two interleaved Archimedean spiral arms | The classic hard case — needs depth/width and many steps |
| **Circles** | Inner disk vs. outer ring | Radially separable; tanh nets solve it quickly |
| **XOR** | Four quadrants, diagonal pairing | The textbook proof that a single linear layer is insufficient |
| **Clusters** | Two Gaussian blobs | Linearly separable sanity check — converges almost instantly |

Each dataset is generated procedurally with uniform noise, ~220 points total, in the domain [−1, 1]².

---

## Getting started

The network itself is a single self-contained React component (`neural-playground.jsx`) with no dependencies beyond React — all styling is inline, so there's no CSS or Tailwind step. Vite is used only to serve and bundle it.

```bash
git clone https://github.com/vanshtaneja23/neuralplayground.git
cd neuralplayground
npm install
npm run dev
```

No environment variables, no backend, no API keys. `npm run build` emits a fully static `dist/` — that's what GitHub Pages serves.

---

## Things to try

1. **Spiral + 3 hidden layers of ~9, tanh, lr 0.1** — reaches ~99% and cleanly separates both arms in under 30k steps (about 40 seconds of training). The default 2×6 at lr 0.3 is deliberately under-provisioned: it stalls around 85% and visibly oscillates, which is the point — it shows you what too-small-and-too-fast looks like before you fix it.
2. **Drop to 1 hidden layer of 2 neurons** on the spiral — watch the network underfit no matter how long it trains. Capacity matters.
3. **ReLU + lr 1.0** — observe loss spikes and dead-zone artifacts in the boundary. Then drop to 0.1 and watch it recover.
4. **XOR with 1 layer of 3** — the minimal-ish architecture that solves it, and a nice demonstration of the universal approximation idea at small scale.

---

## Project structure

```
neuralplayground/
├── README.md
├── index.html                   Vite entry
├── vite.config.js               base: /neuralplayground/ for Pages
├── src/main.jsx                 mounts the component
├── .github/workflows/deploy.yml build + deploy to GitHub Pages
└── neural-playground.jsx        the entire app
    ├── Dataset generators        (spiral / circles / xor / gaussians)
    ├── Network core
    │   ├── makeNet()             weight & bias initialization
    │   ├── forward()             cached activations + pre-activations
    │   ├── trainStep()           backprop + minibatch SGD update
    │   └── accuracy()
    ├── Rendering
    │   ├── drawBoundary()        64×64 probability grid → canvas heatmap
    │   ├── drawLoss()            rolling loss sparkline
    │   └── renderNetSVG()        live weight-encoded architecture diagram
    └── UI                        controls, telemetry, layout
```

---

## Roadmap

- [ ] Momentum / Adam optimizers
- [ ] L2 regularization toggle
- [ ] Noise slider for dataset generation
- [ ] Per-neuron activation heatmaps (hover a node to see its learned feature)
- [ ] Softmax output for 3+ classes

---

## Credits

Built by **Vansh Taneja** — forward pass, backpropagation, and minibatch SGD implemented by hand in JavaScript. Concept inspired by TensorFlow Playground.

## License

MIT
