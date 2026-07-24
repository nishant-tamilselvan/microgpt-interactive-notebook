<div align="center">

# 🧠 microGPT — Interactive Teaching Notebook

### *Watch a GPT think, learn, and dream — one character at a time.*

A single-file, zero-dependency web app that turns Andrej Karpathy's
[`microgpt.py`](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
into a **living, clickable notebook**. Read the story, poke every widget, and see a real
transformer train and hallucinate brand-new names — all running natively in your browser.

<br/>

![Built with](https://img.shields.io/badge/Built%20with-Vanilla%20JS-f7df1e?logo=javascript&logoColor=black)
![No dependencies](https://img.shields.io/badge/Dependencies-0-2ea44f)
![Pure Python](https://img.shields.io/badge/Reference-Pure%20Python-3776ab?logo=python&logoColor=white)
![Theme](https://img.shields.io/badge/Theme-Dark%20%2F%20Light-6ea8fe)
![License](https://img.shields.io/badge/License-MIT-blue)

</div>

---

## ✨ What is this?

Large language models feel like magic. This notebook removes the magic and leaves the **mechanism**.

microGPT teaches a tiny neural network a single skill — **predict the next character** in a name.
After training on ~32,000 real names it can *hallucinate* new, plausible ones like `kamon`,
`areli`, or `keylen`. Every piece of a real GPT is here — autograd, multi-head attention, an MLP,
RMSNorm, residuals, cross-entropy, and the Adam optimizer — written from scratch with **no NumPy,
no PyTorch, no libraries at all**.

> *"This file is the complete algorithm. Everything else is just efficiency."*

The widgets don't fake it: they run a **faithful JavaScript port** of the exact same math, and a
built-in **parity self-test** proves the browser numbers match the Python reference to within `1e-4`.

---

## 🎬 Features

| | |
|---|---|
| 📖 **Story-driven** | 12 guided cells walk from a single character to a full working GPT. |
| 🧩 **Live widgets** | Tokenize a name, step the autograd graph, watch attention light up, train live, and sample new names token-by-token with probability bars. |
| 🌗 **Dark / light theme** | One-click switcher with system-preference detection and persistence — every canvas, chart, and heatmap adapts. |
| 🔬 **Real computation** | Pre-trained weights load instantly; generation and attention are live from the first second. |
| ✅ **Provably faithful** | A parity test compares the JS engine against Python's `parity_test.py` logits. |
| 🎨 **Zero build step** | It's one HTML file. No bundler, no npm, no server framework. |

---

## 🚀 Quick start

Because the app fetches `weights.json`, browsers block that over `file://` — so serve the folder
with any tiny static server:

```bash
# clone
git clone https://github.com/nishant-tamilselvan/microgpt-interactive-notebook.git
cd microgpt-interactive-notebook

# serve (pick one)
python -m http.server 8000
#   → open http://localhost:8000/index.html

npx serve .          # if you prefer Node
```

Then open the printed URL and start scrolling. That's it. ✅

> 💡 Prefer no server? Use the **"Load weights…"** button in the last cell to pick `weights.json`
> manually from disk.

---

## 🐍 Run the real Python model

The reference model is pure Python and trains in about a minute on a laptop:

```bash
python microgpt.py        # downloads names, trains ~1000 steps, saves model.pkl
python export_weights.py  # turns model.pkl → weights.json for the browser app
python parity_test.py     # prints the reference logits the web app checks against
```

Watch the loss fall from ~3.3 (random guessing among 27 tokens) toward ~2.37 — the sound of a
network learning the shape of names.

---

## 🧭 The notebook, cell by cell

```
00  Introduction          the big picture + a preview of the payoff
01  Tokenizer             characters ↔ integer token ids (+ BOS)
02  The Value engine      scalar autograd & the chain rule, live graph
03  Parameters            the model's 4,192-number "knowledge"
04  gpt() forward pass    embed → attention → MLP → logits + heatmap
05  Adam                  the optimizer that nudges every weight
06  Training loop         one name per step, watch it learn live
07  Inference             autoregressive sampling with temperature
08  The whole program     end-to-end flow + JS==Python parity proof
09  From microGPT to ChatGPT   what changes at scale
10  FAQ                   understanding, hallucinations, and more
11  Takeaways             the one-sentence summary
```

---

## 🗂️ Project structure

```
microgpt-interactive-notebook/
├── index.html          ⭐ the interactive notebook (the whole app)
├── weights.json        pre-trained weights loaded by the app at runtime
├── microgpt.py         the reference model — train & save
├── export_weights.py   model.pkl → weights.json
├── parity_test.py      numerical reference for the JS port
├── EXPLANATION.md      the full narrative, in depth
└── README.md           you are here
```

---

## 🧠 How it works, in one breath

microGPT forwards a name through a scalar-autograd GPT to compute a next-character prediction loss,
back-propagates to get gradients, updates the weights with Adam, and samples brand-new names one
character at a time. **The same math, vectorized on a GPU, *is* a frontier language model.**
Everything else is just efficiency.

---

## 🙏 Credits

- **Original algorithm** — Andrej Karpathy's
  [`microgpt.py`](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95),
  the culmination of micrograd, makemore, and nanoGPT.
- **Interactive notebook, browser port & UI** — [Nishant Tamilselvan](https://github.com/nishant-tamilselvan).

---

## 📄 License

Released under the [MIT License](LICENSE). The reference algorithm remains the work of Andrej Karpathy.

<div align="center">

<br/>

*Built with curiosity by [Nishant Tamilselvan](https://github.com/nishant-tamilselvan) — because the best way to understand a language model is to watch one think.* ✨

</div>
