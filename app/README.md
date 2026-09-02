# app/

A minimal Streamlit demo for browsing a trained pd-synth run.

Run it with:

```bash
streamlit run app/streamlit_app.py
```

It loads the generator and classifier(s) saved by `experiments/train_generator.py`,
`experiments/train_classifier.py`, and `experiments/train_classifier_on_synthetic.py`
for a given config (defaults to `configs/mnist_baseline.yaml`; override with the
`PD_SYNTH_CONFIG` environment variable), and shows:

- headline accuracy numbers (real-data classifier vs. synthetic-data classifier)
- a real-vs-generated digit comparison, with a button to sample fresh digits from
  the diffusion model live (class-conditional generators let you pick which digit)
- per-class precision/recall/F1, if an `evaluate.py` run has produced a report

The app is dataset-agnostic - it only depends on the `LabeledImageDataset`
interface and the saved config/checkpoint files, so it works unmodified once a
real dataset config replaces `mnist_baseline.yaml`.

## Deploying to Streamlit Community Cloud

The repo is deploy-ready:

- `requirements.txt` (repo root) pins the runtime deps and installs `pd_synth`
  from the `src/` layout; CPU-only PyTorch wheels keep the image small.
- `.python-version` pins Python 3.10.
- `.streamlit/config.toml` holds app-level preferences.
- The trained `outputs/mnist_baseline/` run (generator + classifier `.pt` files
  + metric YAMLs, ~4 MB) is committed, so the app has something to load. MNIST
  itself is downloaded on first run (`data.download: true`).

Steps: push to GitHub, then on [share.streamlit.io](https://share.streamlit.io)
create an app from this repo with main file `app/streamlit_app.py` (Python 3.10).
It redeploys automatically on every push.

## Layout

| File | Responsibility |
|------|----------------|
| `streamlit_app.py` | Entry point: page config, then calls each section in order. |
| `data_loading.py` | Loading + caching the run (`load_run`) and its metrics YAMLs (`load_metrics`, `accuracy_rows`). The only module that knows the `outputs/<run>/` file layout. |
| `charts.py` | Altair chart builders. Pure functions of plain data - no Streamlit calls, so they're testable in isolation. |
| `sections.py` | The three `render_*` page sections (headline metrics, real-vs-generated digits, per-class metrics). |
