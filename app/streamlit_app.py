"""Streamlit demo for pd-synth: browse a trained run's results.

Loads the generator and classifier(s) saved by the ``experiments/`` scripts
for a given config, and lets a viewer compare real vs. generated digits and
see the headline accuracy numbers. Works with any config whose dataset
implements ``LabeledImageDataset`` - nothing here is MNIST-specific.

The page is split into modules:
    data_loading.py  - loading + caching the run and its metrics
    charts.py        - Altair chart builders
    sections.py      - the three render_* page sections

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os

import streamlit as st
from data_loading import load_run
from sections import render_headline, render_per_class, render_samples

CONFIG_PATH = os.environ.get("PD_SYNTH_CONFIG", "configs/mnist_baseline.yaml")

st.set_page_config(page_title="pd-synth demo", page_icon=":material/blur_on:")
st.title("pd-synth: pipeline demo")
st.caption(
    "Stage 1 pipeline validation on MNIST. MNIST isn't scarce, so this checks the "
    "generation/sampling/classification mechanics work **end-to-end** - it is not a test "
    "of the boundary-sampling hypothesis itself. See README.md."
)

run = load_run(CONFIG_PATH)

render_headline(run)
render_samples(run)
render_per_class(run)
