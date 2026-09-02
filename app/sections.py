"""The three page sections of the demo, each a ``render_*`` function.

Kept as functions (rather than one long script) so the entry point reads as
a table of contents and each section can be reordered or dropped without
untangling shared local state.
"""

from __future__ import annotations

import charts
import streamlit as st
import torch
from data_loading import RunArtifacts, accuracy_rows, load_metrics

from pd_synth.utils import to_grid_image

GRID_NROW = 4


def render_headline(run: RunArtifacts) -> None:
    """Headline accuracy numbers and the training-set comparison chart."""
    st.header("Headline metrics")
    output_dir = run.output_dir

    full_real = load_metrics(str(output_dir / "classifier_metrics.yaml"))
    real_subset = load_metrics(str(output_dir / "classifier_on_real_subset_metrics.yaml"))
    synthetic = load_metrics(str(output_dir / "classifier_on_synthetic_metrics.yaml"))
    synthetic_full = load_metrics(str(output_dir / "classifier_on_synthetic_full_metrics.yaml"))

    full_real_acc = full_real["val_accuracy"] if full_real is not None else None

    with st.container(horizontal=True):
        _accuracy_metric("Full real data (60k)", full_real_acc)

        if real_subset is not None:
            count = real_subset["num_real_train_samples"]
            _accuracy_metric(f"{count:,} real images", real_subset["val_accuracy"])
        else:
            st.metric("Matched real subset", "not run yet", border=True)

        if synthetic is not None:
            count = synthetic["num_synthetic_train_samples"]
            _accuracy_metric(f"{count:,} synthetic images", synthetic["val_accuracy"])
        else:
            st.metric("Synthetic data", "not run yet", border=True)

        st.metric(
            "Generator epochs trained", f"{run.generator_config['num_epochs']}", border=True
        )

    rows = accuracy_rows(output_dir)
    if rows:
        with st.container(border=True):
            st.markdown("**Classifier accuracy by training set**")
            st.altair_chart(charts.accuracy_chart(rows), width="stretch")

    if synthetic_full is not None and full_real_acc is not None:
        _render_full_budget_gap(synthetic_full, full_real_acc)


def _accuracy_metric(label: str, accuracy: float | None) -> None:
    st.metric(label, "n/a" if accuracy is None else f"{accuracy:.2%}", border=True)


def _render_full_budget_gap(synthetic_full: dict, full_real_acc: float) -> None:
    count = synthetic_full["num_synthetic_train_samples"]
    synth_acc = synthetic_full["val_accuracy"]
    gap_pts = (full_real_acc - synth_acc) * 100

    st.subheader("Synthetic at a full data budget")
    st.caption(
        f"The same classifier trained on **{count:,}** generated images - the same count as "
        "the full real training set - so real-vs-synthetic is no longer confounded with "
        f"more-data-vs-less. It reaches **{synth_acc:.2%}** against **{full_real_acc:.2%}** "
        "for real data, both on the real MNIST test set."
    )
    st.metric("Real - synthetic gap", f"{gap_pts:.2f} pts", border=True)


def render_samples(run: RunArtifacts) -> None:
    """Side-by-side real vs. freshly generated digits."""
    st.header("Real vs. generated digits")
    generator = run.generator
    gen_cfg = run.generator_config

    if not generator.class_conditional:
        st.warning(
            "This generator checkpoint isn't class-conditional, so samples below are "
            "random digits rather than the one you pick.",
            icon=":material/warning:",
        )

    controls = st.container(horizontal=True, vertical_alignment="bottom", gap="large")
    digit = controls.segmented_control("Digit", options=list(range(10)), default=0)
    num_samples = controls.slider("Samples", min_value=4, max_value=16, value=8, step=4)
    if digit is None:
        digit = 0

    if st.button("Generate new samples", type="primary", icon=":material/casino:"):
        with st.spinner("Running the diffusion model..."):
            class_labels = (
                torch.full((num_samples,), digit) if generator.class_conditional else None
            )
            st.session_state["generated"] = generator.sample(
                num_samples=num_samples,
                num_inference_steps=gen_cfg.get("num_inference_steps"),
                class_labels=class_labels,
            )
            st.session_state["generated_digit"] = digit

    left, right = st.columns(2)

    with left, st.container(border=True):
        st.markdown(f"**Real** - digit **{digit}**")
        indices = run.indices_by_class.get(digit, [])[:num_samples]
        if indices:
            real_images = torch.stack([run.real_dataset[i][0] for i in indices])
            st.image(to_grid_image(real_images, nrow=GRID_NROW), width="stretch")
        else:
            st.caption("No real images for this digit.")

    with right, st.container(border=True):
        st.markdown(f"**Generated** - digit **{digit}**")
        generated = st.session_state.get("generated")
        if generated is not None and st.session_state.get("generated_digit") == digit:
            st.image(to_grid_image(generated, nrow=GRID_NROW), width="stretch")
        else:
            st.caption(
                "Click **Generate new samples** to sample fresh digits from the diffusion model."
            )


def render_per_class(run: RunArtifacts) -> None:
    """Per-class precision / recall / F1 and the confusion matrix, if evaluated."""
    report = load_metrics(str(run.output_dir / "evaluation_report.yaml"))
    if report is None:
        return

    st.header("Per-class classifier metrics")

    with st.container(border=True):
        st.markdown("**Precision / recall / F1 per digit**")
        st.altair_chart(charts.per_class_metrics_chart(report), width="stretch")

    confusion_matrix = report.get("confusion_matrix")
    if confusion_matrix is not None:
        with st.container(border=True):
            st.markdown("**Confusion matrix** (real MNIST test set)")
            st.altair_chart(charts.confusion_matrix_chart(confusion_matrix), width="stretch")
