"""Chart data + builders for the demo app.

Each function takes plain Python data (lists/dicts as loaded from the run's
YAML files) and returns either an Altair chart or a DataFrame for a native
``st.*_chart``. No Streamlit calls here, so everything is easy to render from
a notebook or test in isolation.
"""

from __future__ import annotations

from typing import Any

import altair as alt
import pandas as pd

_CATEGORY_SCALE = alt.Scale(scheme="tableau10")


def accuracy_chart(rows: list[dict[str, Any]]) -> alt.LayerChart:
    """Horizontal bar chart comparing classifier accuracy across training sets.

    Bars run from zero on a full 0-100% axis - no zoomed baseline, so it
    renders on any Vega-Lite version - and each is labelled with its exact
    percentage just inside the bar end.
    """
    data = pd.DataFrame(rows)

    base = alt.Chart(data).encode(
        y=alt.Y("training_set:N", sort=None, title=None),
        x=alt.X(
            "accuracy:Q",
            title="Accuracy on the real MNIST test set",
            scale=alt.Scale(domain=[0.0, 1.0]),
            axis=alt.Axis(format="%"),
        ),
        tooltip=[
            alt.Tooltip("training_set:N", title="Training set"),
            alt.Tooltip("accuracy:Q", format=".2%", title="Accuracy"),
        ],
    )
    bars = base.mark_bar().encode(
        color=alt.Color("data:N", title="Training data", scale=_CATEGORY_SCALE),
    )
    labels = base.mark_text(align="right", dx=-6, color="white", fontWeight="bold").encode(
        text=alt.Text("accuracy:Q", format=".2%")
    )
    return (bars + labels).properties(height=180)


def per_class_metrics_chart(report: dict[str, Any]) -> alt.LayerChart:
    """Per-class precision / recall / F1 as a labelled digit x metric heatmap.

    Same ``mark_rect`` + text-layer construction as
    :func:`confusion_matrix_chart` (a shape that renders reliably in
    Streamlit). The colour scale is clamped to ``[0.95, 1.0]`` so the small
    but real differences between digits are actually visible; the exact score
    is printed in every cell regardless.
    """
    metric_keys = (
        ("precision", "classifier_precision"),
        ("recall", "classifier_recall"),
        ("f1", "classifier_f1"),
    )
    num_classes = len(report["classifier_precision"])
    data = pd.DataFrame(
        [
            {"digit": str(digit), "metric": metric, "score": float(report[key][digit])}
            for metric, key in metric_keys
            for digit in range(num_classes)
        ]
    )

    heat = alt.Chart(data).mark_rect().encode(
        x=alt.X("digit:N", title="Digit"),
        y=alt.Y("metric:N", title=None, sort=[name for name, _ in metric_keys]),
        color=alt.Color(
            "score:Q",
            title="Score",
            scale=alt.Scale(scheme="blues", domain=[0.95, 1.0], clamp=True),
            legend=alt.Legend(format=".2f"),
        ),
        tooltip=[
            alt.Tooltip("digit:N", title="Digit"),
            alt.Tooltip("metric:N", title="Metric"),
            alt.Tooltip("score:Q", format=".3f", title="Score"),
        ],
    )
    labels = heat.mark_text(baseline="middle", fontSize=12).encode(
        text=alt.Text("score:Q", format=".3f"),
        color=alt.condition(alt.datum.score > 0.975, alt.value("white"), alt.value("black")),
    )
    return (heat + labels).properties(height=140)


def confusion_matrix_chart(confusion_matrix: list[list[int]]) -> alt.LayerChart:
    """Labelled confusion-matrix heatmap (rows = actual, columns = predicted)."""
    data = pd.DataFrame(
        [
            {"actual": str(actual), "predicted": str(predicted), "count": int(count)}
            for actual, row in enumerate(confusion_matrix)
            for predicted, count in enumerate(row)
        ]
    )
    max_count = data["count"].max()

    heat = alt.Chart(data).mark_rect().encode(
        x=alt.X("predicted:N", title="Predicted"),
        y=alt.Y("actual:N", title="Actual"),
        color=alt.Color("count:Q", title="Count", scale=alt.Scale(scheme="blues"), legend=None),
        tooltip=[
            alt.Tooltip("actual:N", title="Actual"),
            alt.Tooltip("predicted:N", title="Predicted"),
            alt.Tooltip("count:Q", title="Count"),
        ],
    )
    labels = heat.mark_text(baseline="middle", fontSize=11).encode(
        text=alt.Text("count:Q"),
        color=alt.condition(
            alt.datum.count > max_count / 2, alt.value("white"), alt.value("black")
        ),
    )
    return (heat + labels).properties(height=360)
