"""Boundary-focused sampling strategy - the core thesis contribution."""

from pd_synth.sampling.boundary import (
    BoundaryFocusedSampler,
    boundary_score,
    select_boundary_samples,
)

__all__ = ["BoundaryFocusedSampler", "boundary_score", "select_boundary_samples"]
