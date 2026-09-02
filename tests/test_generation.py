"""Smoke test for pd_synth.generation.diffusion."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from pd_synth.generation.diffusion import DiffusionConfig, DiffusionGenerator


def test_diffusion_generator_trains_and_samples(tiny_dataset) -> None:
    config = DiffusionConfig(
        image_size=8,
        in_channels=1,
        out_channels=1,
        layers_per_block=1,
        block_out_channels=(4, 8),
        num_train_timesteps=50,
        norm_num_groups=4,
    )
    generator = DiffusionGenerator(config, device="cpu")
    loader = DataLoader(tiny_dataset, batch_size=4)

    losses = generator.fit(loader, num_epochs=1, lr=1e-3)
    assert len(losses) == 1
    assert losses[0] >= 0.0

    samples = generator.sample(num_samples=2, num_inference_steps=2)
    assert samples.shape == (2, 1, 8, 8)
    assert torch.isfinite(samples).all()


def test_diffusion_generator_fit_calls_on_epoch_end(tiny_dataset) -> None:
    config = DiffusionConfig(
        image_size=8,
        in_channels=1,
        out_channels=1,
        layers_per_block=1,
        block_out_channels=(4, 8),
        num_train_timesteps=50,
        norm_num_groups=4,
    )
    generator = DiffusionGenerator(config, device="cpu")
    loader = DataLoader(tiny_dataset, batch_size=4)
    calls: list[tuple[int, float]] = []

    def record(epoch: int, loss: float) -> None:
        calls.append((epoch, loss))

    generator.fit(loader, num_epochs=2, lr=1e-3, on_epoch_end=record)

    assert [epoch for epoch, _loss in calls] == [0, 1]


def test_diffusion_generator_sample_chunks_by_batch_size() -> None:
    config = DiffusionConfig(
        image_size=8,
        in_channels=1,
        out_channels=1,
        layers_per_block=1,
        block_out_channels=(4, 8),
        num_train_timesteps=50,
        norm_num_groups=4,
    )
    generator = DiffusionGenerator(config, device="cpu")

    samples = generator.sample(num_samples=5, num_inference_steps=2, batch_size=2)

    assert samples.shape == (5, 1, 8, 8)
    assert torch.isfinite(samples).all()


def test_diffusion_generator_save_and_load_pretrained(tiny_dataset, tmp_path) -> None:
    config = DiffusionConfig(
        image_size=8,
        in_channels=1,
        out_channels=1,
        layers_per_block=1,
        block_out_channels=(4, 8),
        num_train_timesteps=50,
        norm_num_groups=4,
    )
    generator = DiffusionGenerator(config, device="cpu")
    save_dir = tmp_path / "generator"

    generator.save_pretrained(save_dir)
    generator.load_pretrained(save_dir)

    samples = generator.sample(num_samples=1, num_inference_steps=2)
    assert samples.shape == (1, 1, 8, 8)


def _tiny_conditional_config() -> DiffusionConfig:
    return DiffusionConfig(
        image_size=8,
        in_channels=1,
        out_channels=1,
        layers_per_block=1,
        block_out_channels=(4, 8),
        num_train_timesteps=50,
        norm_num_groups=4,
        num_classes=10,
    )


def test_diffusion_generator_class_conditional_trains_and_samples(tiny_dataset) -> None:
    generator = DiffusionGenerator(_tiny_conditional_config(), device="cpu")
    loader = DataLoader(tiny_dataset, batch_size=4)

    losses = generator.fit(loader, num_epochs=1, lr=1e-3)
    assert losses[0] >= 0.0

    labels = torch.tensor([0, 1, 2])
    samples = generator.sample(num_samples=3, num_inference_steps=2, class_labels=labels)
    assert samples.shape == (3, 1, 8, 8)
    assert torch.isfinite(samples).all()

    # Omitting class_labels on a conditional model falls back to random labels.
    samples_random = generator.sample(num_samples=2, num_inference_steps=2)
    assert samples_random.shape == (2, 1, 8, 8)


def test_diffusion_generator_sample_rejects_mismatched_class_labels_length() -> None:
    generator = DiffusionGenerator(_tiny_conditional_config(), device="cpu")

    with pytest.raises(ValueError):
        generator.sample(num_samples=4, num_inference_steps=2, class_labels=torch.tensor([0, 1]))


def test_diffusion_generator_load_pretrained_updates_class_conditional(tmp_path) -> None:
    conditional_generator = DiffusionGenerator(_tiny_conditional_config(), device="cpu")
    save_dir = tmp_path / "generator"
    conditional_generator.save_pretrained(save_dir)

    unconditional_config = DiffusionConfig(
        image_size=8,
        in_channels=1,
        out_channels=1,
        layers_per_block=1,
        block_out_channels=(4, 8),
        num_train_timesteps=50,
        norm_num_groups=4,
    )
    generator = DiffusionGenerator(unconditional_config, device="cpu")
    assert not generator.class_conditional

    generator.load_pretrained(save_dir)

    assert generator.class_conditional
    assert generator.config.num_classes == 10
    samples = generator.sample(num_samples=2, num_inference_steps=2)
    assert samples.shape == (2, 1, 8, 8)
