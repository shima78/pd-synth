"""A lightweight DDPM generator: ``diffusers`` ``UNet2DModel`` + ``DDPMScheduler``.

The UNet is always trained from scratch on whatever dataset is configured
(MNIST for pipeline validation now, real data later) - this module never
loads a pretrained checkpoint.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers import DDPMScheduler, UNet2DModel
from torch import Tensor
from torch.utils.data import DataLoader


@dataclass
class DiffusionConfig:
    """Architecture and diffusion-process hyperparameters for :class:`DiffusionGenerator`.

    All fields should be populated from a run's YAML config rather than
    hardcoded at the call site.
    """

    image_size: int = 28
    in_channels: int = 1
    out_channels: int = 1
    layers_per_block: int = 2
    block_out_channels: tuple[int, ...] = (32, 64)
    num_train_timesteps: int = 1000
    # Group count for UNet2DModel's GroupNorm layers. Must evenly divide every
    # entry of block_out_channels - the diffusers default (32) is too large
    # for the small channel counts used in tests, so it is exposed here
    # rather than hardcoded.
    norm_num_groups: int = 32
    # Number of classes to condition generation on. Any dataset that exposes
    # LabeledImageDataset.num_classes can drive this - leave None for an
    # unconditional model that ignores labels entirely.
    num_classes: int | None = None


class DiffusionGenerator:
    """DDPM image generator built from a small ``UNet2DModel`` noise predictor.

    Wraps the standard DDPM training loop (predict the noise added at a
    random timestep, minimize MSE against the true noise) and the standard
    DDPM reverse-process sampling loop, using diffusers' building blocks
    directly rather than a pretrained end-to-end pipeline.
    """

    def __init__(self, config: DiffusionConfig, device: str = "cpu") -> None:
        """Build the UNet and scheduler.

        Args:
            config: Architecture/diffusion hyperparameters.
            device: torch device string, e.g. ``"cpu"`` or ``"cuda"``.
        """
        self.config = config
        self.device = torch.device(device)
        self.class_conditional = config.num_classes is not None
        num_blocks = len(config.block_out_channels)
        self.model = UNet2DModel(
            sample_size=config.image_size,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            layers_per_block=config.layers_per_block,
            block_out_channels=config.block_out_channels,
            down_block_types=("DownBlock2D",) * num_blocks,
            up_block_types=("UpBlock2D",) * num_blocks,
            norm_num_groups=config.norm_num_groups,
            num_class_embeds=config.num_classes,
        ).to(self.device)
        self.scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps)

    def training_step(self, images: Tensor, labels: Tensor | None = None) -> Tensor:
        """Compute the DDPM noise-prediction MSE loss for one batch of images.

        Args:
            images: A ``(N, C, H, W)`` batch of images scaled to ``[-1, 1]``.
            labels: ``(N,)`` integer class labels. Required when the
                generator is class-conditional (``config.num_classes`` set),
                ignored otherwise.

        Returns:
            Scalar loss tensor.
        """
        images = images.to(self.device)
        noise = torch.randn_like(images)
        batch_size = images.shape[0]
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (batch_size,), device=self.device
        ).long()
        noisy_images = self.scheduler.add_noise(images, noise, timesteps)
        class_labels = labels.to(self.device) if self.class_conditional else None
        noise_pred = self.model(noisy_images, timesteps, class_labels=class_labels).sample
        return torch.nn.functional.mse_loss(noise_pred, noise)

    def fit(
        self,
        dataloader: DataLoader,
        num_epochs: int,
        lr: float,
        on_epoch_end: Callable[[int, float], None] | None = None,
        on_step_end: Callable[[int, float], None] | None = None,
    ) -> list[float]:
        """Train the noise-prediction UNet.

        Args:
            dataloader: Yields ``(images, labels)`` batches. Labels are used
                only when the generator is class-conditional.
            num_epochs: Number of passes over ``dataloader``.
            lr: Learning rate for the AdamW optimizer.
            on_epoch_end: Optional callback invoked after each epoch with
                ``(epoch_index, mean_epoch_loss)``, e.g. for live progress
                logging on long CPU runs.
            on_step_end: Optional callback invoked after each optimizer step
                with ``(global_step, step_loss)``, e.g. for fine-grained
                logging (TensorBoard, etc.) beyond per-epoch averages.

        Returns:
            Mean training loss for each epoch, in order.
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        epoch_losses: list[float] = []
        global_step = 0
        self.model.train()
        for epoch in range(num_epochs):
            running_loss, num_batches = 0.0, 0
            for images, labels in dataloader:
                optimizer.zero_grad()
                loss = self.training_step(images, labels)
                loss.backward()
                optimizer.step()
                loss_value = loss.item()
                running_loss += loss_value
                num_batches += 1
                if on_step_end is not None:
                    on_step_end(global_step, loss_value)
                global_step += 1
            mean_loss = running_loss / max(num_batches, 1)
            epoch_losses.append(mean_loss)
            if on_epoch_end is not None:
                on_epoch_end(epoch, mean_loss)
        return epoch_losses

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        num_inference_steps: int | None = None,
        batch_size: int | None = None,
        class_labels: Tensor | int | None = None,
    ) -> Tensor:
        """Generate images by running the reverse diffusion process from pure noise.

        Args:
            num_samples: Number of images to generate.
            num_inference_steps: Number of denoising steps. Defaults to the
                scheduler's full ``num_train_timesteps`` when omitted; pass a
                smaller value (e.g. for tests) to trade quality for speed.
            batch_size: Maximum number of images to run through the UNet at
                once. ``num_samples`` images are generated in sequential
                chunks of this size rather than a single large batch, which
                matters on GPUs where a batch scaled to ``num_samples`` may
                not fit in memory. Defaults to generating all of
                ``num_samples`` in one batch.
            class_labels: Only meaningful for a class-conditional generator
                (``config.num_classes`` set). Either a single int (every
                generated image gets that class) or a ``(num_samples,)``
                integer tensor (one label per image). Defaults to a random
                label per image, drawn uniformly from the configured
                classes, when the generator is conditional and this is
                omitted.

        Returns:
            A ``(num_samples, C, H, W)`` tensor of generated images.
        """
        self.model.eval()
        num_steps = num_inference_steps or self.scheduler.config.num_train_timesteps
        chunk_size = batch_size or num_samples

        labels_tensor: Tensor | None = None
        if self.class_conditional:
            if class_labels is None:
                labels_tensor = torch.randint(0, self.config.num_classes, (num_samples,))
            elif isinstance(class_labels, int):
                labels_tensor = torch.full((num_samples,), class_labels, dtype=torch.long)
            else:
                labels_tensor = class_labels
                if labels_tensor.shape[0] != num_samples:
                    raise ValueError("class_labels must have length num_samples")

        chunks: list[Tensor] = []
        offset = 0
        while offset < num_samples:
            n = min(chunk_size, num_samples - offset)
            chunk_labels = labels_tensor[offset : offset + n] if labels_tensor is not None else None
            chunks.append(self._sample_batch(n, num_steps, chunk_labels))
            offset += n
        return torch.cat(chunks, dim=0)

    def _sample_batch(
        self, num_samples: int, num_inference_steps: int, class_labels: Tensor | None
    ) -> Tensor:
        """Run the reverse diffusion process for a single batch of ``num_samples`` images."""
        self.scheduler.set_timesteps(num_inference_steps)
        shape = (
            num_samples,
            self.config.in_channels,
            self.config.image_size,
            self.config.image_size,
        )
        images = torch.randn(shape, device=self.device)
        labels = class_labels.to(self.device) if class_labels is not None else None
        for t in self.scheduler.timesteps:
            noise_pred = self.model(images, t, class_labels=labels).sample
            images = self.scheduler.step(noise_pred, t, images).prev_sample
        return images

    def save_pretrained(self, path: str | Path) -> None:
        """Save the UNet weights/config to ``path`` (via diffusers' save format)."""
        self.model.save_pretrained(str(path))

    def load_pretrained(self, path: str | Path) -> None:
        """Load UNet weights/config from ``path``, replacing the current model.

        Re-derives ``class_conditional`` from the loaded checkpoint's own
        config rather than trusting whatever ``DiffusionConfig`` this
        generator was constructed with, since the checkpoint may have been
        trained with different conditioning than the caller assumed.
        """
        self.model = UNet2DModel.from_pretrained(str(path)).to(self.device)
        self.config.num_classes = self.model.config.num_class_embeds
        self.class_conditional = self.config.num_classes is not None
