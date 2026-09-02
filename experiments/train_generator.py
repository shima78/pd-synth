"""Stage 1 entry point: train the DDPM generator on the configured dataset.

Usage:
    python experiments/train_generator.py --config configs/mnist_baseline.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

from pd_synth.data import get_dataset
from pd_synth.generation import DiffusionConfig, DiffusionGenerator
from pd_synth.utils import load_config, save_config, set_seed


def main(config_path: str) -> None:
    """Train a generator per ``config_path`` and save the model, samples, and config."""
    config = load_config(config_path)
    set_seed(config["seed"])

    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "generator_config.yaml")

    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    writer.add_text("config", f"```yaml\n{yaml.safe_dump(config, sort_keys=False)}```")

    data_cfg = config["data"]
    dataset = get_dataset(
        data_cfg["name"],
        root=data_cfg["root"],
        train=True,
        download=data_cfg.get("download", True),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 0),
    )

    gen_cfg = config["generator"]
    class_conditional = gen_cfg.get("class_conditional", False)
    num_classes = dataset.num_classes if class_conditional else None
    generator = DiffusionGenerator(
        DiffusionConfig(
            image_size=gen_cfg["image_size"],
            in_channels=gen_cfg["in_channels"],
            out_channels=gen_cfg["out_channels"],
            layers_per_block=gen_cfg["layers_per_block"],
            block_out_channels=tuple(gen_cfg["block_out_channels"]),
            num_train_timesteps=gen_cfg["num_train_timesteps"],
            norm_num_groups=gen_cfg["norm_num_groups"],
            num_classes=num_classes,
        ),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    num_epochs = gen_cfg["num_epochs"]
    preview_every = gen_cfg.get("preview_every", 5)
    preview_labels = torch.arange(16) % num_classes if class_conditional else None
    start_time = time.monotonic()

    def log_step(global_step: int, loss: float) -> None:
        writer.add_scalar("train/loss_step", loss, global_step)

    def log_epoch(epoch: int, loss: float) -> None:
        elapsed = time.monotonic() - start_time
        print(
            f"epoch {epoch + 1}/{num_epochs} - loss: {loss:.6f} - elapsed: {elapsed:.1f}s",
            flush=True,
        )
        writer.add_scalar("train/loss_epoch", loss, epoch)

        is_last_epoch = epoch + 1 == num_epochs
        if (epoch + 1) % preview_every == 0 or is_last_epoch:
            preview = generator.sample(
                num_samples=16,
                num_inference_steps=gen_cfg.get("num_inference_steps"),
                class_labels=preview_labels,
            )
            viewable_preview = (preview.detach().cpu().clamp(-1, 1) + 1) / 2
            writer.add_image("samples", make_grid(viewable_preview, nrow=4), epoch + 1)
            generator.model.train()  # sample() leaves the model in eval mode

        writer.flush()

    losses = generator.fit(
        dataloader,
        num_epochs=num_epochs,
        lr=gen_cfg["lr"],
        on_epoch_end=log_epoch,
        on_step_end=log_step,
    )
    print(f"Per-epoch training loss: {losses}")

    generator.save_pretrained(output_dir / "generator")

    samples = generator.sample(
        num_samples=16,
        num_inference_steps=gen_cfg.get("num_inference_steps"),
        class_labels=preview_labels,
    )
    torch.save(samples, output_dir / "sample_grid.pt")

    # Rescale from the model's [-1, 1] training range to [0, 1] for viewing.
    viewable_samples = (samples.detach().cpu().clamp(-1, 1) + 1) / 2
    save_image(viewable_samples, output_dir / "sample_grid.png", nrow=4)
    writer.close()

    print(f"Saved generator, config, and samples to {output_dir}")
    print(f"View training curves with: tensorboard --logdir {output_dir / 'tensorboard'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    args = parser.parse_args()
    main(args.config)
