# pd-synth — the whole pipeline, step by step

End to end, with the code that implements each piece.

---

## 0. What the project is

`pd-synth` is a thesis codebase. The real goal (later stages) is **Parkinson's
detection from scarce gait/pose data**, using a **boundary-focused sampling**
idea: don't accept generator output uniformly — keep the synthetic samples a
classifier finds *most ambiguous* (closest to a decision boundary), on the
hypothesis they carry more training value.

Right now it's **Stage 1: MNIST pipeline validation** — MNIST stands in for the
real data purely to prove the plumbing works. Every result here (98.34%,
96.38%, …) is an integration test, not a scientific claim.

The "loop" is a set of entry-point scripts in `experiments/`, all driven by one
YAML config, all writing to `outputs/mnist_baseline/`:

```
train_generator.py                 →  generator/                 (a DDPM)
train_classifier.py                →  classifier.pt              (real-data baseline)
train_classifier_on_synthetic.py   →  classifier_on_synthetic*.pt
train_classifier_on_real_subset.py →  classifier_on_real_subset.pt   (equal-budget control)
evaluate.py                        →  evaluation_report.yaml     (boundary sampling + quality metrics)
```

---

## 1. The config is the single source of truth

`configs/mnist_baseline.yaml` — nothing is hardcoded. Key blocks:

```yaml
seed: 42
data:      {name: mnist, root: data/mnist, batch_size: 128}
generator: {image_size: 28, block_out_channels: [32,64], num_train_timesteps: 1000,
            num_epochs: 20, lr: 0.0001, num_inference_steps: 50,
            class_conditional: true, preview_every: 5}
classifier:{image_size: 28, num_classes: 10, num_epochs: 5, lr: 0.001}
sampling:  {oversample_factor: 4.0, num_synthetic_samples: 1000}
output:    {dir: outputs/mnist_baseline}
```

Every script starts identically (`train_generator.py:24-30`):

```python
config = load_config(config_path)          # utils/config.py  – yaml.safe_load
set_seed(config["seed"])                    # utils/seed.py    – python+numpy+torch RNGs
output_dir = Path(config["output"]["dir"])
save_config(config, output_dir / "..._config.yaml")   # snapshot next to results
```

So any output file can be traced back to the exact config that produced it.

---

## 2. The data interface

Everything downstream is written against one abstraction, `LabeledImageDataset`
(`data/base.py`): `__getitem__` returns `(image, label)` with `image` a
`(C,H,W)` float tensor, plus a `num_classes` attribute.

- `MNISTDataset` (`data/mnist.py`): wraps `torchvision.datasets.MNIST`, applies
  `ToTensor` then `Normalize((0.5,),(0.5,))` → pixels in **[-1, 1]** (the range
  the diffusion model wants). `num_classes = 10`.
- `InMemoryDataset` (`data/in_memory.py`): wraps an in-RAM `(images, labels)`
  tensor pair behind the same interface — this is how generated samples get fed
  to the classifier without special-casing.
- `get_dataset(name, **kw)` (`data/__init__.py`): a registry lookup. Swapping in
  the real PD data later = add one class + one registry line.

---

## 3. Stage 1 — `train_generator.py` (the DDPM)

### 3.1 What gets built

`DiffusionGenerator` (`generation/diffusion.py:55-77`) is a **from-scratch**
diffusers `UNet2DModel` + `DDPMScheduler`:

```python
self.model = UNet2DModel(
    sample_size=28, in_channels=1, out_channels=1,
    block_out_channels=(32, 64),            # 2 resolution levels
    down_block_types=("DownBlock2D",)*2, up_block_types=("UpBlock2D",)*2,
    num_class_embeds=num_classes,           # 10 → class-conditional; None → not
)
self.scheduler = DDPMScheduler(num_train_timesteps=1000)
```

`train_generator.py:46-48` decides conditioning:
`num_classes = dataset.num_classes if config["generator"]["class_conditional"] else None`.
With `num_class_embeds=10`, the UNet learns a per-class embedding that's added
to the timestep embedding, so generation can be steered by label.

### 3.2 The training step — one batch (`diffusion.py:79-100`)

Standard DDPM "simple" objective (Ho et al. 2020):

```python
noise      = torch.randn_like(images)                       # ε ~ N(0, I)
timesteps  = torch.randint(0, 1000, (batch_size,))          # t ~ Uniform, per image
noisy      = self.scheduler.add_noise(images, noise, timesteps)   # x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε
noise_pred = self.model(noisy, timesteps, class_labels=labels).sample
return F.mse_loss(noise_pred, noise)                        # predict the noise
```

`add_noise` uses the closed-form forward process (no iterative loop for
training). The network's job: given a noised image and the noise level `t`,
predict the noise that was added.

### 3.3 The fit loop (`diffusion.py:102-148`)

```python
optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)   # lr = 1e-4
for epoch in range(num_epochs):                                  # 20
    for images, labels in dataloader:                            # 469 batches of 128
        optimizer.zero_grad()
        loss = self.training_step(images, labels)
        loss.backward(); optimizer.step()
        on_step_end(global_step, loss.item())                    # → TB scalar train/loss_step
    on_epoch_end(epoch, mean_epoch_loss)                         # → print + TB + preview
```

The two callbacks are supplied by `train_generator.py:66-90`:

- `log_step` → `writer.add_scalar("train/loss_step", …)`
- `log_epoch` → prints `epoch N/20 - loss … - elapsed …s`, logs
  `train/loss_epoch`, and **every `preview_every`=5 epochs** (plus the last)
  samples a 16-image grid and writes it to TensorBoard's *Images* tab. That's
  the slider fix — with `preview_every` it has steps at epochs 5/10/15/20
  instead of one.

### 3.4 Sampling — the reverse loop (`diffusion.py:150-222`)

```python
self.scheduler.set_timesteps(num_inference_steps)   # 50-step subsequence of the 1000
images = torch.randn(shape)                          # x_T = pure Gaussian noise
for t in self.scheduler.timesteps:                   # 50 iterations
    noise_pred = self.model(images, t, class_labels=labels).sample
    images = self.scheduler.step(noise_pred, t, images).prev_sample   # x_t → x_{t-1}
```

`scheduler.step` applies the DDPM posterior: subtract the predicted noise,
rescale, add a bit of fresh noise (except the last step). 50 steps instead of
1000 is the standard quality/speed trade. `sample()` chunks by `batch_size` so
memory stays bounded regardless of how many you ask for (why 60,000 samples
didn't OOM). For a conditional model, `class_labels` can be `None` (random per
image), an int (all one class), or a `(N,)` tensor (one per image).

### 3.5 Outputs

`generator.save_pretrained(output_dir / "generator")` → a **directory**
(`diffusion_pytorch_model.safetensors` + `config.json`), *not* a `.pt` file.
Plus `sample_grid.png` and `generator_config.yaml`. `load_pretrained`
(`diffusion.py:228-238`) re-derives `class_conditional` from the checkpoint's
own config, so downstream code doesn't have to be told.

**This session:** the generator directory got deleted during a task switch; it
was reran (seed 42 → reproducible: epoch-1 loss 0.0954 → epoch-20 loss 0.0230,
~33 min on CPU).

---

## 4. Stage 2 — `train_classifier.py` (real-data baseline)

### 4.1 The model (`classifiers/simple_cnn.py`)

```
Conv2d(1→16, 3x3, pad 1) → ReLU → MaxPool2d(2)
Conv2d(16→32, 3x3, pad 1) → ReLU → MaxPool2d(2)
flatten → Linear(32 · (28/4)² = 1568 → 10)
```

Sized entirely from config; `image_size` must be divisible by 4 (two 2×
downsamples).

### 4.2 The train loop (`classifiers/train.py:316-361`)

```python
optimizer = torch.optim.Adam(model.parameters(), lr=lr)   # lr = 1e-3
criterion = nn.CrossEntropyLoss()
for _ in range(num_epochs):                                # 5
    for images, labels in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward(); optimizer.step()
val_accuracy = evaluate_classifier(model, val_loader)      # argmax == label, fraction correct
```

`train_classifier.py` runs this on the **full 60,000-image** MNIST train split,
validates on the **10,000-image** test split → `classifier.pt` +
`classifier_metrics.yaml` → **`val_accuracy: 0.9834`**. Every other classifier
in the project is evaluated on that same test set, so all the numbers are
directly comparable.

---

## 5. Stage 3a — `train_classifier_on_synthetic.py`

The core "can synthetic replace real?" experiment.

```python
generator.load_pretrained(output_dir / "generator")
if not generator.class_conditional:                        # need true labels, not pseudo-labels
    raise RuntimeError(...)

labels = torch.arange(num_synthetic_samples) % num_classes # class-balanced: 0,1,…,9,0,1,…
images = generator.sample(num_synthetic_samples,           # reverse diffusion, 50 steps
                          batch_size=128, class_labels=labels)
synthetic_dataset = InMemoryDataset(images.cpu(), labels, num_classes=10)

model  = SimpleCNN(...)
result = train_classifier(model, synthetic_loader,
                          real_test_loader,                 # ← evaluate on REAL test set
                          num_epochs=5, lr=1e-3)
```

Because the generator is class-conditional, the label used to *generate* an
image is a genuine label for it — no separate labeling step. It trains the
identical `SimpleCNN` on those generated images and tests on real MNIST, then
prints `real 0.9834 vs. synthetic <x>`.

**Change this session:** added a `--label` flag.
`suffix = f"_{label}" if label else ""` gets appended to all three output
filenames (`classifier_on_synthetic_full.pt`, `_full_config.yaml`,
`_full_metrics.yaml`), so a run at a different sample count sits *alongside* the
default instead of overwriting it.

**Two runs now exist:**

- default config (`num_synthetic_samples: 1000`) → `classifier_on_synthetic.pt`
  → **81.10%**
- `configs/mnist_synth_full.yaml` (`num_synthetic_samples: 60000`, same
  `output.dir` so it reuses the generator) run with `--label full` →
  `classifier_on_synthetic_full.pt` → **96.38%** (~12 min)

---

## 6. Stage 3b — `train_classifier_on_real_subset.py` (the control)

Comparing "1k synthetic vs 60k real" confounds *real-vs-synthetic* with
*less-data-vs-more*. This script removes that confound:

```python
n = config["sampling"]["num_synthetic_samples"]            # same count as the synthetic run
indices = balanced_subset_indices(index_by_class(real_train), n, num_classes, seed)  # subset.py
subset  = Subset(real_train_dataset, indices)              # n class-balanced REAL images
result  = train_classifier(SimpleCNN(...), subset_loader, real_test_loader, 5, 1e-3)
```

`balanced_subset_indices` (`data/subset.py:670-700`) splits `n` as evenly as
possible across classes and samples without replacement, seeded. →
`classifier_on_real_subset.pt` → **81.97%** at n=1000. It then prints the 3-way
comparison (60k real / 1k real / 1k synthetic).

---

## 7. Stage 4 — `evaluate.py` (boundary sampling + quality)

Where the actual thesis idea is *measured* (though at Stage 1 it's not yet fed
back into training):

```python
generator.load_pretrained(...);  classifier.load_state_dict(torch.load("classifier.pt"))

def generate_fn(n):  return generator.sample(n, num_inference_steps=50, batch_size=128)  # unconditional
def classify_fn(x):  return torch.softmax(classifier(x), dim=1)

sampler = BoundaryFocusedSampler(generate_fn, classify_fn, oversample_factor=4.0)
synthetic_images, boundary_scores = sampler.sample(1000)
```

`BoundaryFocusedSampler.sample` (`sampling/boundary.py:493-505`):

1. generate `max(1000, 4.0·1000) = 4000` candidates
2. `boundary_score(probs) = p_top1 − p_top2` — the margin between the two most
   likely classes; **small margin ⇒ near a decision boundary**
   (`boundary.py:408-429`)
3. `select_boundary_samples` keeps the **1000 with the smallest margin**
   (`argsort(scores)[:1000]`)

Then it computes and saves to `evaluation_report.yaml`:

- **`mean_image_difference`** — |mean(real) − mean(synthetic)| over pixels;
  catches a collapsed/diverged generator (`evaluation/metrics.py`)
- **`frechet_distance_diagonal`** — a scipy-free, diagonal-covariance Fréchet
  distance on flattened pixels; a lightweight FID stand-in
- **`mean_boundary_score`** — mean margin of the kept set (0.375)
- **per-class precision / recall / F1 + confusion matrix** for `classifier.pt`
  over the real test set (`evaluation/classifier_metrics.py`, implemented
  directly on tensors)

Note the boundary-kept set is reported, not yet used to train anything — the
"train on the boundary subset and show it beats a uniform subset" experiment is
what the real-data stage is for.

---

## 8. The Streamlit app (`app/streamlit_app.py`)

```python
@st.cache_resource
def load_run(config_path):
    config = load_config(config_path)                       # PD_SYNTH_CONFIG env var, default mnist_baseline
    generator.load_pretrained(output_dir / "generator")
    classifier.load_state_dict(torch.load(output_dir / "classifier.pt"))
    real_dataset = get_dataset("mnist", train=False)
    indices_by_class = index_by_class(real_dataset)
```

Then it renders:

- **4 headline metric cards** read straight from the `*_metrics.yaml` files
  (full real 98.34%, 1k real 81.97%, 1k synthetic 81.10%, generator epochs 20).
  Each card degrades to "not run yet" if its file is missing.
- **the "Synthetic at a full data budget" section** (added this session) —
  shows only when `classifier_on_synthetic_full_metrics.yaml` exists:
  `60,000 real` 98.34% · `60,000 synthetic` 96.38% · `Real − synthetic gap`
  1.96 pts.
- **Real vs. generated grids** — a "Generate new samples" button calls
  `generator.sample(num_samples, num_inference_steps=50, class_labels=torch.full((n,), digit))`
  live and shows the result next to real test digits of the same class.
- **per-class metrics table** from `evaluation_report.yaml`.

AppTest (`streamlit.testing.v1`) is how the app was verified to render with no
exception after each change.

---

## 9. What the numbers say

| Classifier training set | Accuracy on the real MNIST test set |
|---|---|
| 60,000 real | **98.34%** |
| 1,000 real | 81.97% |
| 1,000 synthetic | 81.10% |
| 60,000 synthetic | **96.38%** |

- **At equal 1k budget**, synthetic ≈ real (81.10 vs 81.97) — the
  class-conditional DDPM captures MNIST's per-class distribution well enough
  that 1000 generated images teach the classifier about as much as 1000 real
  ones.
- **Scaling synthetic to 60k** recovers most of the remaining gap: 81.10 →
  96.38, within ~2 points of the full-real 98.34.
- **The residual 1.96-point gap** is the generator's limit: its samples don't
  span the full diversity/tail of real MNIST. Symptom — the synthetic
  classifier's *training* loss plateaus at 0.092 vs 0.037 for the real run
  (`classifier_on_synthetic_full_metrics.yaml`): even *fitting* the synthetic
  set is harder, because generated digits have more inter-class overlap /
  residual noise.

---

## 10. How Stage 1 generalizes

Per the README: implement a `LabeledImageDataset` for the real gait/pose data,
register it in `data/__init__.py`, copy the config, point `data.name` at it,
adjust `image_size`/`num_classes`. Nothing in `generation/`, `sampling/`,
`classifiers/`, or `evaluation/` changes — they only ever touch the
`(image, label)` interface and plain tensors/callables. Then the
boundary-focused sampling that `evaluate.py` currently just *measures* becomes
the thing under test: does training on the boundary-kept synthetic subset beat
training on a uniform synthetic subset, when real data is genuinely scarce.
