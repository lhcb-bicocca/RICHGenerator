# rich-generator

[![CI](https://github.com/lhcb-bicocca/RichGenerator/actions/workflows/python-app.yml/badge.svg)](https://github.com/lhcb-bicocca/RichGenerator/actions/workflows/python-app.yml)

`rich-generator` is a lightweight Python library for synthesising Cherenkov ring data similar to what is observed in LHCb RICH detectors. It focuses on:

* Fast generation of events containing multiple rings without full Monte Carlo simulation
* Compact in-memory and HDF5 serialisation formats
* Generation of ring-centered images (Cartesian or polar) for ML training
* Generation of whole-event images

The core API centres on the class [`rich_generator.generator.SCGen`](src/rich_generator/generator.py) and the dataset utility functions in [`src/rich_generator/dataset_utils.py`](src/rich_generator/dataset_utils.py).

---

## Contents

1. Scope & Concepts  
2. Installation  
3. Physics Model Summary  
4. Data Model  
5. Quick Start  
6. Detailed Usage  
7. Image Generation & Radial Stretching  
8. Momentum & Centre Distributions (KDE / Uniform)  
9. Reproducibility & Seeding  
10. Performance Notes 
11. Contributing & License

---

## 1. Scope & Concepts

The library does **not** attempt to be a full detector simulation. It produces idealised circular (optionally noisy) Cherenkov rings projected on a 2D detection plane. Noise is only radial (Gaussian) and not positional dependent. This is intended for use in rapid prototyping of ring-finding or PID (Particle ID) ML pipelines.

Core concepts:

* Event: a collection of rings (one per particle) + metadata.
* Ring: a set of photon hit coordinates `(x, y)` sampled around a centre.
* Dataset: list of events plus global metadata, optionally saved in HDF5.
* MYOLO folder: ring-centred images + split text files (train/val/test).

---

## 2. Installation

```bash
git clone https://github.com/lhcb-bicocca/RichGenerator
cd RichGenerator
pip install .
```

Requires Python ≥ 3.9. Main dependencies (see [pyproject.toml](pyproject.toml)): NumPy, SciPy, h5py, Pillow, tqdm, particle.

---

## 3. Physics Model Summary

Cherenkov condition:

$ \cos \theta_c = \frac{1}{n \beta}, \quad \beta = \frac{p}{\sqrt{p^2 + m^2}} $

Maximum angle (ultra-relativistic):

$ \theta_{c,\max} = \arccos \left( \frac{1}{n} \right) $

Radius mapping to a reference maximum physical radius $R_{\max}$:

$ R(\theta_c) = R_{\max} \frac{\tan \theta_c}{\tan \theta_{c,\max}} $

Implemented helpers:
* [`rich_generator.dataset_utils.calculate_cherenkov_angle`](src/rich_generator/dataset_utils.py)
* [`rich_generator.dataset_utils.calculate_cherenkov_radius`](src/rich_generator/dataset_utils.py)

---

## 4. Data Model

Each event (`dict`) produced by [`rich_generator.generator.SCGen`](src/rich_generator/generator.py):

```
{
  'centers': (N, 2) float32 array
  'momenta': (N,) float array (GeV/c)
  'particle_types': (N,) int array (PDG codes)
  'rings': list of length N; each element (num_hits_i, 2)
  'tracked_rings': int array of indices (pruning bookkeeping)
}
```

HDF5 layout (one group per event) written by `SCGen.save_dataset()`; metadata attributes include refractive index, detector size, masses, etc. Use:
* [`rich_generator.dataset_utils.read_dataset_metadata`](src/rich_generator/dataset_utils.py)
* [`rich_generator.dataset_utils.read_dataset`](src/rich_generator/dataset_utils.py)

---

## 5. Quick Start

Below we use the **shipped KDE distributions** found in the `distributions/` folder of this repository (log-momentum KDEs per PDG code and a 2D centre KDE). Paths are relative to the repository root; if you install the package elsewhere, adjust paths accordingly (currently these data files are not packaged inside the module namespace).

```python
from rich_generator.generator import SCGen
from rich_generator.dataset_utils import load_kde, generate_MYOLO_dataset_folder

# Load provided log-momentum KDEs (they already model log(p); DO NOT exponentiate here)
mom_dists = {
  211: load_kde("distributions/log_momenta_kdes/211-kde.npz"),
  321: load_kde("distributions/log_momenta_kdes/321-kde.npz"),
}

# Load 2D centres KDE (returns samples shaped (2, N))
centers_kde = load_kde("distributions/centers_R1-kde.npz")

gen = SCGen(
  particle_types=[211, 321],                 # PDG codes to sample
  refractive_index=1.0014,                   # example refractive index
  detector_size=((-600, 600), (-600, 600)),  # x/y window (mm or arbitrary)
  momenta_log_distributions=mom_dists,       # dict: PDG -> KDE providing log(p)
  centers_distribution=centers_kde,          # 2D KDE for ring centres
  radial_noise=(0.0, 1.5),                   # mean & sigma radial Gaussian noise
  N_init=50,                                 # photon yield scale
  max_radius=100.0,                          # reference max physical radius
  particle_type_proportions={211: 0.8, 321: 0.2}, # custom particle abundances
)

# Generate a handful of events (each with 5-7 particles)
gen.generate_dataset(num_events=5, num_particles_per_event=(5, 7), parallel=False)
gen.save_dataset("example_dataset.h5")

# Produce MYOLO-style ring-centred polar images WITHOUT radial stretching
generate_MYOLO_dataset_folder(
  data_file="example_dataset.h5",
  base_dir="myolo_no_stretch",
  image_size=128,
  annular_mask=True,
  polar_transform=True,
  stretch_radii=False,
)

# Same, but with legacy per-image radial stretching
generate_MYOLO_dataset_folder(
  data_file="example_dataset.h5",
  base_dir="myolo_legacy",
  image_size=128,
  annular_mask=True,
  polar_transform=True,
  stretch_radii=True,
)
```
**Using default distributions & particle type proportions**
To get events with distributions similar to LHCb RICH1 data, you can use the provided KDEs and a JSON file for particle type proportions. You can find the JSON file in the `distributions/` folder named `particle_type_proportions.json` and the log-momentum KDEs in the `distributions/log_momenta_kdes/` folder. Here's how you can load all of them:

```python
from rich_generator.dataset_utils import load_kde
import json

ptypes = [211, 321, 2212, 11, 13]  # PDG codes for pi+, K+, p, e-, mu-
# Load provided log-momentum KDEs (they already model log(p); DO NOT exponentiate here)
mom_dists = {pt: load_kde(f"distributions/log_momenta_kdes/{pt}-kde.npz") for pt in ptypes}
# Load particle type proportions from JSON
with open("distributions/particle_type_proportions.json", "r") as f:
    particle_type_proportions = json.load(f)
```

---

## 6. Detailed Usage

### 6.1 Event Generation

Class: [`rich_generator.generator.SCGen`](src/rich_generator/generator.py)

Key constructor arguments:
* `particle_types` - PDG codes (masses auto-loaded via `particle` unless overridden).
* `momenta_log_distributions` - dict of PDG → distribution objects exposing `.resample(size)` returning log-momenta.
* `particle_type_proportions` - optional `dict` or `list` to control the relative abundance of sampled particle types. If not provided, sampling is uniform.
* `centers_distribution` - distribution with `.resample(size)` → shape `(2, size)` array (x row, y row).
* `radial_noise` - `(mean, sigma)` or single `sigma` for Gaussian perturbation of ring radius.
* `N_init` - base photon yield scale.
* `max_radius` - defines the reference maximum radius mapping (use 100 for RICH1).
  
Photon count per ring is Poisson with mean scaled by $\sin^2 \theta_c$ relative to $\sin^2 \theta_{c,\max}$.

### 6.2 Inserting Additional Rings

After generation, augment events using:
* [`rich_generator.generator.SCGen.insert_rings`](src/rich_generator/generator.py)

This allows adding rings with specified particle types, momenta, and/or centres to existing events. It is also used internally by the balanced dataset generator.

### 6.3 Pruning Centers

Simulate partial detection (drop ring metadata only):
* [`rich_generator.generator.SCGen.prune_centers`](src/rich_generator/generator.py)

This removes rings from the `tracked_rings` bookkeeping list but leaves the hit coordinates intact. Useful for generating background events or simulating tracking inefficiencies.

### 6.4 Converting to Images

Two levels:

1. Whole event raster: [`rich_generator.generator.SCGen.event_image`](src/rich_generator/generator.py)
2. Ring-centred YOLO style images: [`rich_generator.dataset_utils.generate_MYOLO_dataset_folder`](src/rich_generator/dataset_utils.py)

### 6.5 Generating Balanced Synthetic Datasets

Function: [`rich_generator.dataset_utils.generate_MYOLO_dataset`](src/rich_generator/dataset_utils.py)

Workflow:
1. Build background events via internal `SCGen`.
2. Prune all original rings (`tracked_rings` emptied).
3. Uniformly inject new rings with momenta in a supplied range (flat in linear momentum).
4. Optionally save to an HDF5 file.

---

## 7. Image Generation & Radial Stretching

When `polar_transform=True`, hits are mapped:

* Angle: $x = \mathrm{round}(\theta \cdot \frac{H-1}{2\pi})$
* Radius (non-stretched):  
  $ y = \mathrm{round}\left( (R_{\text{global,max}} - r) \cdot \frac{H-1}{R_{\text{global,max}}} \right)$

“Stretched” legacy mode (`stretch_radii=True`) remaps only the annulus covered by hypothetical rings:

* Let $R_{\min}$ / $R_{\max}$ be the inner / outer radii after mass bounds + noise margin.  
  $ y = \mathrm{round}\left( (R_{\max} - r) \cdot \frac{H-1}{R_{\max} - R_{\min}} \right)$

Effect:
* Non-stretched preserves a consistent vertical scale, that means all images have the same physical size.
* Stretched normalises each ring band to full vertical extent (may aid models expecting full utilisation of dynamic range).

Annular masking uses min/max possible radii for the particle mass extremes (with ±4σ noise padding).

---

## 8. Momentum & Centre Distributions (KDE / Uniform)

You may load shipped KDEs (see `distributions/`) or build new ones.

Helpers:
* [`rich_generator.dataset_utils.save_kde`](src/rich_generator/dataset_utils.py)
* [`rich_generator.dataset_utils.load_kde`](src/rich_generator/dataset_utils.py)
* [`rich_generator.dataset_utils.TruncatedKDE`](src/rich_generator/dataset_utils.py) - adds rejection bounds to `scipy.stats.gaussian_kde`.

**Example (truncated 1D log-momentum KDE):**

```python
from scipy.stats import gaussian_kde
from rich_generator.dataset_utils import TruncatedKDE, save_kde, load_kde
import numpy as np

data = np.log(np.random.uniform(5.0, 50.0, size=5000))[None, :]
kde = TruncatedKDE(data, bw_method='scott', truncate_range=(np.log(5.0), np.log(50.0)))
save_kde(kde, "logmom_kde.npz")
kde2 = load_kde("logmom_kde.npz")
samples = np.exp(kde2.resample(10_000))
```

**Example loading shipped KDEs:**

```python
from rich_generator.dataset_utils import load_kde

kde = load_kde("distributions/log_momenta_kdes/211-kde.npz")
samples = np.exp(kde.resample(10_000))
```

---

## 9. Reproducibility & Seeding

* Global randomness uses `numpy.random`. Set `np.random.seed(seed)` before generation for deterministic datasets.
* The dataset folder split also depends on the seed passed to `generate_MYOLO_dataset_folder`.

---

## 10. Performance Notes

* Parallel flag in `SCGen.generate_dataset` uses Python threads (beneficial mainly for I/O or light CPU; heavy CPU may be GIL-limited).
* Sharding (`images_per_folder`) prevents single directories with tens of thousands of files (speeds up filesystem operations).
* Use uniform distributions (via [`rich_generator.generator.UniformDist`](src/rich_generator/generator.py)) for faster prototyping before switching to KDEs.

---

## 11. Contributing & License

Contributions (issues / PRs) are welcome. Please:
* Keep PRs focused.
* Provide tests for new functionality (test should be made with `pytest` and placed in the `tests/` directory).
* For further information, please contact me at: <g.lagana2@campus.unimib.it>

**License: MIT** - see [LICENSE](LICENSE).

---

## API Reference (Summary)

Symbols (see source docstrings):

* [`rich_generator.generator.SCGen`](src/rich_generator/generator.py)
* [`rich_generator.generator.UniformDist`](src/rich_generator/generator.py)
* [`rich_generator.dataset_utils.calculate_cherenkov_angle`](src/rich_generator/dataset_utils.py)
* [`rich_generator.dataset_utils.calculate_cherenkov_radius`](src/rich_generator/dataset_utils.py)
* [`rich_generator.dataset_utils.read_dataset`](src/rich_generator/dataset_utils.py)
* [`rich_generator.dataset_utils.read_dataset_metadata`](src/rich_generator/dataset_utils.py)
* [`rich_generator.dataset_utils.generate_MYOLO_dataset`](src/rich_generator/dataset_utils.py)
* [`rich_generator.dataset_utils.generate_MYOLO_dataset_folder`](src/rich_generator/dataset_utils.py)
* [`rich_generator.dataset_utils.TruncatedKDE`](src/rich_generator/dataset_utils.py)
* [`rich_generator.dataset_utils.save_kde`](src/rich_generator/dataset_utils.py)
* [`rich_generator.dataset_utils.load_kde`](src/rich_generator/dataset_utils.py)

