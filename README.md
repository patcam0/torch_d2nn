# torchmplc

PyTorch-based tools for designing phase masks for multi-plane light conversion (MPLC).

## Installation

With Python 3.9 or newer, install from the GitHub repository:

```bash
python -m pip install git+https://github.com/YOUR_USERNAME/torchmplc.git
```

For SIC-POVM datasets, also install the optional `sic` extra:

```bash
python -m pip install "torchmplc[sic] @ git+https://github.com/YOUR_USERNAME/torchmplc.git"
```

For local development:

```bash
python -m pip install -e ".[sic]"
```

The SIC-POVM functionality relies on the `qbism` package (which in turn uses QuTiP). Verify its compatibility with your Python environment.

## Usage

```python
from mplc import MPLC
from mplc.data import get_basis_fields, RandomFieldDataset, SICFieldDataset
import mplc.functional as mplcF
```

See [`example.ipynb`](example.ipynb) for a full training and visualization example.

The Python **distribution** is named `torchmplc`, while the **import** is named `mplc`.

### 1D and 2D fields

`MPLC(shape=N, ...)` keeps the existing 1D interface. For 2D fields, use
`shape=(height, width)`. Fields may be unbatched or have leading batch dimensions.
Both `spacing` and `mask_spacing` accept a scalar or `(dy, dx)` in 2D.

```python
from functools import partial
import torch

model = MPLC(
    shape=(64, 96), spacing=10e-6, mask_spacing=20e-6,
    num_masks=3, propagation_distance=0.01,
    beam_waist=100e-6, wavelength=810e-9, device="cpu",
)
fields = mplcF.normalize_field(
    torch.randn(4, 64, 96, dtype=torch.complex64), spatial_ndim=2
)
outputs = model(fields)
loss_func = partial(mplcF.field_overlap_loss, spatial_ndim=2)
loss = loss_func(outputs, fields) + model.smoothness_penalty()
loss.backward()
```

Pass this `loss_func` to `train_mplc` and `test`. The field norm, inner product,
normalization, and overlap helpers default to 1D for compatibility; pass
`spatial_ndim=2` for 2D fields. `FieldDataset`, `RandomFieldDataset`, `encode`,
and `decode` accept a supplied basis shaped `(modes, height, width)`; the existing
basis-generation functions still generate 1D fields.

Visualization selects 1D line plots or 2D amplitude/phase images automatically.
The 2D far-field view uses the full centered FFT. Mask CSVs contain sampled phase
columns in 1D and phase matrices in 2D. Import verifies reconstruction and restores
all masks on failure.

Lower-resolution masks use linear interpolation in 1D and bilinear interpolation
in 2D after sequential phase unwrapping (y, then x). This convention is intended
for smooth phase surfaces; it is not a general phase-vortex unwrapping method.
Masks at the field resolution are applied directly without unwrapping.

## Testing

```bash
python -m pip install pytest
python -m pytest
```

The tests cover existing 1D behavior, rectangular 2D propagation against an explicit
DFT reference, gradients, training, plots, mask CSV round trips and rollback, and
CUDA device transfers when available. They do not validate the SIC-POVM dependency
or establish convergence of a physical MPLC design.

The 2D test module also runs without pytest:

```bash
python -m unittest discover -s tests -p test_mplc_2d.py
```
