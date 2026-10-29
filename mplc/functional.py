import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors

def field_norm(field, spatial_ndim=1):

    return torch.sqrt(torch.abs(field_inner(field, field, spatial_ndim)))

def field_inner(field1, field2, spatial_ndim=1):

    if spatial_ndim not in (1, 2):
        raise ValueError("spatial_ndim must be 1 or 2.")
    return (field1 * torch.conj(field2)).sum(dim=tuple(range(-spatial_ndim, 0)))

def field_overlap_loss(field1, field2, spatial_ndim=1):

    """Overlap loss; pass spatial_ndim=2 for fields shaped (..., H, W)."""
    return 1 - field_inner(field1, field2, spatial_ndim).abs().square().mean()


def normalize_field(field, spatial_ndim=1):
    if spatial_ndim not in (1, 2):
        raise ValueError("spatial_ndim must be 1 or 2.")
    norm = torch.sqrt(torch.sum(torch.abs(field) ** 2,
                               dim=tuple(range(-spatial_ndim, 0)), keepdim=True))
    return field / norm

def phase_gradient(theta, dx = 1, dim=-1):
    
    dtheta = torch.diff(theta, dim=dim)
    dtheta_wrapped = torch.atan2(torch.sin(dtheta), torch.cos(dtheta))
    return dtheta_wrapped/dx

def gradient(y, dx = 1):
    dy = y[..., 1:] - y[..., :-1]
    return dy/dx


def phase_laplacian(theta, dx=1, dim=-1):
    # Wrapped phase differences
    dtheta = torch.diff(theta, dim=dim)
    dtheta_wrapped = torch.atan2(torch.sin(dtheta), torch.cos(dtheta))

    # Second derivative: change in the wrapped phase gradient
    return torch.diff(dtheta_wrapped, dim=dim) / dx**2

def hadamard(D):

    D = torch.tensor(D)
    hadamard_matrix = torch.zeros((D,D), dtype = torch.complex64)
    for i in range(D):
        for j in range(D):
            hadamard_matrix[i][j] = 1/torch.sqrt(D)*torch.exp(1j*2*torch.pi*i*j/D)

    return hadamard_matrix

def xgate(D,m):
    
    # D = torch.tensor(D)
    # m = torch.tensor(m)
    x_matrix = torch.eye(D, dtype=dtype)
    x_matrix = torch.roll(x_matrix,m,0)
    return x_matrix

def unwrap_phase(phase):
    """Unwrap axes in order (y then x in 2D).

    This is a deterministic interpolation convention for smooth surfaces,
    not a general 2D phase-unwrapping algorithm for surfaces with vortices.
    """
    unwrapped = phase
    for dim in range(phase.ndim):
        delta = torch.diff(unwrapped, dim=dim)
        delta = torch.remainder(delta + torch.pi, 2 * torch.pi) - torch.pi
        first = unwrapped.narrow(dim, 0, 1)
        unwrapped = torch.cat((first, first + delta.cumsum(dim)), dim=dim)
    return unwrapped


def interpolate_phase(phase, field_shape):
    """Resize a 1D/2D phase mask using unwrapped linear/bilinear interpolation."""
    size = tuple(int(n) for n in torch.as_tensor(field_shape).reshape(-1))
    if phase.ndim not in (1, 2) or len(size) != phase.ndim:
        raise ValueError("Mask and field must have the same spatial dimensionality (1 or 2).")
    if any(n < 1 for n in size):
        raise ValueError("Field dimensions must be positive.")
    if tuple(phase.shape) == size:
        return torch.remainder(phase + torch.pi, 2 * torch.pi) - torch.pi

    unwrapped = unwrap_phase(phase)
    interpolated_phase_unwrapped = F.interpolate(
        unwrapped[None, None],
        size=size,
        mode="linear" if phase.ndim == 1 else "bilinear",
        align_corners=True,
    )[0, 0]

    interpolated_phase_wrapped = torch.remainder(
        interpolated_phase_unwrapped + torch.pi,
        2 * torch.pi,
    ) - torch.pi

    return interpolated_phase_wrapped

@torch.no_grad()
def pad_to_square(tensor):
    t = tensor.detach().clone()

    rows, cols = t.shape

    pad_top = pad_bottom = 0
    pad_left = pad_right = 0

    if rows < cols:
        diff = cols - rows
        pad_top = diff // 2
        pad_bottom = diff - pad_top

    elif cols < rows:
        diff = rows - cols
        pad_left = diff // 2
        pad_right = diff - pad_left

    padded_tensor = F.pad(
        t,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode="constant",
        value=0,
    )

    crop = (
        slice(pad_top, pad_top + rows),
        slice(pad_left, pad_left + cols),
    )

    return padded_tensor, crop

def plot_field(field):
    
    field = np.asarray(field)
    if field.ndim != 1:
        raise ValueError("field must be a 1D complex array")

    x = np.arange(field.size)
    mag = np.abs(field)
    phase = np.angle(field)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    # --- Main magnitude plot ---
    ax.plot(x, mag, linewidth=2, color = 'black')
    ax.set_xlabel("Position")
    ax.set_ylabel("Field Amplitude")
    ax.grid(alpha=0.2)

    # --- Ribbon geometry ---
    mag_max = mag.max() if mag.size else 1.0
    if mag_max == 0:
        mag_max = 1.0

    ribbon_h = 0.08 * mag_max
    ribbon_gap = 0.06 * mag_max
    ribbon_y1 = -ribbon_gap
    ribbon_y0 = ribbon_y1 - ribbon_h

    # --- Phase colormap ---
    cmap = cm.get_cmap("hsv")
    norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
    rgba = cmap(norm(phase))   # shape (N, 4)

    # --- Fade out ribbon where magnitude is very small ---
    # Phase is only visually emphasized where |psi| is significant.
    cutoff = 0.03 * mag.max() if mag.max() > 0 else 0.0
    alpha = np.clip((mag - cutoff) / (mag.max() - cutoff + 1e-12), 0, 1)
    alpha = alpha**0.7  # slightly smoother ramp
    rgba[:, 3] = alpha

    # Build a 2-pixel-high RGBA image for the ribbon
    ribbon_img = np.repeat(rgba[np.newaxis, :, :], 2, axis=0)

    # Light gray background under the ribbon so "empty" regions stay visible
    ax.axhspan(ribbon_y0, ribbon_y1, color="0.92", zorder=0)

    ax.imshow(
        ribbon_img,
        extent=[x.min(), x.max(), ribbon_y0, ribbon_y1],
        aspect="auto",
        interpolation="nearest",
        zorder=1,
    )

    # --- Limits and cosmetics ---
    ax.set_ylim(ribbon_y0 - 0.03 * mag_max, mag.max() + 0.05 * mag_max)
    #ax.text(x.min(), 0.5 * (ribbon_y0 + ribbon_y1), " phase",
    #        va="center", ha="left")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Colorbar for phase ---
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Phase")
    cbar.set_ticks(
        [-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
        labels=[r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"]
    )

    fig.tight_layout()
    return fig, ax

def fft_convolve_1d(x, h):
    S = h.shape[-1]
    n = 2 * S - 1
    y = torch.fft.ifft(torch.fft.fft(h, n=n, dim=-1) * torch.fft.fft(x, n=n))
    return y[..., :S]
