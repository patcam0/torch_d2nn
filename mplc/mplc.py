import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft2, fftshift, ifft2, ifftshift
from torch.nn.functional import pad
from torch.utils.data import Dataset, DataLoader

from IPython.display import clear_output, display
import matplotlib.pyplot as plt
from matplotlib import colors

from pathlib import Path
import shutil
import numpy as np
import time

from .propagation import Propagator
from .modulation import PhaseMask
from . import functional as mplcF

class MPLC(nn.Module):
    """MPLC for (..., N) or (..., H, W) complex fields.

    Use shape=N (or (N,)) for 1D and shape=(H, W) for 2D. Spacings
    may be scalars or one value per spatial axis, ordered (dy, dx) in 2D.
    Custom losses must reduce all spatial axes; for field_overlap_loss use
    functools.partial(field_overlap_loss, spatial_ndim=model.spatial_ndim).
    """

    def __init__(self, 
        shape, 
        spacing, 
        mask_spacing, 
        num_masks, 
        propagation_distance, 
        beam_waist, 
        wavelength, 
        device, 
        smooth_weight = 1e-1,
        init_scale = 1e-1): 

        super().__init__()

        shape_values = torch.as_tensor(shape, device=device)
        if (shape_values.ndim > 1 or shape_values.numel() not in (1, 2)
                or not torch.isfinite(shape_values).all()
                or (shape_values < 1).any()
                or (shape_values != shape_values.long()).any()):
            raise ValueError("shape must be a positive integer or a tuple of one or two positive integers.")
        self.spatial_shape = tuple(int(n) for n in shape_values.reshape(-1))
        self.spatial_ndim = len(self.spatial_shape)
        # Preserve scalar self.shape for existing 1D notebooks.
        shape_tensor = shape_values.long().reshape(()) if self.spatial_ndim == 1 else shape_values.long()
        self.register_buffer('shape', shape_tensor)
        for name, value in [('spacing', spacing), ('mask_spacing', mask_spacing)]:
            values = torch.as_tensor(value, dtype=torch.float64, device=device).reshape(-1)
            if (values.numel() not in (1, self.spatial_ndim)
                    or not torch.isfinite(values).all() or (values <= 0).any()):
                raise ValueError(f"{name} must be positive and finite, with one value or one per spatial axis.")
            self.register_buffer(name, values.reshape(()) if values.numel() == 1 else values)
        mask_size = self.shape * self.spacing / self.mask_spacing
        # Avoid losing a control point to roundoff near an integer ratio.
        mask_size = torch.where(torch.isclose(mask_size, mask_size.round(), rtol=1e-7, atol=1e-9),
                                mask_size.round(), mask_size.floor()).long()
        if (mask_size < 1).any():
            raise ValueError("mask_spacing leaves fewer than one mask sample along an axis.")
        self.register_buffer('mask_shape', mask_size)
        self.num_masks = num_masks
        self.register_buffer('propagation_distance', torch.as_tensor(propagation_distance, device=device))
        self.register_buffer('beam_waist', torch.as_tensor(beam_waist, device=device))
        self.register_buffer('wavelength', torch.as_tensor(wavelength, device=device))
        self.register_buffer('smooth_weight', torch.tensor(smooth_weight, device = device))
        self.init_scale = init_scale

        self.layers = nn.ModuleList()
        self.add_mplc_stack()
        self.sequence = nn.Sequential(*self.layers)

        self.loss_history = []
        self.time_history = []
        self.epochs = 0

    @property
    def device(self):
        return self.wavelength.device
        
    def add_mplc_stack(self):

        self.phase_masks = nn.ModuleList()
        self.propagators = nn.ModuleList()

        for i in range(self.num_masks):

            phase_mask = PhaseMask(self.mask_shape, self.device, self.init_scale)
            self.layers.append(phase_mask)
            self.phase_masks.append(phase_mask)

            propagator = Propagator(self.shape, self.spacing, self.propagation_distance, self.wavelength, self.device)
            self.layers.append(propagator)
            self.propagators.append(propagator)

    def forward(self, field):
        if field.ndim < self.spatial_ndim or tuple(field.shape[-self.spatial_ndim:]) != self.spatial_shape:
            raise ValueError(f"Expected trailing spatial shape {self.spatial_shape}, got {tuple(field.shape)}.")
        if not torch.is_complex(field):
            raise ValueError("Expected a complex-valued field.")
        return self.sequence(field)
    
    def smoothness_penalty(self):

        # Sum curvature penalties over spatial axes. In 1D this is the
        # existing penalty; small axes have no second differences.
        penalties = []
        for mask in self.phase_masks:
            penalty = mask.theta.sum() * 0
            for dim in range(self.spatial_ndim):
                if mask.theta.shape[dim] >= 3:
                    penalty = penalty + mplcF.phase_laplacian(mask.theta, dim=dim).abs().mean()
            penalties.append(penalty)
        if not penalties:
            return self.smooth_weight * 0
        return self.smooth_weight * torch.stack(penalties).mean()

    def train_mplc(self, num_epochs, dataloader, optimizer, device, loss_func):
    
        def train_epoch():

            self.train()
            total_loss = 0
            count = 0
            for x_batch, y_batch in dataloader:

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                y_pred = self.forward(x_batch)
                
                loss =  loss_func(y_pred, y_batch) + self.smoothness_penalty()
                loss.backward()
                optimizer.step()

                batch_size = x_batch.size(0)
                total_loss += loss.item() * batch_size
            
                count += batch_size

            epoch_loss = total_loss / count

            return epoch_loss

        @torch.no_grad()
        def wrap_phases():
        
            for mask in self.phase_masks:
                mask.theta.copy_(((mask.theta + torch.pi) % (2 * torch.pi)) - torch.pi)


        for i in range(num_epochs):
            start_time = time.time()
            loss = train_epoch()
            wrap_phases()
            self.time_history.append(time.time() - start_time)
            self.loss_history.append(loss)
            self.epochs += 1
            clear_output(wait=True)
            print(f"Epoch {self.epochs:02d} | Loss: {loss:.6f}")
            #fig = self.visualize_model(show = False)
            #display(fig)
            #plt.close(fig)

    @torch.no_grad()
    def test(self, dataloader, loss_func):
        
        self.eval()
        total_loss = 0
        count = 0
        for x_batch, y_batch in dataloader:

            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            y_pred = self.forward(x_batch)
            loss = loss_func(y_pred, y_batch)

            batch_size = x_batch.size(0)
            total_loss += loss.item() * batch_size
            count += batch_size

        test_loss = total_loss / count

        return test_loss

    def visualize_model(self, show = False):
    
        fig_width = 6 + self.num_masks * 4  # Base width + width per mask
        fig, axes = plt.subplots(1, self.num_masks + 1, figsize=(fig_width, 4), squeeze=False)
        axes = axes[0]

        # plot loss
        ax = axes[0]
        ax.plot(np.arange(self.epochs) + 1, self.loss_history, '-o', label='Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_ylim(0,1)
        ax.grid(True)
        ax.legend()

        # plot phase masks
        for i, mask in enumerate(self.phase_masks):
            ax = axes[i + 1]
            theta = mask.get_theta(self.spatial_shape).detach().cpu().numpy()
            im = ax.imshow(theta if self.spatial_ndim == 2 else theta[None, :],
                           vmin=-np.pi, vmax=np.pi, cmap='twilight', origin='lower', aspect='auto')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Phase [rad]')
            ax.set_title(f'Phase Mask {i + 1}')

        fig.tight_layout()

        if show:
            plt.show()

        return fig 

    @torch.no_grad()
    def visualize_field(self, input_fields, show=True, far_field=False):
        """
        Visualize complex fields through the MPLC.

        2D fields use amplitude and phase image pairs at each step. The
        2D far-field view displays the full centered FFT without mode cropping.

        Rows:
            One row per input field.

        Columns:
            Input field, then one column after each mask + propagation step.

        Each subplot shows:
            - field magnitude as a black line
            - field phase as a ribbon underneath

        If far_field=True:
            - performs a regular length-S FFT
            - crops afterwards around the expected Gaussian diffraction modes
            - assumes basis modes of the form

                exp(-x^2 / w0^2) * exp(2π i m x / w0)

            whose Fourier peaks are centred at f = m / w0.
        """

        if torch.is_tensor(input_fields) and input_fields.ndim == self.spatial_ndim:
            input_fields = input_fields.unsqueeze(0)
        if len(input_fields) == 0:
            raise ValueError("Provide at least one input field.")
        for field in input_fields:
            if tuple(field.shape) != self.spatial_shape:
                raise ValueError(f"Expected individual fields of shape {self.spatial_shape}.")
        if self.spatial_ndim == 2:
            return self._visualize_field_2d(input_fields, show, far_field)

        # ------------------------------------------------------------
        # Extract mask / propagation layers
        # ------------------------------------------------------------
        phasemask_layers = [
            layer for layer in self.sequence
            if isinstance(layer, PhaseMask)
        ]

        propagation_layers = [
            layer for layer in self.sequence
            if isinstance(layer, Propagator)
        ]

        n_fields = len(input_fields)

        n_steps = min(
            len(phasemask_layers),
            len(propagation_layers),
        )

        n_cols = n_steps + 1

        # ------------------------------------------------------------
        # Figure layout
        #
        # Last narrow column contains one colourbar per row.
        # ------------------------------------------------------------
        fig = plt.figure(
            figsize=(
                max(4.5 * n_cols + 0.8, 7),
                max(3.2 * n_fields, 3.5),
            )
        )

        gs = fig.add_gridspec(
            n_fields,
            n_cols + 1,
            width_ratios=[1] * n_cols + [0.045],
            wspace=0.30,
            hspace=0.40,
        )

        axes = np.empty(
            (n_fields, n_cols),
            dtype=object,
        )

        caxes = []

        for row in range(n_fields):

            for col in range(n_cols):
                axes[row, col] = fig.add_subplot(
                    gs[row, col]
                )

            caxes.append(
                fig.add_subplot(gs[row, -1])
            )

        # ------------------------------------------------------------
        # Phase colormap
        # ------------------------------------------------------------
        cmap = plt.get_cmap("hsv")

        phase_norm = colors.Normalize(
            vmin=-np.pi,
            vmax=np.pi,
        )

        phase_map = plt.cm.ScalarMappable(
            norm=phase_norm,
            cmap=cmap,
        )

        phase_map.set_array([])

        # ------------------------------------------------------------
        # Maximum expected mode index
        # ------------------------------------------------------------
        if hasattr(self, "m_max"):
            m_max = abs(self.m_max)

        elif hasattr(self, "modes"):
            m_max = max(
                abs(float(m))
                for m in self.modes
            )

        else:
            # Fallback: assume the input fields span symmetric modes
            # about zero.
            m_max = (n_fields - 1) / 2

        # ------------------------------------------------------------
        # Convert field into whatever should actually be displayed
        # ------------------------------------------------------------
        def get_display_field(field):

            S = field.shape[-1]

            # ========================================================
            # Near field
            # ========================================================
            if not far_field:

                x = np.arange(S)

                return field, x

            # ========================================================
            # Far field
            # ========================================================

            # Regular length-S centered FFT.
            fft_field = torch.fft.fftshift(
                torch.fft.fft(
                    torch.fft.ifftshift(
                        field,
                        dim=-1,
                    ),
                    dim=-1,
                    norm="ortho",
                ),
                dim=-1,
            )

            # Spatial-frequency coordinate [cycles / unit length].
            fx = torch.fft.fftshift(
                torch.fft.fftfreq(
                    S,
                    d=float(self.spacing),
                    device=field.device,
                    dtype=field.real.dtype,
                )
            )

            w0 = self.beam_waist

            # --------------------------------------------------------
            # For
            #
            #   exp(-x²/w0²) exp(2π i m x/w0)
            #
            # the Fourier-space Gaussian is centred at
            #
            #   f_m = m/w0
            #
            # and has characteristic amplitude width
            #
            #   Δf = 1/(π w0).
            #
            # Crop two widths beyond the outermost expected mode.
            # --------------------------------------------------------
            f_outer = m_max / w0

            f_margin = 2 / (torch.pi * w0) * 3

            f_max = f_outer + f_margin

            keep = torch.abs(fx) <= f_max

            return (
                fft_field[keep],
                fx[keep].detach().cpu().numpy(),
            )

        # ------------------------------------------------------------
        # Plot a single complex 1D field
        # ------------------------------------------------------------
        def plot_field(ax, field):

            display_field, x = get_display_field(
                field
            )

            field_np = (
                display_field
                .detach()
                .cpu()
                .numpy()
            )

            if field_np.ndim != 1:
                raise ValueError(
                    "Expected a 1D field, "
                    f"got shape {field_np.shape}"
                )

            mag = np.abs(field_np)
            phase = np.angle(field_np)

            # --------------------------------------------------------
            # Magnitude
            # --------------------------------------------------------
            ax.plot(
                x,
                mag,
                color="black",
                linewidth=2,
            )

            ax.grid(alpha=0.18)

            ax.set_xlabel(
                r"Spatial frequency $f_x$"
                if far_field
                else "Position"
            )

            # --------------------------------------------------------
            # Vertical scale
            # --------------------------------------------------------
            if mag.size > 0:
                mag_max = float(mag.max())
            else:
                mag_max = 0.0

            scale = (
                mag_max
                if mag_max > 0
                else 1.0
            )

            # --------------------------------------------------------
            # Phase ribbon geometry
            # --------------------------------------------------------
            ribbon_h = 0.08 * scale
            ribbon_gap = 0.055 * scale

            ribbon_y1 = -ribbon_gap
            ribbon_y0 = ribbon_y1 - ribbon_h

            # --------------------------------------------------------
            # Phase -> colour
            # --------------------------------------------------------
            rgba = cmap(
                phase_norm(phase)
            )

            # Fade the phase where amplitude is negligible.
            cutoff = 0.03 * mag_max

            alpha = np.clip(
                (
                    mag - cutoff
                )
                /
                (
                    mag_max
                    - cutoff
                    + 1e-12
                ),
                0,
                1,
            )

            alpha = alpha ** 0.7

            rgba[:, 3] = alpha

            ribbon_img = np.repeat(
                rgba[np.newaxis, :, :],
                2,
                axis=0,
            )

            # Neutral ribbon background.
            ax.axhspan(
                ribbon_y0,
                ribbon_y1,
                color="0.92",
                zorder=0,
            )

            # --------------------------------------------------------
            # Draw phase ribbon
            # --------------------------------------------------------
            if len(x) > 1:

                ax.imshow(
                    ribbon_img,
                    extent=[
                        x[0],
                        x[-1],
                        ribbon_y0,
                        ribbon_y1,
                    ],
                    aspect="auto",
                    interpolation="nearest",
                    zorder=1,
                )

                ax.set_xlim(
                    x[0],
                    x[-1],
                )

            ax.set_ylim(
                ribbon_y0 - 0.025 * scale,
                mag_max + 0.06 * scale,
            )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # --------------------------------------------------------
            # Total power in the actual field
            # --------------------------------------------------------
            total_power = (
                torch.abs(field)
                .square()
                .sum()
                .real
                .item()
            )

            ax.text(
                0.97,
                0.94,
                f"$P={total_power:.3f}$",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
            )

        # ============================================================
        # One row for each input field
        # ============================================================
        for row, field in enumerate(input_fields):

            current_field = (
                field.detach().clone().to(self.device)
            )

            # --------------------------------------------------------
            # Input field
            # --------------------------------------------------------
            ax = axes[row, 0]

            plot_field(
                ax,
                current_field,
            )

            ax.set_ylabel(
                f"Input {row + 1}\nAmplitude"
            )

            if row == 0:
                ax.set_title(
                    "Input"
                    + (" — FFT" if far_field else "")
                )

            # --------------------------------------------------------
            # Propagate through each mask + propagator pair
            # --------------------------------------------------------
            for step in range(n_steps):

                current_field = phasemask_layers[
                    step
                ](
                    current_field
                )

                current_field = propagation_layers[
                    step
                ](
                    current_field
                )

                ax = axes[
                    row,
                    step + 1,
                ]

                plot_field(
                    ax,
                    current_field,
                )

                if row == 0:

                    ax.set_title(
                        f"After Mask {step + 1}"
                        + (
                            " — FFT"
                            if far_field
                            else ""
                        )
                    )

                if step > 0:
                    ax.set_ylabel("")

            # --------------------------------------------------------
            # Full-height phase colourbar for this row
            # --------------------------------------------------------
            cbar = fig.colorbar(
                phase_map,
                cax=caxes[row],
            )

            cbar.set_ticks(
                [
                    -np.pi,
                    -np.pi / 2,
                    0,
                    np.pi / 2,
                    np.pi,
                ],
                labels=[
                    r"$-\pi$",
                    r"$-\pi/2$",
                    "0",
                    r"$\pi/2$",
                    r"$\pi$",
                ],
            )

            cbar.set_label(
                "Phase",
                rotation=270,
                labelpad=15,
            )

        if show:
            plt.show()

        return fig

    @torch.no_grad()
    def _visualize_field_2d(self, input_fields, show, far_field):
        n_fields = len(input_fields)
        n_cols = self.num_masks + 1
        fig, axes = plt.subplots(2 * n_fields, n_cols, squeeze=False,
                                 figsize=(4 * n_cols, 6 * n_fields))
        spacing = self.spacing.reshape(-1)
        dy, dx = float(spacing[0]), float(spacing[-1])
        height, width = self.spatial_shape

        if far_field:
            fy = np.fft.fftshift(np.fft.fftfreq(height, d=dy))
            fx = np.fft.fftshift(np.fft.fftfreq(width, d=dx))
            extent = [fx[0] - 0.5 / (width * dx), fx[-1] + 0.5 / (width * dx),
                      fy[0] - 0.5 / (height * dy), fy[-1] + 0.5 / (height * dy)]
        else:
            extent = [-width * dx / 2, width * dx / 2,
                      -height * dy / 2, height * dy / 2]

        for row, field in enumerate(input_fields):
            current = field.to(self.device)
            fields = [current]
            for mask, propagator in zip(self.phase_masks, self.propagators):
                current = propagator(mask(current))
                fields.append(current)
            displays = [torch.fft.fftshift(torch.fft.fft2(
                torch.fft.ifftshift(value, dim=(-2, -1)), norm='ortho'), dim=(-2, -1))
                if far_field else value for value in fields]
            vmax = max(float(value.abs().max()) for value in displays) or 1.0
            for col, (value, displayed) in enumerate(zip(fields, displays)):
                amplitude = displayed.abs().cpu().numpy()
                phase = displayed.angle().cpu().numpy()
                amp_ax, phase_ax = axes[2 * row, col], axes[2 * row + 1, col]
                amp_im = amp_ax.imshow(amplitude, origin='lower', extent=extent,
                                       cmap='gray', vmin=0, vmax=vmax)
                phase_im = phase_ax.imshow(phase, origin='lower', extent=extent,
                                           cmap='twilight', vmin=-np.pi, vmax=np.pi)
                label = 'Input' if col == 0 else f'After Mask {col}'
                amp_ax.set_title(f'{label} | P={value.abs().square().sum().item():.3f}')
                phase_ax.set_title('Phase' + (' (FFT)' if far_field else ''))
                for ax in (amp_ax, phase_ax):
                    ax.set_xlabel('Spatial frequency fx' if far_field else 'x')
                    ax.set_ylabel('Spatial frequency fy' if far_field else 'y')
                fig.colorbar(amp_im, ax=amp_ax, label='Amplitude')
                fig.colorbar(phase_im, ax=phase_ax, label='Phase [rad]')
        fig.tight_layout()
        if show:
            plt.show()
        return fig

    def vizualize_phase_gradient(self):

        if self.spatial_ndim == 2:
            return self._visualize_phase_gradient_2d()

        phases = [mask.get_theta(self.shape).detach() for mask in self.phase_masks]
        phase_grads = [mplcF.phase_gradient(phase, self.spacing) for phase in phases]

        for i, grad in enumerate(phase_grads):
            plt.plot(grad.detach().cpu().numpy(), label = f'Phase Mask {i+1}')

        plt.xlabel('x-postition')
        plt.ylabel('Phase Gradient')
        threshold = np.pi/50e-6

        plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
        plt.axhline(y=-threshold, color='r', linestyle='--')
        plt.ylim(-threshold-10000, threshold+10000)

        plt.legend(loc = 'center right')
        plt.show()

    @torch.no_grad()
    def _visualize_phase_gradient_2d(self):
        fig, axes = plt.subplots(self.num_masks, 2, squeeze=False,
                                 figsize=(9, 3.5 * self.num_masks))
        spacing = self.spacing.reshape(-1)
        threshold = np.pi / 50e-6
        for row, mask in enumerate(self.phase_masks):
            phase = mask.get_theta(self.spatial_shape)
            for dim, label in [(0, 'y'), (1, 'x')]:
                grad = mplcF.phase_gradient(phase, spacing[0 if dim == 0 else -1], dim=dim)
                # The last row/column has no forward difference.
                if grad.numel() == 0:
                    values = np.zeros((1, 1))
                else:
                    values = grad.cpu().numpy()
                ax = axes[row, dim]
                im = ax.imshow(values, origin='lower', aspect='auto', cmap='coolwarm',
                               vmin=-threshold, vmax=threshold)
                ax.set_title(f'Mask {row + 1}: d phase / d{label}')
                fig.colorbar(im, ax=ax, label='Phase gradient [rad/m]')
        fig.tight_layout()
        plt.show()
        return fig

    def export_masks(self, filename):
        """Save sampled phases as column CSVs (1D) or H-by-W CSVs (2D)."""

        phases = [mask.get_theta(self.shape).detach() for mask in self.phase_masks]
        
        folder = Path(filename)

        if folder.exists():
            confirm = input(
                f"Folder '{folder}' already exists. Overwrite it? [y/N]: "
            )

            if confirm.lower() not in ("y", "yes"):
                print("Save cancelled.")
                return

            shutil.rmtree(folder)

        folder.mkdir(parents=True)

        for i, phase in enumerate(phases):
            if phase.ndim != self.spatial_ndim:
                raise ValueError(f"phases[{i}] must have {self.spatial_ndim} spatial dimensions.")

            np.savetxt(
                folder / f"phase_{i + 1}.csv",
                phase.detach().cpu().numpy(),
                delimiter=",",
            )

        print(f"Saved {len(phases)} phases to '{folder}'.")

    def import_masks(self, filename, *, atol=1e-4):
        """Fit control phases from sampled CSVs, verifying angular error.

        Each spatial axis must be sampled at least as finely as the mask.
        All masks are restored if reconstruction fails. In 2D the fit uses
        the same sequential-unwrapping convention as phase interpolation.
        """
        folder = Path(filename)

        if not folder.exists():
            raise FileNotFoundError(
                f"Mask folder '{folder}' does not exist."
            )

        if not folder.is_dir():
            raise NotADirectoryError(
                f"'{folder}' is not a directory."
            )

        if not (0 < atol < float("inf")):
            raise ValueError(
                f"atol must be positive and finite; got {atol}."
            )

        n_masks = len(self.phase_masks)
        expected_names = {
            f"phase_{i + 1}.csv" for i in range(n_masks)
        }
        found_names = {
            path.name for path in folder.glob("*.csv")
        }

        if len(found_names) != n_masks:
            raise ValueError(
                f"Wrong number of mask CSVs in '{folder}': "
                f"expected {n_masks}, found {len(found_names)}."
            )

        if found_names != expected_names:
            raise ValueError(
                "Incorrect mask filenames. "
                f"Missing: {sorted(expected_names - found_names)}; "
                f"unexpected: {sorted(found_names - expected_names)}."
            )

        new_thetas = []
        saved_phases = []

        # Load, validate and fit every mask
        for i, mask in enumerate(self.phase_masks, start=1):
            path = folder / f"phase_{i}.csv"

            with torch.no_grad():
                expected_phase = mask.get_theta(self.shape)

            if expected_phase.ndim != self.spatial_ndim or mask.theta.ndim != self.spatial_ndim:
                raise ValueError(
                    f"Mask {i}: expected {self.spatial_ndim}D phase tensors; "
                    f"got output shape {tuple(expected_phase.shape)} "
                    f"and parameter shape {tuple(mask.theta.shape)}."
                )

            try:
                data = np.loadtxt(
                    path, delimiter=",", ndmin=2
                )
            except (OSError, ValueError, UnicodeError) as exc:
                raise ValueError(
                    f"Mask {i}: cannot read numeric CSV "
                    f"'{path}': {exc}"
                ) from exc

            n_samples = expected_phase.numel()
            n_controls = mask.theta.numel()

            expected_csv_shape = (n_samples, 1) if self.spatial_ndim == 1 else tuple(expected_phase.shape)
            if data.shape != expected_csv_shape:
                raise ValueError(
                    f"Mask {i}: wrong CSV shape {data.shape}; "
                    f"expected {expected_csv_shape}."
                )

            if not np.isfinite(data).all():
                raise ValueError(
                    f"Mask {i}: CSV contains NaN or infinity."
                )

            if any(n < c for n, c in zip(expected_phase.shape, mask.theta.shape)):
                raise ValueError(
                    f"Mask {i}: {n_samples} saved samples cannot "
                    f"uniquely determine {n_controls} trainable "
                    "phase values. Export at a field resolution "
                    "at least as large as the mask resolution."
                )

            saved = torch.from_numpy(
                (data[:, 0] if self.spatial_ndim == 1 else data).copy()
            ).to(torch.float64)

            if tuple(expected_phase.shape) == tuple(mask.theta.shape):
                # No interpolation: retain arbitrary pixel phases, including vortices.
                fitted_theta = saved.clone()
            else:
                fitted_theta = mplcF.unwrap_phase(saved)
                # Bilinear interpolation is separable: solve one small system
                # per axis rather than building a huge 2D interpolation matrix.
                for dim, (n, c) in enumerate(zip(expected_phase.shape, mask.theta.shape)):
                    x = torch.linspace(0, c - 1, n, dtype=torch.float64)
                    left = x.floor().long()
                    right = (left + 1).clamp(max=c - 1)
                    alpha = x - left
                    A = torch.zeros((n, c), dtype=torch.float64)
                    rows = torch.arange(n)
                    A[rows, left] = 1 - alpha
                    A[rows, right] += alpha
                    moved = fitted_theta.movedim(dim, 0)
                    solved = torch.linalg.lstsq(A, moved.reshape(n, -1)).solution
                    fitted_theta = solved.reshape((c,) + moved.shape[1:]).movedim(0, dim)

            fitted_theta = fitted_theta.to(
                device=mask.theta.device,
                dtype=mask.theta.dtype
            )

            if not torch.isfinite(fitted_theta).all():
                raise ValueError(
                    f"Mask {i}: fitted theta contains "
                    "non-finite values."
                )

            new_thetas.append(fitted_theta)
            saved_phases.append(saved)

        # Back up existing parameters before modifying them
        original_thetas = [
            mask.theta.detach().clone()
            for mask in self.phase_masks
        ]

        try:
            with torch.no_grad():

                # Update the existing parameters in-place
                for mask, theta in zip(
                    self.phase_masks, new_thetas
                ):
                    mask.theta.copy_(theta)

                # Verify reconstruction through get_theta()
                for i, (mask, target) in enumerate(
                    zip(self.phase_masks, saved_phases),
                    start=1
                ):
                    reconstructed = (
                        mask.get_theta(self.shape)
                        .detach()
                        .cpu()
                        .to(torch.float64)
                    )

                    # Angular difference modulo 2*pi
                    difference = torch.remainder(
                        reconstructed - target + torch.pi,
                        2 * torch.pi
                    ) - torch.pi

                    error = difference.abs().max().item()

                    if (
                        not torch.isfinite(reconstructed).all()
                        or error > atol
                    ):
                        raise ValueError(
                            f"Mask {i}: reconstruction failed. "
                            f"Maximum phase error = {error:.3e} rad; "
                            f"allowed = {atol:.3e} rad. "
                            "The model cannot reproduce this phase "
                            "profile at its current mask resolution. "
                            "Original parameters have been restored."
                        )

        except Exception:
            # Roll back every mask if verification fails
            with torch.no_grad():
                for mask, old_theta in zip(
                    self.phase_masks, original_thetas
                ):
                    mask.theta.copy_(old_theta)
            raise

        print(
            f"Imported and verified {n_masks} phase masks "
            f"from '{folder}'."
        )
    
        
    

        
