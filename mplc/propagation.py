import torch
import torch.nn as nn

class Propagator(nn.Module):

    def __init__(self, shape, spacing, propagation_distance, wavelength, device):
        
        super().__init__()

        self.shape = shape
        self.register_buffer('spacing', torch.as_tensor(spacing, device=device))
        self.register_buffer('propagation_distance', torch.as_tensor(propagation_distance, device=device))
        self.register_buffer('wavelength', torch.as_tensor(wavelength, device=device))
        self.device = device
        
    def forward(self, field):

        # Use the configured spatial shape; leading field axes may be batches.
        spatial_ndim = torch.as_tensor(self.shape).numel()
        if spatial_ndim not in (1, 2):
            raise ValueError("Propagator supports only 1D or 2D spatial shapes.")
        return self.asm1d(field) if spatial_ndim == 1 else self.asm2d(field)


    def asm1d(self, field):
        
        S = field.shape[-1]

        dx = self.spacing.reshape(-1)[0].item()
        lam = self.wavelength
        z = self.propagation_distance

        # Zero-pad to suppress periodic wrap-around
        pad = S // 2
        field_padded = torch.nn.functional.pad(field, (pad, pad))
        N = field_padded.shape[-1]

        fx = torch.fft.fftfreq(
            N,
            d=dx,
            device=field.device,
            dtype=field.real.dtype,
        )
        kx = 2 * torch.pi * fx

        k = 2 * torch.pi / lam
        kz = torch.sqrt((k**2 - kx**2).to(field.dtype))

        H = torch.exp(1j * kz * z)

        propagated = torch.fft.ifft(
            torch.fft.fft(field_padded, dim=-1) * H,
            dim=-1,
        )

        # Return original observation window
        return propagated[..., pad:pad + S]
    
    def asm2d(self, field):
        
        (S_row,S_col) = field.shape[-2:]

        spacing = self.spacing.reshape(-1)
        if spacing.numel() not in (1, 2):
            raise ValueError("2D spacing must be a scalar or (dy, dx).")
        dy = spacing[0].item()
        dx = spacing[-1].item()
        lam = self.wavelength
        z = self.propagation_distance

        # Zero-pad to suppress periodic wrap-around
        pad_row = S_row // 2
        pad_col = S_col // 2
        field_padded = torch.nn.functional.pad(field, (pad_col, pad_col, pad_row, pad_row))
        
        (Ny,Nx) = field_padded.shape[-2:]

        fx = torch.fft.fftfreq(
            Nx,
            d=dx,
            device=field.device,
            dtype=field.real.dtype,
        )
        
        fy = torch.fft.fftfreq(
            Ny,
            d=dy,
            device=field.device,
            dtype=field.real.dtype,
        )
        
        kx = 2 * torch.pi * fx
        ky = 2 * torch.pi * fy
        ky, kx = torch.meshgrid(ky, kx, indexing="ij")
        
        
        k = 2 * torch.pi / lam
        kz = torch.sqrt((k**2 - kx**2 - ky**2).to(field.dtype))

        H = torch.exp(1j * kz * z)

        propagated = torch.fft.ifft2(
            torch.fft.fft2(field_padded, dim=(-2,-1)) * H,
            dim=(-2,-1),
        )

        # Return original observation window
        return propagated[..., pad_row:pad_row + S_row, pad_col:pad_col + S_col]
