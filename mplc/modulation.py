import torch
import torch.nn as nn

from . import functional as mplcF

class PhaseMask(nn.Module):

    def __init__(self, mask_shape, device, init_scale = 2*torch.pi*1e-2):

        super().__init__()

        self.mask_shape = tuple(int(n) for n in torch.as_tensor(mask_shape).reshape(-1))
        if len(self.mask_shape) not in (1, 2) or any(n < 1 for n in self.mask_shape):
            raise ValueError("mask_shape must contain one or two positive dimensions.")
        init_theta = torch.randn(self.mask_shape, device = device)*init_scale
        self.theta = nn.Parameter(init_theta)

        
    def get_theta(self, field_shape, method = 'linear'):
        
        interpolated_theta = mplcF.interpolate_phase(self.theta, field_shape)
        return interpolated_theta


    def forward(self, field):

        phase_mask = torch.exp(1j*self.get_theta(field.shape[-self.theta.ndim:]))
        return field*phase_mask
