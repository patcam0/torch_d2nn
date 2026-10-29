"""Run with pytest or python -m unittest discover -s tests -p test_mplc_2d.py."""
import functools
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from mplc import MPLC
from mplc import functional as mf
from mplc.data import FieldDataset, RandomFieldDataset, encode, decode


def make_model(shape=(8, 12), **kwargs):
    options = dict(shape=shape, spacing=1e-5, mask_spacing=2e-5,
                   num_masks=2, propagation_distance=0.01,
                   beam_waist=1e-4, wavelength=810e-9, device='cpu')
    options.update(kwargs)
    return MPLC(**options)


class TestMPLC2D(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_rectangular_forward_batch_and_gradients(self):
        for shape in [(8, 12), (9, 7), (1, 8), (2, 2)]:
            for leading in [(), (3,), (2, 3)]:
                with self.subTest(shape=shape, leading=leading):
                    model = make_model(shape, mask_spacing=1e-5 if min(shape) < 2 else 2e-5)
                    field = torch.randn(*leading, *shape, dtype=torch.complex64)
                    output = model(field)
                    self.assertEqual(output.shape, field.shape)
                    self.assertTrue(torch.isfinite(output).all())
                    if leading:
                        torch.testing.assert_close(output[0], model(field[0]))
                    loss = output.abs().square().sum() + model.smoothness_penalty()
                    loss.backward()
                    for mask in model.phase_masks:
                        self.assertTrue(torch.isfinite(mask.theta.grad).all())

    def test_anisotropic_propagation_matches_explicit_dft(self):
        model = make_model((5, 8), spacing=(2e-5, 1e-5), mask_spacing=(2e-5, 1e-5))
        p = model.propagators[0]
        field = torch.randn(5, 8, dtype=torch.complex128)
        padded = torch.nn.functional.pad(field, (4, 4, 2, 2))
        ny, nx = padded.shape
        def dft(n):
            indices = torch.arange(n, dtype=torch.float64)
            return torch.exp(-2j * torch.pi / n * indices[:, None] * indices[None, :])
        fy = torch.fft.fftfreq(ny, d=float(model.spacing[0]), dtype=torch.float64)
        fx = torch.fft.fftfreq(nx, d=float(model.spacing[1]), dtype=torch.float64)
        kz = torch.sqrt(((2 * torch.pi / p.wavelength)**2
                         - (2 * torch.pi * fy[:, None])**2
                         - (2 * torch.pi * fx[None, :])**2).to(field.dtype))
        h = torch.exp(1j * kz * p.propagation_distance)
        wy, wx = dft(ny), dft(nx)
        expected = wy.conj().T @ ((wy @ padded @ wx.T) * h) @ wx.conj() / (ny * nx)
        torch.testing.assert_close(p(field), expected[2:7, 4:12], atol=1e-10, rtol=1e-10)

    def test_zero_distance_and_1d_dispatch(self):
        for shape in [16, (16,), torch.tensor(16), (7, 10)]:
            model = make_model(shape, propagation_distance=0)
            field = torch.randn(3, *model.spatial_shape, dtype=torch.complex64)
            propagated = model.propagators[0](field)
            torch.testing.assert_close(propagated, field, atol=1e-6, rtol=1e-6)
            if model.spatial_ndim == 1:
                torch.testing.assert_close(propagated, model.propagators[0].asm1d(field))

    def test_phase_interpolation_crosses_wrap_smoothly(self):
        phase = torch.tensor([[2.8, -2.8], [2.9, -2.7]], requires_grad=True)
        resized = mf.interpolate_phase(phase, (3, 5))
        # Midpoint across the discontinuity stays near pi, not zero.
        self.assertGreater(float(resized[1, 2].abs()), 3.0)
        resized.sin().sum().backward()
        self.assertTrue(torch.isfinite(phase.grad).all())

    def test_1d_interpolation_regression(self):
        phase = torch.tensor([2.8, -2.9, -2.3], dtype=torch.float64)
        delta = (phase[1:] - phase[:-1] + torch.pi) % (2 * torch.pi) - torch.pi
        unwrapped = torch.cat((phase[:1], phase[:1] + delta.cumsum(0)))
        old = torch.nn.functional.interpolate(unwrapped[None, None], size=11,
                                               mode='linear', align_corners=True)[0, 0]
        expected = (old + torch.pi) % (2 * torch.pi) - torch.pi
        torch.testing.assert_close(mf.interpolate_phase(phase, 11), expected)

    def test_smoothness_covers_both_axes(self):
        model = make_model((5, 7), mask_spacing=1e-5, num_masks=1)
        for axis in [0, 1]:
            with torch.no_grad():
                values = 0.02 * torch.arange(model.spatial_shape[axis]).square()
                theta = values[:, None] if axis == 0 else values[None, :]
                model.phase_masks[0].theta.copy_(theta.expand(model.spatial_shape))
            torch.testing.assert_close(model.smoothness_penalty(), model.smooth_weight * 0.04)

    def test_field_helpers_reduce_spatial_axes(self):
        field = torch.randn(3, 5, 7, dtype=torch.complex64)
        normalized = mf.normalize_field(field, spatial_ndim=2)
        torch.testing.assert_close(normalized.abs().square().sum((-2, -1)), torch.ones(3))
        torch.testing.assert_close(mf.field_norm(normalized, 2), torch.ones(3))
        torch.testing.assert_close(mf.field_overlap_loss(normalized, normalized, 2), torch.tensor(0.), atol=1e-6, rtol=0)

    def test_train_and_evaluate(self):
        model = make_model()
        field = mf.normalize_field(torch.randn(4, 8, 12, dtype=torch.complex64), 2)
        loader = DataLoader(TensorDataset(field, field), batch_size=2)
        loss = functools.partial(mf.field_overlap_loss, spatial_ndim=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        before = [mask.theta.detach().clone() for mask in model.phase_masks]
        model.train_mplc(1, loader, optimizer, 'cpu', loss)
        self.assertEqual(model.epochs, 1)
        self.assertTrue(np.isfinite(model.test(loader, loss)))
        self.assertTrue(any(not torch.equal(a, b.theta) for a, b in zip(before, model.phase_masks)))

    def test_datasets_encode_decode_2d_basis(self):
        basis = torch.eye(6, dtype=torch.complex64).reshape(6, 2, 3)
        target = torch.roll(torch.eye(6), 1, 0)
        for dataset in [FieldDataset(target, basis), RandomFieldDataset(target, basis, 9, seed=7)]:
            inputs = decode(dataset.inputs, basis)
            outputs = decode(dataset.outputs, basis)
            torch.testing.assert_close(outputs, torch.roll(inputs, 1, dims=1))
            torch.testing.assert_close(encode(inputs, basis), dataset.inputs)
            torch.testing.assert_close(dataset.inputs.abs().square().sum((-2, -1)),
                                       torch.ones(len(dataset)), atol=1e-6, rtol=1e-6)

    def test_csv_round_trip(self):
        for shape, spacing in [(16, 2e-5), ((9, 13), 2e-5), ((5, 7), 1e-5), ((1, 7), 1e-5)]:
            with self.subTest(shape=shape, spacing=spacing), tempfile.TemporaryDirectory() as root:
                model = make_model(shape, mask_spacing=spacing)
                if spacing == 1e-5:
                    with torch.no_grad():
                        for mask in model.phase_masks:
                            mask.theta.uniform_(-torch.pi, torch.pi)
                original = [mask.get_theta(model.spatial_shape).detach().clone() for mask in model.phase_masks]
                folder = Path(root) / 'masks'
                model.export_masks(folder)
                parameters = [mask.theta for mask in model.phase_masks]
                with torch.no_grad():
                    for mask in model.phase_masks:
                        mask.theta.zero_()
                model.import_masks(folder)
                for parameter, mask, target in zip(parameters, model.phase_masks, original):
                    self.assertIs(parameter, mask.theta)
                    difference = torch.angle(torch.exp(1j * (mask.get_theta(model.spatial_shape) - target)))
                    self.assertLess(float(difference.abs().max()), 1e-4)

    def test_failed_import_rolls_back_all_parameters(self):
        model = make_model((8, 12))
        original = [mask.theta.detach().clone() for mask in model.phase_masks]
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / 'masks'
            model.export_masks(folder)
            values = np.loadtxt(folder / 'phase_2.csv', delimiter=',')
            values[3, 4] += 0.2  # Not representable by this coarse mask.
            np.savetxt(folder / 'phase_2.csv', values, delimiter=',')
            with self.assertRaisesRegex(ValueError, 'reconstruction failed'):
                model.import_masks(folder)
        for mask, target in zip(model.phase_masks, original):
            torch.testing.assert_close(mask.theta, target)

    def test_visualization_1d_and_2d(self):
        for shape in [16, (5, 8)]:
            model = make_model(shape)
            field = torch.randn(*model.spatial_shape, dtype=torch.complex64)
            model.visualize_model(show=False).canvas.draw()
            for far in [False, True]:
                model.visualize_field(field, show=False, far_field=far).canvas.draw()
            with patch('matplotlib.pyplot.show'):
                model.vizualize_phase_gradient()

    def test_validation_and_device_buffers(self):
        for shape in [(3, 4, 5), (0, 4), 4.5]:
            with self.assertRaises(ValueError):
                make_model(shape)
        model = make_model()
        with self.assertRaises(ValueError):
            model(torch.ones(3, 8, 12))
        with self.assertRaises(ValueError):
            model(torch.ones(3, 12, 8, dtype=torch.complex64))
        model.to('cpu')
        self.assertEqual(model.device, torch.device('cpu'))
        for name in ['spacing', 'wavelength', 'propagation_distance']:
            self.assertIn(name, dict(model.propagators[0].named_buffers()))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_construction_and_device_transfer(self):
        for shape, spacing in [(16, 1e-5), ((7, 10), (2e-5, 1e-5))]:
            for device in ['cpu', 'cuda']:
                model = make_model(shape, spacing=spacing, mask_spacing=2e-5, device=device).to('cuda')
                field = torch.randn(2, *model.spatial_shape, dtype=torch.complex64, device='cuda')
                output = model(field)
                self.assertEqual(output.device.type, 'cuda')
                loss = mf.field_overlap_loss(output, field, model.spatial_ndim) + model.smoothness_penalty()
                loss.backward()
                self.assertTrue(all(torch.isfinite(mask.theta.grad).all() for mask in model.phase_masks))
                model.to('cpu')
                self.assertEqual(model.device.type, 'cpu')
                torch.testing.assert_close(model(field.cpu()), output.cpu(), atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
