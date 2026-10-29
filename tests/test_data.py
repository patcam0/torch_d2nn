import torch

from mplc.data import FieldDataset, RandomFieldDataset, get_basis_fields_pos


def test_position_gaussians_have_expected_centers_and_norms():
    # Cover even and odd dimensions and grid sizes, with resolved peaks.
    spacing, waist, separation = 1e-6, 20e-6, 3
    for dimension in (4, 5):
        for shape in (1000, 1001):
            fields = get_basis_fields_pos(
                dimension, shape, spacing, waist, 'cpu', separation
            )
            assert fields.shape == (dimension, shape)
            assert fields.dtype == torch.complex64
            torch.testing.assert_close(
                fields.abs().square().sum(dim=1), torch.ones(dimension)
            )
            peaks = (fields.abs().argmax(dim=1) - shape // 2) * spacing
            expected = (torch.arange(dimension) - (dimension - 1) / 2) * separation * waist
            torch.testing.assert_close(peaks, expected, atol=spacing / 2, rtol=0)


def test_position_gaussian_waist_controls_width():
    for waist in (10e-6, 20e-6):
        field = get_basis_fields_pos(1, 1001, 1e-6, waist, 'cpu')[0]
        # One waist from the center, amplitude must fall to exp(-1).
        offset = round(waist / 1e-6)
        ratio = field[500 + offset].abs() / field[500].abs()
        torch.testing.assert_close(ratio, torch.exp(torch.tensor(-1.0)))


def test_datasets_align_cpu_states_with_basis_device_and_dtype():
    devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
    for device in devices:
        for dtype in (torch.complex64, torch.complex128):
            basis = torch.eye(4, device=device, dtype=dtype)
            target = torch.roll(torch.eye(4), 1, 0)
            datasets = [
                FieldDataset(target, basis),
                RandomFieldDataset(target, basis, 16, seed=7),
            ]
            for dataset in datasets:
                for tensor in (dataset.target, dataset.inputs, dataset.outputs):
                    assert tensor.device == basis.device
                    assert tensor.dtype == dtype
                torch.testing.assert_close(
                    dataset.outputs, torch.roll(dataset.inputs, 1, dims=1)
                )
