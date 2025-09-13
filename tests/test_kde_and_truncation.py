"""Tests for KDE sampling, truncation and serialisation.

This module exercises the :class:`TruncatedKDE` class and the
``save_kde``/``load_kde`` functions from ``rich_generator.dataset_utils``.  The
tests ensure that samples respect truncation bounds, runtime errors are
raised when appropriate and that KDE objects can be saved and reloaded
without loss of functionality or dimensionality.
"""

import numpy as np
import pytest

from rich_generator.dataset_utils import TruncatedKDE, save_kde, load_kde
from scipy.stats import gaussian_kde


def test_truncated_kde_resample_bounds():
    """Samples drawn from a TruncatedKDE should lie within the specified bounds."""
    # Create a simple 1D dataset centred on 0
    data = np.linspace(-1.0, 1.0, 200)
    # The KDE expects shape (d,N)
    dataset = data.reshape(1, -1)
    kde = TruncatedKDE(dataset, truncate_range=(0.0, 0.5))
    samples = kde.resample(1000)
    # Shape should be (1, N)
    assert samples.shape == (1, 1000)
    # All samples should respect the truncation range
    assert (samples >= 0.0).all() and (samples <= 0.5).all()


def test_truncated_kde_runtime_error_on_small_region():
    """A RuntimeError should be raised when the truncation region has negligible acceptance."""
    # Create a dataset far away from the truncation region
    data = np.random.normal(loc=0.0, scale=0.1, size=200)
    dataset = data.reshape(1, -1)
    kde = TruncatedKDE(dataset, truncate_range=(5.0, 6.0))
    # Attempt to sample with a very low iteration cap to force failure
    with pytest.raises(RuntimeError):
        kde.resample(size=5, max_iter=1)


def test_save_and_load_kde_roundtrip(tmp_path):
    """Verify that KDE objects survive a save/load cycle and maintain functionality."""
    # Build a standard gaussian_kde on a simple dataset
    x = np.random.randn(500)
    dataset = x.reshape(1, -1)
    kde_orig = gaussian_kde(dataset)
    # Save and reload
    out_file = tmp_path / "kde.npz"
    save_kde(kde_orig, out_file.as_posix())
    kde_loaded = load_kde(out_file.as_posix())
    # Loaded object should be a gaussian_kde instance
    assert isinstance(kde_loaded, gaussian_kde)
    # Resample and check shape/dimensionality
    samples = kde_loaded.resample(10)
    assert samples.shape == (1, 10)
    # Save and reload a truncated KDE
    truncated = TruncatedKDE(dataset, truncate_range=(0.0, 1.0))
    out_file2 = tmp_path / "kde_trunc.npz"
    save_kde(truncated, out_file2.as_posix())
    truncated_loaded = load_kde(out_file2.as_posix())
    # The returned object should have the truncate_range attribute set
    assert isinstance(truncated_loaded, TruncatedKDE)
    assert truncated_loaded.truncate_range == truncated.truncate_range
    # Samples should obey the truncation range
    samples_trunc = truncated_loaded.resample(20)
    assert (samples_trunc >= 0.0).all() and (samples_trunc <= 1.0).all()