"""Pytest configuration and helper fixtures for the rich-generator tests.

This file defines a few reusable fixtures that simplify the construction of
deterministic synthetic data for testing the public API of the package.  In
particular we provide utilities to seed the global NumPy RNG, build simple
centre distributions with predictable outputs and create small KDE objects
without having to rely on the heavy distributions shipped with the package.

Fixtures:

* ``rng_seed`` - helper to deterministically seed ``numpy.random``.  Many
  tests rely on reproducible random numbers; by calling this fixture at the
  start of a test you ensure repeatability.

* ``simple_centers_distribution`` - returns a minimal distribution object
  exposing a ``resample(size)`` method that produces ring centres on demand.
  The default implementation returns centres at the origin; individual tests
  may subclass or monkeypatch this fixture to customise the behaviour.

* ``small_kde_factory`` - a factory returning either a standard
  ``scipy.stats.gaussian_kde`` or the package's ``TruncatedKDE`` depending
  on whether a truncation range is provided.  This utility facilitates
  constructing lightweight KDEs entirely in memory for tests that exercise
  KDE sampling and serialisation logic.
"""

import numpy as np
import pytest

from rich_generator.dataset_utils import TruncatedKDE
from scipy.stats import gaussian_kde


@pytest.fixture
def rng_seed():
    """Return a callable that seeds NumPy's RNG.

    Many of the generator functions rely on the global random state of
    ``numpy.random``.  To make tests deterministic you can call
    ``rng_seed(some_int)`` at the beginning of a test.  The returned
    value is the seed that was used which can be asserted if desired.
    """

    def _seed(seed: int) -> int:
        np.random.seed(seed)
        return seed

    return _seed


@pytest.fixture
def simple_centers_distribution():
    """Provide a trivial centres distribution for tests.

    The returned object implements a ``resample(size)`` method that
    returns a 2-``size`` array of zeroes.  The transpose of this array
    yields an (N,2) array of (x,y) centres all located at the origin.
    Tests that require different centre positions can override this
    fixture or monkeypatch the ``resample`` method at runtime.
    """

    class _SimpleCenters:
        def resample(self, size: int) -> np.ndarray:
            # shape (2, size) as expected by SCGen
            return np.zeros((2, size), dtype=float)

    return _SimpleCenters()


@pytest.fixture
def small_kde_factory():
    """Return a factory for small KDE objects.

    The returned function accepts a 2D array ``data`` of shape (N, d)
    representing N samples in d dimensions and an optional ``truncate_range``
    argument.  When ``truncate_range`` is ``None`` the factory returns a
    standard ``gaussian_kde`` built from the transposed data; otherwise it
    returns a ``TruncatedKDE`` with the provided bounds.  This helper
    avoids touching the on-disk distributions shipped with the package.
    """

    def factory(data: np.ndarray, truncate_range=None):
        # ensure data is two dimensional (N,d) for convenience
        data = np.atleast_2d(data)
        if data.ndim == 1:
            data = data[:, None]
        # scipy and our implementation expect shape (d,N)
        dataset = data.T
        if truncate_range is None:
            return gaussian_kde(dataset)
        else:
            return TruncatedKDE(dataset, truncate_range=truncate_range)

    return factory