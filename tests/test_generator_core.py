"""Tests for the core event generator functionality.

This module contains unit tests that exercise the public API of the
``rich_generator.generator`` module.  The focus is on verifying the
structure of the data produced by the generator, correct handling of
Cherenkov thresholds, and basic sampling behaviour of helper classes
such as :class:`UniformDist`.  Integration with internal helpers is
covered via the ``insert_rings`` method, which implicitly uses the
private ``_create_rings`` routine.
"""

import math
import numpy as np
import pytest

from rich_generator.generator import SCGen, UniformDist
from rich_generator.dataset_utils import (
    calculate_cherenkov_angle,
    calculate_cherenkov_radius,
)


def test_generate_event_output_structure(rng_seed, simple_centers_distribution):
    """Verify that ``SCGen.generate_event`` returns a well-formed event.

    The event dictionary should contain the expected keys and the types
    and shapes of each entry should be consistent.  Seeding the RNG
    ensures reproducibility.
    """
    rng_seed(123)
    # provide a deterministic log-momentum distribution: logs all zero → p=1
    momenta_log_dists = {211: (0.0, 0.0)}
    scgen = SCGen(
        particle_types=[211],
        refractive_index=1.0014,
        detector_size=((-10, 10), (-10, 10)),
        momenta_log_distributions=momenta_log_dists,
        centers_distribution=simple_centers_distribution,
        radial_noise=(0.0, 0.0),
        N_init=20,
        max_radius=100.0,
        masses={211: 0.13957},
    )
    event = scgen.generate_event(3)
    # expected keys
    assert set(event.keys()) == {
        "centers",
        "momenta",
        "particle_types",
        "rings",
        "tracked_rings",
    }
    # array shapes
    assert event["centers"].shape == (3, 2)
    assert event["momenta"].shape == (3,)
    assert event["particle_types"].shape == (3,)
    assert isinstance(event["rings"], list) and len(event["rings"]) == 3
    assert event["tracked_rings"].shape == (3,)
    # types
    assert event["centers"].dtype.kind == "f"
    assert event["momenta"].dtype.kind == "f"
    assert event["particle_types"].dtype.kind in ("i", "u")
    # ring shapes and dtypes
    for ring in event["rings"]:
        assert ring.ndim == 2 and ring.shape[1] == 2
        # zero-hit rings are empty float arrays
        assert ring.dtype.kind == "f"


def test_cherenkov_threshold_and_ultra_relativistic():
    """Check Cherenkov angle and radius helpers near threshold and in the ultra-relativistic limit."""
    # choose a realistic pion mass and refractive index
    m = 0.13957  # GeV/c²
    n = 1.0014
    # threshold momentum: p_th = m / sqrt(n^2 - 1)
    p_threshold = m / math.sqrt(n * n - 1.0)
    # slightly below threshold → no Cherenkov radiation
    p_below = 0.99 * p_threshold
    theta_below = calculate_cherenkov_angle(p_below, m, n)
    radius_below = calculate_cherenkov_radius(theta_below, n, MAX_RADIUS=100.0)
    assert theta_below == 0.0
    assert radius_below == 0.0
    # slightly above threshold → positive angle and radius
    p_above = 1.1 * p_threshold
    theta_above = calculate_cherenkov_angle(p_above, m, n)
    radius_above = calculate_cherenkov_radius(theta_above, n, MAX_RADIUS=100.0)
    assert theta_above > 0.0
    assert radius_above > 0.0
    # ultra-relativistic momentum approximates the maximum radius
    p_high = 1e6
    theta_high = calculate_cherenkov_angle(p_high, m, n)
    radius_high = calculate_cherenkov_radius(theta_high, n, MAX_RADIUS=100.0)
    assert pytest.approx(radius_high, rel=1e-3) == 100.0


def test_uniformdist_resample_bounds():
    """Ensure that ``UniformDist.resample`` samples within the requested bounds."""
    dist = UniformDist(0.2, 1.8)
    samples = dist.resample(1000)
    # correct shape and dtype
    assert samples.shape == (1000,)
    assert samples.dtype.kind == "f"
    # within bounds (inclusive of floating errors)
    assert samples.min() >= 0.2 - 1e-8
    assert samples.max() <= 1.8 + 1e-8


def test_insert_rings_zero_and_positive(simple_centers_distribution):
    """Verify that ``insert_rings`` correctly handles zero and positive hit counts."""
    # Build a generator with no background rings
    scgen = SCGen(
        particle_types=[211],
        refractive_index=1.0014,
        detector_size=((-5, 5), (-5, 5)),
        momenta_log_distributions={211: (0.0, 0.0)},
        centers_distribution=simple_centers_distribution,
        radial_noise=(0.0, 0.0),
        N_init=10,
        max_radius=50.0,
        masses={211: 0.13957},
    )
    # generate a dataset with one event and no particles
    scgen.generate_dataset(num_events=1, num_particles_per_event=0, parallel=False, progress_bar=False)
    assert len(scgen.dataset) == 1
    assert len(scgen.dataset[0]["rings"]) == 0
    # prepare insertion parameters: one ring at the origin with p=1 GeV
    centres = np.array([[0.0, 0.0]])
    momenta = np.array([1.0])
    ptypes = np.array([211])
    # insert a ring with zero hits
    scgen.insert_rings(event=0, centers=centres, momenta=momenta, particle_types=ptypes, N_hits=np.array([0]))
    assert len(scgen.dataset[0]["rings"]) == 1
    assert scgen.dataset[0]["rings"][0].shape == (0, 2)
    # insert another ring with three hits
    scgen.insert_rings(event=0, centers=centres, momenta=momenta, particle_types=ptypes, N_hits=np.array([3]))
    assert len(scgen.dataset[0]["rings"]) == 2
    last_ring = scgen.dataset[0]["rings"][-1]
    assert last_ring.shape == (3, 2)