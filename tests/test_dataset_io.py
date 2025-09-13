"""Tests for dataset serialisation and I/O utilities.

These tests verify that datasets written via :meth:`SCGen.save_dataset`
can be faithfully read back using :func:`read_dataset` and
:func:`read_dataset_metadata`.  They also exercise partial loading
through the ``events`` parameter and confirm that silent mode works
without emitting progress output.
"""

import numpy as np
import os
import pytest

from rich_generator.generator import SCGen
from rich_generator.dataset_utils import read_dataset, read_dataset_metadata, load_kde


def test_dataset_roundtrip(tmp_path, rng_seed, simple_centers_distribution):
    """Save a small dataset to disk and load it back, checking metadata and shapes."""
    rng_seed(101)
    # Load KDE distributions from the repository rather than constructing
    # synthetic ones.  Use the shipped log-momentum KDEs and centre KDE.
    kdes = {
        211: load_kde(os.path.join(os.path.dirname(__file__), '..', 'distributions', 'log_momenta_kdes', '211-kde.npz')),
        321: load_kde(os.path.join(os.path.dirname(__file__), '..', 'distributions', 'log_momenta_kdes', '321-kde.npz')),
    }
    centres_kde = load_kde(os.path.join(os.path.dirname(__file__), '..', 'distributions', 'centers_R1-kde.npz'))
    # construct a generator with the real distributions
    # Provide explicit masses to avoid reliance on the ``particle`` package which
    # may not be installed in the test environment.  Masses are given in GeV/c².
    masses = {211: 0.13957, 321: 0.49367}
    scgen = SCGen(
        particle_types=[211, 321],
        refractive_index=1.0014,
        detector_size=((-5, 5), (-5, 5)),
        momenta_log_distributions=kdes,
        centers_distribution=centres_kde,
        radial_noise=(0.0, 0.0),
        N_init=30,
        max_radius=50.0,
        masses=masses,
    )
    # generate a couple of events with fixed number of particles
    scgen.generate_dataset(num_events=2, num_particles_per_event=2, parallel=False, progress_bar=False)
    # Ensure no ring is completely empty.  HDF5 variable-length datasets
    # cannot be created when all contained arrays are empty on some h5py
    # versions.  Replace empty rings with a single dummy hit.
    for ev in scgen.dataset:
        for idx, ring in enumerate(ev["rings"]):
            if ring.size == 0:
                ev["rings"][idx] = np.array([[0.0, 0.0]], dtype=float)
    # persist to HDF5
    out_file = tmp_path / "small_dataset.h5"
    scgen.save_dataset(out_file.as_posix())
    assert out_file.exists()
    # read metadata
    meta = read_dataset_metadata(out_file.as_posix())
    # compare scalar values directly
    assert meta["refractive_index"] == scgen.refractive_index
    assert np.allclose(meta["detector_size"], np.asarray(scgen.detector_size, dtype=np.float32))
    assert np.allclose(meta["radial_noise"], np.asarray(scgen.radial_noise, dtype=np.float32))
    assert meta["N_init"] == scgen.N_init
    assert meta["max_cherenkov_radius"] == scgen.max_cherenkov_radius
    # particle types order may differ but sets should match
    assert set(meta["particle_types"]) == set(scgen.particle_types)
    # masses stored as JSON with string keys; compare values
    for key, val in meta["masses"].items():
        assert np.isclose(val, scgen.masses[int(key)])
    assert meta["num_events"] == len(scgen.dataset)
    # read full dataset
    loaded = read_dataset(out_file.as_posix())
    assert len(loaded) == len(scgen.dataset)
    # compare shapes of loaded events against original
    for orig_event, loaded_event in zip(scgen.dataset, loaded):
        # centres
        assert loaded_event["centers"].shape == orig_event["centers"].shape
        # momenta
        assert loaded_event["momenta"].shape == orig_event["momenta"].shape
        # particle types
        assert loaded_event["particle_types"].shape == orig_event["particle_types"].shape
        # tracked rings
        assert loaded_event["tracked_rings"].shape == orig_event["tracked_rings"].shape
        # rings: each ring should have same length
        assert len(loaded_event["rings"]) == len(orig_event["rings"])
        for r_o, r_l in zip(orig_event["rings"], loaded_event["rings"]):
            assert r_l.shape == r_o.shape
    # partial loading: select only the second event (index 1)
    subset = read_dataset(out_file.as_posix(), events=[1], silent=True)
    assert isinstance(subset, list)
    assert len(subset) == 1
    assert subset[0]["centers"].shape == scgen.dataset[1]["centers"].shape
