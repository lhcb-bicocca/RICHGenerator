"""Tests for balanced dataset generation and MYOLO folder creation.

This module exercises the high-level dataset generation routines
``generate_MYOLO_dataset`` and ``generate_MYOLO_dataset_folder``.  It
constructs tiny datasets using simple uniform distributions and verifies
that returned objects and on-disk outputs match expectations.  Both the
in-memory return path and the file output path are tested, as is the
conversion into image folders with train/val/test splits.
"""

import os
import numpy as np
import pytest

from rich_generator.dataset_utils import (
    generate_MYOLO_dataset,
    generate_MYOLO_dataset_folder,
    read_dataset,
    load_kde,
)
from rich_generator.utility import _get_distributions_path


def get_distribution_path(relative_path: str) -> str:
    """Get the full path to a distribution file within the package."""
    distributions_dir = _get_distributions_path()
    return str(distributions_dir / relative_path)


def test_generate_myolo_dataset_and_folder(tmp_path, simple_centers_distribution):
    """Create a tiny MYOLO dataset and verify generation and folder contents."""
    # Define parameters for a minimal dataset
    num_events = 2
    num_added_per_event = 2
    # Use a single particle type to keep things simple
    particles = [211]
    # For this test we use the KDE files bundled with the package.  When
    # ``momenta_log_distributions`` and ``centers_distribution`` are omitted
    # the ``generate_MYOLO_dataset`` function will automatically load the
    # appropriate distributions from the ``distributions`` folder.  We also
    # set a large ``N_init`` to reduce the likelihood of Poisson sampling
    # zero hits.
    kdes = {
        211: load_kde(get_distribution_path('log_momenta_kdes/211-kde.npz')),
    }
    centres_kde = load_kde(get_distribution_path('centers_R1-kde.npz'))
    common_kwargs = dict(
        particle_types=particles,
        refractive_index=1.0014,
        detector_size=((-5.0, 5.0), (-5.0, 5.0)),
        momenta_log_distributions=kdes,
        centers_distribution=centres_kde,
        radial_noise=(0.0, 0.0),
        N_init=1000,
        max_radius=50.0,
        masses={211: 0.13957},
        num_events=num_events,
        num_particles_per_event=0,
        parallel=False,
        progress_bar=False,
    )
    # Generate dataset in memory (output_file=None)
    dataset = generate_MYOLO_dataset(
        momenta_range=(1.0, 1.0),
        output_file=None,
        particles=particles,
        num_added_ring_per_event=num_added_per_event,
        **common_kwargs
    )
    # Should return a list of events
    assert isinstance(dataset, list)
    assert len(dataset) == num_events
    # Each event should now contain exactly num_added_per_event rings
    for ev in dataset:
        assert len(ev["rings"]) == num_added_per_event
        assert len(ev["tracked_rings"]) == num_added_per_event
    # Now generate a dataset on disk
    h5_path = tmp_path / "small_myolo_dataset.h5"
    result = generate_MYOLO_dataset(
        momenta_range=(1.0, 1.0),
        output_file=h5_path.as_posix(),
        particles=particles,
        num_added_ring_per_event=num_added_per_event,
        **common_kwargs
    )
    # When output_file is provided, the function returns None
    assert result is None
    assert h5_path.exists()
    # Load the saved HDF5 and verify ring counts
    loaded = read_dataset(h5_path.as_posix())
    assert len(loaded) == num_events
    for ev in loaded:
        assert len(ev["rings"]) == num_added_per_event
        assert len(ev["tracked_rings"]) == num_added_per_event
    # Prepare to build MYOLO folder
    base_dir = tmp_path / "myolo_out"
    generate_MYOLO_dataset_folder(
        data_file=h5_path.as_posix(),
        base_dir=base_dir.as_posix(),
        image_size=16,
        max_photons=1.0,
        train_val_test_split=(0.8, 0.1, 0.1),
        polar_transform=True,
        stretch_radii=False,
        seed=0,
        as_png=False,
    )
    # Verify directory structure and files
    assert base_dir.exists()
    images_dir = base_dir / "images"
    assert images_dir.exists() and images_dir.is_dir()
    train_file = base_dir / "train.txt"
    val_file = base_dir / "val.txt"
    test_file = base_dir / "test.txt"
    particles_file = base_dir / "particles.txt"
    for f in [train_file, val_file, test_file, particles_file]:
        assert f.exists()
    # Count total lines; should equal total rings (num_events * num_added_per_event)
    with open(train_file) as f:
        train_lines = f.readlines()
    with open(val_file) as f:
        val_lines = f.readlines()
    with open(test_file) as f:
        test_lines = f.readlines()
    total = len(train_lines) + len(val_lines) + len(test_lines)
    assert total == num_events * num_added_per_event
    # Check that particles.txt lists the single particle type
    with open(particles_file) as f:
        types = [int(line.strip()) for line in f if line.strip()]
    assert types == particles
    # At least one image file should exist in images_dir; load it and check shape
    npy_files = [p for p in images_dir.rglob("*.npy")]  # includes shards if any
    assert npy_files, "No image files were generated"
    sample_img = np.load(npy_files[0])
    assert sample_img.shape == (16, 16)
    assert sample_img.dtype == np.float32
    # Repeat the folder generation with polar_transform=False and stretching enabled
    base_dir2 = tmp_path / "myolo_out_cartesian"
    generate_MYOLO_dataset_folder(
        data_file=h5_path.as_posix(),
        base_dir=base_dir2.as_posix(),
        image_size=16,
        max_photons=1.0,
        train_val_test_split=(0.8, 0.1, 0.1),
        polar_transform=False,
        stretch_radii=True,
        seed=1,
        as_png=False,
    )
    # Check that the second base_dir was created
    assert base_dir2.exists()
    assert (base_dir2 / "images").exists()