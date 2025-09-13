"""Tests for metadata comparison utilities and input validation.

This module covers the private ``_compare_metadata`` helper to ensure
order-insensitive comparison of metadata dictionaries and exercises
argument validation logic in ``generate_MYOLO_dataset_folder``.  The
tests deliberately supply malformed inputs to confirm that
``ValueError`` is raised when required conditions are not met.
"""

import numpy as np
import pytest

from rich_generator.dataset_utils import _compare_metadata, generate_MYOLO_dataset_folder
from rich_generator.generator import SCGen


def test_compare_metadata_matches_and_mismatches():
    """Metadata comparison should ignore ordering but detect genuine differences."""
    meta1 = {
        "refractive_index": 1.0014,
        "detector_size": np.array([[-1.0, 1.0], [-2.0, 2.0]], dtype=np.float32),
        "radial_noise": np.array([0.0, 1.0], dtype=np.float32),
        "N_init": 50,
        "max_cherenkov_radius": 100.0,
        "particle_types": [211, 321],
        "masses": {"211": 0.13957, "321": 0.49367},
        "num_events": 10,
    }
    # meta2 has arrays and lists in different order but is otherwise identical
    meta2 = {
        "refractive_index": 1.0014,
        "detector_size": np.array([[-2.0, 2.0], [-1.0, 1.0]], dtype=np.float32),
        "radial_noise": np.array([0.0, 1.0], dtype=np.float32),
        "N_init": 50,
        "max_cherenkov_radius": 100.0,
        "particle_types": [321, 211],
        "masses": {"321": 0.49367, "211": 0.13957},
        "num_events": 10,
    }
    match, mismatches = _compare_metadata(meta1, meta2)
    assert match is True
    assert mismatches == []
    # meta3 differs in refractive index
    meta3 = meta1.copy()
    meta3["refractive_index"] = 1.01
    match2, mismatches2 = _compare_metadata(meta1, meta3)
    assert match2 is False
    assert mismatches2 and any("refractive_index" in m for m in mismatches2)


def test_generate_myolo_dataset_folder_invalid_arguments(tmp_path):
    """Invalid arguments to ``generate_MYOLO_dataset_folder`` should raise ValueError."""
    # Create a tiny valid dataset file for basic validation
    # Provide a minimal centres distribution instead of None to avoid AttributeError
    class _DummyCenters:
        def resample(self, size: int):
            return np.zeros((2, size), dtype=float)
    scgen = SCGen(
        particle_types=[211],
        refractive_index=1.0014,
        detector_size=((-1.0, 1.0), (-1.0, 1.0)),
        momenta_log_distributions={211: (0.0, 0.0)},
        centers_distribution=_DummyCenters(),
        radial_noise=(0.0, 0.0),
        N_init=10,
        max_radius=50.0,
        masses={211: 0.13957},
    )
    scgen.generate_dataset(num_events=1, num_particles_per_event=0, parallel=False, progress_bar=False)
    valid_path = tmp_path / "valid_dataset.h5"
    scgen.save_dataset(valid_path.as_posix())
    # Non-existent data file
    with pytest.raises(ValueError):
        generate_MYOLO_dataset_folder(
            data_file=str(tmp_path / "nonexistent.h5"),
            base_dir=str(tmp_path / "foo"),
        )
    # Invalid split length
    with pytest.raises(ValueError):
        generate_MYOLO_dataset_folder(
            data_file=valid_path.as_posix(),
            base_dir=(tmp_path / "foo1").as_posix(),
            train_val_test_split=(0.8, 0.2),
        )
    # Split does not sum to 1
    with pytest.raises(ValueError):
        generate_MYOLO_dataset_folder(
            data_file=valid_path.as_posix(),
            base_dir=(tmp_path / "foo2").as_posix(),
            train_val_test_split=(0.5, 0.4, 0.2),
        )
    # Negative or zero image size
    with pytest.raises(ValueError):
        generate_MYOLO_dataset_folder(
            data_file=valid_path.as_posix(),
            base_dir=(tmp_path / "foo3").as_posix(),
            image_size=-16,
        )
    # Non-positive max_photons
    with pytest.raises(ValueError):
        generate_MYOLO_dataset_folder(
            data_file=valid_path.as_posix(),
            base_dir=(tmp_path / "foo4").as_posix(),
            max_photons=0.0,
        )
    # Seed must be an integer
    with pytest.raises(ValueError):
        generate_MYOLO_dataset_folder(
            data_file=valid_path.as_posix(),
            base_dir=(tmp_path / "foo5").as_posix(),
            seed=1.234,
        )