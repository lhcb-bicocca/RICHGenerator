"""Tests for the whole-event image rasterisation functionality.

This module constructs a simple synthetic event with known hit positions
and verifies that the :meth:`SCGen.event_image` method correctly maps
these positions onto pixel indices.  It also checks normalisation and
output data types.
"""

import numpy as np
import pytest

from rich_generator.generator import SCGen


def test_event_image_mapping_and_normalization():
    """Render a synthetic event and verify pixel mapping and normalisation."""
    # Define a simple detector spanning [-1,1] in both axes
    detector_size = ((-1.0, 1.0), (-1.0, 1.0))
    # Build an SCGen instance with arbitrary distributions (unused here)
    # Provide explicit masses to avoid requiring the ``particle`` package.
    scgen = SCGen(
        particle_types=[211],
        refractive_index=1.0014,
        detector_size=detector_size,
        momenta_log_distributions={211: (0.0, 0.0)},
        centers_distribution=None,
        radial_noise=(0.0, 0.0),
        N_init=10,
        max_radius=50.0,
        masses={211: 0.13957},
    )
    # Construct a deterministic event: single ring with four hits in the detector corners
    ring_hits = np.array([
        [-1.0, -1.0],  # bottom-left
        [-1.0,  1.0],  # top-left
        [ 1.0, -1.0],  # bottom-right
        [ 1.0,  1.0],  # top-right
    ], dtype=float)
    event = {
        "centers": np.array([[0.0, 0.0]], dtype=float),
        "momenta": np.array([1.0], dtype=float),
        "particle_types": np.array([211], dtype=int),
        "rings": [ring_hits],
        "tracked_rings": np.array([0], dtype=int),
    }
    scgen.dataset = [event]
    # Render the event at 10-10 pixels without normalisation
    img_size = (10, 10)
    image = scgen.event_image(0, img_size, normalize=False)
    # The output should be a float32 array of the correct shape
    assert image.shape == img_size
    assert image.dtype == np.float32
    # Compute expected pixel positions
    h, w = img_size
    (x_min, x_max), (y_min, y_max) = detector_size
    sx = (w - 1) / (x_max - x_min)
    sy = (h - 1) / (y_max - y_min)
    expected_coords = []
    for x, y in ring_hits:
        x_pix = int(round((x - x_min) * sx))
        y_pix = int(round((y_max - y) * sy))
        expected_coords.append((y_pix, x_pix))
    # All expected pixels should have value 1
    for yp, xp in expected_coords:
        assert image[yp, xp] == 1.0
    # All other pixels should be zero
    total_hits = len(ring_hits)
    assert image.sum() == pytest.approx(float(total_hits))
    # Now test normalisation: all nonzero pixels become 1.0 / max
    image_norm = scgen.event_image(0, img_size, normalize=True)
    # The maximum pixel value should be 1
    assert image_norm.max() == 1.0
    # The sum should still equal the number of hits (all ones)
    assert image_norm.sum() == pytest.approx(float(total_hits))