"""
Top-level package for the synthetic Cherenkov ring generator.

This package exposes the core `SCGen` class for synthesising events, a
`UniformDist` helper for simple uniform sampling, and a handful of utility
functions to work with datasets and produce image files suitable for object
detection training.

Example usage::

    from rich_generator import SCGen, UniformDist
    from rich_generator.dataset_utils import generate_myolo_dataset_folder

    # initialise generator and build a dataset ...
    # see README.md for a complete example.
"""

from .generator import SCGen, UniformDist
# re-export selected dataset utilities.  Use lower-case aliases for
# backward compatibility with the original project naming.  The
# underlying implementation resides in :mod:`rich_generator.dataset_utils`.
from .dataset_utils import (
    calculate_cherenkov_angle,
    calculate_cherenkov_radius,
    read_dataset,
    read_dataset_metadata,
    generate_MYOLO_dataset as generate_myolo_dataset,
    generate_MYOLO_dataset_folder as generate_myolo_dataset_folder,
)
# re-export utility functions for loading default distributions
from .utility import (
    load_default_log_momenta_kdes,
    load_default_centers,
    load_default_particle_weights,
    list_available_particle_types,
    list_available_center_distributions,
    create_scgen_with_defaults,
)

__all__ = [
    "SCGen",
    "UniformDist",
    "calculate_cherenkov_angle",
    "calculate_cherenkov_radius",
    "read_dataset",
    "read_dataset_metadata",
    "generate_myolo_dataset",
    "generate_myolo_dataset_folder",
    "load_default_log_momenta_kdes",
    "load_default_centers",
    "load_default_particle_weights",
    "list_available_particle_types",
    "list_available_center_distributions",
    "create_scgen_with_defaults",
]


__version__ = "0.1.0"