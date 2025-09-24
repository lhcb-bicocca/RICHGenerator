"""
Utility functions for loading default KDEs and particle weights.

This module provides convenience functions to load the default distributions
bundled with the rich_generator package, including:

- KDE distributions for log-momenta of different particle types
- KDE distributions for ring centers (R1 and R2 detectors)
- Default particle type proportions

These utilities simplify the creation of :class:`SCGen` instances by providing
pre-trained distributions based on realistic LHCb RICH detector data.
"""

from __future__ import annotations

import os
import json
from typing import Dict, Any, Optional, Union

import numpy as np

__all__ = [
    "load_default_log_momenta_kdes",
    "load_default_centers",
    "load_default_particle_weights",
    "list_available_particle_types",
    "list_available_center_distributions",
]


def _get_distributions_path() -> str:
    """Get the path to the distributions folder."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(this_dir, '..', '..', 'distributions')


def list_available_particle_types() -> list[int]:
    """List all particle types for which log-momenta KDEs are available.
    
    Returns
    -------
    list[int]
        Sorted list of PDG particle codes with available KDE distributions.
    """
    distributions_path = _get_distributions_path()
    log_momenta_dir = os.path.join(distributions_path, 'log_momenta_kdes')
    
    if not os.path.exists(log_momenta_dir):
        raise FileNotFoundError(f"Log momenta KDEs directory not found: {log_momenta_dir}")
    
    particle_types = []
    for filename in os.listdir(log_momenta_dir):
        if filename.endswith('-kde.npz'):
            particle_type_str = filename.replace('-kde.npz', '')
            try:
                particle_type = int(particle_type_str)
                particle_types.append(particle_type)
            except ValueError:
                continue  # Skip invalid filenames
    
    return sorted(particle_types)


def list_available_center_distributions() -> list[str]:
    """List all available center distribution files.
    
    Returns
    -------
    list[str]
        List of available center distribution identifiers (e.g., 'R1', 'R2').
    """
    distributions_path = _get_distributions_path()
    
    if not os.path.exists(distributions_path):
        raise FileNotFoundError(f"Distributions directory not found: {distributions_path}")
    
    center_files = []
    for filename in os.listdir(distributions_path):
        if filename.startswith('centers_') and filename.endswith('-kde.npz'):
            # Extract identifier between 'centers_' and '-kde.npz'
            identifier = filename[len('centers_'):-len('-kde.npz')]
            center_files.append(identifier)
    
    return sorted(center_files)


def load_default_log_momenta_kdes(
    particle_types: Optional[list[int]] = None
) -> Dict[int, Any]:
    """Load default log-momenta KDE distributions for particle types.
    
    This function loads pre-trained KDE distributions for the log-momenta
    of various particle types based on LHCb RICH detector simulation data.
    
    Parameters
    ----------
    particle_types : list[int], optional
        List of PDG particle codes to load. If None, loads all available
        particle types. Default particle types include common particles
        like electrons (11), muons (13), pions (211), kaons (321), and 
        protons (2212).
    
    Returns
    -------
    dict[int, Any]
        Dictionary mapping PDG codes to KDE objects that implement
        the ``resample(size)`` method.
    
    Raises
    ------
    FileNotFoundError
        If the distributions directory or specific KDE files are not found.
    ValueError
        If no KDE files are found for the requested particle types.
    
    Examples
    --------
    Load KDEs for all available particle types:
    
    >>> momenta_kdes = load_default_log_momenta_kdes()
    >>> print(f"Available particles: {list(momenta_kdes.keys())}")
    
    Load KDEs for specific particles only:
    
    >>> momenta_kdes = load_default_log_momenta_kdes([211, 321, 2212])
    >>> samples = momenta_kdes[211].resample(100)  # Sample 100 log-momenta values
    """
    from .dataset_utils import load_kde
    
    distributions_path = _get_distributions_path()
    log_momenta_dir = os.path.join(distributions_path, 'log_momenta_kdes')
    
    if not os.path.exists(log_momenta_dir):
        raise FileNotFoundError(f"Log momenta KDEs directory not found: {log_momenta_dir}")
    
    # If no specific particle types requested, load all available
    if particle_types is None:
        particle_types = list_available_particle_types()
    
    momenta_distributions = {}
    missing_particles = []
    
    for ptype in particle_types:
        kde_path = os.path.join(log_momenta_dir, f'{ptype}-kde.npz')
        if os.path.exists(kde_path):
            try:
                momenta_distributions[ptype] = load_kde(kde_path)
            except Exception as e:
                raise RuntimeError(f"Failed to load KDE for particle {ptype}: {e}")
        else:
            missing_particles.append(ptype)
    
    if missing_particles and len(momenta_distributions) == 0:
        raise ValueError(f"No KDE files found for any of the requested particles: {missing_particles}")
    elif missing_particles:
        print(f"Warning: KDE files not found for particles: {missing_particles}")
    
    return momenta_distributions


def load_default_centers(detector: str = "R1") -> Any:
    """Load default center distribution KDE for RICH detectors.
    
    This function loads a pre-trained 2D KDE distribution for ring centers
    based on realistic LHCb RICH detector geometry.
    
    Parameters
    ----------
    detector : str, optional
        RICH detector identifier. Available options can be found using
        :func:`list_available_center_distributions`. Common values are
        'R1' (RICH1) and 'R2' (RICH2). Default is 'R1'.
    
    Returns
    -------
    Any
        A 2D KDE object that implements the ``resample(size)`` method.
        The resample method returns a (2, size) array with x and y coordinates.
    
    Raises
    ------
    FileNotFoundError
        If the distributions directory or the specified center KDE file
        is not found.
    ValueError
        If the detector identifier is not available.
    
    Examples
    --------
    Load default R1 center distribution:
    
    >>> centers_kde = load_default_centers()
    >>> centers = centers_kde.resample(100)  # Sample 100 center positions
    >>> print(f"Centers shape: {centers.shape}")  # Should be (2, 100)
    
    Load R2 center distribution:
    
    >>> centers_kde = load_default_centers("R2")
    """
    from .dataset_utils import load_kde
    
    distributions_path = _get_distributions_path()
    centers_file = os.path.join(distributions_path, f'centers_{detector}-kde.npz')
    
    if not os.path.exists(centers_file):
        available = list_available_center_distributions()
        raise ValueError(
            f"Center KDE file not found for detector '{detector}'. "
            f"Available options: {available}"
        )
    
    try:
        return load_kde(centers_file)
    except Exception as e:
        raise RuntimeError(f"Failed to load center KDE for detector {detector}: {e}")


def load_default_particle_weights() -> Dict[int, float]:
    """Load default particle type proportions.
    
    This function loads the default relative proportions of different
    particle types based on typical LHCb RICH detector event compositions.
    
    Returns
    -------
    dict[int, float]
        Dictionary mapping PDG particle codes to their relative proportions.
        The values sum to 1.0.
    
    Raises
    ------
    FileNotFoundError
        If the particle proportions file is not found.
    ValueError
        If the proportions file is malformed or proportions don't sum to 1.0.
    
    Examples
    --------
    Load default particle weights:
    
    >>> weights = load_default_particle_weights()
    >>> print(f"Particle weights: {weights}")
    >>> print(f"Total weight: {sum(weights.values())}")  # Should be 1.0
    """
    distributions_path = _get_distributions_path()
    proportions_file = os.path.join(distributions_path, 'particle_proportions.json')
    
    if not os.path.exists(proportions_file):
        raise FileNotFoundError(f"Particle proportions file not found: {proportions_file}")
    
    try:
        with open(proportions_file, 'r') as f:
            proportions_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in particle proportions file: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to read particle proportions file: {e}")
    
    # Convert string keys to integers and validate
    try:
        proportions = {int(k): float(v) for k, v in proportions_data.items()}
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid particle type or proportion in file: {e}")
    
    # Validate that proportions are positive and sum to approximately 1.0
    if any(p < 0 for p in proportions.values()):
        raise ValueError("All particle proportions must be non-negative")
    
    total = sum(proportions.values())
    if not np.isclose(total, 1.0, rtol=1e-6):
        # Normalize proportions to sum to 1.0
        proportions = {k: v / total for k, v in proportions.items()}
    
    return proportions


def create_scgen_with_defaults(
    particle_types: Optional[list[int]] = None,
    detector: str = "R1",
    refractive_index: float = 1.0014,
    detector_size: tuple[tuple[float, float], tuple[float, float]] = ((-600, 600), (-600, 600)),
    radial_noise: Union[float, tuple[float, float]] = 1.5,
    N_init: int = 60,
    max_radius: float = 100.0,
    **kwargs
) -> Any:
    """Create an SCGen instance with default distributions.
    
    This convenience function creates a fully configured :class:`SCGen` instance
    using the default KDE distributions and particle proportions bundled with
    the package.
    
    Parameters
    ----------
    particle_types : list[int], optional
        List of PDG particle codes. If None, uses all available particles
        from the default proportions file.
    detector : str, optional
        RICH detector identifier for center distribution ('R1' or 'R2').
        Default is 'R1'.
    refractive_index : float, optional
        Refractive index of the radiator material. Default is 1.0014.
    detector_size : tuple[tuple[float, float], tuple[float, float]], optional
        Detector boundaries as ((x_min, x_max), (y_min, y_max)) in mm.
        Default is ((-600, 600), (-600, 600)).
    radial_noise : float or tuple[float, float], optional
        Gaussian noise for ring radius. Either std deviation or (mean, std).
        Default is 1.5.
    N_init : int, optional
        Base photon yield. Default is 60.
    max_radius : float, optional
        Maximum physical ring radius in mm. Default is 100.0.
    **kwargs
        Additional parameters passed to :class:`SCGen`.
    
    Returns
    -------
    SCGen
        Configured generator instance ready for event generation.
    
    Raises
    ------
    FileNotFoundError
        If default distribution files are not found.
    ValueError
        If particle types are invalid or incompatible.
    
    Examples
    --------
    Create generator with all defaults:
    
    >>> gen = create_scgen_with_defaults()
    >>> event = gen.generate_event(100)
    
    Create generator with specific particles:
    
    >>> gen = create_scgen_with_defaults(
    ...     particle_types=[211, 321, 2212],
    ...     detector="R2"
    ... )
    """
    from .generator import SCGen
    
    # Load default particle weights first to determine particle types if not specified
    default_weights = load_default_particle_weights()
    
    if particle_types is None:
        # Use all particles from the default weights
        particle_types = list(default_weights.keys())
    
    # Ensure all requested particles have weights
    particle_weights = {pt: default_weights.get(pt, 0.0) for pt in particle_types}
    
    # Remove particles with zero weight and warn
    particle_weights = {k: v for k, v in particle_weights.items() if v > 0}
    if len(particle_weights) != len(particle_types):
        removed = set(particle_types) - set(particle_weights.keys())
        print(f"Warning: Removed particles with zero weight: {removed}")
        particle_types = list(particle_weights.keys())
    
    # Load distributions
    momenta_kdes = load_default_log_momenta_kdes(particle_types)
    centers_kde = load_default_centers(detector)
    
    # Create SCGen instance
    return SCGen(
        particle_types=particle_types,
        refractive_index=refractive_index,
        detector_size=detector_size,
        momenta_log_distributions=momenta_kdes,
        centers_distribution=centers_kde,
        radial_noise=radial_noise,
        N_init=N_init,
        max_radius=max_radius,
        particle_type_proportions=particle_weights,
        **kwargs
    )
