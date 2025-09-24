"""Tests for the utility functions for loading default distributions.

This module tests the utility functions that load default KDE distributions
for log-momenta, centers, and particle type proportions from the bundled
distributions folder.
"""

import pytest
import numpy as np

from rich_generator.utility import (
    list_available_particle_types,
    list_available_center_distributions,
    load_default_log_momenta_kdes,
    load_default_centers,
    load_default_particle_weights,
    create_scgen_with_defaults,
)


def test_list_available_particle_types():
    """Test that list_available_particle_types returns a valid list of integers."""
    particle_types = list_available_particle_types()
    
    assert isinstance(particle_types, list)
    assert len(particle_types) > 0
    assert all(isinstance(pt, int) for pt in particle_types)
    # Should be sorted
    assert particle_types == sorted(particle_types)
    
    # Common particles should be present
    common_particles = [11, 13, 211, 321, 2212]  # e-, mu-, pi+, K+, p
    for pt in common_particles:
        assert pt in particle_types


def test_list_available_center_distributions():
    """Test that list_available_center_distributions returns valid identifiers."""
    center_dists = list_available_center_distributions()
    
    assert isinstance(center_dists, list)
    assert len(center_dists) >= 1  # At least R1 should be available
    assert all(isinstance(cd, str) for cd in center_dists)
    # Should be sorted
    assert center_dists == sorted(center_dists)
    
    # R1 should be available at minimum
    assert 'R1' in center_dists


def test_load_default_particle_weights():
    """Test loading default particle weights."""
    weights = load_default_particle_weights()
    
    assert isinstance(weights, dict)
    assert len(weights) > 0
    
    # All keys should be integers (PDG codes)
    assert all(isinstance(k, int) for k in weights.keys())
    # All values should be positive floats
    assert all(isinstance(v, float) and v > 0 for v in weights.values())
    
    # Total should sum to approximately 1.0
    total = sum(weights.values())
    assert pytest.approx(total, abs=1e-6) == 1.0
    
    # Common particles from the default file should be present
    assert 11 in weights    # electron
    assert 13 in weights    # muon  
    assert 211 in weights   # pion
    assert 321 in weights   # kaon
    assert 2212 in weights  # proton


def test_load_default_log_momenta_kdes():
    """Test loading default log-momenta KDEs."""
    # Test loading all available KDEs
    all_kdes = load_default_log_momenta_kdes()
    
    assert isinstance(all_kdes, dict)
    assert len(all_kdes) > 0
    
    # All keys should be integers
    assert all(isinstance(k, int) for k in all_kdes.keys())
    # All values should have resample method
    assert all(hasattr(kde, 'resample') for kde in all_kdes.values())
    
    # Test specific particle types
    specific_particles = [211, 321, 2212]  # pion, kaon, proton
    specific_kdes = load_default_log_momenta_kdes(specific_particles)
    
    assert len(specific_kdes) == len(specific_particles)
    for pt in specific_particles:
        assert pt in specific_kdes
        # Test that we can sample from the KDE
        samples = specific_kdes[pt].resample(10)
        assert samples.shape == (1, 10)  # KDEs return (d, N) arrays
        assert np.all(np.isfinite(samples))


def test_load_default_log_momenta_kdes_invalid_particles():
    """Test loading KDEs for particles that don't exist."""
    # Test with some invalid particle types
    invalid_particles = [999999, 123456]  # Non-existent particle codes
    
    # Should warn about missing particles but not raise error if some valid ones exist
    mixed_particles = [211, 999999, 321]  # mix of valid and invalid
    kdes = load_default_log_momenta_kdes(mixed_particles)
    
    # Should get KDEs for the valid particles
    assert 211 in kdes
    assert 321 in kdes
    assert 999999 not in kdes
    
    # Test with only invalid particles - should raise ValueError
    with pytest.raises(ValueError, match="No KDE files found"):
        load_default_log_momenta_kdes(invalid_particles)


def test_load_default_centers():
    """Test loading default center distributions."""
    # Test loading R1 (default)
    centers_r1 = load_default_centers()
    assert hasattr(centers_r1, 'resample')
    
    # Test sampling
    samples = centers_r1.resample(5)
    assert samples.shape == (2, 5)  # 2D centers: (x, y) for 5 samples
    assert np.all(np.isfinite(samples))
    
    # Test loading R1 explicitly
    centers_r1_explicit = load_default_centers("R1")
    assert hasattr(centers_r1_explicit, 'resample')
    
    # If R2 is available, test it too
    available_centers = list_available_center_distributions()
    if 'R2' in available_centers:
        centers_r2 = load_default_centers("R2")
        assert hasattr(centers_r2, 'resample')
        samples_r2 = centers_r2.resample(3)
        assert samples_r2.shape == (2, 3)


def test_load_default_centers_invalid():
    """Test loading centers with invalid detector name."""
    with pytest.raises(ValueError, match="Center KDE file not found"):
        load_default_centers("InvalidDetector")


def test_create_scgen_with_defaults():
    """Test creating SCGen with default distributions."""
    # Test with all defaults
    gen = create_scgen_with_defaults()
    
    assert hasattr(gen, 'particle_types')
    assert hasattr(gen, 'refractive_index')
    assert hasattr(gen, 'centers_distribution')
    assert hasattr(gen, 'momenta_log_distributions')
    
    # Check that default values are set correctly
    assert gen.refractive_index == 1.0014
    assert gen.N_init == 60
    assert gen.max_cherenkov_radius == 100.0
    
    # Check that we have particle types and they match proportions
    assert len(gen.particle_types) > 0
    assert gen.particle_type_proportions is not None
    assert len(gen.particle_type_proportions) == len(gen.particle_types)
    
    # Test that we can generate events
    event = gen.generate_event(3)
    assert 'centers' in event
    assert 'momenta' in event
    assert 'particle_types' in event
    assert 'rings' in event
    
    # Test with specific particles
    gen_specific = create_scgen_with_defaults(
        particle_types=[211, 321, 2212],
        detector="R1",
        refractive_index=1.002,
        N_init=50
    )
    
    assert gen_specific.particle_types == [211, 321, 2212]
    assert gen_specific.refractive_index == 1.002
    assert gen_specific.N_init == 50
    
    # Check that proportions are normalized for the specific particles
    total_prop = np.sum(gen_specific.particle_type_proportions)
    assert pytest.approx(total_prop, abs=1e-6) == 1.0


def test_create_scgen_with_defaults_invalid_particles():
    """Test creating SCGen with invalid particle types."""
    # Test with particles not in default weights
    invalid_particles = [999999]
    
    # Should remove particles with zero weight and warn
    gen = create_scgen_with_defaults(particle_types=invalid_particles + [211])
    
    # Should only have the valid particle
    assert 211 in gen.particle_types
    assert 999999 not in gen.particle_types


def test_kde_sampling_consistency():
    """Test that KDEs produce consistent, reasonable samples."""
    # Load a specific KDE and test its properties
    kdes = load_default_log_momenta_kdes([211])  # pion
    pion_kde = kdes[211]
    
    # Generate multiple samples and check consistency
    samples1 = pion_kde.resample(100)
    samples2 = pion_kde.resample(100)
    
    # Should have correct shape
    assert samples1.shape == (1, 100)
    assert samples2.shape == (1, 100)
    
    # Should be finite numbers
    assert np.all(np.isfinite(samples1))
    assert np.all(np.isfinite(samples2))
    
    # For log-momenta, values should be reasonable (typically 0-15 range)
    assert np.all(samples1 >= -2)  # log(0.1) ≈ -2.3
    assert np.all(samples1 <= 15)  # log(3e6) ≈ 15
    assert np.all(samples2 >= -2) 
    assert np.all(samples2 <= 15)


def test_integration_with_existing_api():
    """Test that utility functions integrate well with existing SCGen API."""
    # Load defaults manually
    momenta_kdes = load_default_log_momenta_kdes([211, 321])
    centers_kde = load_default_centers("R1")
    weights = load_default_particle_weights()
    
    # Create SCGen manually using loaded distributions
    from rich_generator import SCGen
    
    gen = SCGen(
        particle_types=[211, 321],
        refractive_index=1.0014,
        detector_size=((-600, 600), (-600, 600)),
        momenta_log_distributions=momenta_kdes,
        centers_distribution=centers_kde,
        radial_noise=1.5,
        N_init=60,
        particle_type_proportions={211: weights[211], 321: weights[321]}
    )
    
    # Should work exactly the same as create_scgen_with_defaults
    event = gen.generate_event(5)
    assert len(event['particle_types']) == 5
    assert all(pt in [211, 321] for pt in event['particle_types'])
    
    # Compare with create_scgen_with_defaults
    gen_defaults = create_scgen_with_defaults(
        particle_types=[211, 321],
        detector="R1"
    )
    
    event_defaults = gen_defaults.generate_event(5)
    assert len(event_defaults['particle_types']) == 5
    assert all(pt in [211, 321] for pt in event_defaults['particle_types'])


class TestUtilityFunctionErrorHandling:
    """Test error handling in utility functions."""
    
    def test_missing_distributions_directory(self, monkeypatch):
        """Test behavior when distributions directory is missing."""
        # Mock the _get_distributions_path to return non-existent directory
        def mock_get_dist_path():
            return "/non/existent/path"
        
        monkeypatch.setattr("rich_generator.utility._get_distributions_path", mock_get_dist_path)
        
        with pytest.raises(FileNotFoundError):
            list_available_particle_types()
        
        with pytest.raises(FileNotFoundError):
            load_default_particle_weights()
        
        with pytest.raises(FileNotFoundError):
            load_default_centers("R1")
    
    def test_corrupted_particle_proportions(self, tmp_path, monkeypatch):
        """Test handling of corrupted particle proportions file."""
        # Create a temporary distributions directory with corrupted file
        dist_dir = tmp_path / "distributions"
        dist_dir.mkdir()
        
        # Create corrupted JSON file
        prop_file = dist_dir / "particle_proportions.json"
        prop_file.write_text("{ invalid json")
        
        monkeypatch.setattr("rich_generator.utility._get_distributions_path", lambda: str(dist_dir))
        
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_default_particle_weights()
    
    def test_invalid_particle_proportions_data(self, tmp_path, monkeypatch):
        """Test handling of invalid data in particle proportions file."""
        # Create temporary distributions directory
        dist_dir = tmp_path / "distributions"  
        dist_dir.mkdir()
        
        # Create file with invalid data types
        prop_file = dist_dir / "particle_proportions.json"
        prop_file.write_text('{"not_a_number": "not_a_float"}')
        
        monkeypatch.setattr("rich_generator.utility._get_distributions_path", lambda: str(dist_dir))
        
        with pytest.raises(ValueError, match="Invalid particle type"):
            load_default_particle_weights()