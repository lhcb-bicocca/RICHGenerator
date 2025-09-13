"""
Event generator for synthetic RICH detector Cherenkov rings.

The :class:`SCGen` class synthesises events for a RICH detector by
sampling particle types, momenta, centres and photon hit positions from given distributions.
See the README for usage examples.

The generator stores events in memory as dictionaries with the following
keys:

``centers``: an ``(N,2)`` array of ring centres
``momenta``: an ``(N,)`` array of particle momenta
``particle_types``: an ``(N,)`` array of PDG codes
``rings``: a list of length ``N`` containing arrays of shape ``(num_hits_i,2)`` with the (x,y) coordinates of each photon hit
``tracked_rings``: an array indexing the rings; used when pruning

Events can be written to or loaded from HDF5 with the functions in
``rich_generator.dataset_utils``.
"""

from __future__ import annotations

import numpy as np
import os
import json
import h5py
from particle import Particle
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
from typing import Any, Dict, Tuple, Optional, Iterable

__all__ = ["UniformDist", "SCGen"]


class UniformDist:
    """Simple uniform distribution helper.

    When providing momentum distributions to :class:`SCGen` you can either
    supply a kernel density estimator (KDE) or a pair of numbers to sample
    uniformly within some range.  This helper class implements the same API
    (a ``resample`` method) expected of KDEs.
    """

    def __init__(self, min_val: float, max_val: float) -> None:
        self.min_val = float(min_val)
        self.max_val = float(max_val)

    def resample(self, size: int) -> np.ndarray:
        return np.random.uniform(self.min_val, self.max_val, size)


class SCGen:
    """
    Synthetic LHCb RICH detector data generator.

    Parameters
    ----------
    particle_types : list[int]
        List of PDG particle codes for which to generate rings.  Masses will
        be looked up automatically via the ``particle`` package unless you
        provide the optional ``masses`` mapping.
    refractive_index : float
        Refractive index of the radiator material.
    detector_size : tuple[tuple[float, float], tuple[float, float]]
        Size of the detector in millimetres in the form ``((x_min,x_max),
        (y_min,y_max))``.  Hits outside this region are discarded.
    momenta_log_distributions : dict[int, Any]
        Distributions of log-momenta for each particle type.  Keys should
        match elements of ``particle_types``.  Values may be KDE objects
        exposing a ``resample`` method or tuples ``(min, max)`` describing
        uniform sampling ranges (in which case a :class:`UniformDist` will
        be created internally).
    centers_distribution : Any
        Distribution object for the ring centres.  It must expose a
        ``resample(size)`` method returning a 2x``size`` array with the x
        and y coordinates (for example a 2D KDE object).
    radial_noise : tuple[float, float] | float
        A Gaussian noise model applied to the ring radius.  Either a
        single standard deviation (in which case zero mean is assumed) or a pair ``(mean, std_dev)``.
    N_init : int
        Base photon yield used to compute the expected number of photons per ring.
        The mean photon count for a ring with Cherenkov angle ``theta`` is computed as::

            N_mean = round(N_init * sin(theta)**2 / sin(max_cherenkov_angle)**2)

        for rings above threshold (zero otherwise). Individual ring hit counts
        are then drawn from a Poisson distribution with mean ``N_mean``.
    max_radius : float, optional
        Maximum physical radius of a Cherenkov ring in in millimetres.  This is used
        when normalising radii for image creation.
    masses : dict[int, float], optional
        Optional mapping from PDG codes to particle masses (GeV/c²).  If
        omitted the values are looked up via :mod:`particle`.
    """

    def __init__(
        self,
        particle_types: Iterable[int],
        refractive_index: float,
        detector_size: Tuple[Tuple[float, float], Tuple[float, float]],
        momenta_log_distributions: Dict[int, Any],
        centers_distribution: Any,
        radial_noise: Tuple[float, float] | float,
        N_init: int,
        max_radius: float = 100.0,
        masses: Optional[Dict[int, float]] = None,
    ) -> None:
        self.particle_types = list(particle_types)
        self.refractive_index = float(refractive_index)
        self.detector_size = detector_size
        self.centers_distribution = centers_distribution

        # normalise radial_noise to a tuple
        if isinstance(radial_noise, float):
            self.radial_noise: Tuple[float, float] = (0.0, float(radial_noise))
        else:
            self.radial_noise = (float(radial_noise[0]), float(radial_noise[1]))

        # convert uniform tuples into UniformDist instances
        self.momenta_log_distributions: Dict[int, Any] = {}
        for ptype, dist in momenta_log_distributions.items():
            if isinstance(dist, tuple) and len(dist) == 2:
                self.momenta_log_distributions[ptype] = UniformDist(*dist)
            else:
                self.momenta_log_distributions[ptype] = dist

        self.N_init = int(N_init)
        self.max_cherenkov_radius = float(max_radius)
        self.max_cherenkov_angle = np.arccos(1.0 / refractive_index)

        # initialise masses
        if masses is None:
            # use the particle package to look up masses in GeV
            self.masses = {pt: Particle.from_pdgid(pt).mass for pt in self.particle_types}
        else:
            self.masses = {int(k): float(v) for k, v in masses.items()}

        self.dataset: list[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # internal helpers

    def _create_rings(
        self, centers: np.ndarray, radii: np.ndarray, N_hits: np.ndarray
    ) -> list[np.ndarray]:
        """Sample photon hits for a set of rings.

        Parameters
        ----------
        centers : array-like of shape (N, 2)
            Ring centres in millimetres.
        radii : array-like of shape (N,)
            Physical radii for each ring.
        N_hits : array-like of shape (N,)
            Number of photon hits for each ring.

        Returns
        -------
        list[np.ndarray]
            A list of 2D arrays with shape ``(num_hits_i, 2)`` containing
            the sampled (x,y) positions of photon hits.
        """
        rings: list[np.ndarray] = [np.empty((0, 2))] * len(centers)
        for i in range(len(centers)):
            nh = int(N_hits[i])
            if nh > 0:
                radial_noise = np.random.normal(self.radial_noise[0], self.radial_noise[1], nh)
                angles = np.random.uniform(0.0, 2.0 * np.pi, nh)
                x = centers[i][0] + (radii[i] + radial_noise) * np.cos(angles)
                y = centers[i][1] + (radii[i] + radial_noise) * np.sin(angles)
                rings[i] = np.column_stack((x, y))
            else:
                rings[i] = np.empty((0, 2))
        return rings

    # ------------------------------------------------------------------
    # event generation API

    def generate_event(self, num_particles: int) -> Dict[str, Any]:
        """Generate a single synthetic event.

        This method samples particle types, centres and momenta, computes the
        Cherenkov angles and radii, samples a number of photon hits and
        constructs an event dictionary.  Rings lying outside the detector
        boundaries are automatically pruned.

        Parameters
        ----------
        num_particles : int
            Number of particles (rings) in the event.  Individual rings may
            end up with zero hits if they are below Cherenkov threshold or
            fall outside the detector.

        Returns
        -------
        dict
            Event dictionary containing centres, momenta, particle types,
            rings and tracked indices.
        """
        # sample particle types
        ptypes = np.random.choice(self.particle_types, size=num_particles)
        # sample centres from the supplied distribution
        centers = self.centers_distribution.resample(num_particles)
        centers = centers.T  # expected shape (N,2)
        # sample log momenta and convert back to linear scale
        momenta = np.array([
            self.momenta_log_distributions[int(pt)].resample(1)[0] for pt in ptypes
        ], dtype=float)
        momenta = np.exp(momenta).flatten()
        # compute Cherenkov angles and radii
        masses = np.array([self.masses[int(pt)] for pt in ptypes])
        beta = momenta / np.sqrt(np.square(momenta) + np.square(masses))
        arccos_arg = 1.0 / (self.refractive_index * beta)
        cherenkov_angles = np.where(arccos_arg <= 1.0, np.arccos(arccos_arg), 0.0)
        radii = np.tan(cherenkov_angles) * self.max_cherenkov_radius / np.tan(self.max_cherenkov_angle)
        # expected photon yield
        N_mean = np.where(
            radii > 0.0,
            np.round(
                self.N_init * np.square(np.sin(cherenkov_angles)) / np.square(np.sin(self.max_cherenkov_angle))
            ),
            0,
        ).astype(int)
        N_hits = np.random.poisson(N_mean)
        # generate rings
        rings = self._create_rings(centers, radii, N_hits)
        # prune points outside the detector
        (x_min, x_max), (y_min, y_max) = self.detector_size
        rings = [ring[(ring[:, 0] >= x_min) & (ring[:, 0] <= x_max) & (ring[:, 1] >= y_min) & (ring[:, 1] <= y_max)] for ring in rings]
        # track ring indices (useful for pruning later)
        tracked_rings = np.arange(num_particles)
        return {
            'centers': centers,
            'momenta': momenta,
            'particle_types': ptypes,
            'rings': rings,
            'tracked_rings': tracked_rings,
        }

    def generate_dataset(
        self,
        num_events: int,
        num_particles_per_event: int | Tuple[int, int],
        parallel: bool | int = False,
        progress_bar: bool = True,
    ) -> list[Dict[str, Any]]:
        """Generate a collection of events.

        Parameters
        ----------
        num_events : int
            Number of events to generate.
        num_particles_per_event : int or tuple
            Either a fixed number of particles for every event or a
            ``(min,max)`` range from which to sample the count uniformly.
        parallel : bool or int, optional
            If ``True`` or an integer, events are generated using a thread
            pool.  If an integer it specifies the number of workers.  If
            ``False`` (default) generation is done sequentially.  Note that
            Python's GIL can limit the benefits of threads.
        progress_bar : bool, optional
            Display a tqdm progress bar during generation.

        Returns
        -------
        list of dict
            The generated dataset (a list of event dictionaries).  The
            internal ``dataset`` attribute is also updated.
        """
        if isinstance(num_particles_per_event, tuple):
            counts = np.random.randint(
                num_particles_per_event[0], num_particles_per_event[1] + 1, size=num_events
            ).tolist()
        else:
            counts = [int(num_particles_per_event)] * num_events
        events: list[Dict[str, Any]] = []
        if not parallel:
            it = range(num_events)
            bar = tqdm(total=num_events, disable=not progress_bar)
            for i in it:
                events.append(self.generate_event(counts[i]))
                bar.update(1)
            bar.close()
        else:
            # set number of workers
            n_workers = os.cpu_count() if parallel is True else int(parallel)
            n_workers = max(1, n_workers)
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(self.generate_event, c) for c in counts]
                for fut in tqdm(futures, disable=not progress_bar):
                    events.append(fut.result())
        self.dataset = events
        return events

    # ------------------------------------------------------------------
    # dataset manipulation

    def save_dataset(self, filename: str) -> None:
        """Persist the currently stored dataset to an HDF5 file.

        The output file contains a top level attribute block with
        generator metadata (refractive index, detector size, radial noise,
        etc.) and one group per event.  Within each event group the hits for
        each ring are stored in a variable-length dataset of `(x,y)` pairs.

        Parameters
        ----------
        filename : str
            Name of the output HDF5 file.  If the name does not end in ``.h5`` it will be added automatically.
        """
        if not filename.endswith('.h5'):
            filename += '.h5'
        # prepare ring dtype for efficient storage
        xy_dtype = np.dtype([('x', np.float32), ('y', np.float32)])
        vlen_xy = h5py.vlen_dtype(xy_dtype)
        with h5py.File(filename, 'w', libver='latest') as f:
            # global metadata
            f.attrs.update({
                'refractive_index': self.refractive_index,
                'detector_size': np.asarray(self.detector_size, dtype=np.float32),
                'radial_noise': np.asarray(self.radial_noise, dtype=np.float32),
                'N_init': self.N_init,
                'max_cherenkov_radius': self.max_cherenkov_radius,
                'particle_types': np.asarray(self.particle_types, dtype=np.int32),
                'masses': json.dumps(self.masses),
                'num_events': len(self.dataset),
            })
            for i, event in enumerate(self.dataset):
                grp = f.create_group(f'event_{i}')
                grp.attrs['num_particles'] = len(event['rings'])
                grp.create_dataset('centers', data=event['centers'].astype('float32'))
                grp.create_dataset('momenta', data=event['momenta'].astype('float32'))
                grp.create_dataset('particle_types', data=event['particle_types'].astype('int32'))
                grp.create_dataset('tracked_rings', data=event['tracked_rings'].astype('int32'))
                # Convert each ring (N_i,2) array into a structured array of dtype xy_dtype.
                # h5py expects a 1D object array where each element is a variable-length
                # structured array. For empty rings produce an empty structured array
                # with shape (0,) so lengths match correctly.
                rings_structs = []
                for ring in event['rings']:
                    # ensure contiguous and convert to sequence of (x,y) tuples
                    arr = np.ascontiguousarray(list(map(tuple, ring)), dtype=xy_dtype)
                    rings_structs.append(arr)
                # create a 1D numpy object array to hold the variable-length entries
                rings_obj = np.empty((len(rings_structs),), dtype=object)
                for idx, r in enumerate(rings_structs):
                    rings_obj[idx] = r
                grp.create_dataset('rings', shape=(len(rings_obj),), dtype=vlen_xy, data=rings_obj, compression='gzip', compression_opts=4)

    def prune_centers(self, keep_probability: float) -> list[Dict[str, Any]]:
        """Randomly drop ring centres from each event.

        This method updates the ``tracked_rings`` array of each stored event
        according to the given probability.  It leaves the underlying rings
        untouched.  The updated dataset is returned and stored back in the
        generator.

        Parameters
        ----------
        keep_probability : float
            Probability of keeping a ring centre (between 0 and 1).

        Returns
        -------
        list of dict
            The updated dataset with pruned ``tracked_rings``.
        """
        new_dataset: list[Dict[str, Any]] = []
        for event in self.dataset:
            centres = event['centers']
            mask = np.random.rand(len(centres)) < keep_probability
            # update tracked_rings; other fields are left untouched
            event = event.copy()
            event['tracked_rings'] = event['tracked_rings'][mask]
            new_dataset.append(event)
        self.dataset = new_dataset
        return new_dataset

    def insert_rings(
        self,
        event: int,
        centers: np.ndarray,
        momenta: np.ndarray,
        particle_types: np.ndarray,
        N_hits: Optional[np.ndarray] = None,
    ) -> None:
        """Insert additional rings into an existing event.

        Parameters
        ----------
        event : int
            Index of the event into which to insert rings.
        centers : ndarray of shape (M,2)
            Coordinates of the new ring centres.
        momenta : ndarray of shape (M,)
            Particle momenta.
        particle_types : ndarray of shape (M,)
            PDG codes of the particles.
        N_hits : ndarray, optional
            Explicit numbers of hits per ring.  If omitted the values are
            computed identically to :meth:`generate_event`.
        """
        if event < 0 or event >= len(self.dataset):
            raise IndexError("Event index out of range.")
        if centers.shape[0] != momenta.shape[0] or centers.shape[0] != particle_types.shape[0]:
            raise ValueError("Shapes of centers, momenta and particle_types must match.")
        masses = np.array([self.masses[int(pt)] for pt in particle_types])
        beta = momenta / np.sqrt(np.square(momenta) + np.square(masses))
        arccos_arg = 1.0 / (self.refractive_index * beta)
        cherenkov_angles = np.where(arccos_arg <= 1.0, np.arccos(arccos_arg), 0.0)
        radii = np.tan(cherenkov_angles) * self.max_cherenkov_radius / np.tan(self.max_cherenkov_angle)
        if N_hits is None:
            N_mean = np.where(
                radii > 0.0,
                np.round(
                    self.N_init * np.square(np.sin(cherenkov_angles)) / np.square(np.sin(self.max_cherenkov_angle))
                ),
                0,
            ).astype(int)
            N_hits = np.random.poisson(N_mean)
        new_rings = self._create_rings(centers, radii, N_hits)
        (x_min, x_max), (y_min, y_max) = self.detector_size
        new_rings = [ring[(ring[:, 0] >= x_min) & (ring[:, 0] <= x_max) & (ring[:, 1] >= y_min) & (ring[:, 1] <= y_max)] for ring in new_rings]
        # update event in place
        ev = self.dataset[event]
        ev['centers'] = np.vstack((ev['centers'], centers))
        ev['momenta'] = np.concatenate((ev['momenta'], momenta))
        ev['particle_types'] = np.concatenate((ev['particle_types'], particle_types))
        ev['rings'] = list(ev['rings']) + list(new_rings)
        new_indices = np.arange(len(ev['tracked_rings']), len(ev['tracked_rings']) + len(new_rings))
        ev['tracked_rings'] = np.concatenate((ev['tracked_rings'], new_indices))

    def event_image(self, idx: int, img_size: Tuple[int, int], normalize: bool = False) -> np.ndarray:
        """Rasterise an entire event to a 2D hit count map.

        Parameters
        ----------
        idx : int
            Index of the event to render.
        img_size : tuple[int, int]
            Height and width (H,W) in pixels of the output image.
        normalize : bool, optional
            If ``True`` the image is scaled to [0,1] by dividing by its
            maximum value.

        Returns
        -------
        ndarray
            A 2D array with dtype ``float32`` containing the number of hits
            per pixel.
        """
        if idx < 0 or idx >= len(self.dataset):
            raise IndexError("Index out of range.")
        h, w = img_size
        (x_min, x_max), (y_min, y_max) = self.detector_size
        sx = (w - 1) / (x_max - x_min)
        sy = (h - 1) / (y_max - y_min)
        img = np.zeros((h, w), dtype=np.float32)
        for ring in self.dataset[idx]['rings']:
            if len(ring) == 0:
                continue
            x_pix = np.rint((ring[:, 0] - x_min) * sx).astype(int)
            y_pix = np.rint((y_max - ring[:, 1]) * sy).astype(int)
            m = (x_pix >= 0) & (x_pix < w) & (y_pix >= 0) & (y_pix < h)
            img[y_pix[m], x_pix[m]] += 1.0
        if normalize and img.max() > 0:
            img = img / img.max()
        return img

    def num_rings(self) -> int:
        """Return the total number of rings in the current dataset."""
        return sum(len(ev['rings']) for ev in self.dataset)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= len(self.dataset):
            raise IndexError("Index out of range.")
        return self.dataset[idx]
