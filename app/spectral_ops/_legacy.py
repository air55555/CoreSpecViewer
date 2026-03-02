"""
Large, monolithic module for performing operations. A few of these functions are
useless without the GUI (con_dict, slice_from_sensor), but the vast majority are
UI agnostic and can be run on any hyperspectral cube in npy format.

NB. This will be broken up in a re-factor eventually.
"""
import os
import glob
import time
import xml.etree.ElementTree as ET
import logging


import cv2
from gfit.util import remove_hull
import hylite
from hylite.analyse import minimum_wavelength
from hylite.sensors import Fenix as HyliteFenix
from hylite.io.images import loadWithNumpy
import matplotlib
from numba import jit
import numpy as np
from PIL import Image
import scipy as sc
import spectral as sp
import spectral.io.envi as envi
from ..config import config  # mutable module singleton
from .fenix_smile import fenix_smile_correction
from .analysis import Combined_MWL
logger = logging.getLogger(__name__)
my_map = matplotlib.colormaps['viridis']
my_map.set_bad('black')


def est_peaks_cube_scipy(data, bands, wavrange=(2300, 2340)):
    """
    Detect spectral absorption peaks within a specified wavelength range
    for every pixel in a hyperspectral cube using SciPy peak detection.

    For each pixel spectrum, local maxima of ``(1 - reflectance)`` are
    identified using :func:`scipy.signal.find_peaks`. The first peak whose
    wavelength lies inside ``wavrange`` is returned; if no such peak is found,
    a sentinel value of ``-999`` is assigned.

    Parameters
    ----------
    data : np.ndarray, shape (H, W, B)
        Hyperspectral image or array of continuum-removed or inverted
        reflectance values. The last axis must represent spectral bands.

    bands : np.ndarray, shape (B,)
        Wavelength values (in nm) corresponding to the third axis of ``data``.

    wavrange : tuple(float, float), optional
        Minimum and maximum wavelength (in nm) to search for peaks.
        Default is ``(2300, 2340)``.

    Returns
    -------
    arr : np.ndarray, shape (H, W)
        Estimated peak wavelength for each pixel. If no valid peak is found
        in the target range, the value is ``-999``.

    Notes
    -----
    - Peaks are detected on ``1 - data[i, j]``, making absorption dips into
      positive peaks.
    - Only the *first* valid peak inside ``wavrange`` is reported per pixel.
    - No thresholding by peak height or prominence is applied; downstream
      filtering may be required for reliability.
    """
    w, l, b = data.shape
    arr = np.zeros((w,l))
    for i in range(w):
        for j in range(l):
            peak_indices, peak_dict = sc.signal.find_peaks(1-data[i,j], height=(None, None))
            x = [bands[i] for i in peak_indices]
            for k in x:
                if k > wavrange[0] and k < wavrange[1]:
                     arr[i,j] = k
                     break
                else:
                    arr[i,j] = -999
    return arr


def est_peaks_cube_scipy_multi_thresh(
    data,
    bands,
    wavrange=(2300, 2340),
    depth_thresh=0.3,
    prom_thresh=None,
    min_width_nm=10.0,
    max_width_nm=None,
):
    """
    Robust peak detector for use as a 'real feature?' gate.

    Parameters
    ----------
    data : (H, W, B) ndarray
        Continuum-removed reflectance cube (values ~1 with dips).
    bands : (B,) ndarray
        Wavelengths in nm.
    wavrange : (float, float)
        Target feature window [min_nm, max_nm].
    depth_thresh : float
        Minimum feature depth (1 - R_CR at the minimum).
    prom_thresh : float or None
        Minimum prominence. If None, defaults to depth_thresh.
    min_width_nm : float or None
        Minimum allowed feature width (FWHM) in nm.
    max_width_nm : float or None
        Maximum allowed feature width in nm (optional).

    Returns
    -------
    arr : (H, W) ndarray
        For each pixel, the wavelength (nm) of the best peak in wavrange,
        or -999.0 if no acceptable feature was found.
    """
    H, W, B = data.shape
    arr = np.full((H, W), -999.0, dtype=float)

    if prom_thresh is None:
        prom_thresh = depth_thresh

    # Approximate band step (assumed almost regular)
    band_step = float(np.median(np.diff(bands)))

    for i in range(H):
        for j in range(W):
            spec = data[i, j, :]
            y = 1.0 - spec  # turn absorption into positive peaks

            # Get peaks + measurements.
            # height filters by depth; prominence/width=0 just request props.
            peaks, props = sc.signal.find_peaks(
                y,
                height=depth_thresh,
                prominence=0,
                width=0,
            )

            if peaks.size == 0:
                # Leave arr[i,j] = -999.0 (no feature)
                continue

            heights = props["peak_heights"]
            prom    = props.get("prominences", np.zeros_like(heights))
            widths_idx = props.get("widths", np.zeros_like(heights))
            widths_nm  = widths_idx * band_step
            lambdas    = bands[peaks]

            # Basic quality masks
            valid = np.ones_like(heights, dtype=bool)
            valid &= heights >= depth_thresh
            valid &= prom    >= prom_thresh

            if min_width_nm is not None:
                valid &= widths_nm >= min_width_nm
            if max_width_nm is not None:
                valid &= widths_nm <= max_width_nm

            # Restrict to the target wavelength window
            if np.any(valid):
                valid &= (lambdas >= wavrange[0]) & (lambdas <= wavrange[1])

            if not np.any(valid):
                # No valid peak in the window → leave -999
                continue

            # Choose the 'best' peak: highest prominence (or depth if you prefer)
            v_idx = np.where(valid)[0]
            best_local = v_idx[np.argmax(prom[v_idx])]
            arr[i, j] = lambdas[best_local]

    return arr




def get_SQM_peak_finder_vectorized(data, bands, atol=1e-12):
    """
    Vectorised implementation of the Simple Quadratic Method (SQM).
    Inputs:
        data  : (M,N,B) or (N,B) array of continuum-removed spectra (baseline ~1), cropped to a single feature.
        bands : (B,) array of band centres (same units as desired output, e.g., nm).
        atol  : numerical tolerance for flatness / zero-division guards.
    Returns:
        SQM(np.squeeze(tru), np.squeeze(dep))
    Notes:
        - Edge minima (b==0 or b==B-1) and flat triplets fall back to the discrete band centre;
          depth is left as 0.0 for those pixels (as in your original).
        - Depth is computed from the fitted parabola: depth = 1 - f(min).
    """

    # Ensure 3D
    if data.ndim == 2:
        data = np.expand_dims(data, 0)  # (1,N,B)
    M, N, B = data.shape
    logger.debgug(f"shape {M,N,B}, bands {bands.shape}")
    # Argmin index at each pixel
    b = np.argmin(data, axis=-1)  # (M,N)

    # Neighbor indices
    bL = np.clip(b - 1, 0, B - 1)
    bR = np.clip(b + 1, 0, B - 1)

    # Gather spectral values at L, 0, R
    # Build broadcast indices
    ii, jj = np.indices((M, N), sparse=False)
    D0  = data[ii, jj, b ]   # at minimum band
    DL  = data[ii, jj, bL]   # left neighbor
    DR  = data[ii, jj, bR]   # right neighbor

    # Gather wavelengths
    W0  = bands[b ]
    WL  = bands[bL]
    WR  = bands[bR]

    # Start outputs with fallbacks (band centre), depth 0
    tru = W0.copy()
    dep = np.zeros_like(D0, dtype=float)

    # Valid pixels: not at edges and not flat around the min
    not_edge = (b > 0) & (b < (B - 1))
    not_flat = (~np.isclose(D0, DR, atol=atol)) & (~np.isclose(DL, D0, atol=atol))
    mask = not_edge & not_flat

    if np.any(mask):
        # Compute quadratic coefficients using the 3-point formula (Rodger et al. 2012)
        # A = an/ad, Bc = ...
        an = (D0[mask] - DL[mask]) * (WL[mask] - WR[mask]) + (DR[mask] - D0[mask]) * (W0[mask] - WL[mask])
        ad = (WL[mask] - WR[mask]) * (W0[mask]**2 - WL[mask]**2) + (W0[mask] - WL[mask]) * (WR[mask]**2 - WL[mask]**2)

        # Guards against degenerate geometry (tiny denominator or identical wavelengths)
        good_geom = (~np.isclose(ad, 0.0, atol=atol)) & (~np.isclose(W0[mask] - WL[mask], 0.0, atol=atol))

        # Submask of truly valid pixels
        if np.any(good_geom):
            A = an[good_geom] / ad[good_geom]
            Bc = ((D0[mask][good_geom] - DL[mask][good_geom]) - A * (W0[mask][good_geom]**2 - WL[mask][good_geom]**2)) / (W0[mask][good_geom] - WL[mask][good_geom])

            # Avoid division by ~0 for A
            nondeg = ~np.isclose(A, 0.0, atol=atol)

            # Final valid set
            valid = np.zeros_like(mask, dtype=bool)
            # Map good_geom & nondeg back into full image mask positions
            idx_mask = np.argwhere(mask)
            idx_good = idx_mask[good_geom]
            idx_final = idx_good[nondeg]
            if idx_final.size > 0:
                vi = idx_final[:, 0]
                vj = idx_final[:, 1]

                A  = A[nondeg]
                Bc = Bc[nondeg]
                # Vertex (refined minimum)
                m = -Bc / (2.0 * A)

                # Compute C via left point, then depth = 1 - f(m)
                C = DL[vi, vj] - (A * (WL[vi, vj]**2)) - (Bc * WL[vi, vj])
                fmin = A * (m**2) + Bc * m + C
                d = 1.0 - fmin

                tru[vi, vj] = m
                dep[vi, vj] = d

    # Report fallback usage (edges/flat/degenerate)
    total = M * N
    used_band_centre = np.count_nonzero(dep == 0.0)  # depth stays 0 only where we fell back (by construction)


    return np.squeeze(tru), np.squeeze(dep)


@jit(nopython=True)
def numpy_pearson_stackexemplar_threshed(data, exemplar_stack):
    """
    Classify each pixel spectrum by maximum Pearson correlation against a stack
    of exemplar spectra, applying a fixed correlation threshold.

    For every pixel spectrum in ``data``, Pearson correlation coefficients are
    computed against each exemplar in ``exemplar_stack``. The pixel is assigned
    to the exemplar with the highest coefficient **only if the maximum
    correlation exceeds 0.7**. Otherwise, the pixel is marked as unclassified
    using a sentinel index (``-999``). Correlation scores are returned in a
    separate confidence array.

    Parameters
    ----------
    data : np.ndarray, shape (H, W, B)
        Hyperspectral data cube (typically reflectance or continuum-removed
        spectra). Pixel spectra are along the last axis.

    exemplar_stack : np.ndarray, shape (N, B)
        Stack of N reference/exemplar spectra to match against. Each spectrum
        must have the same band length as ``data`` along the spectral axis.

    Returns
    -------
    coeffs : np.ndarray, shape (H, W)
        Index of the best-matching exemplar for each pixel. Pixels failing the
        correlation threshold are assigned ``-999``.

    confidence : np.ndarray, shape (H, W)
        Maximum Pearson correlation coefficient for each pixel, regardless of
        whether it passed the threshold.

    Notes
    -----
    - Pearson correlation is computed via ``np.corrcoef`` for each exemplar.
    - Threshold is currently fixed at 0.7; modify in the code for different
      confidence limits.
    - No spectral preprocessing is performed—spectra should ideally be
      normalized or continuum-removed beforehand to avoid bias.
    - Returned ``coeffs`` is an integer index map; it can be used to build
      classification images or masks.
    """
    num = exemplar_stack.shape[0]
    coeffs = np.zeros((data.shape[0], data.shape[1]))
    confidence = np.zeros((data.shape[0], data.shape[1]))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            c_list = np.zeros(num)
            for n in range(num):
                c_list[n] = np.corrcoef(data[i,j], exemplar_stack[n])[1,0]
            if np.max(c_list) > 0.7:
                coeffs[i,j] = np.argmax(c_list)
                confidence[i,j] = np.max(c_list)

            else:
                coeffs[i,j] = -999
                confidence[i,j] = np.max(c_list)
    return coeffs, confidence


# ======== Unsure why these are in here, but leave for now ====================
#TODO: Is this called anywhere? Delete
def crop_with_mask_cv2(cube, mask, margin=0, invert=False, min_area=0):
    """
    Crop a hyperspectral cube to the bounding box of a boolean mask.

    Parameters
    ----------
    cube : np.ndarray
        3D array of shape (y, x, bands)
    mask : np.ndarray
        2D boolean array, same (y, x) shape as cube. True = foreground.
    margin : int
        Extra pixels of padding around the detected box.
    invert : bool
        If True, treat False as foreground instead of True.
    min_area : int
        Ignore connected components smaller than this area (optional).

    Returns
    -------
    cropped : np.ndarray
        Cropped hyperspectral cube (y, x, bands)
    bbox : tuple[int]
        (y0, y1, x0, x1) bounding box used for cropping
    """
    if cube.ndim != 3:
        raise ValueError("cube must be 3D (y, x, bands)")
    if mask.shape != cube.shape[:2]:
        raise ValueError("mask must match cube spatial dimensions (y, x)")
    if mask.dtype != bool:
        raise ValueError("mask must be boolean")

    # Optionally invert
    if invert:
        mask = ~mask

    # Convert for cv2 (expects 0/255 uint8)
    m = (mask.astype(np.uint8)) * 255

    # Find external contours (connected regions)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in mask")

    # Optionally ignore tiny components
    if min_area > 0:
        contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        if not contours:
            raise ValueError("No contours above min_area")

    # Largest component
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    # Add margin and clip to image bounds
    H, W = mask.shape
    x0 = max(0, x - margin); y0 = max(0, y - margin)
    x1 = min(W, x + w + margin); y1 = min(H, y + h + margin)

    cropped = cube[y0:y1, x0:x1, :]
    return cropped, (y0, y1, x0, x1)


#TODO: This has been replaced by strict version, dont think this is called
def mineral_map_wta(data, exemplar_stack, thresh=0.70, invalid_value=-999):
    """
    Vectorized winner-takes-all Pearson to match np.corrcoef semantics.

    data:            (H, W, B) float
    exemplar_stack:  (K, B)    float
    Returns:
        class_idx: (H, W) int32  (=-999 for low confidence)
        best_corr: (H, W) float32  Pearson r in [-1, 1]
    """
    H, W, B = data.shape
    K = exemplar_stack.shape[0]

    # --- z-score with sample std (ddof=1) to match np.corrcoef
    # data -> (N, B)
    X = data.reshape(-1, B).astype(np.float32)
    X_mean = X.mean(axis=1, keepdims=True)
    X_std  = X.std(axis=1, ddof=1, keepdims=True)
    X_std  = np.where(X_std < 1e-12, 1e-12, X_std)
    Xz = (X - X_mean) / X_std                              # (N, B)

    # exemplars -> (K, B)
    E = exemplar_stack.astype(np.float32)
    E_mean = E.mean(axis=1, keepdims=True)
    E_std  = E.std(axis=1, ddof=1, keepdims=True)
    E_std  = np.where(E_std < 1e-12, 1e-12, E_std)
    Ez = (E - E_mean) / E_std                              # (K, B)

    # --- Pearson matrix: divide by (B-1) to match corrcoef scaling
    corr = (Xz @ Ez.T) / max(B - 1, 1)                    # (N, K), in [-1, 1]
    best_corr = corr.max(axis=1)
    idx = corr.argmax(axis=1).astype(np.int32)

    # --- apply threshold like your nested loops
    idx = np.where(best_corr >= thresh, idx, invalid_value).astype(np.int32)

    return idx.reshape(H, W), best_corr.reshape(H, W).astype(np.float32)

#======== Author specific functions - unlikely to have wide use

def carbonate_facies(savgol, savgol_cr, mask, bands, technique = 'QUAD'):
    feats = {'1400W':(	1387,	1445,	1350,	1450),
    '1480W':(	1471,	1491,	1440,	1520),
    '1550W':(	1520,	1563,	1510,	1610),
    '1760W':(	1751,	1764,	1730,	1790),
    '1850W':(	1749,	1949,	1820,	1880),
    '1900W': (1840,1990, 1850, 1970),
    '2080W':(	1980,	2180,	2060,	2100),
    '2160W':(	2159,	2166,	2138,	2179),
    '2200W':(	2185,	2215,	2120,	2245),
    '2250W':(	2248,	2268,	2230,	2280),
    '2290W':(	2279,	2338,	2270,	2320),
    '2320W':(	2300,	2340,	2295,	2355),
    '2350W':(	2320,	2366,	2310,	2370),
    '2390W':(	2377,	2406,	2375,	2435),
    '2950W':(2920, 2980, 2900, 3000),
    '2950AW':(2900, 2960, 2900, 3000),
    '2830W':(	2677,	2890,	2790,	2890),
    '3000W':(	2900,	3100,	2795,	3900),
    '3500W':(	3400,	3600,	3300,	3700),#NOT from laukamp
    '4000W':(	3930,	4150,	3800,	4200),
    '4000WIDEW':(3910,	4150,	3800,	4200),
    '4470TRUEW':(	4460,	4490,	4350,	4550),
    '4500SW':(4570,	4850,	4090,	5040),
    '4500CW':(4625,	4770,	4090,	5040),
    '4670W': (4300, 4800, 4300, 4800),
    '4920W': (4850, 5100, 4850, 5157),
    '4000V_NARROWW': (3930,4150,3800,4200),
    '4000shortW': (3850,4000,3800,4200),
    '2950BW':(2920, 2990, 2790, 3200),
    }
    cr_crop_min_22 = feats['2200W'][2]
    cr_crop_max_22 = feats['2200W'][3]
    cr_crop_min_index_22 = np.argmin(np.abs(np.array(bands)-(feats['2200W'][2])))
    cr_crop_max_index_22 = np.argmin(np.abs(np.array(bands)-(feats['2200W'][3])))
    wav_min_22 = feats['2200W'][0]
    wav_max_22 = feats['2200W'][1]
    wav_min_index_22 = np.argmin(np.abs(np.array(bands)-(feats['2200W'][0])))
    wav_max_index_22 = np.argmin(np.abs(np.array(bands)-(feats['2200W'][1])))

    cr_crop_min_23 = feats['2320W'][2]
    cr_crop_max_23 = feats['2320W'][3]
    cr_crop_min_index_23 = np.argmin(np.abs(np.array(bands)-(feats['2320W'][2])))
    cr_crop_max_index_23 = np.argmin(np.abs(np.array(bands)-(feats['2320W'][3])))
    wav_min_23 = feats['2320W'][0]
    wav_max_23 = feats['2320W'][1]
    wav_min_index_23 = np.argmin(np.abs(np.array(bands)-(feats['2320W'][0])))
    wav_max_index_23 = np.argmin(np.abs(np.array(bands)-(feats['2320W'][1])))


    logger.debug('checking dirty or clean')
    dirty_or_clean = est_peaks_cube_scipy(savgol_cr, bands, wavrange=(wav_min_22, wav_max_22))
    logger.debug('checking how calcitic')
    calcitic_or_not = est_peaks_cube_scipy(savgol_cr, bands, wavrange=(wav_min_23, wav_max_23))
    #carb wavelength position
    logger.debug('MWL-ing')
    calc_or_dolo, _, feat_mask = Combined_MWL(savgol, savgol_cr, mask, bands, '2320W', technique = technique)


    # ==========#Facies colours"===================================================================
    clean_calcite = [0, 0, 255] #1 in data
    clean_dolomitic_calcite = [255, 0, 0]#2 in data
    clean_calcitic_dolomite = [0, 255, 255] #3 in data
    clean_dolomite =[204, 255, 153]#4 in data
    dirty_calcite = [0, 255, 0]#5 in data
    dirty_dolomitic_calcite = [255, 255, 0]#6 in data
    dirty_calcitic_dolomite = [255, 0, 255]#7 in data
    dirty_dolomite = [204, 153, 255]#8 in data
# =============================================================================
    M, N, B = savgol_cr.shape
    output_image = np.zeros((M,N, 3))
    output_data = np.zeros((M,N))
#decision tree
# 8 part facies
    for i in range(M):
        for j in range(N):
            if dirty_or_clean[i,j] > 0:
                #dirty
                if calc_or_dolo[i,j] >= 2330:
                    output_image[i, j] = dirty_calcite
                    output_data[i, j] = 5
                elif calc_or_dolo[i,j] < 2330 and calc_or_dolo[i,j] >= 2320:
                    output_image[i, j] = dirty_dolomitic_calcite
                    output_data[i, j] = 6
                elif calc_or_dolo[i,j] < 2320 and calc_or_dolo[i,j] >= 2310:
                    output_image[i, j] = dirty_calcitic_dolomite
                    output_data[i, j] = 7
                else:
                    output_image[i, j] = dirty_dolomite
                    output_data[i, j] = 8
            else:
                #Clean
                if calc_or_dolo[i,j] >= 2330:
                    output_image[i, j] = clean_calcite
                    output_data[i, j] = 1
                elif calc_or_dolo[i,j] < 2330 and calc_or_dolo[i,j] >= 2320:
                    output_image[i, j] = clean_dolomitic_calcite
                    output_data[i, j] = 2
                elif calc_or_dolo[i,j] < 2320 and calc_or_dolo[i,j] >= 2310:
                    output_image[i, j] = clean_calcitic_dolomite
                    output_data[i, j] = 3
                else:
                    output_image[i, j] = clean_dolomite
                    output_data[i, j] = 4
            if calcitic_or_not[i,j] < 0:
                if dirty_or_clean[i,j] < 0:
                    #not calcitic not siliciclastic
                   output_image[i, j] = [255, 255, 255]# 10 non-carbonaceous response
                   output_data[i, j] = 10
                else:
                    #not carbonaceous but siliciclastic
                    output_image[i, j] = [96, 96, 96] # 9
                    output_data[i, j] = 9

    output_data[mask==1] = 0
    output_image[mask==1] = [0,0,0]
    return output_data, output_image


def carbonate_facies_original(savgol, savgol_cr, mask, bands, technique = 'QUAD'):
    feats = {'1400W':(	1387,	1445,	1350,	1450),
    '1480W':(	1471,	1491,	1440,	1520),
    '1550W':(	1520,	1563,	1510,	1610),
    '1760W':(	1751,	1764,	1730,	1790),
    '1850W':(	1749,	1949,	1820,	1880),
    '1900W': (1840,1990, 1850, 1970),
    '2080W':(	1980,	2180,	2060,	2100),
    '2160W':(	2159,	2166,	2138,	2179),
    '2200W':(	2185,	2215,	2120,	2245),
    '2250W':(	2248,	2268,	2230,	2280),
    '2290W':(	2279,	2338,	2270,	2320),
    '2320W':(	2300,	2340,	2295,	2355),
    '2350W':(	2320,	2366,	2310,	2370),
    '2390W':(	2377,	2406,	2375,	2435),
    '2950W':(2920, 2980, 2900, 3000),
    '2950AW':(2900, 2960, 2900, 3000),
    '2830W':(	2677,	2890,	2790,	2890),
    '3000W':(	2900,	3100,	2795,	3900),
    '3500W':(	3400,	3600,	3300,	3700),#NOT from laukamp
    '4000W':(	3930,	4150,	3800,	4200),
    '4000WIDEW':(3910,	4150,	3800,	4200),
    '4470TRUEW':(	4460,	4490,	4350,	4550),
    '4500SW':(4570,	4850,	4090,	5040),
    '4500CW':(4625,	4770,	4090,	5040),
    '4670W': (4300, 4800, 4300, 4800),
    '4920W': (4850, 5100, 4850, 5157),
    '4000V_NARROWW': (3930,4150,3800,4200),
    '4000shortW': (3850,4000,3800,4200),
    '2950BW':(2920, 2990, 2790, 3200),
    }
    cr_crop_min_22 = feats['2200W'][2]
    cr_crop_max_22 = feats['2200W'][3]
    cr_crop_min_index_22 = np.argmin(np.abs(np.array(bands)-(feats['2200W'][2])))
    cr_crop_max_index_22 = np.argmin(np.abs(np.array(bands)-(feats['2200W'][3])))
    wav_min_22 = feats['2200W'][0]
    wav_max_22 = feats['2200W'][1]
    wav_min_index_22 = np.argmin(np.abs(np.array(bands)-(feats['2200W'][0])))
    wav_max_index_22 = np.argmin(np.abs(np.array(bands)-(feats['2200W'][1])))

    cr_crop_min_23 = feats['2320W'][2]
    cr_crop_max_23 = feats['2320W'][3]
    cr_crop_min_index_23 = np.argmin(np.abs(np.array(bands)-(feats['2320W'][2])))
    cr_crop_max_index_23 = np.argmin(np.abs(np.array(bands)-(feats['2320W'][3])))
    wav_min_23 = feats['2320W'][0]
    wav_max_23 = feats['2320W'][1]
    wav_min_index_23 = np.argmin(np.abs(np.array(bands)-(feats['2320W'][0])))
    wav_max_index_23 = np.argmin(np.abs(np.array(bands)-(feats['2320W'][1])))


    logger.debug('checking dirty or clean')
    dirty_or_clean = est_peaks_cube_scipy(savgol_cr, bands, wavrange=(wav_min_22, wav_max_22))
    logger.debug('checking how calcitic')
    calcitic_or_not = est_peaks_cube_scipy(savgol_cr, bands, wavrange=(wav_min_23, wav_max_23))
    #carb wavelength position
    logger.debug('MWL-ing')
    
    calc_or_dolo, _ = get_SQM_peak_finder_vectorized(remove_hull(savgol[:,:,cr_crop_min_index_23:cr_crop_max_index_23]), bands[cr_crop_min_index_23:cr_crop_max_index_23])

    # ==========#Facies colours"===================================================================
    clean_calcite = [0, 0, 255] #1 in data
    clean_dolomitic_calcite = [255, 0, 0]#2 in data
    clean_calcitic_dolomite = [0, 255, 255] #3 in data
    clean_dolomite =[204, 255, 153]#4 in data
    dirty_calcite = [0, 255, 0]#5 in data
    dirty_dolomitic_calcite = [255, 255, 0]#6 in data
    dirty_calcitic_dolomite = [255, 0, 255]#7 in data
    dirty_dolomite = [204, 153, 255]#8 in data
# =============================================================================
    M, N, B = savgol_cr.shape
    output_image = np.zeros((M,N, 3))
    output_data = np.zeros((M,N))
    #decision tree
    # 8 part facies
    for i in range(M):
        for j in range(N):
            if dirty_or_clean[i,j] > 0:
                #dirty
                if calc_or_dolo[i,j] >= 2330:
                    output_image[i, j] = dirty_calcite
                    output_data[i, j] = 5
                elif calc_or_dolo[i,j] < 2330 and calc_or_dolo[i,j] >= 2320:
                    output_image[i, j] = dirty_dolomitic_calcite
                    output_data[i, j] = 6
                elif calc_or_dolo[i,j] < 2320 and calc_or_dolo[i,j] >= 2310:
                    output_image[i, j] = dirty_calcitic_dolomite
                    output_data[i, j] = 7
                else:
                    output_image[i, j] = dirty_dolomite
                    output_data[i, j] = 8
            else:
                #Clean
                if calc_or_dolo[i,j] >= 2330:
                    output_image[i, j] = clean_calcite
                    output_data[i, j] = 1
                elif calc_or_dolo[i,j] < 2330 and calc_or_dolo[i,j] >= 2320:
                    output_image[i, j] = clean_dolomitic_calcite
                    output_data[i, j] = 2
                elif calc_or_dolo[i,j] < 2320 and calc_or_dolo[i,j] >= 2310:
                    output_image[i, j] = clean_calcitic_dolomite
                    output_data[i, j] = 3
                else:
                    output_image[i, j] = clean_dolomite
                    output_data[i, j] = 4
            if calcitic_or_not[i,j] < 0:
                if dirty_or_clean[i,j] < 0:
                    #not calcitic not siliciclastic
                   output_image[i, j] = [255, 255, 255]# 10 non-carbonaceous response
                   output_data[i, j] = 10
                else:
                    #not carbonaceous but siliciclastic
                    output_image[i, j] = [96, 96, 96] # 9
                    output_data[i, j] = 9

    output_data[mask==1] = 0
    output_image[mask==1] = [0,0,0]
    return output_data, output_image


# Older version of unwrap_from_stats, not mask aware. Dont think it is used in GUI
def seg_from_stats(image, stats, MIN_AREA=300, MIN_WIDTH=10):
    """
    Extract bounding boxes from component stats, pad to a common width,
    and stack them vertically (right-to-left, then top-to-bottom ordering).

    Parameters
    ----------
    image : ndarray
        Input 2D or 3D image to segment.
    stats : ndarray, shape (N, 5)
        (x, y, width, height, area) rows from `cv2.connectedComponentsWithStats`.
    MIN_AREA : int, optional
        Minimum area to keep.
    MIN_WIDTH : int, optional
        Minimum width to keep.

    Returns
    -------
    ndarray
        Vertically concatenated segments (plain array; padding filled with zeros).

    Notes
    -----
    Segments are sorted into columns using an x-based binning tolerance,
    then into rows by ascending y. 
    """


    segments = []

    for i in range(1, stats.shape[0]): # Skip background (label 0)
        x, y, w, h, area = stats[i]
        if area < MIN_AREA or w < MIN_WIDTH:
            continue # Skip small regions
        else:
            segment = image[y:y+h, x:x+w]
                        # Store top-left x, y for sorting
            segments.append(((x, y), segment))

    # Sort segments: right to left (x descending), top to bottom (y ascending)
    tolerance = 15
    segments_sorted = sorted(segments, key=lambda s: (round(-s[0][0]/tolerance), s[0][1]))



    # Determine max width
    max_width = max(s[1].shape[1] for s in segments_sorted)
    # Pad segments to same width
    padded_segments = []

    for _, seg in segments_sorted:
        h, w = seg.shape[:2]
        pad_total = max_width - w

        if pad_total > 0:
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            if seg.ndim == 2:
                pad_shape = ((0, 0), (pad_left, pad_right))
            else:
                pad_shape = ((0, 0), (pad_left, pad_right), (0, 0))

            seg_padded = np.pad(seg, pad_shape, mode='constant', constant_values=0)
        else:
            seg_padded = seg

        padded_segments.append(seg_padded)


    # Stack vertically
    concatenated = np.vstack(padded_segments)

    return concatenated








