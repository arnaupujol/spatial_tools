#This module contains methods for clusteirng analyses.

import numpy as np
from stat_tools import errors

def get_dist(px1, py1, px2, py2, mode = 'latlon'):
    """
    This method calculates the distance between two populations.

    Parameters:
    -----------
    px1: np.array
        x position of population 1
    py1: np.array
        y position of population 1
    px2: np.array
        x position of population 2
    py2: np.array
        y position of population 2
    mode: str {'xy', 'latlon'}
        Mode to calculate the distance

            'xy':
                Euclidean distance is calculated

            'latlon':
                Distance in Km is obtained from latitude and
                longitude coordinates, from Haversine formula

    Returns:
    --------
    d: np.ndarray
        Array with all the pair distances
    """
    if mode == 'xy':
        d = np.sqrt((px1 - px2)**2. + (py1 - py2)**2.)
    elif mode == 'latlon':
        #Earth radius in km
        R = 6373.0
        #Coordinates to radians
        lon1, lon2 = np.radians(px1), np.radians(px2)
        lat1, lat2 = np.radians(py1), np.radians(py2)
        #Coordinate differences
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = (np.sin(dlat/2.))**2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon/2))**2
        c = 2 * np.arctan2( np.sqrt(a), np.sqrt(1-a) )
        d = R * c
    else:
        print("Error: wrong distance calculation mode: " + mode)
    return d


def all_distances(x1, y1, x2 = None, y2 = None, mode = 'latlon'):
    """
    This method outputs all the distances between all
    positions 1 and 2.

    Parameters:
    -----------
    x1: np.array
        x position of population 1
    y1: np.array
        y position of population 1
    x2: np.array
        x position of population 2. If None, distance within x1,y1 are obtained
    y2: np.array
        y position of population 2. If None, distance within x1,y1 are obtained
    mode: str {'xy', 'latlon'}
        Mode to calculate the distance

            'xy':
                Euclidean distance is calculated

            'latlon':
                Distance in Km is obtained from latitude and
                    longitude coordinates, from Haversine formula
    Returns:
    --------
    dists: np.ndarray
        Matrix (size1, size2) with all the pair distances
    """

    if x2 is None or y2 is None:
        dists = []
        for i in range(len(x1)):
            dists.append(get_dist(x1[i], y1[i], x1[i+1:], y1[i+1:], mode = mode))
        dists = np.concatenate(dists)
    else:
        dists = np.zeros((len(x1), len(x2)))
        for i in range(len(x1)):
            dists[i] = get_dist(x1[i], y1[i], x2, y2, mode = mode)
    return dists

def num_pairs(x1, y1, x2 = None, y2 = None, bins = 10, ranges = None, mode = 'latlon'):
    """
    This method counts the number of pairs between two populations
    in different distance bins.

    Parameters:
    -----------
    x1: np.array
        x position of population 1
    y1: np.array
        y position of population 1
    x2: np.array
        x position of population 2. If None, distance within x1,y1 are obtained
    y2: np.array
        y position of population 2. If None, distance within x1,y1 are obtained
    bins: int
        Number of distance bins
    ranges: None or [float, float]
        It defines the range values of the binning
    mode: str {'xy', 'latlon'}
        Mode to calculate the distance

            'xy':
                Euclidean distance is calculated

            'latlon':
                Distance in Km is obtained from latitude and
                    longitude coordinates, from Haversine formula

    Returns:
    --------
    numpairs: np.array
        Number of pairs per distance bin
    weighted_bins: np.array
        Mean distance for each bin
    edges: np.array
        Values of the edges of the distance bins
    """
    distances = all_distances(x1, y1, x2, y2, mode = mode)
    distances = distances[distances>=0]
    numpairs, edges = np.histogram(distances, bins, range = ranges)
    #Obtain mean distances per bin
    weighted_bins, edges = np.histogram(distances, bins = edges, weights = distances)
    weighted_bins /= numpairs
    return numpairs, weighted_bins, edges

def num_pairs_bootstrap(x1, y1, x2 = None, y2 = None, bins = 10, ranges = None, nrands = 10, mode = 'latlon', out_subsamples = False):
    """
    This method counts the number of pairs between two populations
    in different distance bins and its bootstrap error.

    Parameters:
    -----------
    x1: np.array
        x position of population 1
    y1: np.array
        y position of population 1
    x2: np.array
        x position of population 2. If None, distance within x1,y1 are obtained
    y2: np.array
        y position of population 2. If None, distance within x1,y1 are obtained
    bins: int
        Number of distance bins
    ranges: None or [float, float]
        It defines the range values of the binning
    nrands: int
        Number of random resamples for Bootstrap error
    mode: str {'xy', 'latlon'}
        Mode to calculate the distance

            'xy':
                Euclidean distance is calculated

            'latlon':
                Distance in Km is obtained from latitude and
                    longitude coordinates, from Haversine formula
    out_subsamples: bool
        If True, it returns the number of pairs for the Bootstrap subsamples

    Returns:
    --------
    numpairs: np.array
        Number of pairs per distance bin
    errors: np.array
        Error in the number of pairs
    weighted_bins: np.array
        Mean distance for each bin
    edges: np.array
        Values of the edges of the distance bins
    mean_boostrap: np.array
        Mean number of pairs per bin from Bootstrap subsamples
    numpairs_rands: np.ndarray (if out_subsamples)
        Number of pairs per distance bin for all Bootstrap subsamples
    """
    numpairs, weighted_bins, edges = num_pairs(x1 = x1, y1 = y1, x2 = x2, y2 = y2, bins = bins, ranges = ranges, mode = mode)
    numpairs_rands = []
    for i in range(nrands):
        resample_indeces_1 = errors.bootstrap_resample(np.arange(len(x1)))
        x1r, y1r = x1[resample_indeces_1], y1[resample_indeces_1]
        if x2 is None or y2 is None:
            x2r, y2r = None, None
        else:
            resample_indeces_2 = errors.bootstrap_resample(np.arange(len(x2)))
            x2r, y2r = x2[resample_indeces_2], y2[resample_indeces_2]
        numpairs_r, mean_bin_r, edges_r = num_pairs(x1 = x1r, y1 = y1r, x2 = x2r, y2 = y2r, bins = bins, ranges = ranges, mode = mode)
        numpairs_rands.append(numpairs_r)
    numpairs_rands = np.array(numpairs_rands)
    errors = np.std(numpairs_rands, axis = 0)
    mean_bootstrap = np.mean(numpairs_rands, axis = 0)
    if out_subsamples:
        return numpairs, errors, weighted_bins, edges, mean_bootstrap, numpairs_rands
    else:
        return numpairs, errors, weighted_bins, edges, mean_bootstrap

def correlation_function(x1, y1, x2, y2, bins = 10, ranges = None, mode = 'latlon', get_error = True, nrands = 10):
    """
    This method calculates the 2-point correlation function between two populations.

    Parameters:
    -----------
    x1: np.array
        x position of population 1
    y1: np.array
        y position of population 1
    x2: np.array
        x position of population 2. If None, distance within x1,y1 are obtained
    y2: np.array
        y position of population 2. If None, distance within x1,y1 are obtained
    bins: int
        Number of distance bins
    ranges: None or [float, float]
        It defines the range values of the binning
    mode: str {'xy', 'latlon'}
        Mode to calculate the distance

            'xy':
                Euclidean distance is calculated

            'latlon':
                Distance in Km is obtained from latitude and
                    longitude coordinates, from Haversine formula
    get_error: bool
        If True, the Bootstrap error is obtained
    nrands: int
        Number of random resamples for Bootstrap error

    Returns:
    --------
    corr: np.array
        The 2-point correlation function in distance bins
    weighted_bins_1: np.array
        The mean distance in each bin for the first population
    edges_1: np.array
        The edges of the binning
    errors: np.array
        The errors of the 2PCF
    """
    #Population sizes
    len_1, len_2 = float(len(x1)), float(len(x2))
    #Get number of pairs for the two population, forcing the same bin ranges
    if get_error:
        numpairs_1, errors_1, weighted_bins_1, edges_1, mean_bootstrap_1, numpairs_rands_1 = num_pairs_bootstrap(x1, y1, bins = bins, ranges = ranges, mode = mode, nrands = nrands, out_subsamples = True)
        numpairs_2, errors_2, weighted_bins_2, edges_2, mean_bootstrap_2, numpairs_rands_2 = num_pairs_bootstrap(x2, y2, bins = bins, ranges = [edges_1[0], edges_1[-1]], mode = mode, nrands = nrands, out_subsamples = True)
        corr_rands = ((len_2/len_1)**2.)*numpairs_rands_1/numpairs_rands_2 - 1.
        errors = np.std(corr_rands, axis = 0)
    else:
        numpairs_1, weighted_bins_1, edges_1 = num_pairs(x1, y1, bins = bins, ranges = ranges, mode = mode)
        numpairs_2, weighted_bins_2, edges_2 = num_pairs(x2, y2, bins = bins, ranges = [edges_1[0], edges_1[-1]], mode = mode)
    #2-Point Correlation Function
    corr = ((len_2/len_1)**2.)*numpairs_1/numpairs_2 - 1.
    if get_error:
        return corr, weighted_bins_1, edges_1, errors
    else:
        return corr, weighted_bins_1, edges_1
