#This module contains methods for clusteirng analyses.

import numpy as np

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


def all_distances(x1, y1, x2 = None, y2 = None, mode = 'latlon', \
                w1 = None, w2 = None):
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
    w1: np.array
        Weights applied to the pairs from population 1
    w2: np.array
        Weights applied to the pairs from population 2

    Returns:
    --------
    dists: np.ndarray
        Matrix (size1, size2) with all the pair distances
    """
    #Check and adapt usage of weights
    if w1 is None and w2 is None:
        use_weights = False
    else:
        use_weights = True
        if w1 is None:
            w1 = np.ones_like(x1)
        if w2 is None:
            w2 = np.ones_like(x2)
    #Use distances within population 1 or between 1 and 2
    if x2 is None or y2 is None:
        dists = []
        weights = [] #only used if w1 is not None
        for i in range(len(x1)):
            dists.append(get_dist(x1[i], y1[i], x1[i+1:], y1[i+1:], mode = mode))
            if use_weights:
                weights.append(w1[i]*w1[i+1:])
        dists = np.concatenate(dists)
        if use_weights:
            weights = np.concatenate(weights)
    else:
        dists = np.zeros((len(x1), len(x2)))
        if use_weights:
            weights = np.zeros((len(x1), len(x2)))
        for i in range(len(x1)):
            dists[i] = get_dist(x1[i], y1[i], x2, y2, mode = mode)
            if use_weights:
                weights[i] = w1[i]*w2
    if use_weights:
        return dists, weights
    else:
        return dists

def num_pairs(x1, y1, x2 = None, y2 = None, bins = 10, ranges = None, \
            mode = 'latlon', w1 = None, w2 = None):
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
    w1: np.array
        Weights applied to the pairs from population 1
    w2: np.array
        Weights applied to the pairs from population 2

    Returns:
    --------
    numpairs: np.array
        Number of pairs per distance bin
    weighted_bins: np.array
        Mean distance for each bin
    edges: np.array
        Values of the edges of the distance bins
    """
    #Check and adapt usage of weights
    if w1 is None and w2 is None:
        use_weights = False
        weights = None
        distances = all_distances(x1, y1, x2, y2, mode = mode, w1 = w1, w2 = w2)
    else:
        use_weights = True
        distances, weights = all_distances(x1, y1, x2, y2, mode = mode, \
                                            w1 = w1, w2 = w2)
    if use_weights:
        weights = weights[distances>=0]
    distances = distances[distances>=0]
    numpairs, edges = np.histogram(distances, bins, range = ranges, \
                                    weights = weights)
    #Obtain mean distances per bin
    if weights is None:
        weights = 1.
    weighted_bins, edges = np.histogram(distances, bins = edges, \
                                        weights = distances*weights)
    weighted_bins /= numpairs
    return numpairs, weighted_bins, edges

def num_pairs_bootstrap(x1, y1, x2 = None, y2 = None, bins = 10, ranges = None, \
                        nrands = 10, mode = 'latlon', out_subsamples = False, \
                        w1 = None, w2 = None):
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
    w1: np.array
        Weights applied to the pairs from population 1
    w2: np.array
        Weights applied to the pairs from population 2

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
    numpairs, weighted_bins, edges = num_pairs(x1 = x1, y1 = y1, x2 = x2, \
                                            y2 = y2, bins = bins, \
                                            ranges = ranges, mode = mode, \
                                            w1 = w1, w2 = w2)
    numpairs_rands = []
    for i in range(nrands):
        resample_indeces_1 = bootstrap_resample(np.arange(len(x1)))
        x1r, y1r = x1[resample_indeces_1], y1[resample_indeces_1]
        if w1 is not None:
            w1r = w1[resample_indeces_1]
        else:
            w1r = None
        if x2 is None or y2 is None:
            x2r, y2r, w2r = None, None, None
        else:
            resample_indeces_2 = bootstrap_resample(np.arange(len(x2)))
            x2r, y2r = x2[resample_indeces_2], y2[resample_indeces_2]
            if w2 is not None:
                w2r = w2[resample_indeces_2]
            else:
                w2r = None
        numpairs_r, mean_bin_r, edges_r = num_pairs(x1 = x1r, y1 = y1r, \
                                                x2 = x2r, y2 = y2r, \
                                                bins = bins, ranges = ranges, \
                                                mode = mode, w1 = w1r, w2 = w2r)
        numpairs_rands.append(numpairs_r)
    numpairs_rands = np.array(numpairs_rands)
    errors = np.std(numpairs_rands, axis = 0)
    mean_bootstrap = np.mean(numpairs_rands, axis = 0)
    if out_subsamples:
        return numpairs, errors, weighted_bins, edges, mean_bootstrap, numpairs_rands
    else:
        return numpairs, errors, weighted_bins, edges, mean_bootstrap

def correlation_function(x1, y1, x2, y2, bins = 10, ranges = None, \
                        mode = 'latlon', get_error = True, nrands = 10, \
                        w1 = None, w2 = None, x1x = None, y1x = None, \
                        w1x = None, x2x = None, y2x = None, w2x = None):
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
    w1: np.array
        Weights applied to the pairs from population 1
    w2: np.array
        Weights applied to the pairs from population 2
    x1x: np.array
        x position of population to be cross-paired with population 1 (if any)
    y1x: np.array
        y position of population to be cross-paired with population 1 (if any)
    x2x: np.array
        x position of population to be cross-paired with population 2 (if any)
    y2x: np.array
        y position of population to be cross-paired with population 2 (if any)
    w1x: np.array
        Weights applied to the pairs from population to be cross-paired with
        population  1 (if any)
    w2x: np.array
        Weights applied to the pairs from population to be cross-paired with
        population  2 (if any)

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
    len_1 = get_lengths(x1, w1)
    len_2 = get_lengths(x2, w2)
    if x1x is not None:
        len_1 = (len_1*get_lengths(x1x, w1x))**.5
    if x2x is not None:
        len_2 = (len_2*get_lengths(x2x, w2x))**.5
    #Get number of pairs for the two population, forcing the same bin ranges
    if get_error:
        numpairs_1, errors_1, weighted_bins_1, edges_1, mean_bootstrap_1, \
        numpairs_rands_1 = num_pairs_bootstrap(x1, y1, x1x, y1x, bins = bins, \
                                                ranges = ranges, mode = mode, \
                                                nrands = nrands, \
                                                out_subsamples = True, \
                                                w1 = w1, w2 = w1x)
        numpairs_2, errors_2, weighted_bins_2, edges_2, mean_bootstrap_2, \
        numpairs_rands_2 = num_pairs_bootstrap(x2, y2, x2x, y2x, bins = bins, \
                                                ranges = [edges_1[0], edges_1[-1]], \
                                                mode = mode, nrands = nrands, \
                                                out_subsamples = True, w1 = w2, \
                                                w2 = w2x)
        corr_rands = ((len_2/len_1)**2.)*numpairs_rands_1/numpairs_rands_2 - 1.
        errors = np.std(corr_rands, axis = 0)
    else:
        numpairs_1, weighted_bins_1, edges_1 = num_pairs(x1, y1, x1x, y1x, bins = bins, \
                                                        ranges = ranges, mode = mode, \
                                                        w1 = w1, w2 = w1x)
        numpairs_2, weighted_bins_2, edges_2 = num_pairs(x2, y2, x2x, y2x, bins = bins, \
                                                        ranges = [edges_1[0], edges_1[-1]], \
                                                        mode = mode, w1 = w2, \
                                                        w2 = w2x)
    #2-Point Correlation Function
    corr = ((len_2/len_1)**2.)*numpairs_1/numpairs_2 - 1.
    if get_error:
        return corr, weighted_bins_1, edges_1, errors
    else:
        return corr, weighted_bins_1, edges_1

def get_lengths(x1, w1 = None):
    """
    This method calculates the total length of a population based on their
    number of elements and weigths.

    Parameters:
    -----------
    x: np.array
        Array of positions (one of the dimensions) of the population
    w: np.array
        Array of weights of the elements of the population
    """
    if w1 is None:#TODO make get_len function #TODO repeat this for crosses and get final len
        len_1 = float(len(x1))
    else:
        len_1 = np.sum(w1)
    return len_1

def bootstrap_resample(data):
    """
    This method creates a shuffled version of the data
    with resampling (so repetitions can happen).

    Parameters:
    -----------
    data: np.ndarray
        Data with shape (samples, values)

    Returns:
    --------
    new_data: np.ndarray
        New data resample from the original, resampling
        the samples with their data
    """
    data_len = len(data)
    rand_ints = np.random.randint(0, data_len, data_len)
    new_data = data[rand_ints]
    return new_data
