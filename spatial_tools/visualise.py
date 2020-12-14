#This module contains functions to make plots.

import numpy as np
import matplotlib.pyplot as plt

def scatter_geo_map(lon, lat, title = '', figsize = [10,8], s = 10, c = 'tab:blue', cmap = 'viridis', alpha = 1., list_locs = None):
    """
    This method shows a scatter plot of geographical positions on top of
    a cartographic map around the data.

    Parameters:
    -----------
    lon: np.array
        Values of longitude
    lat: np.array
        Values of latitude
    title: str
        Title of the plot
    figsize: list of two values
        Dimensions of the figure to be plotted
    s: float
        Size of data points
    c: str or array
        Colour of the data points
    cmap: str
        Colormap in case c is an array
    alpha: float [0,1]
        Transparency of points
    list_locs: dict
        Dictionary with location names and [lon, lat] positions

    Returns:
    --------
    Geographic map with the latitude and longitude positions
    """
    plt.figure(figsize = figsize)
    plt.scatter(lon, lat, s = s, c = c, cmap = cmap, alpha = alpha)
    if list_locs is not None:
        plot_locations(list_locs)
    plt.title(title)
    plt.xlabel('longitude')
    plt.ylabel('latitude')

def locations(list_locs, c = 'k'):
    """
    This method plots the main data locations on a map.

    Parameters:
    -----------
    list_locs: dict
        Dictionary with location names and [lon, lat] positions
    c: str or colour code
        Color of dots and text

    Returns:
    --------
    Plotted points and location names
    """
    for loc in list_locs:
        plt.scatter(list_locs[loc][0], list_locs[loc][1], c = c, s = 30)
        plt.annotate(loc, xy = [list_locs[loc][0] + .02, list_locs[loc][1]], color = c)

def spatial_hist2d_map(lon, lat, list_locs, title = '', bins = 10, ranges = None, cmap = 'binary', loc_col = 'tab:red', vmin = None, vmax = None):
    """
    This method shows a map of densities from a 2d histogram of geographical
    positions.

    Parameters:
    -----------
    lon: np.array
        Longitude coordinates
    lat: np.array
        Latitude coordinates
    list_locs: dict
        Dictionary with location names and [lon, lat] positions
    title: str
        Title of the plot
    bins: int or [int,int]
        Number of bins for the 2d histogram
    ranges: list of shape (2,2)
        It specifies the lat and lon ranges in the histogram
    cmap: str
        Colour map of 2d histogram
    loc_col: str or colour code
        Color of dots and text for the location of positions from list_locs
    vmin: float
        Minimum value for the colormap
    vmax: float
        Maximum value for the colormap


    Returns:
    --------
    Plotted points and location names

    """
    if ranges is None:
        ranges = [[np.min(lon), np.max(lon)],[np.min(lat), np.max(lat)]]
    plt.figure(figsize=[8,6])
    plt.hist2d(lon, lat, bins = bins, range = ranges, cmap = cmap, vmin = vmin, vmax = vmax)
    plt.colorbar()
    plot_locations(list_locs, c = loc_col)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(title)
    plt.show()

def spatial_hist2d_ratio(lon1, lat1, lon2, lat2, list_locs, title = '', bins = 10, ranges = None, cmap = 'viridis', loc_col = 'r', vmin = None, vmax = None):
    """
    This method shows the ratio of two 2d histograms in spatial coordinates.

    Parameters:
    -----------
    lon1: np.array
        Longitude coordinates for the first histogram
    lat1: np.array
        Latitude coordinates for the first histogram
    lon2: np.array
        Longitude coordinates for the second histogram
    lat2: np.array
        Latitude coordinates for the second histogram
    list_locs: dict
        Dictionary with location names and [lon, lat] positions
    title: str
        Title of the plot
    bins: int or [int,int]
        Number of bins for the 2d histogram
    ranges: list of shape (2,2)
        It specifies the lat and lon ranges in the histogram
    cmap: str
        Colour map of 2d histogram
    loc_col: str or colour code
        Color of dots and text for the location of positions from list_locs
    vmin: float
        Minimum value for the colormap
    vmax: float
        Maximum value for the colormap

    Returns:
    --------
    ratio: np.ndarray
        Ratio of the two histograms
    xedges: np.array
        Values of edges in x bins
    yedges: np.array
        Values of edges in y bins
    """
    #Define lon and lat ranges
    if ranges is None:
        min_lon1, min_lon2 = np.min(lon1), np.min(lon2)
        max_lon1, max_lon2 = np.max(lon1), np.max(lon2)
        min_lon, max_lon = min([min_lon1, min_lon2]), max([max_lon1, max_lon2])
        min_lat1, min_lat2 = np.min(lat1), np.min(lat2)
        max_lat1, max_lat2 = np.max(lat1), np.max(lat2)
        min_lat, max_lat = min([min_lat1, min_lat2]), max([max_lat1, max_lat2])
        ranges = [[min_lon, max_lon],[min_lat, max_lat]]
    #Calculating histograms and ratio
    h1, xedges, yedges = np.histogram2d(lon1, lat1, bins = bins, range = ranges)
    h2, xedges, yedges = np.histogram2d(lon2, lat2, bins = bins, range = ranges)
    ratio = h1/h2

    #Make figure
    plt.figure(figsize=[8,6])
    plt.pcolormesh(xedges, yedges, ratio.T, cmap = cmap, vmin = vmin, vmax = vmax)
    plt.colorbar()
    plot_locations(list_locs, c = 'r')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(title)

    return ratio, xedges, yedges
