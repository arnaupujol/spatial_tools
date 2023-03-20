#This module contains methods to identify Friends of Friends groups.

import numpy as np
import pandas as pd
import geopandas
from scipy import spatial, stats

def get_friends_indeces(positions, scale, tree):
    """
    This method returns the indeces of all the friends
    of each position from positions given a KDTree.

    Parameters:
    -----------
    positions: np.ndarray
        An array with the position parameters with shape (n,2),
        where n is the number of positions
    scale: float
        The linking scale of the FoF
    tree: scipy.spatial.KDTree
        A KDTree build from the positions of the target data

    Returns:
    --------
    indeces: list
        List with an array of the indeces of the friends of each
        position
    """
    indeces = []
    for i in range(len(positions)):
        indeces.append([])
        dist = 0
        kth = 0
        while dist <= scale:
            kth += 1
            dist, index = tree.query([positions[i]], k = [kth])
            if dist == 0 and kth > 1:#avoiding issue for >1 point with dist == 0
                d, index = tree.query([positions[i]], k = kth)
                indeces[i] = index[0].tolist()
            elif dist <= scale:
                indeces[i].append(index[0][0])
            else:
                break
        indeces[i] = np.array(indeces[i], dtype = int)
    return indeces

def get_fofid(positions, scale, min_neighbours = 2):
    """
    This method finds the Friends of Friends (FoF) groups from
    a set of positions and returns their FoF IDs.

    Parameters:
    -----------
    positions: np.ndarray
        An array with the position parameters with shape (n,2),
        where n is the number of positions
    scale: float
        The linking scale of the FoF
    min_neighbours: int
        Minium number of neighbours in the radius < scale needed to consider
        the FOF group

    Returns:
    --------
    fofid: np.array
        List of the FoF IDs of each position, with 0 for those
        without a FoF.
    """
    #Create FOF id
    fofid = np.zeros(len(positions))

    #Create KDTree
    tree =spatial.KDTree(positions)
    #Query KDTree
    indeces = get_friends_indeces(positions, scale, tree)

    last_fofid = 0
    for i in range(len(positions)):
        #check if ith position has any neighbour
        if len(indeces[i]) < min_neighbours:
            continue
        else:
            #Define indeces of selected friends
            indeces_friends = indeces[i]
            #FOFids of these friends
            fofid_friends = fofid[indeces_friends]
            #Unique valies of fofids
            unique_fofids = np.unique(fofid_friends)
            #check values of fofid in these neighbours
            if len(unique_fofids) == 1:
                if unique_fofids[0] == 0:
                    #assign to ith and friends last_fofid
                    fofid[indeces_friends] = last_fofid + 1
                    last_fofid+=1
                else:
                    #if one fofid different than 0, assign it to ith and friends
                    fofid[indeces_friends] = unique_fofids[0]
            else:
                #Define the fofid to assign for merging several fof
                min_fofid = np.min(unique_fofids[unique_fofids != 0])
                #Assign this fofis to ith and its friends
                fofid[indeces_friends] = min_fofid
                #Assign it to all cases with any of these fofid_friends
                for j in unique_fofids[unique_fofids != 0]:
                    fofid[fofid == j] = min_fofid
    #Rename FOFid to continuous integers
    for i, f in enumerate(np.unique(fofid[fofid>0])):
        fofid[fofid == f] = i+1
    return fofid

def get_fof_PR(positions, test_result, scale, fofid = None, min_neighbours = 2):
    """
    This method runs the Friends of Friends (FOF) algorithm (if fofid is None) and obtains the mean
    positivity rate (PR) of each FOF extended with the non-infected cases closer than the scale.

    Parameters:
    -----------

    Parameters:
    -----------
    positions: np.ndarray
        An array with the position parameters with shape (n,2),
        where n is the number of positions
    test_result: np.array
        An array with the test results (0 or 1)
    scale: float
        The linking scale of the FoF
    fofid: np.array
        An array with the FOF ids of the positive cases
    min_neighbours: int
        Minium number of neighbours in the radius < scale needed to consider
        the FOF group

    Returns:
    --------
    fofid: np.array
        List of the FoF IDs of each position, with 0 for those
        without a FoF.
    mean_pr_fof: np.array
        Mean PR corresponding to fofid
    pval_fof: np.array
        P-value corresponding to fofid
    fof_catalogue: geopandas.DataFrame
        Catalogue of the FOF groups and its main characteristics
    """
    #Define positions of positive cases
    positive_positions = positions[test_result == 1]
    #Computing fofid is needed
    if fofid is None:
        fofid = get_fofid(positive_positions, scale, \
                            min_neighbours = min_neighbours)
    #Create KDTree for all populations
    tree =spatial.KDTree(positions)
    #Define total number of positive and cases
    total_positives = np.sum(test_result)
    total_n = len(test_result)

    #Mean PR and p-value for the positive cases in FOFs
    mean_pr_fof = np.zeros_like(fofid)
    pval_fof = np.ones_like(fofid)
    #FOF catalogue
    fof_catalogue = {'id' : [], #FOF id
                     'mean_pos' : [], #Mean position of positive cases
                     'mean_pos_ext' : [], #Mean position of extended FOF
                     'mean_pr' : [], #Positivity rate
                     'positives' : [], #Number of positive cases
                     'negatives' : [], #Number of negative cases in extended FOF
                     'total' : [], #Total number of positions in extended FOF
                     'indeces' : [], #Indeces of all positions in extended FOF
                    }
    for i,f in enumerate(np.unique(fofid[fofid>0])):
        #get all indeces with this FOF id
        has_this_fofid = fofid == f
        fofid_indeces = np.arange(len(positive_positions))[has_this_fofid]
        #for all these indeces, get list of friends from all positions
        all_friends_indeces = get_friends_indeces(positive_positions[fofid_indeces], scale, tree)
        #get unique values of such indeces
        total_friends_indeces = np.unique(np.concatenate(all_friends_indeces))
        #get mean infection from all the unique indeces
        mean_pr = np.mean(test_result[total_friends_indeces])
        #assign mean PR and p-value to each fofid for the positive cases
        mean_pr_fof[fofid_indeces] = mean_pr
        pval_fof[fofid_indeces] = 1 - stats.binom.cdf(np.sum(test_result[total_friends_indeces]) - 1, \
                                                      len(total_friends_indeces), \
                                                      total_positives/total_n)
        #setting FOF catalogue
        fof_catalogue['id'].append(f)
        fof_catalogue['mean_pos'].append(np.mean(positive_positions[fofid_indeces], axis = 0))
        fof_catalogue['mean_pos_ext'].append(np.mean(positions[total_friends_indeces], axis = 0))
        fof_catalogue['mean_pr'].append(mean_pr)
        fof_catalogue['positives'].append(len(fofid_indeces))
        fof_catalogue['negatives'].append(len(total_friends_indeces) - len(fofid_indeces))
        fof_catalogue['total'].append(len(total_friends_indeces))
        fof_catalogue['indeces'].append(total_friends_indeces)
    #Make the fof_catalogue a geopandas dataframe
    fof_catalogue = fof2geodf(fof_catalogue)
    #Calculate p-value from binomial distribution assuming random infections
    fof_catalogue['p'] = 1 - stats.binom.cdf(fof_catalogue['positives']-1, \
                                             fof_catalogue['total'], \
                                             total_positives/total_n)#p-value of FOF
    return fofid, mean_pr_fof, pval_fof, fof_catalogue

def fof2geodf(fof_catalogue, epsg = 3857):
    """
    This method transforms the FOF catalogue dictionary
    into a geopandas dataframe.

    Parameters:
    -----------
    fof_catalogue: dict
        Dictionary with the FOF catalogue
    epsg: int
        GIS spatial projection of coordinates

    Returns:
    --------
    fof_geocat: geopandas.GeoDataFrame
        Data frame of FOF catalogue with GIS coordinates
    """
    fof_geocat = pd.DataFrame(fof_catalogue)
    x_points = np.array([i[0] for i in fof_geocat['mean_pos']])
    y_points = np.array([i[1] for i in fof_geocat['mean_pos']])
    fof_geocat = geopandas.GeoDataFrame(fof_geocat, \
                        geometry = geopandas.points_from_xy(x_points, y_points))
    fof_geocat = fof_geocat.set_crs(epsg=epsg)
    return fof_geocat

def get_dist(pos_a, pos_b):
    """
    This method calculates the Euclidean distance
    between two positions.

    Parameters:
    -----------
    pos_a: np.ndarray
        First position
    pos_b: np.ndarray
        Second position

    Returns:
    --------
    dist: float
        Distance between positions
    """
    dist = np.sqrt(np.sum((pos_a - pos_b)**2))
    return dist

def get_temp_id(fof_cat_list, linking_time, linking_dist, get_timelife = True):
    """
    This method generates the temporal ID of hotspots by linking
    hotspots from different time frames, assigning the same
    temporal ID to them when they are close enough in time and space.

    Parameters:
    -----------
    fof_cat_list: list of pandas.DataFrame
        List of hotspot catalogues, each element of the list
        corresponding to the hotspot catalogue of each timestep
    linking_time: int
        Maximum number of timesteps of distance to link hotspots with
        the same temporal ID
    linking_dist: float
        Linking distance used to link the hotspots from the different
        time frames
    get_timelife: bool
        It specifies if time periods and timelife of hotspots is obtained

    Returns:
    --------
    fof_cat_list: list of pandas.DataFrame
        List of hotspot catalogues with the added variable 'tempID'
    """
    #setting empty values of temp_id
    for t in range(len(fof_cat_list)):
        fof_cat_list[t]['tempID'] = pd.Series(dtype = int)
    #Initialising tempID value to assign
    next_temp_id = 0
    #Loop over all timesteps
    for t in range(len(fof_cat_list)):
        #Loop over all FOFs in a timestep
        for f in fof_cat_list[t].T:
            #Loop over all timesteps within linking_time
            for t2 in range(t + 1, min(t + linking_time, len(fof_cat_list))):
                #Loop over all FOFs in the linked timesteps
                for f2 in fof_cat_list[t2].T:
                    #Calculating distance between hotspots
                    dist = get_dist(fof_cat_list[t].loc[f]['mean_pos'], \
                                    fof_cat_list[t2].loc[f2]['mean_pos'])
                    if dist <= linking_dist:
                        temp_id1 = fof_cat_list[t].loc[f]['tempID']
                        temp_id2 = fof_cat_list[t2].loc[f2]['tempID']
                        #Assign tempIDs to linked hotspots
                        if np.isnan(temp_id1) and np.isnan(temp_id2):
                            fof_cat_list[t]['tempID'].loc[f] = next_temp_id
                            fof_cat_list[t2]['tempID'].loc[f2] = next_temp_id
                            next_temp_id += 1
                        elif np.isnan(temp_id1):
                            fof_cat_list[t]['tempID'].loc[f] = temp_id2
                        elif np.isnan(temp_id2):
                            fof_cat_list[t2]['tempID'].loc[f2] = temp_id1
                        elif temp_id1 != temp_id2:
                            for t3 in range(len(fof_cat_list)):
                                fof_cat_list[t3]['tempID'].loc[fof_cat_list[t3]['tempID'] == temp_id2] = temp_id1
    if get_timelife:
        fof_cat_list = get_lifetimes(fof_cat_list)
    return fof_cat_list

def get_lifetimes(fof_cat_list):
    """
    This method obtains the first and last time frames for each
    temporal ID from a list of hotspot catalogues and the corresponding
    timelife.

    Parameters:
    -----------
    fof_cat_list: list of pandas.DataFrame
        List of hotspot catalogues, each element of the list
        corresponding to the hotspot catalogue of each timestep

    Returns:
    --------
    fof_cat_list: list of pandas.DataFrame
        List of hotspot catalogues with the added fields 'first_timestep',
        'last_timestep' and 'lifetime'
    """
    #getting list of timeIDs appearing in fof_cat_list
    timeid_list = get_label_list(fof_cat_list, label = 'tempID')
    #Creating empty columns for first timestep, last timestep and lifteime
    for t in range(len(fof_cat_list)):
            fof_cat_list[t]['first_timestep'] = pd.Series(dtype = int)
            fof_cat_list[t]['last_timestep'] = pd.Series(dtype = int)
            fof_cat_list[t]['lifetime'] = 0
    for timeid_num in timeid_list:
        appearances = []
        for i in range(len(fof_cat_list)):
            if timeid_num in fof_cat_list[i]['tempID'].unique():
                appearances.append(i)
        min_appearance = min(appearances)
        max_appearance = max(appearances)
        lifetime = max_appearance - min_appearance
        for i in range(min_appearance, max_appearance + 1):
            fof_cat_list[i]['first_timestep'].loc[fof_cat_list[i]['tempID'] == timeid_num] = min_appearance
            fof_cat_list[i]['last_timestep'].loc[fof_cat_list[i]['tempID'] == timeid_num] = max_appearance
            fof_cat_list[i]['lifetime'].loc[fof_cat_list[i]['tempID'] == timeid_num] = lifetime
    return fof_cat_list

def get_label_list(df_list, label = 'tempID'):
    """
    This method gives the unique values of a column in a list
    of data frames.

    Parameters:
    -----------
    df_list: list of pandas.DataFrames
        List of dataframes
    label: str
        Name of column to select

    Returns:
    --------
    label_list: list
        List of unique values of the column over all dataframes
    """
    for i in range(len(df_list)):
        mask = df_list[i][label].notnull()
        if i == 0:
            label_list = df_list[i][label].loc[mask].unique()
        else:
            label_list = np.unique(np.concatenate((label_list, df_list[i][label].loc[mask].unique())))
    return label_list
