#This module contains methods to identify Friends of Friends groups.

import numpy as np
from scipy import spatial

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
            if dist <= scale:
                indeces[i].append(index[0][0])
            else:
                break
        indeces[i] = np.array(indeces[i], dtype = int)
    return indeces

def get_fofid(positions, scale):
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
        if len(indeces[i]) == 1:
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

def get_fof_PR(positions, test_result, scale, fofid = None):
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

    Returns:
    --------
    fofid: np.array
        List of the FoF IDs of each position, with 0 for those
        without a FoF.
    mean_pr_fof: np.array
        Mean PR corresponding to fofid
    fof_catalogue: dict
        Dictionary of a catalogue of the FOF groups and its main characteristics
    """
    #Define positions of positive cases
    positive_positions = positions[test_result == 1]
    #Computing fofid is needed
    if fofid is None:
        fofid = fof.get_fofid(positive_positions, scale)
    #Create KDTree for all populations
    tree =spatial.KDTree(positions)

    #Mean PR for the positive cases in FOFs
    mean_pr_fof = np.zeros_like(fofid)
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
        all_friends_indeces = fof.get_friends_indeces(positive_positions[fofid_indeces], scale, tree)
        #get unique values of such indeces
        total_friends_indeces = np.unique(np.concatenate(all_friends_indeces))
        #get mean infection from all the unique indeces
        mean_pr = np.mean(test_result[total_friends_indeces])
        #assign mean PR to each fofid for the positive cases
        mean_pr_fof[fofid_indeces] = mean_pr
        #assign mean PR to each extended fofid
        mean_pr_ext_fof[total_friends_indeces] = mean_pr
        #assign FOF id to non-infected positions
        extended_fofid[total_friends_indeces] = f
        #setting FOF catalogue
        fof_catalogue['id'].append(f)
        fof_catalogue['mean_pos'].append(np.mean(positive_positions[fofid_indeces], axis = 0))
        fof_catalogue['mean_pos_ext'].append(np.mean(positions[total_friends_indeces], axis = 0))
        fof_catalogue['mean_pr'].append(mean_pr)
        fof_catalogue['positives'].append(len(fofid_indeces))
        fof_catalogue['negatives'].append(len(total_friends_indeces) - len(fofid_indeces))
        fof_catalogue['total'].append(len(total_friends_indeces))
        fof_catalogue['indeces'].append(total_friends_indeces)
    return fofid, mean_pr_fof, fof_catalogue
