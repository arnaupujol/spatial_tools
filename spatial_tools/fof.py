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
    #distances, indeces = tree.query(positions, k = len(positions), distance_upper_bound = scale)
    indeces = get_friends_indeces(positions, scale, tree)

    last_fofid = 0
    for i in range(len(positions)):
        #check if ith position has any neighbour
        if len(indeces[i]) == 1:
            continue
        else:
            #Mask indeces farther than scale
            #indeces_mask = indeces[i] < len(positions)
            #Define indeces of selected friends
            indeces_friends = indeces[i]#[indeces_mask]
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
