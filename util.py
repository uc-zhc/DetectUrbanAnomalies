import numpy as np

def pairPearson(A, B, precompute_p1):
    # Get number of rows in either A or B
    N = B.shape[0]

    # Store columnw-wise in A and B, as they would be used at few places
    sA = A.sum(0)
    sB = B.sum(0)

    # Basically there are four parts in the formula. We would compute them one-by-one
    p1 = N*precompute_p1
    p2 = sA*sB[:,None]
    p3 = N*((B**2).sum(0)) - (sB**2)
    p4 = N*((A**2).sum(0)) - (sA**2)

    # Finally compute Pearson Correlation Coefficient as 2D array 
    pcorr = ((p1 - p2)/np.sqrt(p4*p3[:,None]))
    pcorr[np.isinf(pcorr)] = 0
    pcorr[np.isnan(pcorr)] = 0
    return pcorr

 