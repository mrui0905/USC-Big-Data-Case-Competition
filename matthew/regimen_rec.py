import pandas as pd
import numpy as np

from sklearn.neighbors import NearestNeighbors

def label(x):
    '''
    x should be array of baseline [vl, cd4, relcd4]
    '''

    x = np.asarray(x).reshape(1, -1)

    centers = np.asarray([
        [1.04668581e+03, 6.70628946e+02, 3.93743832e+01],
        [7.38870903e+04, 1.12735257e+03, 1.70159312e+01],
        [3.69610218e+04, 8.43187852e+02, 1.85942520e+01]])

    nn = NearestNeighbors().fit(centers)
    return int(nn.kneighbors(n_neighbors = 1, X = x)[1])

def regimen_reccomend(gender, ethnicity, vl_levels, cd4_levels, cd4_percent):
    cluster = label([float(vl_levels), float(cd4_levels), float(cd4_percent)])

    hazard_estimates = pd.read_csv(f'{ethnicity}_{gender}/label={cluster}.csv')
    weight_vl, weight_cd4 = np.round(float(vl_levels)/50, 2), np.round(500/float(cd4_levels), 2)
    denom = 2*weight_vl + weight_cd4
    hazard_estimates['weighted_score'] = np.power(hazard_estimates['effect_vl50'], weight_vl/denom) * np.power(hazard_estimates['effect_vl250'], weight_vl/denom) * np.power(hazard_estimates['effect_cd500'], weight_cd4/denom)
    hazard_estimates = hazard_estimates.sort_values(['weighted_score'], ascending=False)
    return hazard_estimates