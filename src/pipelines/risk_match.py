#################################################################################
# Imports + Package Setup                                                       #
#################################################################################
import numpy as np
import pandas as pd
from tqdm import tqdm


#################################################################################
# Helper Functions                                                              #
#################################################################################
def weighted_risk(df, ft_inds=range(3, 12)):
    weights = [i for i in range(9)]
    curr_df = df.iloc[:, ft_inds]*weights
    weighted_risks = round(curr_df.sum(axis=1))
    weighted_df = df.copy().loc[:, ['Patient', 'Day', 'Label']]
    weighted_df['risk'] = weighted_risks
    weighted_df.columns = ['PatientSeqID', 'DSB', 'Label', 'Risk']
    return weighted_df


def risk_grading(df, risk_mapping={0: 'g0', 1: 'g1', 2: 'g1', 3: 'g2', 4: 'g2', 5: 'g2', 6: 'g3', 7: 'g3', 8: 'g-1'}):
    curr_df = df.copy()
    curr_df['cat_risk'] = curr_df['Risk'].map(risk_mapping)
    return curr_df







