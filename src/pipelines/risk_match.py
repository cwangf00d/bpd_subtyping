#################################################################################
# Imports + Package Setup                                                       #
#################################################################################
import numpy as np
import pandas as pd
from tqdm import tqdm


#################################################################################
# Helper Functions                                                              #
#################################################################################
def expected_risk(df):
    # create new columns for each grade's probability
    w_df = df.copy().loc[:, ['Patient', 'Day', 'Label']]
    w_df['grade_0_prob'] = df['prob_0']
    w_df['grade_1_prob'] = df['prob_1'] + df['prob_2']
    w_df['grade_2_prob'] = df['prob_3'] + df['prob_4'] + df['prob_5']
    w_df['grade_3_prob'] = df['prob_6'] + df['prob_7']
    w_df['grade_minus1_prob'] = df['prob_8']
    w_df['expected_grade'] = 4*w_df['grade_minus1_prob'] + 0*w_df['grade_0_prob'] + 1*w_df['grade_1_prob'] + \
                             2*w_df['grade_2_prob'] + 3*w_df['grade_3_prob']
    w_df['int_exp_grade'] = round(w_df['expected_grade'])
    w_df.columns = ['PatientSeqID', 'DSB', 'Label'] + list(w_df.columns[3:])
    return w_df


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
    curr_df['Cat_Risk'] = curr_df['Risk'].map(risk_mapping)
    return curr_df







