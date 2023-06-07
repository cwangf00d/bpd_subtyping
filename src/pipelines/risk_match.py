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
    w_df.columns = ['PatientSeqID', 'DSB', 'Label'] + list(w_df.columns[3:])
    return w_df


def matched_risk(df, o_df, day, output_path):
    mr_suffix = '_mr_df.csv'
    r_suffix = '_r_df.csv'
    # take a df with risk broken down into the 5 classes; draw histograms for each class; choose top 10%
    results = {}
    # Loop over the columns, excluding the 'Patient_ID' column
    for i, column in enumerate(df.columns.drop(['PatientSeqID', 'DSB', 'Label'])):
        # Sort dataframe by column values in descending order
        df_sorted = df.loc[df['DSB'] == day].sort_values(by=column, ascending=False)
        # Get the top 10% rows
        top_10_percent = df_sorted.head(round(int(len(df_sorted) * 0.1)))
        # Store the patient ID and the relevant class's probability into the dictionary
        top_10_percent.to_csv(output_path + 'd' + str(day) + 'g' + str(i) + mr_suffix, index=False)
        pids = list(top_10_percent['PatientSeqID'])
        # make dataframe with all patient data
        risk_df = o_df.loc[o_df['PatientSeqID'].isin(pids)]
        risk_df.to_csv(output_path + 'd' + str(day) + 'g' + str(i) + r_suffix, index=False)
        results[column] = top_10_percent[['PatientSeqID', column]]
    return results


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







