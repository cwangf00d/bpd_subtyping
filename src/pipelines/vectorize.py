#################################################################################
# Imports + Package Setup                                                       #
#################################################################################
import numpy as np
import pandas as pd
from tqdm import tqdm


#################################################################################
# Helper Functions                                                              #
#################################################################################
def fup_day(df, day):
    return df.loc[df['DSB'] < day]


# function to create time windows where data is averaged, should work for window = 1 and up
# windows should be divisible by days, w % d == 0
def make_windows(df, w_size, grades, days):
    # filtering for correct grades
    df = df.loc[df['BPD Grade'].isin(grades)]
    # identifying patients with sufficient data to have data in all windows
    id_lists = list()
    window_arrs = np.array([i for i in range(days)]).reshape(int(days/w_size), w_size).tolist()
    for w_arr in window_arrs:
        new_list = list(df.loc[df['DSB'].isin(w_arr)].groupby('PatientSeqID').count().reset_index()['PatientSeqID'])
        id_lists.append(new_list)
    valid_ids = set(id_lists[0])
    for id_list in id_lists[1:]:
        valid_ids = valid_ids & set(id_list)
    f_df = df.loc[df['PatientSeqID'].isin(valid_ids)]
    # take filtered data + create averages by time window
    mean_intervals = list()
    for w_arr in window_arrs:
        mean_intervals.append(f_df.loc[f_df['DSB'].isin(w_arr)].groupby('PatientSeqID').mean().reset_index())
    return pd.concat(mean_intervals).sort_values(by=['PatientSeqID', 'DSB'])


def vectorize(df, dft_cols, sft_cols, windows):
    df = df.sort_values(by=['PatientSeqID', 'DSB'])
    # vectorizing patient data into rows
    vs = list()
    ids = df.groupby('PatientSeqID').count().reset_index()['PatientSeqID']
    for p_id in tqdm(ids):
        p_df = df.loc[df['PatientSeqID'] == p_id]
        curr_v = [p_id]
        for ft in dft_cols:
            curr_v += list(p_df[ft])
        curr_v += list(np.array(p_df.groupby('PatientSeqID').mean()[sft_cols]).squeeze())
        vs.append(curr_v)
    # column names for vectorized dataframe
    cols = ['PatientSeqID']
    for col in dft_cols:
        cols += [col + str(i) for i in range(windows)]
    cols += sft_cols
    return pd.DataFrame(vs, columns=cols).set_index('PatientSeqID')


def make_dfs(df, grades, window_size, days, dft_cols, sft_cols, output_path):
    """
    will return full averaged dataframe acc to window size + will return a separate vectorized dataframe as tuple
    """
    w_df = make_windows(df, window_size, grades, days)
    v_df = vectorize(w_df, dft_cols, sft_cols, int(days/window_size))
    w_df.to_csv(output_path + 'w_df.csv', index=False)
    v_df.to_csv(output_path + 'v_df.csv')
    return w_df, v_df

