#################################################################################
# Imports + Package Setup                                                       #
#################################################################################
import pandas as pd
from src.pipelines.vectorize import fup_day, make_dfs
from src.pipelines.cluster import make_clusters
from src.pipelines.validate import run_bootstrap, get_silhouette_scores
from src.pipelines.risk_match import expected_risk

#################################################################################
# Data Handling                                                                 #
#################################################################################
# choose a location: local or o2
DATA_DIR_PATH = '/Users/cindywang/PycharmProjects/bpd-subtyping/data/'  # local
# DATA_DIR_PATH = '/n/data2/hms/dbmi/beamlab/cindy/bpd-subtyping/data/'  # o2

# other path shortcuts
VDATA_DIR_PATH = DATA_DIR_PATH + 'processed/vector/'
CDATA_DIR_PATH = DATA_DIR_PATH + 'processed/clusters/'
SDATA_DIR_PATH = DATA_DIR_PATH + 'processed/scores/'
# OUTPUT_PATH = DATA_DIR_PATH + 'processed/vector/'
print('variables set')

# timeframes
DAYS = [1, 3, 7, 14, 21, 28]


##########################################
# Vectorize                              #
##########################################
# TRANSFORMER DATASETS
# reading in internal/external embeddings + combining into one transformer csv
# ti_df = pd.read_csv(DATA_DIR_PATH + 'predictions/internal_embeddings.csv')
# te_df = pd.read_csv(DATA_DIR_PATH + 'predictions/external_embeddings.csv')
# t_cols = ['PatientSeqID', 'DSB', 'Support_Level_36'] + list(ti_df.columns)[3:]
# ti_df.columns, te_df.columns = t_cols, t_cols
# ta_df = pd.concat([ti_df, te_df])
# ta_df = ta_df.drop(columns=['Support_Level_36'])
# # merging to get bpd grade status + filter by inclusion criteria across specific days
# dis_df = pd.read_csv(DATA_DIR_PATH + 'discharge_bpd_status.csv')
# pm_df = pd.read_csv(DATA_DIR_PATH + 'patient_manifest.csv')
# for day in DAYS:
#     tv_df = fup_day(ta_df, day).merge((dis_df.merge(pm_df)).loc[:, ['PatientSeqID', 'BPD Grade']])
#     tv_df.to_csv(DATA_DIR_PATH + '/processed/full/tv_' + str(day) + '_df.csv', index=False)
#     T_DFT_COLS = list(tv_df.columns)[2:-1]  # daily feature columns in transformer dataframe
#     T_SFT_COLS = list()  # single value columns in transformer dataframe
#     t_all_output = VDATA_DIR_PATH + 'tga_d' + str(day) + '_'
#     make_dfs(tv_df, [0, 1, 2, 3], 1, day, T_DFT_COLS, T_SFT_COLS, t_all_output)
# ''' now, the dataframes will all be in the format where each row = unique patient, and the rows are formatted
#     [PatientSeqID, DSB 0 dim 0, DSB 0 dim 1, ..., DSB 0 dim 127, DSB 1 dim 0, ..., DSB day dim 127]'''

##########################################
# Risk Matching                          #
##########################################
'''The point is to make a dataframe that has Patient ID, DSB, Risk altogether for ease in separating
    moving forward'''
# # reading in internal/external risk predictions + combining into one transformer csv
# tir_df = pd.read_csv(DATA_DIR_PATH + 'predictions/test_internal.csv')
# ter_df = pd.read_csv(DATA_DIR_PATH + 'predictions/test_external.csv')
# tar_df = pd.concat([tir_df, ter_df])
# inclusion_criteria = list(pm_df['PatientSeqID'])
# tar_df = tar_df.loc[tar_df['Patient'].isin(inclusion_criteria)]
# # assign risk
# tar_df = expected_risk(tar_df)
# # saving to csv for easy access later
# tar_df.to_csv(DATA_DIR_PATH + '/processed/full/tar_df.csv', index=False)


##########################################
# Clustering                             #
##########################################
"""
To take vectorized datasets + find ideal clusterings with UMAP dimensionality reduction + KMeans clustering. Will run
two main sets of experiments: one clustering over all patients for a given timeframe, next clustering over risk-matched
groups for a given timeframe
"""
'''
setup:
    to_cluster = ['data_name', 'data_name', ..., 'data_name']
    for dataset in to_cluster:
        print('loading', dataset)
        curr_df = pd.read_csv(VDATA_DIR_PATH + dataset + v_suffix)
        make_clusters(curr_df, CDATA_DIR_PATH + dataset + c_suffix)
        print(dataset, 'cluster complete')
    print('all clusters complete')
'''

# v_suffix = 'v_df.csv'
# c_suffix = 'c_df.csv'
# tgat1c_df = pd.read_csv(CDATA_DIR_PATH + 'tgat1c_df.csv')
# tgat1cb_pids = list(tgat1c_df.loc[tgat1c_df['umap_KMeans'] == 1]['PatientSeqID'])
# tgat1v_df = pd.read_csv(VDATA_DIR_PATH + 'tgat1v_df.csv')
# tgat1cbv_df = tgat1v_df.loc[tgat1v_df['PatientSeqID'].isin(tgat1cb_pids)]
# tgat1cbv_df.to_csv(VDATA_DIR_PATH + 'tgat1cbv_df.csv', index=False)
# to_cluster = ['tgat1cb']
# for dataset in to_cluster:
#     print('loading', dataset)
#     curr_df = pd.read_csv(VDATA_DIR_PATH + dataset + v_suffix)
#     make_clusters(curr_df, CDATA_DIR_PATH + dataset + c_suffix)
#     print(dataset, 'cluster complete')
# print('all clusters complete')

# TODO get this working for risk matching

# to_cluster = ['tg0d7r', 'tg1d7r', 'tg2d7r', 'tg3d7r']  # risk matched groups
# day = 7
# tar_df = pd.read_csv(DATA_DIR_PATH + '/processed/full/tar_df.csv')
# for i, dataset in enumerate(to_cluster):
#     print('loading', dataset)
#     curr_df = pd.read_csv(VDATA_DIR_PATH + 'tgat1' + v_suffix)
#     patient_ids = list(tar_df.loc[(tar_df['Risk'] == i) & (tar_df['DSB'] == day)]['PatientSeqID'])
#     curr_df = curr_df.loc[curr_df['PatientSeqID'].isin(patient_ids)]
#     make_clusters(curr_df, CDATA_DIR_PATH + 'tg0r' + c_suffix)
#     print(dataset, 'cluster done')
# print('all clusters done')


##########################################
# Cluster Validation                     #
##########################################
"""
This will perform a set number of bootstrap replications for goal 2 of the three-pronged validation + silhouette scores
"""
# BOOTSTRAP + SILHOUETTE SCORES
BOOTSTRAP_REPS = 100
v_suffix = 'v_df.csv'
c_suffix = 'c_df.csv'
'''
setup:
    to_bootstrap = ['name']
    for data in to_bootstrap:
        print('loading in', data)
        curr_c_df = pd.read_csv(CDATA_DIR_PATH + data + c_suffix)
        curr_v_df = pd.read_csv(VDATA_DIR_PATH + data + v_suffix)
        curr_n = curr_c_df.shape[0]
        bs_output = SDATA_DIR_PATH + data + '_bs.json'
        ss_output = SDATA_DIR_PATH + data + '_ss.json'
        run_bootstrap(curr_v_df, curr_c_df, curr_n, BOOTSTRAP_REPS, bs_output)
        print('bootstrap complete!')
'''
to_bootstrap = ['tgat1cb']
for data in to_bootstrap:
    print('loading in', data)
    curr_c_df = pd.read_csv(CDATA_DIR_PATH + data + c_suffix)
    curr_v_df = pd.read_csv(VDATA_DIR_PATH + data + v_suffix)
    curr_n = curr_c_df.shape[0]
    bs_output = SDATA_DIR_PATH + data + '_bs.json'
    ss_output = SDATA_DIR_PATH + data + '_ss.json'
    run_bootstrap(curr_v_df, curr_c_df, curr_n, BOOTSTRAP_REPS, bs_output)
    print('bootstrap complete!')

# TODO: figure out unit-testing of get_silhouette_scores
# v_dfs = [mg0t1_df, mg0t3_df, mg1t1_df, mg1t3_df, mg2t1_df, mg2t3_df, mg3t1_df, mg3t3_df, mgat1_df, mgat3_df,
#          tg0t1_df, tg0t3_df, tg1t1_df, tg1t3_df, tg2t1_df, tg2t3_df, tg3t1_df, tg3t3_df, tgat1_df, tgat3_df]
# cluster_df_paths = ['mg0t1c_df.csv', 'mg0t3c_df.csv', 'mg1t1c_df.csv', 'mg1t3c_df.csv', 'mg2t1c_df.csv',
#                     'mg2t3c_df.csv', 'mg3t1c_df.csv', 'mg3t3c_df.csv', 'mgat1c_df.csv', 'mgat3c_df.csv',
#                     'tg0t1c_df.csv', 'tg0t3c_df.csv', 'tg1t1c_df.csv', 'tg1t3c_df.csv', 'tg2t1c_df.csv',
#                     'tg2t3c_df.csv', 'tg3t1c_df.csv', 'tg3t3c_df.csv', 'tgat1c_df.csv', 'tgat3c_df.csv']
# for suffix, curr_v_df in zip(cluster_df_paths, v_dfs):
#     curr_c_df = pd.read_csv(CDATA_DIR_PATH + suffix)
#     curr_n = curr_c_df.shape[0]
#     bs_output = SDATA_DIR_PATH + suffix.split('c')[0] + '_bs.json'
#     ss_output = SDATA_DIR_PATH + suffix.split('c')[0] + '_ss.json'
#     run_bootstrap(curr_v_df, curr_c_df, curr_n, BOOTSTRAP_REPS, bs_output)
#     cols = list(curr_c_df.columns)[1:]
#     get_silhouette_scores(curr_v_df, curr_c_df, cols, ss_output)
