#################################################################################
# Imports + Package Setup                                                       #
#################################################################################
import pandas as pd
from src.pipelines.vectorize import fup_day, make_dfs
from src.pipelines.cluster import make_clusters_sl
from src.pipelines.validate import run_bootstrap_sl
from src.pipelines.risk_match import expected_risk, matched_risk, jensen_risk

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
RDATA_DIR_PATH = DATA_DIR_PATH + 'processed/risk/'
print('variables set')

# timeframes
DAYS = [1, 3, 7, 14, 21, 27]


##########################################
# Vectorize                              #
##########################################
# # TRANSFORMER DATASETS
# # reading in internal/external embeddings + combining into one transformer csv
# ti_df = pd.read_csv(DATA_DIR_PATH + 'predictions/internal_embeddings.csv')
# te_df = pd.read_csv(DATA_DIR_PATH + 'predictions/external_embeddings.csv')
# t_cols = ['PatientSeqID', 'DSB', 'Support_Level_36'] + list(ti_df.columns)[3:]
# ti_df.columns, te_df.columns = t_cols, t_cols
# ta_df = pd.concat([ti_df, te_df])
# ta_df = ta_df.drop(columns=['Support_Level_36'])
# # merging to get bpd grade status + filter by inclusion criteria across specific days
dis_df = pd.read_csv(DATA_DIR_PATH + 'discharge_bpd_status.csv')
pm_df = pd.read_csv(DATA_DIR_PATH + 'patient_manifest.csv')
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
# reading in internal/external risk predictions + combining into one transformer csv
tir_df = pd.read_csv(DATA_DIR_PATH + 'predictions/test_internal.csv')
ter_df = pd.read_csv(DATA_DIR_PATH + 'predictions/test_external.csv')
tar_df = pd.concat([tir_df, ter_df])
inclusion_criteria = list(pm_df['PatientSeqID'])
tar_df = tar_df.loc[tar_df['Patient'].isin(inclusion_criteria)]
# assign risk
tjr_df = jensen_risk(tar_df)
tar_df = expected_risk(tar_df)
# saving to csv for easy access later
tar_df.to_csv(DATA_DIR_PATH + '/processed/full/tar_df.csv', index=False)
tjr_df.to_csv(DATA_DIR_PATH + '/processed/full/tjr_df.csv', index=False)
# # to run over various timeframes and create risk datasets for each, we can use the following:
# v_suffix = 'v_df.csv'
# v_dfs = ['tga_d1_', 'tga_d3_', 'tga_d7_', 'tga_d14_', 'tga_d21_', 'tga_d27_']
# for day, vdf in zip(DAYS, v_dfs):
#     o_df = pd.read_csv(VDATA_DIR_PATH + vdf + v_suffix)
#     matched_risk(tar_df, o_df, day, DATA_DIR_PATH + '/processed/risk/tga_')

# to run over various timeframes and create risk datasets for each, we can use the following:
v_suffix = 'v_df.csv'
o_df = pd.read_csv(VDATA_DIR_PATH + 'tga_d27_' + v_suffix)
for day in DAYS:
    matched_risk(tar_df, o_df, day, DATA_DIR_PATH + '/processed/risk/t27g_')

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

# # PASS 1
v_suffix = 'v_df.csv'
c_suffix = 'c_df.csv'
u_suffix = 'u_df.csv'
# to_cluster = ['tga_d1_', 'tga_d3_', 'tga_d7_', 'tga_d14_', 'tga_d21_', 'tga_d28_']
# for dataset in to_cluster:
#     print('loading', dataset)
#     curr_df = pd.read_csv(VDATA_DIR_PATH + dataset + v_suffix)
#     make_clusters_sl(curr_df, CDATA_DIR_PATH + 'new/' + dataset + c_suffix, CDATA_DIR_PATH + 'new/' + dataset + u_suffix,
#     visualize=False)
#     print(dataset, 'cluster complete')
# print('all clusters complete')

# # PASS 1.5: trying to not use random state to find most replicable clustering
# v_suffix = 'v_df.csv'
# c_suffix = 'c_df.csv'
# u_suffix = 'u_df.csv'
# to_cluster = ['tga_d1_', 'tga_d14_']
# for dataset in to_cluster:
#     print('loading', dataset)
#     curr_df = pd.read_csv(VDATA_DIR_PATH + dataset + v_suffix)
#     best_c_df, best_ss = make_clusters_sl(curr_df, 'no output', CDATA_DIR_PATH + dataset +
#                                           u_suffix, save_csv=False, visualize=False, random=True)
#     for i in range(50):
#         curr_c_df, curr_ss = make_clusters_sl(curr_df, 'no output', CDATA_DIR_PATH + dataset +
#                                               u_suffix, save_csv=False, visualize=False, random=True)
#         if curr_ss > best_ss:
#             best_c_df, best_ss = curr_c_df.copy(), curr_ss
#             print('current best ss:', best_ss)
#     best_c_df.to_csv(CDATA_DIR_PATH + dataset + c_suffix, index=False)
#     print(dataset, 'cluster complete')
# print('all clusters complete')

# # PASS 2
# to_cluster = ['tga_d1g0_', 'tga_d1g1_', 'tga_d1g2_', 'tga_d1g3_', 'tga_d1g4_',
#               'tga_d3g0_', 'tga_d3g1_', 'tga_d3g2_', 'tga_d3g3_', 'tga_d3g4_',
#               'tga_d7g0_', 'tga_d7g1_', 'tga_d7g2_', 'tga_d7g3_', 'tga_d7g4_',
#               'tga_d14g0_', 'tga_d14g1_', 'tga_d14g2_', 'tga_d14g3_', 'tga_d14g4_',
#               'tga_d21g0_', 'tga_d21g1_', 'tga_d21g2_', 'tga_d21g3_', 'tga_d21g4_',
#               'tga_d27g0_', 'tga_d27g1_', 'tga_d27g2_', 'tga_d27g3_', 'tga_d27g4_']
#
# r_suffix = 'r_df.csv'
# for dataset in to_cluster:
#     print('loading', dataset)
#     curr_df = pd.read_csv(RDATA_DIR_PATH + dataset + r_suffix)
#     make_clusters_sl(curr_df, CDATA_DIR_PATH + dataset + c_suffix, CDATA_DIR_PATH + dataset + u_suffix,
#                      visualize=False)
#     print(dataset, 'cluster complete')
# print('all clusters complete')


# PASS 2.5 risk with all data up to D27
to_cluster = ['t27g_d1g0_', 't27g_d1g1_', 't27g_d1g2_', 't27g_d1g3_', 't27g_d1g4_',
              't27g_d3g0_', 't27g_d3g1_', 't27g_d3g2_', 't27g_d3g3_', 't27g_d3g4_',
              't27g_d7g0_', 't27g_d7g1_', 't27g_d7g2_', 't27g_d7g3_', 't27g_d7g4_',
              't27g_d14g0_', 't27g_d14g1_', 't27g_d14g2_', 't27g_d14g3_', 't27g_d14g4_',
              't27g_d21g0_', 't27g_d21g1_', 't27g_d21g2_', 't27g_d21g3_', 't27g_d21g4_',
              't27g_d27g0_', 't27g_d27g1_', 't27g_d27g2_', 't27g_d27g3_', 't27g_d27g4_']

r_suffix = 'r_df.csv'
for dataset in to_cluster:
    print('loading', dataset)
    curr_df = pd.read_csv(RDATA_DIR_PATH + dataset + r_suffix)
    make_clusters_sl(curr_df, CDATA_DIR_PATH + dataset + c_suffix, CDATA_DIR_PATH + dataset + u_suffix,
                     visualize=False)
    print(dataset, 'cluster complete')
print('all clusters complete')

##########################################
# Cluster Validation                     #
##########################################
"""
This will perform a set number of bootstrap replications for goal 2 of the three-pronged validation + silhouette scores
"""
# BOOTSTRAP + SILHOUETTE SCORES
BOOTSTRAP_REPS = 100
u_suffix = 'u_df.csv'
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
# to_bootstrap = ['tga_d1_', 'tga_d3_', 'tga_d7_', 'tga_d14_', 'tga_d21_', 'tga_d28_']
# cluster_col = 'umap_KMeans'
# # to_bootstrap = ['tga_d1_', 'tga_d14_']
# for data in to_bootstrap:
#     print('loading in', data)
#     curr_c_df = pd.read_csv(CDATA_DIR_PATH + 'new/' + data + c_suffix)
#     num_clusters = len(list(set(curr_c_df[cluster_col])))
#     curr_u_df = pd.read_csv(CDATA_DIR_PATH + 'new/' + data + u_suffix)
#     curr_n = curr_c_df.shape[0]
#     bs_output = SDATA_DIR_PATH + 'new/' + data + 'bs.json'
#     # ss_output = SDATA_DIR_PATH + data + 'ss.json'
#     run_bootstrap_sl(curr_u_df, curr_c_df, curr_n, BOOTSTRAP_REPS, bs_output, num_clusters)
#     print('bootstrap complete!')

# to_bootstrap = ['tga_d14g0_', 'tga_d14g1_', 'tga_d14g2_', 'tga_d14g3_', 'tga_d14g4_',
#                 'tga_d21g0_', 'tga_d21g1_', 'tga_d21g2_', 'tga_d21g3_', 'tga_d21g4_',
#                 'tga_d27g0_', 'tga_d27g1_', 'tga_d27g2_', 'tga_d27g3_', 'tga_d27g4_']
# cluster_col = 'umap_KMeans'
# for data in to_bootstrap:
#     print('loading in', data)
#     curr_c_df = pd.read_csv(CDATA_DIR_PATH + data + c_suffix)
#     num_clusters = len(list(set(curr_c_df[cluster_col])))
#     curr_u_df = pd.read_csv(CDATA_DIR_PATH + data + u_suffix)
#     curr_n = curr_c_df.shape[0]
#     bs_output = SDATA_DIR_PATH + data + 'bs.json'
#     run_bootstrap_sl(curr_u_df, curr_c_df, curr_n, BOOTSTRAP_REPS, bs_output, num_clusters)
#     print('bootstrap complete!')

to_bootstrap = ['t27g_d14g0_', 't27g_d14g1_', 't27g_d14g2_', 't27g_d14g3_', 't27g_d14g4_',
                't27g_d21g0_', 't27g_d21g1_', 't27g_d21g2_', 't27g_d21g3_', 't27g_d21g4_',
                't27g_d27g0_', 't27g_d27g1_', 't27g_d27g2_', 't27g_d27g3_', 't27g_d27g4_']
cluster_col = 'umap_KMeans'
for data in to_bootstrap:
    print('loading in', data)
    curr_c_df = pd.read_csv(CDATA_DIR_PATH + data + c_suffix)
    num_clusters = len(list(set(curr_c_df[cluster_col])))
    curr_u_df = pd.read_csv(CDATA_DIR_PATH + data + u_suffix)
    curr_n = curr_c_df.shape[0]
    bs_output = SDATA_DIR_PATH + data + 'bs.json'
    run_bootstrap_sl(curr_u_df, curr_c_df, curr_n, BOOTSTRAP_REPS, bs_output, num_clusters)
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
