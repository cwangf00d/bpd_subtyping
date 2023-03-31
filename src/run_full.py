#################################################################################
# Imports + Package Setup                                                       #
#################################################################################
import pandas as pd
from src.pipelines.vectorize import fup_day, make_dfs
from src.pipelines.cluster import make_clusters
from src.pipelines.validate import run_bootstrap, get_silhouette_scores
from src.pipelines.risk_match import weighted_risk, risk_grading

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


##########################################
# Vectorize                              #
##########################################
"""
Making initial datasets mv_df.csv, tv_df.csv with all the combined information. Can skip if mv_df, tv_df already
created and go to line 77.
"""
# MANUAL DATASETS
# pm_df = pd.read_csv(DATA_DIR_PATH + 'patient_manifest.csv') # selected patients from transformer inclusion criteria
# ds_df = pd.read_csv(DATA_DIR_PATH + 'daily_support.csv')
# dw_df = pd.read_csv(DATA_DIR_PATH + 'daily_weight.csv')
# dis_df = pd.read_csv(DATA_DIR_PATH + 'discharge_bpd_status.csv')
# dm_df = pd.read_csv(DATA_DIR_PATH + 'daily_medications.csv')
# dp_df = pd.read_csv(DATA_DIR_PATH + 'daily_procedures.csv')
# mf_df = pd.read_csv(DATA_DIR_PATH + 'maternal_facts.csv')
# pf_df = pd.read_csv(DATA_DIR_PATH + 'patient_facts.csv')
#
# # adjust column name in dw_df
# dw_df.columns = ['PatientSeqID', 'DSB', 'PMA', 'Weight']
# # remove support max column because it contains nan values
# ds_df = ds_df.drop(columns='SupportMax')
# # data adjustments to account for duplicated names (Albumin, Epinephrine) in dp_df, dm_df
# dm_cols, dp_cols = list(dm_df.columns), list(dp_df.columns)
# # changing column names w/suffixes
# dm_cols[dm_cols.index('Albumin')], dm_cols[dm_cols.index('Epinephrine')] = 'Albumin_m', 'Epinephrine_m'
# dp_cols[dp_cols.index('Albumin')], dp_cols[dp_cols.index('Epinephrine')] = 'Albumin_p', 'Epinephrine_p'
# # saving to dataframe
# dm_df.columns, dp_df.columns = dm_cols, dp_cols
# # data adjustment to encode data values in mf_df
# mf_none_cols = list(mf_df.columns)[2:-2] + ['Chorioamnionitis']
# for col in mf_none_cols:
#     mf_df[col] = (mf_df[col] != 'None').astype(int)
# # data adjustment to encode data values in pf_df
# pf_cat_cols = ['Gender.Code', 'Delivery.Code', 'Race.Code', 'AliveDied.Transfer']
# pf_num_cols = ['PatientSeqID'] + list(pf_df.columns)[2:7] + list(pf_df.columns)[-2:]
# pf_cat_cols_mapping = {col: {n: cat for n, cat in enumerate(pf_df[col].astype('category').cat.categories)}
#      for col in pf_cat_cols}
# pf_cat_df = pd.DataFrame({col: pf_df[col].astype('category').cat.codes for col in pf_cat_cols}, index=pf_df.index)
# pf_c_df = pd.concat([pf_df.loc[:, pf_num_cols], pf_cat_df], axis=1)
# pf_df = pf_c_df.copy()
# # merging patient information datasets
# pinfo_df = pm_df.merge(mf_df).merge(pf_df).merge(dis_df.loc[:, ['PatientSeqID', 'BPD Grade']])
# pinfo_df = pinfo_df.drop(columns=['AliveDied.Transfer', 'APGAR1', 'APGAR5', 'APGAR10'])
# # massive merging of all dataframes up to 21 days
# dsw_df = fup_day(ds_df, 21).merge(fup_day(dw_df, 21))
# dswm_df = dsw_df.merge(fup_day(dm_df, 21))
# dswmp_df = dswm_df.merge(fup_day(dp_df, 21))
# mv_df = dswm_df.merge(pinfo_df)
# # saving to csv for easy access
# mv_df.to_csv(DATA_DIR_PATH + '/processed/full/mv_df.csv', index=False)
#
# TRANSFORMER DATASETS
# reading in internal/external embeddings + combining into one transformer csv
# ti_df = pd.read_csv(DATA_DIR_PATH + 'predictions/internal_embeddings.csv')
# te_df = pd.read_csv(DATA_DIR_PATH + 'predictions/external_embeddings.csv')
# t_cols = ['PatientSeqID', 'DSB', 'Support_Level_36'] + list(ti_df.columns)[3:]
# ti_df.columns, te_df.columns = t_cols, t_cols
# ta_df = pd.concat([ti_df, te_df])
# ta_df = ta_df.drop(columns=['Support_Level_36'])
# # merging to get bpd grade status + filter by inclusion criteria
# tv_df = fup_day(ta_df, 27).merge((dis_df.merge(pm_df)).loc[:, ['PatientSeqID', 'BPD Grade']])
# # saving to csv for easy access
# tv_df.to_csv(DATA_DIR_PATH + '/processed/full/tv_27_df.csv', index=False)

"""
If mv_df.csv, tv_df.csv already created, skip to here. If all vectorized datasets already created, 
can skip to line 110 and begin with clustering.
"""
# mv_df = pd.read_csv(DATA_DIR_PATH + 'processed/full/mv_df.csv')
# tv_df = pd.read_csv(DATA_DIR_PATH + 'processed/full/tv_df.csv')
# # should have dataframes already set so we can have columns labeled
# DFT_COLS = list(mv_df.columns)[3:435]  # daily feature columns in manual dataframe
# SFT_COLS = list(mv_df.columns)[435:-1]  # single value columns in manual dataframe
# T_DFT_COLS = list(tv_df.columns)[2:-1]  # daily feature columns in transformer dataframe
# T_SFT_COLS = list()  # single value columns in transformer dataframe
# OUTPUT_PATH = DATA_DIR_PATH + '/processed/vector/'
# DAYS = 21
# # manual vectorization, separate grades
# for g in range(3, -1, -1):
#     for time_window in [1, 3]:
#         output = VDATA_DIR_PATH + 'mg' + str(g) + 't' + str(time_window)
#         print(output)
#         make_dfs(mv_df, [g], time_window, DAYS, DFT_COLS, SFT_COLS, output)
# # manual vectorization, all grades
# m_all_output = VDATA_DIR_PATH + 'mga'
# for time_window in [1, 3]:
#     make_dfs(mv_df, [0, 1, 2, 3], time_window, DAYS, DFT_COLS, SFT_COLS, m_all_output + 't' + str(time_window))
# # transformer vectorization, separate grades
# for g in range(3, -1, -1):
#     for time_window in [1, 3]:
#         output = VDATA_DIR_PATH + 'tg' + str(g) + 't' + str(time_window)
#         print(output)
#         make_dfs(tv_df, [g], time_window, DAYS, T_DFT_COLS, T_SFT_COLS, output)
# # transformer vectorization, all grades
# t_all_output = VDATA_DIR_PATH + 'tga'
# for time_window in [1, 3]:
#     make_dfs(tv_df, [0, 1, 2, 3], time_window, DAYS, T_DFT_COLS, T_SFT_COLS, t_all_output + 't' + str(time_window))

##########################################
# Risk Matching                          #
##########################################
# TODO: figure out this risk matching!!
# reading in internal/external risk predictions + combining into one transformer csv
# tir_df = pd.read_csv(DATA_DIR_PATH + 'predictions/test_internal.csv')
# ter_df = pd.read_csv(DATA_DIR_PATH + 'predictions/test_external.csv')
# tar_df = pd.concat([tir_df, ter_df])
# inclusion_criteria = list(pm_df['PatientSeqID'])
# tar_df = tar_df.loc[tar_df['Patient'].isin(inclusion_criteria)]
# # assign risk
# tar_df = weighted_risk(tar_df)
# tar_df = risk_grading(tar_df)
# print(tar_df.head())
# # saving to csv for easy access
# tar_df.to_csv(DATA_DIR_PATH + '/processed/full/tar_df.csv', index=False)


##########################################
# Clustering                             #
##########################################
"""
To take vectorized datasets + find ideal clusterings with 2 reduction methods, 3 clustering algorithms, 2 time windows.
If already created clustered datasets, please skip to line 167. If you want to still run the following validation,
you should still import in all the datasets.
"""
''' 
setup:
    dataset_names = ['name', 'name2', ..., 'name']
    to_cluster = ['data_name', 'data_name', ..., 'data_name']
    for dataset in to_cluster:
        print('loading', dataset)
        curr_df = pd.read_csv(VDATA_DIR_PATH + dataset + v_suffix)
        make_clusters(curr_df, CDATA_DIR_PATH + dataset + c_suffix)
        print(dataset, 'cluster complete')
    print('all clusters complete')
'''

'''
history:
    ['mg3t3', 'mg2t3', 'mg1t3']
    ['tg3t1', 'tg2t1', 'tg1t1']
    ['tgat1']
    ['tgat1'] filtered to be essentially ['tgbt1'] 
'''
v_suffix = 'v_df.csv'
c_suffix = 'c_df.csv'


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
# gee whix this is a weird part.
# # print('loading data')
# # # importing in manual datasets
# # # mg0t1_df = pd.read_csv(VDATA_DIR_PATH + 'mg0t1v_df.csv')
# # # mg0t3_df = pd.read_csv(VDATA_DIR_PATH + 'mg0t3v_df.csv')
# #
# # # mg1t1_df = pd.read_csv(VDATA_DIR_PATH + 'mg1t1v_df.csv')
# # mg1t3_df = pd.read_csv(VDATA_DIR_PATH + 'mg1t3v_df.csv')
# #
# # # mg2t1_df = pd.read_csv(VDATA_DIR_PATH + 'mg2t1v_df.csv')
# # mg2t3_df = pd.read_csv(VDATA_DIR_PATH + 'mg2t3v_df.csv')
# #
# # # mg3t1_df = pd.read_csv(VDATA_DIR_PATH + 'mg3t1v_df.csv')
# # mg3t3_df = pd.read_csv(VDATA_DIR_PATH + 'mg3t3v_df.csv')
# #
# # # mgat1_df = pd.read_csv(VDATA_DIR_PATH + 'mgat1v_df.csv')
# # mgat3_df = pd.read_csv(VDATA_DIR_PATH + 'mgat3v_df.csv')
# # print('manual done')
# # # importing in transformer datasets
# # # tg0t1_df = pd.read_csv(VDATA_DIR_PATH + 'tg0t1v_df.csv')
# # # tg0t3_df = pd.read_csv(VDATA_DIR_PATH + 'tg0t3v_df.csv')
# #
# # tg1t1_df = pd.read_csv(VDATA_DIR_PATH + 'tg1t1v_df.csv')
# # # tg1t3_df = pd.read_csv(VDATA_DIR_PATH + 'tg1t3v_df.csv')
# #
# # tg2t1_df = pd.read_csv(VDATA_DIR_PATH + 'tg2t1v_df.csv')
# # # tg2t3_df = pd.read_csv(VDATA_DIR_PATH + 'tg2t3v_df.csv')
# #
# # tg3t1_df = pd.read_csv(VDATA_DIR_PATH + 'tg3t1v_df.csv')
# # # tg3t3_df = pd.read_csv(VDATA_DIR_PATH + 'tg3t3v_df.csv')
# #
# # tgat1_df = pd.read_csv(VDATA_DIR_PATH + 'tgat1v_df.csv')
# # # tgat3_df = pd.read_csv(VDATA_DIR_PATH + 'tgat3v_df.csv')
# #
# # # running clustering
# # # manual vectorization
# # print('loading done')
# # ''' code for full-run-through of all datasets in transformer + manual (not frequently run bc data too big) '''
# # # m_to_cluster = [mg0t1_df, mg0t3_df, mg1t1_df, mg1t3_df, mg2t1_df, mg2t3_df, mg3t1_df, mg3t3_df, mgat1_df, mgat3_df]
# # # m_suffixes = ['mg0t1c_df.csv', 'mg0t3c_df.csv', 'mg1t1c_df.csv', 'mg1t3c_df.csv', 'mg2t1c_df.csv', 'mg2t3c_df.csv',
# # #            'mg3t1c_df.csv', 'mg3t3c_df.csv', 'mgat1c_df.csv', 'mgat3c_df.csv']
# # # for df, suffix in zip(m_to_cluster, m_suffixes):
# # #     make_clusters(df, CDATA_DIR_PATH + suffix)
# # #     print('manual cluster done')
# # # print('all manual clusters done')
# # # # transformer vectorization
# # # t_to_cluster = [tg0t1_df, tg0t3_df, tg1t1_df, tg1t3_df, tg2t1_df, tg2t3_df, tg3t1_df, tg3t3_df, tgat1_df, tgat3_df]
# # # t_suffixes = ['tg0t1c_df.csv', 'tg0t3c_df.csv', 'tg1t1c_df.csv', 'tg1t3c_df.csv', 'tg2t1c_df.csv', 'tg2t3c_df.csv',
# # #            'tg3t1c_df.csv', 'tg3t3c_df.csv', 'tgat1c_df.csv', 'tgat3c_df.csv']
# # # for df, suffix in zip(t_to_cluster, t_suffixes):
# # #     make_clusters(df, CDATA_DIR_PATH + suffix)
# # ''' code for partial run-through of datasets, focusing on important datasets'''
# # m_to_cluster = [mg1t3_df, mg2t3_df, mg3t3_df, mgat3_df]
# # m_suffixes = ['mg1t3c_df.csv', 'mg2t3c_df.csv', 'mg3t3c_df.csv', 'mgat3c_df.csv']
# # for df, suffix in zip(m_to_cluster, m_suffixes):
# #     make_clusters(df, CDATA_DIR_PATH + suffix)
# #     print('manual cluster done')
# # print('all manual clusters done')

# # # transformer vectorization
# # t_to_cluster = [tg1t1_df, tg2t1_df, tg3t1_df, tgat1_df]
# # t_suffixes = ['tg1t1c_df.csv', 'tg2t1c_df.csv', 'tg3t1c_df.csv', 'tgat1c_df.csv']
# # for df, suffix in zip(t_to_cluster, t_suffixes):
# #     make_clusters(df, CDATA_DIR_PATH + suffix)
# # print('all done!')

##########################################
# Cluster Validation                     #
##########################################
"""
This will perform a set number of bootstrap replications for goal 2 of the three-pronged validation + silhouette scores
"""
# BOOTSTRAP + SILHOUETTE SCORES
# BOOTSTRAP_REPS = 100
# to_bootstrap = ['tgat1']
# # ['tg3t1', 'tg2t1', 'tg1t1', 'tgat1', 'tgbt1']
#
# v_suffix = 'v_df.csv'
# c_suffix = 'c_df.csv'
# for data in to_bootstrap:
#     print('loading in', data)
#     curr_c_df = pd.read_csv(CDATA_DIR_PATH + data + c_suffix)
#     curr_v_df = pd.read_csv(VDATA_DIR_PATH + data + v_suffix)
#     curr_n = curr_c_df.shape[0]
#     bs_output = SDATA_DIR_PATH + data + '_bs.json'
#     ss_output = SDATA_DIR_PATH + data + '_ss.json'
#     run_bootstrap(curr_v_df, curr_c_df, curr_n, BOOTSTRAP_REPS, bs_output)
#     print('bootstrap complete!')
#     cols = list(curr_c_df.columns)[1:]
#     # get_silhouette_scores(curr_v_df, curr_c_df, cols, ss_output)

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
