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

# timeframes
DAYS = [1, 3, 7, 14, 21, 27]


# suffixes
v_suffix = 'v_df.csv'
c_suffix = 'c_df.csv'
u_suffix = 'u_df.csv'
r_suffix = 'r_df.csv'

print('variables set')

##########################################
# Risk Match Trials                      #
##########################################
'''
we are assuming that I already have a base transformer dataset <tar_df>, and jensen risk dataset <tjr_df> that I can 
access and use
'''

to_cluster = ['t27g_d1g0_', 't27g_d1g1_', 't27g_d1g2_', 't27g_d1g3_', 't27g_d1g4_',
              't27g_d3g0_', 't27g_d3g1_', 't27g_d3g2_', 't27g_d3g3_', 't27g_d3g4_',
              't27g_d7g0_', 't27g_d7g1_', 't27g_d7g2_', 't27g_d7g3_', 't27g_d7g4_',
              't27g_d14g0_', 't27g_d14g1_', 't27g_d14g2_', 't27g_d14g3_', 't27g_d14g4_',
              't27g_d21g0_', 't27g_d21g1_', 't27g_d21g2_', 't27g_d21g3_', 't27g_d21g4_',
              't27g_d27g0_', 't27g_d27g1_', 't27g_d27g2_', 't27g_d27g3_', 't27g_d27g4_']

# just changing PCA/UMAP constraints
for dataset in to_cluster:
    print('loading', dataset)
    curr_df = pd.read_csv(RDATA_DIR_PATH + dataset + r_suffix)
    make_clusters_sl(curr_df, CDATA_DIR_PATH + 'risk/jp_' + dataset + c_suffix, CDATA_DIR_PATH + 'risk/jp_' + dataset + u_suffix,
                     visualize=False)
    print(dataset, 'cluster complete')
print('all clusters complete')
# changing UMAP to set number of components
for dataset in to_cluster:
    print('loading', dataset)
    curr_df = pd.read_csv(RDATA_DIR_PATH + dataset + r_suffix)
    make_clusters_sl(curr_df, CDATA_DIR_PATH  + 'risk/'+ dataset + c_suffix, CDATA_DIR_PATH + 'risk/' + dataset + u_suffix,
                     visualize=False, pca=True)
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

cluster_col = 'umap_KMeans'
for data in to_cluster:
    print('loading in', data)
    curr_c_df = pd.read_csv(CDATA_DIR_PATH + 'risk/jp_' + data + c_suffix)
    num_clusters = len(list(set(curr_c_df[cluster_col])))
    curr_u_df = pd.read_csv(CDATA_DIR_PATH + data + u_suffix)
    curr_n = curr_c_df.shape[0]
    bs_output = SDATA_DIR_PATH + 'jp_' + data + 'bs.json'
    run_bootstrap_sl(curr_u_df, curr_c_df, curr_n, BOOTSTRAP_REPS, bs_output, num_clusters)
    print('bootstrap complete!')

for data in to_cluster:
    print('loading in', data)
    curr_c_df = pd.read_csv(CDATA_DIR_PATH + 'risk/' + data + c_suffix)
    num_clusters = len(list(set(curr_c_df[cluster_col])))
    curr_u_df = pd.read_csv(CDATA_DIR_PATH + data + u_suffix)
    curr_n = curr_c_df.shape[0]
    bs_output = SDATA_DIR_PATH + 'set_' + data + 'bs.json'
    run_bootstrap_sl(curr_u_df, curr_c_df, curr_n, BOOTSTRAP_REPS, bs_output, num_clusters)
    print('bootstrap complete!')