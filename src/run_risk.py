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

# suffixes
v_suffix = 'v_df.csv'
c_suffix = 'c_df.csv'
u_suffix = 'u_df.csv'
r_suffix = 'r_df.csv'

##########################################
# Vectorize                              #
##########################################
# # TRANSFORMER DATASETS
tv_df = pd.read_csv(DATA_DIR_PATH + 'processed/full/tv_df.csv')  # includes all transformer encodings + BPD grade @ end
for day in DAYS:
    curr_df = tv_df.loc[tv_df['DSB'] >= day]
    T_DFT_COLS = list(tv_df.columns)[2:-1]  # daily feature columns in transformer dataframe
    T_SFT_COLS = list()  # single value columns in transformer dataframe
    # making new vectorized dataframes from day or risk matching onwards
    t_all_output = VDATA_DIR_PATH + 'tgac_d' + str(day) + '_'
    make_dfs(curr_df, [0, 1, 2, 3], 1, day, T_DFT_COLS, T_SFT_COLS, t_all_output)
''' now, the dataframes will all be in the format where each row = unique patient, and the rows are formatted
    [PatientSeqID, DSB 0 dim 0, DSB 0 dim 1, ..., DSB 0 dim 127, DSB 1 dim 0, ..., DSB day dim 127]'''


##########################################
# Risk Matching                          #
##########################################
'''
we are assuming that I already have a base transformer dataset <tar_df>, and jensen risk dataset <tjr_df> that I can 
access and use
'''
# to run over various timeframes and create risk datasets for each, we can use the following:
tar_df = pd.read_csv(DATA_DIR_PATH + 'processed/full')
vdfs = ['tgac_d1_', 'tgac_d3_', 'tgac_d7_', 'tgac_d14_', 'tgac_d21_', 'tgac_d27_']
for day, vdf in zip(DAYS, vdfs):
    v_df = pd.read_csv(VDATA_DIR_PATH + vdf + v_suffix)
    matched_risk(tar_df, v_df, day, DATA_DIR_PATH + '/processed/risk/tcon_')


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
# clustering for each of the separate risk datasets, with embeddings starting the day of risk prediction
to_cluster = ['tcon_d1g0_', 'tcon_d1g1_', 'tcon_d1g2_', 'tcon_d1g3_', 'tcon_d1g4_',
              'tcon_d3g0_', 'tcon_d3g1_', 'tcon_d3g2_', 'tcon_d3g3_', 'tcon_d3g4_',
              'tcon_d7g0_', 'tcon_d7g1_', 'tcon_d7g2_', 'tcon_d7g3_', 'tcon_d7g4_',
              'tcon_d14g0_', 'tcon_d14g1_', 'tcon_d14g2_', 'tcon_d14g3_', 'tcon_d14g4_',
              'tcon_d21g0_', 'tcon_d21g1_', 'tcon_d21g2_', 'tcon_d21g3_', 'tcon_d21g4_',
              'tcon_d27g0_', 'tcon_d27g1_', 'tcon_d27g2_', 'tcon_d27g3_', 'tcon_d27g4_']
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

to_bootstrap = ['tcon_d1g0_', 'tcon_d1g1_', 'tcon_d1g2_', 'tcon_d1g3_', 'tcon_d1g4_',
                'tcon_d3g0_', 'tcon_d3g1_', 'tcon_d3g2_', 'tcon_d3g3_', 'tcon_d3g4_',
                'tcon_d7g0_', 'tcon_d7g1_', 'tcon_d7g2_', 'tcon_d7g3_', 'tcon_d7g4_',
                'tcon_d14g0_', 'tcon_d14g1_', 'tcon_d14g2_', 'tcon_d14g3_', 'tcon_d14g4_',
                'tcon_d21g0_', 'tcon_d21g1_', 'tcon_d21g2_', 'tcon_d21g3_', 'tcon_d21g4_',
                'tcon_d27g0_', 'tcon_d27g1_', 'tcon_d27g2_', 'tcon_d27g3_', 'tcon_d27g4_']
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
