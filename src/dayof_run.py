import pandas as pd
from src.pipelines.vectorize import fup_day, make_dfs
from src.pipelines.cluster import make_clusters_sl
from src.pipelines.validate import run_bootstrap_sl
from src.pipelines.risk_match import expected_risk, matched_risk, jensen_risk
from src.pipelines.cluster import make_clusters_tsl
from src.pipelines.validate import run_bootstrap_tsl

DATA_DIR_PATH = '/Users/cindywang/PycharmProjects/bpd-subtyping/data/'  # local

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

# we have a larger vectorization with all the grades in the lr_1_v_df.csv, etc. We need to risk match
# RISK MATCHING
tar_df = pd.read_csv(DATA_DIR_PATH + 'processed/full/tar_df.csv')
for day in DAYS:
    v_df = pd.read_csv(VDATA_DIR_PATH + 'lr_' + str(day) + '_' + v_suffix)
    matched_risk(tar_df, v_df, day, DATA_DIR_PATH + '/processed/risk/do_')

# CLUSTERING, will focus on grade 3
to_cluster = ['do_d1g0_', 'do_d1g1_', 'do_d1g2_', 'do_d1g3_', 'do_d1g4_',
              'do_d3g0_', 'do_d3g1_', 'do_d3g2_', 'do_d3g3_', 'do_d3g4_',
              'do_d7g0_', 'do_d7g1_', 'do_d7g2_', 'do_d7g3_', 'do_d7g4_',
              'do_d14g0_', 'do_d14g1_', 'do_d14g2_', 'do_d14g3_', 'do_d14g4_',
              'do_d21g0_', 'do_d21g1_', 'do_d21g2_', 'do_d21g3_', 'do_d21g4_',
              'do_d27g0_', 'do_d27g1_', 'do_d27g2_', 'do_d27g3_', 'do_d27g4_']

for dataset in to_cluster:
    print('loading', dataset)
    curr_df = pd.read_csv(RDATA_DIR_PATH + dataset + r_suffix)
    make_clusters_tsl(curr_df, CDATA_DIR_PATH + dataset + 't' + c_suffix, CDATA_DIR_PATH + dataset + 't' + u_suffix)
    print(dataset, 'cluster complete')
print('all clusters complete')

# BOOTSTRAP REPLICATIONS!
BOOTSTRAP_REPS = 100
cluster_cols = 'umap_KMeans'
for data in to_cluster:
    print('loading in', data)
    curr_c_df = pd.read_csv(CDATA_DIR_PATH + data + 't' + c_suffix)
    curr_u_df = pd.read_csv(CDATA_DIR_PATH + data + 't' + u_suffix)
    curr_n = curr_c_df.shape[0]
    bs_output = SDATA_DIR_PATH + data + 'tbs.json'
    run_bootstrap_tsl(curr_u_df, curr_c_df, curr_n, BOOTSTRAP_REPS, bs_output)
    print('bootstrap complete!')

