import pandas as pd
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

# same vectorization, risk matching as in last_run.py
# CLUSTERING, will focus on grade 3
to_cluster = ['lr_d1g3_', 'lr_d3g3_', 'lr_d7g3_', 'lr_d14g3_', 'lr_d21g3_', 'lr_d27g3_']

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

