import pandas as pd
from src.pipelines.vectorize import fup_day, make_dfs
from src.pipelines.cluster import make_clusters_sl
from src.pipelines.validate import run_bootstrap_sl
from src.pipelines.risk_match import expected_risk, matched_risk, jensen_risk

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

# NEW VECTORIZATION
# TRANSFORMER DATASETS
# reading in internal/external embeddings + combining into one transformer csv
ti_df = pd.read_csv(DATA_DIR_PATH + 'predictions/internal_embeddings.csv')
te_df = pd.read_csv(DATA_DIR_PATH + 'predictions/external_embeddings.csv')
t_cols = ['PatientSeqID', 'DSB', 'Support_Level_36'] + list(ti_df.columns)[3:]
ti_df.columns, te_df.columns = t_cols, t_cols
ta_df = pd.concat([ti_df, te_df])
ta_df = ta_df.drop(columns=['Support_Level_36'])
# merging to get bpd grade status + filter by inclusion criteria across specific days
dis_df = pd.read_csv(DATA_DIR_PATH + 'discharge_bpd_status.csv')
pm_df = pd.read_csv(DATA_DIR_PATH + 'patient_manifest.csv')
tv_df = ta_df.merge((dis_df.merge(pm_df)).loc[:, ['PatientSeqID', 'BPD Grade']])
tv_df.to_csv(DATA_DIR_PATH + 'processed/full/tv_df.csv', index=False)  # includes all transformer encodings + BPD grade @ end
tv_27_df = tv_df.loc[tv_df['DSB'] == 27]
T_DFT_COLS = list(tv_27_df.columns)[2:-1]
T_SFT_COLS = list()
t_all_output = VDATA_DIR_PATH + 'lr_27_'
make_dfs(tv_27_df, [-1, 0, 1, 2, 3], 1, 27, T_DFT_COLS, T_SFT_COLS, t_all_output)

# RISK MATCHING
tar_df = pd.read_csv(DATA_DIR_PATH + 'processed/full/tar_df.csv')
for day in DAYS:
    v_df = pd.read_csv(VDATA_DIR_PATH + 'lr_27_' + v_suffix)
    matched_risk(tar_df, v_df, day, DATA_DIR_PATH + '/processed/risk/lr_')

# CLUSTERING
to_cluster = ['lr_d1g0_', 'lr_d1g1_', 'lr_d1g2_', 'lr_d1g3_', 'lr_d1g4_',
              'lr_d3g0_', 'lr_d3g1_', 'lr_d3g2_', 'lr_d3g3_', 'lr_d3g4_',
              'lr_d7g0_', 'lr_d7g1_', 'lr_d7g2_', 'lr_d7g3_', 'lr_d7g4_',
              'lr_d14g0_', 'lr_d14g1_', 'lr_d14g2_', 'lr_d14g3_', 'lr_d14g4_',
              'lr_d21g0_', 'lr_d21g1_', 'lr_d21g2_', 'lr_d21g3_', 'lr_d21g4_',
              'lr_d27g0_', 'lr_d27g1_', 'lr_d27g2_', 'lr_d27g3_', 'lr_d27g4_']

for dataset in to_cluster:
    print('loading', dataset)
    curr_df = pd.read_csv(RDATA_DIR_PATH + dataset + r_suffix)
    make_clusters_sl(curr_df, CDATA_DIR_PATH + dataset + c_suffix, CDATA_DIR_PATH + dataset + u_suffix,
                     visualize=False)
    print(dataset, 'cluster complete')
print('all clusters complete')

BOOTSTRAP_REPS = 100
cluster_col = 'umap_KMeans'
for data in to_cluster:
    print('loading in', data)
    curr_c_df = pd.read_csv(CDATA_DIR_PATH + data + c_suffix)
    num_clusters = len(list(set(curr_c_df[cluster_col])))
    curr_u_df = pd.read_csv(CDATA_DIR_PATH + data + u_suffix)
    curr_n = curr_c_df.shape[0]
    bs_output = SDATA_DIR_PATH + data + 'bs.json'
    run_bootstrap_sl(curr_u_df, curr_c_df, curr_n, BOOTSTRAP_REPS, bs_output, num_clusters)
    print('bootstrap complete!')
