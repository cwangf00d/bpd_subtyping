import pandas as pd
from src.pipelines.cluster import make_clusters_sl
from src.pipelines.validate import run_bootstrap_sl
from src.pipelines.vectorize import fup_day, make_dfs


DATA_DIR_PATH = '/Users/cindywang/PycharmProjects/bpd-subtyping/data/'  # local

VDATA_DIR_PATH = DATA_DIR_PATH + 'processed/vector/'
CDATA_DIR_PATH = DATA_DIR_PATH + 'processed/clusters/'
SDATA_DIR_PATH = DATA_DIR_PATH + 'processed/scores/'
c_suffix, u_suffix, v_suffix = 'c_df.csv', 'u_df.csv', 'v_df.csv'
DAYS = [1, 3, 7, 14, 21, 27]

# vectorization
# tv_df = pd.read_csv(DATA_DIR_PATH + 'processed/full/tv_df.csv')
# for day in DAYS:
#     t_all_output = VDATA_DIR_PATH + 'lrf_' + str(day) + '_'
#     curr_tv_df = tv_df.loc[tv_df['DSB'] == day]  # trial 1
#     t_dft_cols = list(curr_tv_df.columns)[2:-1]
#     t_sft_cols = list()
#     make_dfs(curr_tv_df, [-1, 0, 1, 2, 3], 1, 27, t_dft_cols, t_sft_cols, t_all_output)

# to_cluster = ['lrf_1_', 'lrf_3_', 'lrf_7_', 'lrf_14_', 'lrf_21_', 'lrf_27_']
to_cluster = ['tga_d1_', 'tga_d3_', 'tga_d7_', 'tga_d14_', 'tga_d21_', 'tga_d27_', 'tgat1']

for dataset in to_cluster:
    print('loading', dataset)
    curr_df = pd.read_csv(VDATA_DIR_PATH + dataset + v_suffix)
    make_clusters_sl(curr_df, CDATA_DIR_PATH + dataset + 'r' + c_suffix, CDATA_DIR_PATH + dataset + 'r' + u_suffix,
                     visualize=False)
    print(dataset, 'complete')
print('all clusters complete')

BOOTSTRAP_REPS = 100
cluster_col = 'umap_KMeans'
for data in to_cluster:
    print('loading in', data)
    curr_c_df = pd.read_csv(CDATA_DIR_PATH + data + 'r' + c_suffix)
    num_clusters = len(list(set(curr_c_df[cluster_col])))
    curr_u_df = pd.read_csv(CDATA_DIR_PATH + data + 'r' + u_suffix)
    curr_n = curr_c_df.shape[0]
    bs_output = SDATA_DIR_PATH + data + 'rbs.json'
    run_bootstrap_sl(curr_u_df, curr_c_df, curr_n, BOOTSTRAP_REPS, bs_output, num_clusters)
    print('bootstrap complete!')
