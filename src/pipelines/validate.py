#################################################################################
# Imports + Package Setup                                                       #
#################################################################################
import numpy as np
from sklearn.metrics import silhouette_score, jaccard_score
import json
from src.pipelines.cluster import make_clusters, make_UMAP, make_X, make_95_PCA, make_clusters_sl


#################################################################################
# Helper Functions                                                              #
#################################################################################
def calc_jaccard(newc_df, origc_df, columns, scores_dict):
    for c_alg in columns:
        nc_clusters, oc_clusters = list(set(newc_df[c_alg])), list(set(origc_df[c_alg]))
        for o_clust in oc_clusters:
            curr_js = list()
            o_clust_ids = list(origc_df.loc[origc_df[c_alg] == o_clust]['PatientSeqID'])
            f_newc = np.array(newc_df.loc[newc_df['PatientSeqID'].isin(o_clust_ids)][c_alg])
            for n_clust in nc_clusters:
                f_origc = np.array([n_clust for id in o_clust_ids])
                curr_js.append(jaccard_score(f_newc, f_origc, average='weighted'))
            scores_dict[c_alg][o_clust].append(np.max(np.array(curr_js)))
    # print(scores_dict)
    return scores_dict


def create_bootstrap_dict(columns, orig_df):
    b_dict = dict()
    for alg in columns:
        alg_clusters = list(set(orig_df[alg]))
        inner_dict = dict()
        for cluster in alg_clusters:
            inner_dict[cluster] = list()
        b_dict[alg] = inner_dict
    return b_dict


def run_bootstrap(v_df, c_df, n, num_reps, json_output_path):
    # combine v_df, c_df so no errors when sampling
    cv_df = v_df.merge(c_df)
    # derive cols from the dataframe
    cluster_cols = list(c_df.columns)[4:]  # check if this is right when debugging, umap only
    ft_cols = list(v_df.columns)[1:]  # also check this one
    jcs_dict = create_bootstrap_dict(cluster_cols, c_df)
    for i in range(num_reps):
        # sampling + setting up data
        sample_df = cv_df.sample(n=n, replace=True)
        sample_df = sample_df.dropna(axis=1)
        sample_ids = list(sample_df['PatientSeqID'])
        oc_df = sample_df.loc[:, ['PatientSeqID'] + cluster_cols]  # storing original clusters for comparison
        sample_v_df = sample_df.loc[:, ['PatientSeqID'] + ft_cols]
        nc_df = make_clusters(sample_v_df, 'no_output', save_csv=False, visualize=False)
        # calculating Jaccard coefficients
        jcs_dict = calc_jaccard(nc_df, oc_df, cluster_cols, jcs_dict)
    # bootstrap averaging
    avg_jcs_dict = create_bootstrap_dict(cluster_cols, c_df)
    for alg in cluster_cols:
        for cl in jcs_dict[alg].keys():
            avg_jcs_dict[alg][cl] = np.mean(np.array(jcs_dict[alg][cl]))
    # save to json
    with open(json_output_path, "w") as outfile:
        json.dump(avg_jcs_dict, outfile)
    return avg_jcs_dict


def run_bootstrap_sl(v_df, c_df, n, num_reps, json_output_path):
    # combine v_df, c_df so no errors when sampling
    cv_df = v_df.merge(c_df)
    # derive cols from the dataframe
    cluster_cols = list(c_df.columns)[1:]
    ft_cols = list(v_df.columns)[1:]
    jcs_dict = create_bootstrap_dict(cluster_cols, c_df)
    for i in range(num_reps):
        # sampling + setting up data
        sample_df = cv_df.sample(n=n, replace=True)
        sample_df = sample_df.dropna(axis=1)
        oc_df = sample_df.loc[:, ['PatientSeqID'] + cluster_cols]  # storing original clusters for comparison
        sample_v_df = sample_df.loc[:, ['PatientSeqID'] + ft_cols]
        nc_df = make_clusters_sl(sample_v_df, 'no_output', save_csv=False, visualize=False)
        # calculating Jaccard coefficients
        jcs_dict = calc_jaccard(nc_df, oc_df, cluster_cols, jcs_dict)
    # bootstrap averaging
    avg_jcs_dict = create_bootstrap_dict(cluster_cols, c_df)
    for alg in cluster_cols:
        for cl in jcs_dict[alg].keys():
            avg_jcs_dict[alg][cl] = np.mean(np.array(jcs_dict[alg][cl]))
    # save to json
    with open(json_output_path, "w") as outfile:
        json.dump(avg_jcs_dict, outfile)
    return avg_jcs_dict


# TODO: need to unit-test
def get_silhouette_scores(v_df, c_df, columns, json_output_path):
    s_scores = dict()
    curr_v_df = v_df.dropna(axis=0)
    X = make_X(curr_v_df)
    pca_X, umap_X = make_95_PCA(X).values, make_UMAP(X).values
    for col in columns:
        col_rd_type = 'pca' if 'pca' in col else 'umap'
        n_clusters = list(set(c_df[col]))
        n_clusters.remove(-1)
        n_clusters = len(n_clusters)
        if n_clusters >= 2:
            if col_rd_type == 'pca':
                s_scores[col] = silhouette_score(pca_X, np.array(list(c_df[col])))
            else:
                s_scores[col] = silhouette_score(umap_X, np.array(list(c_df[col])))
        else:
            s_scores[col] = 0.0
    # save to json
    with open(json_output_path, "w") as outfile:
        json.dump(s_scores, outfile)
    return s_scores

