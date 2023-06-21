#################################################################################
# Imports + Package Setup                                                       #
#################################################################################
import numpy as np
from sklearn.metrics import silhouette_score, jaccard_score
from sklearn.metrics.cluster import adjusted_rand_score
import json
from src.pipelines.cluster import make_clusters, make_UMAP, make_X, make_95_PCA, make_clusters_sl


#################################################################################
# Helper Functions                                                              #
#################################################################################
def calc_jaccard(newc_df, origc_df, columns, scores_dict):
    """
    takes two cluster dataframes and calculates Jaccard scores for each cluster in clustering
    :param newc_df: pd dataframe with new cluster designations from a bootstrap replication
    :param origc_df: pd dataframe with original cluster designations
    :param columns: list of names of columns to calculate Jaccard score for
    :param scores_dict: dictionary structured, {clustering_alg: {cluster 1: [score1, score2, ...], cluster 2: [...], }}
    :return: scores_dict with jaccard scores appended for each cluster in each clustering
    """
    for c_alg in columns:
        nc_clusters, oc_clusters = list(set(newc_df[c_alg])), list(set(origc_df[c_alg]))
        for o_clust in oc_clusters:
            curr_js = list()
            o_clust_ids = list(origc_df.loc[origc_df[c_alg] == o_clust]['PatientSeqID'])
            f_newc = np.array(newc_df.loc[newc_df['PatientSeqID'].isin(o_clust_ids)][c_alg])
            for n_clust in nc_clusters:
                f_origc = np.array([n_clust for _ in o_clust_ids])
                curr_js.append(jaccard_score(f_newc, f_origc, average='weighted'))
            scores_dict[c_alg][o_clust].append(np.max(np.array(curr_js)))
    # print(scores_dict)
    return scores_dict


def calc_ari(newc_df, origc_df, columns, ari_scores_dict):
    """
    takes two cluster dataframes and calculates the adjusted rand index for similarity of clusterings
    :param newc_df: pd dataframe with new cluster designations from a bootstrap replication
    :param origc_df: pd dataframe with original cluster designations
    :param columns: list of names of columns to calculate Jaccard score for
    :param ari_scores_dict: dictionary structured, {clustering_alg: [score1, score2, ...]}
    :return: ari_scores_dict with ari scores appended to list for each clustering algorithm
    """
    for c_alg in columns:
        nc_clusters, oc_clusters = list((newc_df[c_alg])), list((origc_df[c_alg]))
        ari_score = adjusted_rand_score(nc_clusters, oc_clusters)
        ari_scores_dict[c_alg].append(ari_score)
    return ari_scores_dict


def create_bootstrap_dict(columns, orig_df):
    """
    creates dictionary structure for the jaccard score_dict
    :param columns: list of names of columns representing different clustering algorithms
    :param orig_df: pd dataframe with cluster designations from original clustering
    :return: dictionary ready to be used as scores_dict for jaccard scoring
    """
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
        nc_df, ss = make_clusters(sample_v_df, 'no_output', save_csv=False, visualize=False)
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


def run_bootstrap_sl(u_df, c_df, n, num_reps, json_output_path, num_clusters):
    """
    takes umap dataframe and original cluster clustering dataframe and creates num_reps bootstrap replications of size
    n from the umap dataframe and evaluates ARI and Jaccard scores averaged over all bootstrap replications
    :param u_df: pd dataframe with umap dimension reduced data
    :param c_df: pd dataframe with clusterings
    :param n: number of samples in bootstrap replication, size of bootstrap dataset
    :param num_reps: number of replications
    :param json_output_path: where to save evaluation metric scores
    :param num_clusters: number of clusters to find in the clustering
    :return:
    """
    # combine u_df, c_df so no errors when sampling
    cu_df = u_df.merge(c_df)
    # derive cols from the dataframe
    cluster_cols = list(c_df.columns)[1:]
    ft_cols = list(u_df.columns)[1:]
    jcs_dict = create_bootstrap_dict(cluster_cols, c_df)
    ari_scores = {key: [] for key in cluster_cols}
    for i in range(num_reps):
        # sampling + setting up data
        sample_df = cu_df.sample(n=n, replace=True)
        sample_df = sample_df.dropna(axis=1)
        oc_df = sample_df.loc[:, ['PatientSeqID'] + cluster_cols]  # storing original clusters for comparison
        sample_u_df = sample_df.loc[:, ['PatientSeqID'] + ft_cols]
        nc_df, ss = make_clusters_sl(sample_u_df, 'no_output', 'no_output', save_csv=False, visualize=False, umap=True,
                                     random=True, bootstrap=num_clusters)
        # calculating Jaccard coefficients
        jcs_dict = calc_jaccard(nc_df, oc_df, cluster_cols, jcs_dict)
        # calculating Adjusted Rand Index
        ari_scores = calc_ari(nc_df, oc_df, cluster_cols, ari_scores)
    # bootstrap averaging
    avg_jcs_dict = create_bootstrap_dict(cluster_cols, c_df)
    avg_ari_scores = {key: [] for key in cluster_cols}
    for alg in cluster_cols:
        avg_ari_scores[alg] = np.mean(np.array(ari_scores[alg]))
        for cl in jcs_dict[alg].keys():
            avg_jcs_dict[alg][cl] = np.mean(np.array(jcs_dict[alg][cl]))
    # save to json
    with open(json_output_path, "w") as outfile:
        bootstrap_dict = {'jaccard': avg_jcs_dict, 'ari': avg_ari_scores}
        json.dump(bootstrap_dict, outfile)
    return avg_jcs_dict, avg_ari_scores


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

