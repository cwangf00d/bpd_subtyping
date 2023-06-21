#################################################################################
# Imports + Package Setup                                                       #
#################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
import math
import umap.plot
import umap.umap_ as umap_
from kneed import KneeLocator
import scipy.cluster.hierarchy as shc
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from tqdm import tqdm


#################################################################################
# Helper Functions                                                              #
#################################################################################
def make_X(df):
    """
    takes Pandas dataframe where first column is PatientSeqID, all other columns are encodings and returns a numpy
    array of just data; order preserved
    """
    # assuming the vectorized dataframes are all constructed with only column 0, PatientSeqID,
    # to be removed from the values
    X = df.iloc[:, 1:].values
    X = StandardScaler().fit_transform(X)
    return X


# Reduction #
def make_2D_PCA(X, df, to_add=['PatientSeqID']):
    """
    makes 2D PCA reduction of data
    :param X: np array of data
    :param df: Pandas dataframe with columns to append to 2D reduction dataframe
    :param to_add: list of names of columns to add
    :return: dataframe with 2D PCA reduction and all other columns to add
    """
    pca_2 = PCA(n_components=2)
    pc_2 = pca_2.fit_transform(X)
    pca_2_df = pd.DataFrame(data=pc_2, columns=['pc1', 'pc2'])
    for col in to_add:
        pca_2_df[col] = list(df.loc[:, col])
        if col != 'PatientSeqID':
            plt.figure()
            sns.scatterplot(data=pca_2_df, x='pc1', y='pc2', hue=col)
    print(pca_2_df.head())
    print('explained variance ratio:', pca_2.explained_variance_ratio_)
    return pca_2_df


def make_95_PCA(X):
    """
    creates a Pandas dataframe with data reduced in dimensionality by PCA to cover 95% variance
    :param X: np array with data
    :return: pd dataframe with data only
    """
    pca_95 = PCA(.95)
    pc_95 = pca_95.fit_transform(X)
    pca_95_df = pd.DataFrame(data=pc_95)
    print('num_components:', pca_95_df.shape[1])
    return pca_95_df


def make_2D_UMAP(X, df, subgroups=[]):
    """
    takes data from np array and reduces dimensionality to 2 dimensions using UMAP, will plot according to labeled
    subgroups in dataframe with same order of points
    :param X: np array with data
    :param df: pd dataframe with same ordering, other labels
    :param subgroups: list of names of columns to plot
    """
    mapper = umap_.UMAP().fit(X)
    for sg in subgroups:
        plt.figure()
        umap.plot.points(mapper, labels=df[sg], theme='fire')


def make_UMAP(X, n_n=15, n_c=10, min_d=0.1):
    """
    takes data from np array and reduces dimensionality to 10 dimensions using UMAP, with options to adjust number of
    nearest neighbors, number of components, and effective minimum distance between points
    :param X: np array with data
    :param n_n: size of local neighborhood; default set to 15
    :param n_c: number of components; dimension of space to embed into; default set to 10
    :param min_d: effective minimum distance between embedding points; default set to 0.1
    :return: pd dataframe with dimension reduced data, ordering preserved
    """
    reducer = umap.umap_.UMAP(n_neighbors=n_n,  # default 15, The size of local neighborhood (in terms of number of
                              # neighboring sample points) used for manifold approximation.
                              n_components=n_c,  # default 2, The dimension of the space to embed into.
                              metric='euclidean',  # default 'euclidean', The metric to use to compute distances in
                              # high dimensional space.
                              n_epochs=1000,  # default None, The number of training epochs to be used in optimizing the
                              # low dimensional embedding. Larger values result in more accurate embeddings.
                              learning_rate=1.0,  # default 1.0, initial learning rate for the embedding optimization.
                              init='spectral',  # default 'spectral', How to initialize the low dimensional embedding.
                              # Options are: {'spectral', 'random', A numpy array of initial embedding positions}.
                              min_dist=min_d,  # default 0.1, The effective minimum distance between embedded points.
                              spread=1.0,  # default 1.0, The effective scale of embedded points. In combination with
                              # ``min_dist`` this determines how clustered/clumped the embedded points are.
                              low_memory=False,  # default False, For some datasets the nearest neighbor computation can
                              # consume a lot of memory. If you find that UMAP is failing due to memory constraints
                              # consider setting this option to True.
                              set_op_mix_ratio=1.0,  # default 1.0, The value of this parameter should be between 0.0
                              # and 1.0; a value of 1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
                              # intersection.
                              local_connectivity=1,  # default 1, The local connectivity required -- i.e. the number of
                              # nearest neighbors that should be assumed to be connected at a local level.
                              repulsion_strength=1.0,  # default 1.0, Weighting applied to negative samples in low
                              # dimensional embedding optimization.
                              negative_sample_rate=5,  # default 5, Increasing this value will result in greater
                              # repulsive force being applied, greater optimization cost, but slightly more accuracy.
                              transform_queue_size=4.0,  # default 4.0, Larger values will result in slower performance
                              # but more accurate nearest neighbor evaluation.
                              a=None,  # default None, More specific parameters controlling the embedding. If None these
                              # values are set automatically as determined by ``min_dist`` and ``spread``.
                              b=None, # default None, More specific parameters controlling the embedding. If None these
                              # values are set automatically as determined by ``min_dist`` and ``spread``.
                              random_state=42,  # default: None, If int, random_state is the seed used by the random
                              # number generator;
                              metric_kwds=None,  # default None) Arguments to pass on to the metric, such as the ``p``
                              # value for Minkowski distance.
                              angular_rp_forest=False,  # default False, Whether to use an angular random projection
                              # forest to initialise the approximate nearest neighbor search.
                              target_n_neighbors=-1,  # default -1, The number of nearest neighbors to use to construct
                              # the target simplcial set. If set to -1 use the ``n_neighbors`` value.
                              # target_metric='categorical', # default 'categorical', The metric used to measure
                              # distance for a target array is using supervised dimension reduction. By default this is
                              # 'categorical' which will measure distance in terms of whether categories match or are
                              # different.
                              # target_metric_kwds=None, # dict, default None, Keyword argument to pass to the target
                              # metric when performing supervised dimension reduction. If None then no arguments are
                              # passed on.
                              # target_weight=0.5, # default 0.5, weighting factor bw data topology and target topology.
                              transform_seed=42,  # default 42, Random seed used for the stochastic aspects of the
                              # transform operation.
                              verbose=False,  # default False, Controls verbosity of logging.
                              unique=False,  # default False, Controls if the rows of your data should be uniqued before
                              # being embedded.
                              )
    umap_data = reducer.fit_transform(X)
    # Check the shape of the new data
    print('Shape of reduced matrix: ', umap_data.shape)
    umap_df = pd.DataFrame(data=umap_data)
    return umap_df


def create_cluster_df(df, to_add=['PatientSeqID']):
    """
    creates a new dataframe for storing cluster identities
    :param df: pd dataframe from which we have columns to add to the new cluster df
    :param to_add: list of names of columns to add
    :return: new pd dataframe with columns kept
    """
    new_df = pd.DataFrame()
    for col in to_add:
        new_df[col] = list(df.loc[:, col])
    return new_df


def round_down_to_odd(f):
    """
    takes a number and rounds to get an odd number
    :param f: float to round
    :return: returns odd number rounded form of this float
    """
    f = int(np.ceil(f))
    return f - 1 if f % 2 == 0 else f


def find_distances(df, X, visualize=True):
    """
    takes dataframe with data and returns distances, for use in DBScan method
    :param df: pd dataframe with data values
    :param X: np array with data values
    :param visualize: whether or not to plot
    :return: distances for dbscan usage
    """
    nearest_neighbors = NearestNeighbors(n_neighbors=round_down_to_odd(math.sqrt(df.shape[0])))
    neighbors = nearest_neighbors.fit(df)
    distances, indices = neighbors.kneighbors(df)
    distances = np.sort(distances[:, 10], axis=0)
    if visualize:
        fig = plt.figure(figsize=(5, 5))
        plt.plot(distances)
        plt.xlabel("Points")
        plt.ylabel("Distance")
    return distances


def find_dbscan_clusters(df, X, distances, min_dim, cluster_col_name, visualize=True):
    """
    using given distances and data, find number of DBScan clusters
    :param df: pd dataframe with data
    :param X: np array with data
    :param distances: distances derived from find_distances function
    :param min_dim: smallest dimension
    :param cluster_col_name: string for name to put in as new column
    :param visualize: whether or not to plot the clusters
    :return: new cluster dataframe with DBScan clusters added as a column
    """
    # finding eps
    i = np.arange(len(distances))
    knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')
    if visualize:
        fig = plt.figure(figsize=(5, 5))
        knee.plot_knee()
        plt.xlabel("Points")
        plt.ylabel("Distance")
    curr_eps = distances[knee.knee]
    # fitting dbscan cluster
    dbscan_cluster = DBSCAN(eps=curr_eps, min_samples=min_dim+1)
    dbscan_cluster.fit(X)
    # printing results
    labels = dbscan_cluster.labels_
    N_clus = len(set(labels))-(1 if -1 in labels else 0)
    print('Estimated no. of clusters: %d' % N_clus)
    n_noise = list(dbscan_cluster.labels_).count(-1)
    print('Estimated no. of noise points: %d' % n_noise)
    # returning dataframe with cluster results
    df[cluster_col_name] = dbscan_cluster.labels_
    return df


def make_dbscan(data_df, cluster_df, X, cluster_col_name, visualize=True):
    """
    take data dataframe and find dbscan clusters, combines all in one
    :param data_df: pd dataframe with data
    :param cluster_df: pd dataframe for storing cluster identities
    :param X: np array with data
    :param cluster_col_name: name of column for new cluster col in col df
    :param visualize: whether or not to plot
    """
    distances = find_distances(data_df, X, visualize)
    min_dim = X.shape[1]
    cluster_df = find_dbscan_clusters(cluster_df, X, distances, min_dim, cluster_col_name, visualize)
    if visualize:
        plt.figure()
        sns.histplot(cluster_df[cluster_col_name])


def make_dendrogram(X, graph_title, visualize=True):
    """
    creating dendrogram to decide how many clusters to use in hierarchical clustering
    :param X: np array with just data
    :param graph_title: title of graph if visualizing
    :param visualize: whether or not to plot dendrogram
    :return:
    """
    Z = shc.linkage(X, method='ward', metric='euclidean')
    c, cophenetic_dists = cophenet(Z, pdist(X))
    print('cophenetic coefficient:', c)
    if visualize:
        plt.figure(figsize=(10, 7))
        plt.title(graph_title)
        dend = shc.dendrogram(Z)


def make_hierarchical(X, cluster_df, num_clusters, col_name, visualize=True):
    """
    find the hierarchical clustering given number of clusters
    :param X: np array with data
    :param cluster_df: pd dataframe for storing clusters
    :param num_clusters: number of clusters to find
    :param col_name: name of column
    :param visualize: whether or not to plot clusters
    """
    cluster = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='ward')
    if visualize:
        plt.figure()
        sns.histplot(cluster.fit_predict(X))
    cluster_df[col_name] = cluster.fit_predict(X)


def silhouette_hier_cluster(X, visualize=True):
    """
    takes data and finds optimal hierarchical clustering
    :param X: np array with data
    :param visualize:whether or not to plot
    :return: number of clusters to find based on silhouette scores
    """
    range_n_clusters = [2, 3, 4, 5, 6]
    avgs = []
    avg_dict = {}
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        y_predict = clusterer.fit_predict(X)
        cluster_labels = clusterer.labels_

        silhouette_avg = silhouette_score(X, cluster_labels)
        avgs.append(silhouette_avg)
        avg_dict[silhouette_avg] = n_clusters
        if visualize:
            if silhouette_avg > 0.0:
                print("For n_clusters =", n_clusters,
                      "The average silhouette_score is :", silhouette_avg)
                fig, (ax1, ax2) = plt.subplots(1, 2)

                fig.set_size_inches(15, 5)

                ax1.set_xlim([-0.1, 1])
                ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

                sample_silhouette_values = silhouette_samples(X, cluster_labels)

                y_lower = 10
                for i in range(n_clusters):
                    ith_cluster_silhouette_values = \
                        sample_silhouette_values[cluster_labels == i]

                    ith_cluster_silhouette_values.sort()

                    size_cluster_i = ith_cluster_silhouette_values.shape[0]
                    y_upper = y_lower + size_cluster_i

                    color = cm.nipy_spectral(float(i) / n_clusters)
                    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                      0, ith_cluster_silhouette_values,
                                      facecolor=color, edgecolor=color, alpha=0.7)
                    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                    y_lower = y_upper + 10  # 10 for the 0 samples

                ax1.set_title("The silhouette plot for the various clusters.")
                ax1.set_xlabel("The silhouette coefficient values")
                ax1.set_ylabel("Cluster label")
                ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

                ax1.set_yticks([])  # Clear the yaxis labels / ticks
                ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
                ax2 = Axes3D(fig, auto_add_to_figure=False)
                fig.add_axes(ax2)
                colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
                ax2.scatter(X[:, 1], X[:, 2], X[:, 0], marker='o', s=20, lw=0, alpha=0.7,
                            c=colors, edgecolor='k')

                plt.suptitle(("Silhouette analysis for HAC-ward clustering on sample data "
                              "with n_clusters = %d" % n_clusters),
                             fontsize=14, fontweight='bold')
            plt.show()
    return avg_dict[np.max(np.array(avgs))]


def silhouette_KM_clusterer(X, visualize=True, random=False, range_n_clusters=list(range(2, 7))):
    """
    takes data and tries out different numbers of clusters + measures silhouette score to find the number of clusters
    with best silhouette score
    :param X: np array with data
    :param visualize: whether or not to plot the silhouette plots
    :param random: whether or not to set a random state, False = not random, True = random
    :param range_n_clusters: list of numbers of clusters to test
    :return: number of clusters with highest silhouette score
    """
    avgs = []
    avg_dict = {}
    for n_clusters in range_n_clusters:
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        if not random:
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        else:
            clusterer = KMeans(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )
        avgs.append(silhouette_avg)
        avg_dict[silhouette_avg] = n_clusters

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        if visualize:
            # Create a subplot with 1 row and 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(
                X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
            )

            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(
                centers[:, 0],
                centers[:, 1],
                marker="o",
                c="white",
                alpha=1,
                s=200,
                edgecolor="k",
            )

            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

            plt.suptitle(
                "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
                % n_clusters,
                fontsize=14,
                fontweight="bold",
            )
            plt.show()
    return avg_dict[np.max(np.array(avgs))], np.max(np.array(avgs))


def make_kmeans(X, cluster_df, num_clusters, col_name, visualize=True):
    """
    takes data and fits a KMeans clustering with specified number of clusters
    :param X: np array with data
    :param cluster_df: pd dataframe for storing cluster designations as columns
    :param num_clusters: number of clusters to find
    :param col_name: name of column for storing cluster designations in df
    :param visualize: whether to plot
    :return: cluster dataframe with KMeans cluster column included
    """
    kmeans = KMeans(n_clusters=num_clusters)
    y = kmeans.fit_predict(X)
    cluster_df[col_name] = y
    if visualize:
        sns.histplot(cluster_df[col_name])
    return cluster_df


def make_clusters(curr_df, output_path, save_csv=True, visualize=True):
    """
    takes dataframe and output path and finds clusters using DBScan, Hierarchical, KMeans clustering with PCA or UMAP
    dimensionality reduction
    :param curr_df: data to be clustered in dataframe
    :param output_path: where to save the file
    :param save_csv: whether to save a csv with cluster df
    :param visualize: whether to plot
    :return: dataframe with clusters
    """
    # prep data + perform reduction
    curr_df = curr_df.dropna(axis=0)
    curr_X = make_X(curr_df)
    pca_X = make_95_PCA(curr_X).values
    umap_X = make_UMAP(curr_X, n_n=15, n_c=10, min_d=0.1).values
    # prepare storage dataframe for clusters
    curr_cluster_df = create_cluster_df(curr_df, to_add=['PatientSeqID'])
    # run clustering algorithms
    if visualize:
        for data_X, dr_type in tqdm(zip([pca_X, umap_X], ['pca', 'umap'])):
            # dbscan
            make_dbscan(curr_df, curr_cluster_df, data_X, dr_type + '_DBScan')
            # hierarchical
            make_dendrogram(data_X, dr_type + ' dendrograms')
            make_hierarchical(data_X, curr_cluster_df, silhouette_hier_cluster(data_X), dr_type + '_Hier')
            # kmeans
            km, ss = silhouette_KM_clusterer(data_X)
            make_kmeans(data_X, curr_cluster_df, km, dr_type + '_KMeans')
    else:
        for data_X, dr_type in tqdm(zip([pca_X, umap_X], ['pca', 'umap'])):
            # dbscan
            make_dbscan(curr_df, curr_cluster_df, data_X, dr_type + '_DBScan', visualize=False)
            # hierarchical
            make_dendrogram(data_X, dr_type + ' dendrograms', visualize=False)
            make_hierarchical(data_X, curr_cluster_df, silhouette_hier_cluster(data_X, visualize=False),
                              dr_type + '_Hier', visualize=False)
            # kmeans
            km, ss = silhouette_KM_clusterer(data_X, visualize=False)
            make_kmeans(data_X, curr_cluster_df, km, dr_type + '_KMeans', visualize=False)
    # save to csv
    if save_csv:
        curr_cluster_df.to_csv(output_path, index=False)
    else:
        return curr_cluster_df


def make_clusters_sl(curr_df, c_output_path, u_output_path, save_csv=True, visualize=True, umap=False,
                     random=False, bootstrap=None):
    """
    streamlined clustering function, will take data and apply UMAP reduction + KMeans clustering
    :param curr_df: dataframe with data
    :param c_output_path: where to save cluster dataframe
    :param u_output_path: where to save umap reduced dataframe
    :param save_csv: whether to save csv (not used in bootstrap)
    :param visualize: whether to visualize
    :param umap: whether to create new umap reduction (not used in bootstrap)
    :param random: whether to use a random_state, True == no random_state, False == random_state = 10
    :param bootstrap: number of clusters (not used in bootstrap, is specified)
    :return: tuple of cluster dataframe and silhouette score
    """
    curr_df = curr_df.dropna(axis=0)
    umap_X = make_X(curr_df)
    if not umap:
        # prep data + perform reduction
        # do pca upper bound for number of components in umap
        umap_df = make_UMAP(umap_X, n_n=15, n_c=make_95_PCA(umap_X).shape[1], min_d=0.1)  # 128, 256,
        umap_X = umap_df.values
        umap_df.insert(loc=0, column='PatientSeqID', value=curr_df['PatientSeqID'])
        umap_df.to_csv(u_output_path, index=False)
    # prepare storage dataframe for clusters
    curr_cluster_df = create_cluster_df(curr_df, to_add=['PatientSeqID'])
    # run clustering algorithms
    if not bootstrap:  # if set to default, None, will go through this if statement
        km, ss = silhouette_KM_clusterer(umap_X, visualize, random)
        make_kmeans(umap_X, curr_cluster_df, km, 'umap_KMeans', visualize)
    else:
        km, ss = silhouette_KM_clusterer(umap_X, visualize, random, range_n_clusters=[bootstrap])
        make_kmeans(umap_X, curr_cluster_df, bootstrap, 'umap_KMeans', visualize)
    # save to csv
    if save_csv:
        curr_cluster_df.to_csv(c_output_path, index=False)
    else:
        return curr_cluster_df, ss
