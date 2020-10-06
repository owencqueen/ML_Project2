import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import clustering_model as cmodels

def run_clustering_for_img(data, k, model_type = "kmeans", show = True):
    '''
    Arguments:
    ----------
    data: pandas dataframe
        - Image data on which to cluster
    k: int
        - k value to run the clustering model on
    model_type: string, optional 
        - Default: kmeans
        - Specifies which model to run
        - Options: "kmeans" - runs kmeans model or "wta" - runs WTA model
    show: bool, optional    
        - Default: True
        - If True, shows the statistics for the runs

    Returns:
    --------
    new_image: np.array
        - This is the final image created by the routine
        - In the shape of the original data fed to the function
    rmse: float
        - Root mean squared error for the image to the original
    '''

    start_time = time.time()

    # Choose between kmeans or WTA
    if (model_type == "kmeans"):
        model = cmodels.kmeans_classifier(data, labeled_data = False)
        clusters, cluster_inds, labels = model.make_clusters(k = k, stop_tolerance = 0.1, \
                                    show_runtime = True, return_clusters = True)

    elif (model_type == "wta"):
        model = cmodels.wta_classifier(data, labeled_data = False)
        clusters, cluster_inds, labels = model.make_clusters(epsilon = 0.01, k = k, stop_tolerance = 0.1, \
                                    show_runtime = True, return_clusters = True)

    # Need to compute value for each pixel
    cluster_means = { i:(cmodels.compute_cluster_mean(clusters[i], num_features = data.shape[1])) \
                        for i in range(0, len(clusters)) }

    # Go over each index within clusters, assign to new numpy array
    new_image = np.zeros(shape = data.shape)

    total_squared_error = 0 
    num_points = new_image.shape[0]

    # Computes RMSE for the image
    for i in range(0, num_points):
        new_image[i, :] = cluster_means[labels[i]]
        total_squared_error += np.linalg.norm(np.subtract(np.array(data.iloc[i, :]), new_image[i, :]))

    total_squared_error /= num_points
    rmse = np.sqrt(total_squared_error)

    return new_image, rmse
