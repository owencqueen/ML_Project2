import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

def euc_dist(x0, x1):
    '''
    Computes euclidean distance between two points

    Arguments:
    ----------
    x0 and x1: both np.arrays
        - Takes difference of these vectors and returns the norm of the difference

    Returns:
    --------
    euc_diff: float
        - length of difference between x0 and x1
    '''

    return np.linalg.norm(x0 - x1)

def assign_to_cluster(means, point, dist_fn = "euclidean"):
    '''
    Assigns a given point to a cluster given different cluster means

    Arguments:
    ----------
    means: list of np arrays
        - List of means for each class
    point: np array
        - Point that we are classifying
    dist_fn: string
        - Specifies name for distance function to use in classifying points to clusters
        - Default: "euclidean"
        - Options: "euclidean"
        
    Returns:
    --------
    new_class: int
        - int corresponding to new class
    '''

    min_dist = 9999999 # Very big number

    for i in range(0, len(means)):

        if (dist_fn == "euclidean"): # Current support for euclidean distance function

            dist = euc_dist(point, means[i])
            if (dist < min_dist):
                min_dist = dist
                min_dist_i = i

    return min_dist_i

def compute_cluster_mean(cluster, num_features):
    '''
    Arguments:
    ----------
    cluster: list of np arrays of floats
        - Cluster of points to calculate mean for
    num_features: int
        - Number of features in dataset
    
    Returns:
    --------
    means: list of floats
        - Mean point for the cluster
        - Will be a 1xn vector where n = number of dimensions of points
    '''
    
    mean = []

    # Size of the whole cluster
    size_cluster = len(cluster)

    # Go through and get feature-wise averages
    for i in range(0, num_features):
        #cluster[:][i]
        feature_sum = np.sum([cluster[j][i] for j in range(0, size_cluster)])
        mean.append(feature_sum / size_cluster)

    return mean

def random_means(train_data, k, num_features):
    '''
    Computes random means within the bounds of the training data

    Arguments:
    ----------
    train_data: pandas dataframe
        - Training data on which to base the generation of means
        - Assumed to contain labels
    k: int
        - Number of random means which to generate
        - Corresponds to k value in kMeans of WTA
    num_features: int
        - Number of features in dataset

    Returns:
    --------
    means: list of lists
        - Returns a (k) x (num_features) list
        - i.e. generates k different means
    '''
    
    # Create empty lists for means
    means = []
    for i in range(0, k):
        means.append([])

    # Choose two random points within the range of samples:
    for i in range(0, num_features):
        min_feat = min(list(train_data.iloc[:, i]))
        max_feat = max(list(train_data.iloc[:, i]))

        rand_feats = random.sample((list(np.linspace(min_feat, max_feat, k * 20))), k)
        
        for j in range(0, k):
            means[j].append(rand_feats[j])

    return means

def random_means_kmeans_plus_plus(train_data, k, num_features):
    '''
    Chooses the random means based on kmeans++ algorithm

    Arguments:
    ----------
    train_data: pandas dataframe
        - Data which we will choose random clusters for 
    k: int
        - Number of means to choose
    num_features: int
        - Number of features in the dataset

    Returns:
    --------
    means: list of k length
        - Random means
    '''

    #sampling_data = train_data.iloc[:, 0:-1]
    sampling_data = train_data.iloc[:, 0:num_features]

    # First is chosen randomly
    means = []

    rand_ind = np.random.choice(range(0, sampling_data.shape[0]))

    means.append(sampling_data.iloc[rand_ind, :])

    for i in range(1, k):
        
        distribution = [D_x(means, point) for index, point in sampling_data.iterrows()]

        # Turn the distribution into a probability mass function:
        distribution = [distribution[i] / sum(distribution) for i in range(0, len(distribution))]
        
        rand_ind = np.random.choice(range(0, sampling_data.shape[0]), p = distribution)

        means.append(sampling_data.iloc[rand_ind, :])

    return means

def D_x(means, point):
    ''' 
    Computes D_x value for kmeans++ algorithm

    Arguments:
    ----------
    means: list of np.arrays
        - mean points to use to classify D_x
    point: np.array
        - Point on which to calculate the D(x)

    Returns:
    --------
    sq_norm: float
        - Scalar value of distance from point to nearest mean squared
    '''

    mean_for_D = assign_to_cluster(means, point)

    return (np.linalg.norm(np.array(mean_for_D) - np.array(point)) ** 2)

def split_data_by_class(train_data):
    '''
    Splits the training data into its individual classes

    Arguments:
    ----------
    train_data: pandas dataframe
        - Dataframe that you want to split
        - Assumption is that labels are contained in the last row (row -1)
    
    Returns:
    --------
    split_data: dictionary
        - Keyed on labels found in last row
        - Groups the data by label found in last row
        - Note: split data does still contain labels
    '''
    
    labels = train_data.iloc[:,-1]  # Get last row (labels)

    labels_unique = set(labels) # Convert to a set - will isolate unique values
    labels_unique = list(labels_unique) 

    split_data = {l:[] for l in labels_unique}  

    for l in labels_unique:
        split_data[l] = train_data.loc[lambda d: d.iloc[:,-1] == l, :]

    return split_data

def accuracy_stats(self, labels, k, show = True, return_all = False):
    '''
    Prints the overall accuracy and classwise accuracy after a prediction procedure has finished
    Arguments:
    ----------
    test_data: pandas dataframe
        - Dataframe containing the test data
    k: int
        - k parameter being used on the classification routine
    show: bool, optional
        - Default: True
        - If true, prints the accuracy statistics
    return_all: bool, optional
        - If true, returns both overall and classwise accuracies

    Returns:
    --------
    overall_acc: float
        - Overall accuracy of classification routine calculated based on labels given
    '''

    # First, need to go through and adjust labels based on majority membership in clusters:
    labels_with_inds = list(enumerate(labels))
    unique_l = set(labels)
    unique_l = list(unique_l)

    # Separate based on labels
    clusters = {l: [] for l in unique_l}
    
    # Create clusters from classification (based on index in training data)
    for i in range(len(labels)):
        clusters[labels_with_inds[i][1]].append(labels_with_inds[i][0])

    # Get cluster labels
    true_labels = self.train.iloc[:,-1]  # Get last row (labels)
    unique_true_l = set(true_labels)
    unique_true_l = list(unique_true_l)

    # Stores the transformed cluster labels
    true_cluster_labels = {i:i for i in unique_l}
    labels_not_used = [i for i in unique_l]

    for c in clusters.keys(): # Iterate over the labels for the clusters

        # Stores real label counts in the cluster
        real_labels_in_cluster = {l: 0 for l in unique_true_l}

        for ind in clusters[c]: # Go through and count the true prescence of labels in clusters
            train_class = true_labels[ind]
            real_labels_in_cluster[train_class] += 1

        max_val   = max(real_labels_in_cluster.values())
        label_max = [key for key, val in real_labels_in_cluster.items() \
                        if val == max_val] 

        if (len(label_max) == 1):
            true_cluster_labels[c] = label_max[0] # Get maximum label and store
            labels_not_used.remove(c)

    # Go cluster labels not assigned, set to zero
    for c in labels_not_used:
        true_cluster_labels[c] = 0

    # Adjust the keys in the dictionary
    new_clusters = {}
    for o,n in true_cluster_labels.items(): # o,n == old, new
        new_clusters[n] = clusters[o] 

    overall_correct = 0
    class_wise_correct = {l:0 for l in self.labels_unique}
    class_wise_total = {l:0 for l in self.labels_unique}
    
    for c, cluster_inds in new_clusters.items():

        for i in cluster_inds:

            if (true_labels[i] == c):
                overall_correct += 1 # Add to correct count if they match

                # Add to class-wise accuracy
                class_wise_correct[c] += 1

            # Computes specificity for negatives, sensitivity for positives
            class_wise_total[true_labels[i]] += 1

    overall_acc = overall_correct / len(true_labels)
    class_wise_accuracy = {i:(class_wise_correct[i] / class_wise_total[i]) \
                            for i in self.labels_unique}

    if (show):  # Prints accuracies
        print("Overall Accuracy:    {acc:.5f}".format(acc = overall_acc))

        for l in self.labels_unique:
            print("Classwise accuracy for \"{cl}\" class: {acc:.5f}"\
                    .format(cl = l, acc = class_wise_accuracy[l]))

    if (return_all):
        return (class_wise_accuracy, overall_acc)
    else:
        return overall_acc

def plot_mem_changes(kmeans_model, wta_model):
    '''
    Plots membership changes for kmeans and wta
        - Must run classification routines for each model before calling this function

    Arguments:
    ----------
    kmeans_model: kmeans_classifier instance
        - - kMeans model which to use for plotting membership changes over epochs
    wta_model: wta_classifier instance
        - WTA model which to use for plotting membership changes over epochs

    Returns:
    --------
    No explicit return
    '''
    fig, (ax1, ax2) = plt.subplots(2)

    # Plots change_membership member list for each class:
    ax1.plot(range(0, len(kmeans_model.change_membership)), kmeans_model.change_membership, c = "orange") 
    ax2.plot(range(0, len(wta_model.change_membership)), wta_model.change_membership, c = "green") 

    fig.suptitle("Convergence of Clustering Algorithms")

    ax1.text(range(0, len(kmeans_model.change_membership))[-2], 190, "kMeans")
    ax2.text(range(0, len(wta_model.change_membership))[-2], 190, "WTA")

    ax1.set_ylabel("Samples Changing Class")
    ax1.set_xlabel("Epochs")

    ax2.set_ylabel("Samples Changing Class")
    ax2.set_xlabel("Epochs")

    plt.show()


class kmeans_classifier:

    # Sets the accuracy stats function for this class
    acc_stats = accuracy_stats

    def __init__(self, training_df, labeled_data = True):
        '''
        Instantiate a kmeans clusering object

        Arguments:
        ----------
        training_df: pandas dataframe
            - Training data for the model
        labeled_data: bool, optional
            - Default: True
            - If true, last label in training data is the labels
                - i.e. this last row is not considered in calculations

        Returns:
        --------
        No explicit return value
        '''
        
        self.train = training_df

        # Subtract one to account for labels
        if (labeled_data):
            self.num_features = (training_df.shape[1] - 1)
        else:
            self.num_features = training_df.shape[1]

        if (labeled_data):
            # Extract label information - will need for testing
            labels = training_df.iloc[:,-1]  # Get last row (labels)

            self.labels_unique = set(labels) # Convert to a set - will isolate unique values
            self.labels_unique = list(self.labels_unique) 

    def make_clusters(self, k, stop_tolerance = 0, show_runtime = False, return_clusters = False, \
                        sampling_method = "regular"):
        '''
        Runs the clustering routine based on a given k value

        Arguments:
        ----------
        k: int
            - Specifies the k hyperparameter in the kmeans algorithm
        stop_tolerance: float, optional
            - Default: 0 (i.e. stop clustering when no samples have changed class)
            - Algorithm stops when this percentage or less of samples are changing classes in an epoch
        show_runtime: bool, optional
            - Default: False
            - If true, prints the runtime & number of epochs
        return_clusters: bool, optional
            - Default: False
            - If true, returns (clusters, labels) where clusters is a list
                of np arrays
        sampling_method: string, optional
            - Default: "regular"
            - Defines sampling method for initial cluster centers
            - See the discussion in "Bonus" section about sampling method for initial clusters
            - Options: "regular", "++" - regular method and kMeans++ method respectively

        Returns:
        --------
        labels: list of ints
            - List contains labels for each of the derived
            - Note: labels are arbitrary in that the only significance they carry is 
                the difference between them
        Note: could be (clusters, labels) depending on value of return_clusters
        '''

        if (show_runtime):
            start_time = time.time()

        # Choose random starting points:
        if (sampling_method == "regular"):
            means = random_means(train_data = self.train, k = k, num_features = self.num_features)
        elif (sampling_method == "++"):
            means = random_means_kmeans_plus_plus(train_data = self.train, k = k, \
                                num_features = self.num_features)

        # Put all labels to -1 because we know that no label will be -1
        labels = [-1] * self.train.shape[0]

        self.change_membership = []

        while (True):

            # Compute cluster means

            num_change = 0

            clusters = [] # Initialize empty clusters
            cluster_inds = []
            for i in range(0, k):
                clusters.append([])
                cluster_inds.append([])

            # Assign points to clusters
            for i in range(0, self.train.shape[0]):
                sample = np.array(self.train.iloc[i, 0:self.num_features])

                new_label = assign_to_cluster(means = means, point = sample)

                # If this sample changed labels, 
                if (new_label != labels[i]):
                    num_change += 1

                labels[i] = new_label

                clusters[new_label].append(sample) # Append to clusters list
                cluster_inds[new_label].append(i)

                # Else, we don't do anything

            # Need for keeping track of points changing membership
            #   - Also tracks number of iterations (len)
            self.change_membership.append(num_change)

            if (num_change / self.train.shape[0]) <= stop_tolerance: # Break condition
                break

            # Compute new cluster means
            for i in range(0, k):
                means[i] = compute_cluster_mean(np.array(clusters[i]), self.num_features)

        if (show_runtime):
            print("")
            print("Epochs to Convergence:", len(self.change_membership))
            print("kmeans (k = {}) Runtime: {} seconds".format(k, time.time() - start_time))

        # Return clusters or not based on specification
        if (return_clusters): 
            return (clusters, cluster_inds, labels)
        else:
            return labels

    def plot_membership_changes(self, show = True):
        '''
        Plots the number of membership changes during clustering routine per epoch
            - Must run clustering routine before running this
            - Automatically shows the plot

        Arguments:
        ----------
        show: bool, optional
            - Default: True
            - If true, shows the plot

        Returns:
        --------
        No explicit return  value
        '''
        
        plt.plot(self.change_membership)
        plt.xlabel("Epoch")
        plt.ylabel("Points Changing Label")
        plt.title("Points Changing Labels per Iteration in k-Means")
        if(show):
            plt.show()
        else:
            return plt.gca()

    def run_multiple(self, k = 2, times = 10):
        '''
        Runs multiple routines for clustering with kMeans model and training data

        Arguments:
        ----------
        k: int
            - Corresponds to k value used in clustering
        times: int
            - Number of times you want to run the model

        Returns:
        --------
        No explicit return value
        '''

        overall_acc = 0

        for i in range(times): # Gets overall accuracies for different runs
            labels = self.make_clusters(k = k, show_runtime = True)
            overall_acc += self.acc_stats(labels = labels, k = 2, show = True)

        overall_acc /= times

        print("Average Overall Accuracy ({t} runs):    {acc:.5f}".format(t = times, acc = overall_acc))

class wta_classifier:

    # Sets the accuracy stats function for this class
    acc_stats = accuracy_stats

    def __init__(self, training_df, labeled_data = True):
        '''
        Instantiate a winner-takes-all clustering object

        Arguments:
        ----------
        training_df: pandas dataframe
            - Training data for the model
        labeled_data: bool, optional
            - Default: True
            - If true, last label in training data is the labels
                - i.e. this last row is not considered in calculations

        Returns:
        --------
        No explicit return value
        '''
        self.train = training_df

        if (labeled_data):
            # Subtract one to account for labels
            self.num_features = (training_df.shape[1] - 1)
        else:
            self.num_features = training_df.shape[1]

        # Extract label information - will need for testing
        if (labeled_data):
            labels = training_df.iloc[:,-1]  # Get last row (labels)

            self.labels_unique = set(labels) # Convert to a set - will isolate unique values
            self.labels_unique = list(self.labels_unique) 

        self.change_membership = [] # Keeps track of how many points switch labels

    def make_clusters(self, epsilon, k, stop_tolerance = 0, show_runtime = True, \
                        return_clusters = False, sampling_method = "regular"):
        '''
        Runs the clustering routine based on a given k value
        
        Arguments:
        ----------
        epsilon: float
            - Learning rate for WTA
            - Assume to be in range [0, 1]
        k: int
            - Specifies the k hyperparameter in the kmeans algorithm
        stop_tolerance: float, optional
            - Default: 0 (i.e. stop clustering when no samples have changed class)
            - Algorithm stops when this percentage or less of samples are changing classes in an epoch
        show_runtime: bool, optional
            - Default: False
            - If true, prints the runtime & number of epochs
        return_clusters: bool, optional
            - Default: False
            - If true, returns (clusters, labels) where clusters is a dictionary
                of pandas dataframes keyed on label values
        sampling_method: string, optional
            - Default: "regular"
            - Defines sampling method for initial cluster centers
            - See the discussion in "Bonus" section about sampling method for initial clusters
            - Options: "regular", "++" - regular method and kMeans++ method respectively

        Returns:
        --------
        labels: list of ints
            - List contains labels for each of the derived
            - Note: labels are arbitrary in that the only significance they carry is 
                the difference between them
        '''

        if (show_runtime):
            start_time = time.time()

        self.change_membership = []

        # Choose arbitrary cluster means
        if (sampling_method == "regular"):
            means = random_means(train_data = self.train, k = k, num_features = self.num_features)
        elif (sampling_method == "++"):
            means = random_means_kmeans_plus_plus(train_data = self.train, k = k, \
                                num_features = self.num_features)

        labels = [-1] * self.train.shape[0]

        while True:

            num_change = 0

            for i in range(0, self.train.shape[0]):
                sample = np.array(self.train.iloc[i, 0:self.num_features]) # Get sample from data

                # Find nearest cluster mean, classify those points
                new_label = assign_to_cluster(means = means, point = sample)

                # For each sample, move the cluster center towards the sample
                move_vec = np.multiply(epsilon, np.subtract(sample, means[new_label]))
                means[new_label] = np.add(means[new_label], move_vec)

                if (new_label != labels[i]):
                    num_change += 1

                labels[i] = new_label

            self.change_membership.append(num_change)

            # Stop when classification of samples has not changed (after epoch)
            # OR stop when we reach the tolerance level (0 by default)
            if (num_change / self.train.shape[0]) <= stop_tolerance:
                break

        if (show_runtime):
            print("")
            print("Epochs to Convergence:", len(self.change_membership))
            print("WTA (k = {}) Runtime: {} seconds".format(k, time.time() - start_time))

        if (return_clusters): # User can provide option to return tuple containing clusters and labels

            # Need to build clusters based on classes:
            clusters = []
            cluster_inds = []
            for i in range(0, k):
                clusters.append([])
                cluster_inds.append([])

            for i in range(0, self.train.shape[0]): # Gets the whole bunch of samples
                clusters[labels[i]].append(np.array(self.train.iloc[i, :]))
                cluster_inds[labels[i]].append(i)

            return (clusters, cluster_inds, labels)

        else: # If the user doesn't want clusters
            return labels

    def plot_membership_changes(self, show = True):
        '''
        Plots the number of membership changes during clustering routine per epoch
            - Must run clustering routine before running this
            - Automatically shows the plot

        Arguments:
        ----------
        show: bool, optional
            - Default: True
            - If true, shows the plot

        Returns:
        --------
        No explicit return  value
        '''
        
        plt.plot(self.change_membership)
        plt.xlabel("Epoch")
        plt.ylabel("Points Changing Label")
        plt.title("Points Changing Labels per Iteration in Winner-Takes-All")
        
        plt.show()

    def run_multiple(self, epsilon, k = 2, times = 10):
        '''
        Runs multiple routines for clustering with kMeans model and training data
        Arguments:
        ----------
        epsilon: int or list
            - If an int, every run is ran with this epsilon value
            - If a list, must have length equal to times
        k: int
            - Corresponds to k value used in clustering
        times: int
            - Number of times you want to run the model

        Returns:
        --------
        No explicit return value
        '''

        try:
            test_it = iter(epsilon)
        except TypeError:
            epsilon = [epsilon] * times

        if (len(epsilon) != times):
            print("Length of epsilon list must be times length")
            return -1

        overall_acc = 0

        for i in range(times):
            labels = self.make_clusters(epsilon[i], k = k)
            overall_acc += self.acc_stats(labels = labels, k = 2, show = True)

        overall_acc /= times

        print("Average Overall Accuracy ({t} runs):    {acc:.5f}".format(t = times, acc = overall_acc))




