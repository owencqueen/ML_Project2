import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from clustering_model import *

def plot_classwise_accs(regular_accs, pp_accs, title = "Accuracy vs. Sampling Strategy"):
    '''
    Generates 3D plot of overall accuracy against classwise accuracies for regular and kmeans++
        sampling methods

    Arguments:
    ----------
    regular_accs: tuple of (list, list, float)
        - See return value of run_sampling_strategies for more details
    pp_accs: tuple of (list, list, float)
        - See return value of run_sampling_strategies for more details
    title: string
        - Title of plot to be displayed

    Returns:
    --------
    No explicit return value
    '''
    
    x_reg = [acc[0] for acc in regular_accs]
    y_reg = [acc[1] for acc in regular_accs]
    z_reg = [acc[2] for acc in regular_accs]

    x_pp = [acc[0] for acc in pp_accs]
    y_pp = [acc[1] for acc in pp_accs]
    z_pp = [acc[2] for acc in pp_accs]

    ax = plt.axes(projection = '3d')

    ax.scatter3D(x_reg, y_reg, z_reg, c = "blue", label = "Regular Sampling")
    ax.scatter3D(x_pp, y_pp, z_pp, c = "red", label = "kMeans++ Sampling")

    ax.set_xlabel("Class 0 Accuracies")
    ax.set_ylabel("Class 1 Accuracies")
    ax.set_zlabel("Overall Accuracy")

    plt.title(title)
    plt.legend()
    plt.show()

def plot_sum_classwise(regular_accs, pp_accs, title = "Sum of Classwise Accuracies"):
    '''
    Generates histogram of sum of classwise accuracies for regular and kmeans++
        sampling methods

    Arguments:
    ----------
    regular_accs: tuple of (list, list, list)
        - See return value of run_sampling_strategies for more details
    pp_accs: tuple of (list, list, list)
        - See return value of run_sampling_strategies for more details
    title: string
        - Title of plot to be displayed

    Returns:
    --------
    No explicit return value
    '''
    
    classwise_sums_reg = [(c[0] + c[1]) for c in regular_accs]
    classwise_sums_pp = [(c[0] + c[1]) for c in pp_accs]

    fig, ax = plt.subplots()

    ax.hist(classwise_sums_reg, bins = 20, alpha = 0.4, color = "blue", label = "Regular")
    ax.hist(classwise_sums_pp, bins = 20, alpha = 0.4, color = "red", label = "kMeans++")

    ax.set_title(title)
    ax.set_xlabel("Class 1 Acc + Class 2 Acc")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.show()

def run_sampling_strategies(test_data, times, mod_name = "kmeans"):
    '''
    Returns a dictionary with lists specifying classwise and total accuracies for model
        with different sampling strategies used.
    Assumes k = 2

    Arguments:
    ----------
    test_data: pandas dataframe
        - testing data on which to run sampling strategies
    times: int
        - Number of trials to run
    mod_name: string
        - Options: "kmeans" or "wta"
        - Specifies model to be ran (kmeans or Winner-Takes-All respectively)

    Returns:
    --------
    results: dictionary
        - Keys: "regular" and "++"
            - Correspond to regular and kmeans++ sampling methods, respectively
        - Holds tuples of form (list, list, list)
            - First list is Class 0 accuracies for each run
            - Second list is Class 1 accuracies for each run
            - Third list is overall accuracy for each run
    '''

    if (mod_name == "kmeans"):
        model = kmeans_classifier(test_data)

        results = {"regular":[], "++":[]}

        for i in range(0, times):
            labels = model.make_clusters(k = 2, show_runtime = False, sampling_method = "regular")
            class_acc, overall_acc = model.acc_stats(labels, k = 2, show = False, return_all = True)

            results["regular"].append( (class_acc[0], class_acc[1], overall_acc) )

        # Run the kmeans++ algorithm for sampling, gather data
        for i in range(0, times):
            labels = model.make_clusters(k = 2, show_runtime = False, sampling_method = "++")
            class_acc, overall_acc = model.acc_stats(labels, k = 2, show = False, return_all = True)

            results["++"].append( (class_acc[0], class_acc[1], overall_acc) )

    elif(mod_name == "wta"): # Run WTA

        model = wta_classifier(test_data)

        results = {"regular":[], "++":[]}

        for i in range(0, times):
            labels = model.make_clusters(epsilon = 0.01, k = 2, show_runtime = False, sampling_method = "regular")
            class_acc, overall_acc = model.acc_stats(labels, k = 2, show = False, return_all = True)

            results["regular"].append( (class_acc[0], class_acc[1], overall_acc) )

        # Run the kmeans++ algorithm for sampling, gather data
        for i in range(0, times):
            labels = model.make_clusters(epsilon = 0.01, k = 2, show_runtime = False, sampling_method = "++")
            class_acc, overall_acc = model.acc_stats(labels, k = 2, show = False, return_all = True)

            results["++"].append( (class_acc[0], class_acc[1], overall_acc) )

    return results

def run_varying_epsilon_values(test_data, epsilons):
    '''
    Runs varying epsilon values on WTA

    Arguments:
    ----------
    test_data: pandas dataframe
        - testing data on which to run the classification
    epsilons: list of floats
        - List of epsilon values to try
        - Must be between 0 and 1

    Returns:
    --------
    No explicit return value
    '''
    model = wta_classifier(test_data)

    classwise_0 = []
    classwise_1 = []
    total_acc = []

    for e in epsilons:
        labels = labels = model.make_clusters(epsilon = e, k = 2, show_runtime = False, sampling_method = "++")
        class_acc, overall_acc = model.acc_stats(labels, k = 2, show = False, return_all = True)

        classwise_0.append(class_acc[0])
        classwise_1.append(class_acc[1])
        total_acc.append(overall_acc)

    # Plot each separate curve:
    plt.plot(epsilons, classwise_0, c = "blue", label = "Class 0")
    plt.plot(epsilons, classwise_1, c = "red", label = "Class 1")
    plt.plot(epsilons, total_acc, c = "green", label = "Total Classwise")

    plt.ylabel('Accuracy')
    plt.xlabel('Epsilon Value')
    plt.title('Varying Epsilon Values Accuracies')
    plt.legend()
    plt.show()
