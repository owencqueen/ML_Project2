import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors

class kNN_classifier:

    def __init__(self, train_data):
        '''
        Initializes a parametric learning classifer class variable in Python
        Arguments:
        ----------
        training_df: pandas dataframe
            - Training data for the class

        Returns:
        --------
        Trains the class on the training data and returns the class
        '''

        self.train = train_data

        self.num_features = train_data.shape[1] - 1

        labels = train_data.iloc[:,-1]
        labels = set(labels)
        self.labels = list(labels) # Get individual labels
        self.labels = [int(i) for i in self.labels]

        self.num_samples = train_data.shape[0]

        self.overall_acc_by_k = {}

        self.overall_acc_stats = {} # Initialize dictionary to keep up with accuracies

    def classify(self, testing_data, k, pp_weights = [1, 1], show_statistics = True, df_provided = False, progress_bar = False):
        '''
        Arguments:
        ----------
        testing_data: string or pandas dataframe
            - If testing_data is a string, will call read_csv to turn that data into pandas df
            - If a dataframe, simply copies it to use in the function
            - If testing_data is a dataframe, df_provided must be True
        k: int
            - Number of k-Nearest Neighbors to consider when classifying testing sample
        pp_weights: list of floats or dictionary with floats as values, optional
            - Contains the prior probabilities for each class
            - Preferably a dictionary with keyed on labels
        show_statistics: bool, optional
            - If true, prints the statistics for the classification at the end of the routine
        df_provided: bool, optional
            - Must be true if testing_data is a pandas dataframe (not a string)
        progress_bar: bool, optional
            - If true, prints a message to indicate the progress of the classification routine

        Returns:
        --------
        No explicit return value
        Stores the predictions into the class (self.predictions)
        '''

        start_time = time.time() # Start the clock

        if (k not in self.overall_acc_by_k.keys()):
            # Initialize key for this given list if it hasn't been done already
            self.overall_acc_by_k[k] = []

        if (df_provided):
            test = testing_data
        else:
            test = pd.read_csv(testing_data)

        self.predictions = []

        dtype = [("dist", float), ("label", int)]

        run = 0
        total_runs = test.shape[0]

        for i in range(total_runs):

            x = test.iloc[i, 0:-1]

            dists = []

            # Calculate all euclidean distances:
            for sample in range(self.num_samples):
                sample_x = self.train.iloc[sample,0:-1]
                sample_l = self.train.iloc[sample, -1]
                dists.append( (euclidean_dist(sample_x, x), sample_l) )    
                    
            conj_dists = np.array(dists, dtype = dtype)  # Create structured array with labels
            conj_dists = np.sort(conj_dists, order = 'dist') # Sort structured array based on distance

            k_shortest = conj_dists[0:k] # Get k smallest dists

            k_labels = []

            for i in range(len(k_shortest)):
                k_labels.append(k_shortest[i][1])

            label_counts = [0] * len(self.labels) # Will be used to count frequencies

            # Take majority vote:
            for label in k_labels:
                label_counts[self.labels.index(label)] += 1

            # Implement weighting of the votes:
            label_counts = [(label_counts[i] * pp_weights[i]) for i in self.labels]

            max_label_ind = label_counts.index(max(label_counts))
            #print(self.labels[max_label_ind])
            self.predictions.append(self.labels[max_label_ind])

            if (progress_bar): # Print the progress bar if specified
                print("{} of {} runs".format(run, total_runs))
                #if ((run / total_runs * 100) % 10 == 0):
                #    print ("{} percent finished".format(int(run / total_runs * 100)))
            run += 1
            #run += 1

        if (show_statistics):
            self.accuracy_stats(test, k = k, save_overall = True) # Prints statistics for classification algorithm
            print("k = {} Runtime: {} seconds".format(k, time.time() - start_time))
            print("") # Need newline


    def accuracy_stats(self, test_data, k, save_overall = True):
        '''
        Prints the overall accuracy and classwise accuracy after a prediction procedure has finished
        Arguments:
        ----------
        test_data: pandas dataframe
            - Dataframe containing the test data
        k: int
            - k parameter being used on the classification routine
        save_overall: bool
            - If true, saves the overall accuracy in the class in a dict (keyed on k value)
        Returns:
        --------
        No explicit return value, only prints the stats to the console
        '''
        # Need:
        #   1. overall classification accuracy
        #   2. classwise accuracy - all classes
        #   3. run time - printed in classify function

        # Overall classification accuracy:
        true_labels = list(test_data.iloc[:, -1])

        overall_correct = 0
        class_wise_correct = [0] * len(self.labels)
        class_wise_total = [0] * len(self.labels)
        # For classwise accuracy, numbers will be stored in 

        for i in range(0, len(true_labels)):

            predict = self.predictions[i]

            if (true_labels[i] == predict):
                overall_correct += 1 # Add to correct count if they match

                # Add to class-wise accuracy
                class_wise_correct[self.labels.index(predict)] += 1

            class_wise_total[self.labels.index(true_labels[i])] += 1

        overall_acc = overall_correct / len(true_labels)
        class_wise_accuracy = [(class_wise_correct[i] / class_wise_total[i]) \
                                for i in range(0, len(self.labels))]

        print("Overall Accuracy:    {acc:.5f}".format(acc = overall_acc))

        for i in range(0, len(self.labels)):
            print("Classwise accuracy for \"{cl}\" class: {acc:.5f}"\
                    .format(cl = self.labels[i], acc = class_wise_accuracy[i]))

        if (save_overall): # Save overall accuracy to class
            self.overall_acc_stats[k] = overall_acc 
            self.overall_acc_by_k[k].append(overall_acc)

    def plot_overall_acc(self, plot_pp = False, pp = [0, 0]):
        '''
        Plots the overall accuracy for given k values
        Arguments:
        ----------
        plot_pp: bool, optional
            - Tells whether to include prior probability values in title or not
        pp: list of floats or dictionary with floats as values, optional
            - Contains the prior probabilities for each class
            - Preferably a dictionary with keyed on labels
            - Will be plotted on the output graph if plot_pp == True

        No explicit return, only shows the plot
        '''
        x = self.overall_acc_stats.keys()
        y = self.overall_acc_stats.values()

        title = "Classification Accuracy vs. k"
        if (plot_pp):
            title += ":: P(0) ={}, P(1) ={}".format(pp[0], pp[1])

        plt.bar(x, y)
        plt.xlabel("k value")
        plt.ylabel("Overall Classification Accuracy")
        plt.ylim(0.5, 1)
        plt.title(title)
        plt.show()

    def plot_overall_acc_w_varying(self, pp_varied):
        '''
        Used to plot the varying accuracies given the prior probabilities

        Arguments:
        ----------
        pp_varied: list
            - List of varying prior probability values used to train the model
        
        Returns:
        --------
        No explicit return, just displays the plot and prints the most accurate parameters
        '''

        max_acc = 0
        max_acc_i = 0
        max_acc_k = 0

        for key in self.overall_acc_by_k.keys():
            if (max(self.overall_acc_by_k[key]) > max_acc):
                max_acc = max(self.overall_acc_by_k[key])
                max_acc_i = self.overall_acc_by_k[key].index(max_acc)
                max_acc_k = key

            plt.plot(pp_varied, self.overall_acc_by_k[key], label = "k = {}".format(key))

        plt.title("Overall accuracy for PP vs. k")
        plt.xlabel("Prior Prob value for Class 0")
        plt.ylabel("Overall Classification Accuracy")
        plt.axvline(0.5, c='r', linestyle = '--', label = "Original Probability")
        plt.legend()
        plt.show()
        print("Highest overall = {}: k = {} at P(0) = {:.5f}".format(max_acc, int(max_acc_k), pp_varied[max_acc_i]))

    def plot_boundaries(self, k = 13, mesh_resolution = 0.03):
        '''
        Plots the decision boundary for kNN
            - Note: only for use in synth dataset
            - Hardcoded to work with only synth dataset

        Arguments:
        ----------
        k: int
            - k parameter to train data on
        mesh_resolution: float
            - Resolution to use when computing mesh values

        No explicit return
        '''
        area_colors  = colors.ListedColormap(["salmon", "lightskyblue"])

        labeled_lists = [ [], [] ] 

        # Plot the points with corresponding colors:
        for i in range(0, self.train.shape[0]):
            label = int(self.train.iloc[i, -1])
            labeled_lists[label].append(list(self.train.iloc[i, :]))

        # Get the labels for each of the training data points
        label0_x = [i[0] for i in labeled_lists[0]]
        label0_y = [i[1] for i in labeled_lists[0]]

        label1_x = [i[0] for i in labeled_lists[1]]
        label1_y = [i[1] for i in labeled_lists[1]]

        # Need to assign a color to each point in the mesh:
        min_x = self.train.iloc[:, 0].min() - 1
        max_x = self.train.iloc[:, 0].max() + 1

        min_y = self.train.iloc[:, 1].min() - 1
        max_y = self.train.iloc[:, 1].max() + 1

        xlist = np.arange(min_x, max_x, mesh_resolution)
        ylist = np.arange(min_y, max_y, mesh_resolution)

        # Create mesh
        xmesh, ymesh = np.meshgrid(xlist, ylist)

        # Need to get new predictions based on each point in the mesh:
        dict_data = {'x': xmesh.ravel(), 'y': ymesh.ravel(), "label": ([0] * len(xmesh.ravel()))}
        mesh_data = pd.DataFrame(dict_data)

        # Classify each of the points:
        self.classify(testing_data = mesh_data, k = k, show_statistics = False, \
                        df_provided = True, progress_bar = True)
        preds = np.array(self.predictions)

        # Shape our predictions to fit the mesh data points
        preds = preds.reshape(xmesh.shape)

        # Plot each of the points in the mesh (effectively decision boundary)
        plt.figure()
        plt.pcolormesh(xmesh, ymesh, preds, cmap = area_colors)

        # Plot the original training data points
        plt.scatter(label0_x, label0_y, c = "red")  
        plt.scatter(label1_x, label1_y, c = "blue")

        plt.xlabel("x")
        plt.ylabel('y')
        plt.title("kNN (k = {}) Decision Boundary on Synth Dataset".format(int(k)))
        
        plt.show()
        

def euclidean_dist(x_p, x_i):
    '''
    Computes euclidean distance given two vectors (x_p and x_i)
    '''
    # Get the list of norms for vector each other one
    x_p = np.array(x_p)
    x_i = np.array(x_i)
    return np.linalg.norm(x_p - x_i)
