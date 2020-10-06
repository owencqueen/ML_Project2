import time
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#from scatterplot_data import scatterplot_for_synth as plot_points
from parametric_model.discriminant_fns import euclidean, mahalanobis, quadratic
from parametric_model.discriminant_fns import euclidean_decision_bd, mahalanobis_decision_bd, quadratic_decision_bd
#from ec2_fit_distributions import multimodal_Gaussian_functions

class min_dist_classifier:

    def __init__(self, training_df):
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
        self.train = training_df

        self.num_features = training_df.shape[1] - 1 # Must subtract 1 to account for labels

        # Need to calculate mean and store vector:
        labels = training_df.iloc[:,-1]  # Get last row (labels)

        self.labels_unique = set(labels) # Convert to a set - will isolate unique values
        self.labels_unique = list(self.labels_unique) 
            # Convert back to a list so we can iterate

        self.labels_unique = [int(i) for i in self.labels_unique]

        self.class_data = [ [] for i in range(0, len(self.labels_unique))] # Create lists of empty lists

        self.class_0_acc_e = [] # Will hold class_0_accuracy values for euclidean
        self.class_0_acc_m = [] # Will hold class_0_accuracy values for mahalanobis
        self.class_0_acc_q = [] # Will hold class_0_accuracy values for quadratic

        # Need to create list of matrices that store values based on labels
        for i in range(0, self.train.shape[0]): 
            current_label = int(self.train.iloc[i,-1])            # Get current label
            label_ind = self.labels_unique.index(current_label)   # Get index of label in class_data
            self.class_data[label_ind].append( self.train.iloc[i, 0:-1] ) # Append row to class_data
        
        # Convert to np array:
        self.class_data = [np.array(i) for i in self.class_data]

        # Calculate mu vectors:
        self.mu = []

        # Iterate over class_data:
        for label in range(0, len(self.labels_unique)):

            # Get matrix of values for current class
            current_matrix = self.class_data[label]

            feature_wise_sums = [0] * self.num_features # Will add to this as we iterate

            # Iterate over the matrix and calculate sum for each feature
            for row in range(0, len(current_matrix)):
                for col in range(0, len(current_matrix[row])):
                    feature_wise_sums[col] += current_matrix[row][col]

            # Get averages for each of the classes
            feature_wise_sums = [(i / len(current_matrix)) for i in feature_wise_sums]

            self.mu.append(np.array(feature_wise_sums)) # Append feature_wise avg's for this class to list of mu_i lists

    def calc_variance(self):
        '''
        Calculates the variance for the class
        Arguments:
        ----------
        No arguments, just calculates based off data stored after __init__ is called
        '''
        return np.var(self.train.iloc[:,0])

    def calc_cov(self, label_ind):
        '''
        Calculates the covariance matrix for the data with a given label

        Arguments:
        ----------
        label_ind: int
            - Index of label as it's stored in the class

        Returns:
        --------
        covariance matrix: <number of labels> x <number of labels> np array
            - Covariance matrix of the data with that given label
            - Calculted using numpy's covariance matrix function
        '''
        return np.cov(np.transpose(self.class_data[label_ind]))

    def classify(self, test_data, discriminant_type = "euclidean", prior_probs = [0, 0],
                show_statistics = True, plot_predictions = False, test_is_pd = False, 
								a_vals = [0.5, 0.5]):
        '''
        Arguments:
        ----------
        test_data: string
            - String refering to dataframe where the testing data is located
        discriminant_type: string
            - Options:
                1. "euclidean"
                2. "mahalanobis"
                3. "quadratic"
        prior_probs: list of floats
            - List corresponding to the prior probability to use for each class
                in the classification problem
            - Must be same length as number of classes
        show_statistics: bool
            - If true, prints the statistics for the classification at the end of the routine
        plot_predictions: bool
            - If true, plots each point in the testing sample (color coded for predicted class) 
                along with the decision boundary made by the discriminant function.
        a_vals: list of floats
            - List of a parameters to use in calculation of multimodel Gaussian distributions
            - Will only be used if discriminant_type == "bimodal"
            - See ec2_fit_distribution.py and report for more explanation

        Returns:
        --------
        No explicit return value
        Stores the predictions into the class (self.predictions)
        '''

        start_time = time.time()

        if (test_is_pd):
          test = test_data
        else:
          test = pd.read_csv(test_data) # Read in our test data

        self.predictions = [] # Will store predictions for each class based on row

        if (discriminant_type == "euclidean"):
            # Need to calculate variance:
            #   Case 1 Assumption: Variance for all classes is equal
            #   Therefore, only calculate for one class
            var = self.calc_variance()

        elif (discriminant_type == "mahalanobis"):
            # Case 2 assumption: Covariance matrix equal for all classes
            cov_mat_0 = self.calc_cov(label_ind = 0) # Get covariance matrix for first class 
                # Choice  is arbitrary b/c of our assumption
            
            # Need to invert for efficiency
            cov_mat_0_inv = np.linalg.inv(cov_mat_0)

        elif (discriminant_type == "quadratic"):

            cov_mat_list = []

            # Case 3 assumption: Covariance matrix different for all classes
            for i in range(0, len(self.labels_unique)):
                cov_mat_i = self.calc_cov(label_ind = i)
                cov_mat_list.append(cov_mat_i)

        #elif (discriminant_type == "bimodal"):
        #    pdf_class0,pdf_class1 = multimodal_Gaussian_functions(a_vals[0], a_vals[1])

        # Note: don't need to calculate anything initially for quadratic: all calcs done during iteration

        # Iterate over dataframe:
        for i in range(0, test.shape[0]):

            x = test.iloc[i,0:-1]

            if (discriminant_type != "bimodal"):

                val_per_class = []

                for i in range(0, len(self.labels_unique)):

                    # Calculate discriminant based on user specification:
                    if (discriminant_type == "euclidean"):
                        curr_val = euclidean(x, self.mu[i], var, prior_probs[i])
                    
                    elif (discriminant_type == "mahalanobis"):
                        curr_val = mahalanobis(x, self.mu[i], cov_mat_0_inv, prior_probs[i])

                    elif (discriminant_type == "quadratic"):
                        curr_val = quadratic(x, self.mu[i], cov_mat_list[i], prior_probs[i])

                    val_per_class.append(curr_val)

                highest_p = max(val_per_class) # Get highest probability

                prediction = self.labels_unique[val_per_class.index(highest_p)] # Set prediction based on probability

            #elif (discriminant_type == "bimodal"):

                #if prob_eq:
                    # This is if prior probabilities are equal
                #    val0 = pdf_class0(x)
                #    val1 = pdf_class1(x)
                #else:
                    # This bit is hardcoded to only work with 
                #    val0 = pdf_class0(x) * prior_probs[0]
                #    val1 = pdf_class1(x) * prior_probs[1]

                # Choose prediction with highest posterior probability
                #prediction = 0 if (val0 > val1) else 1

            self.predictions.append(prediction) # Adds prediction


        if (show_statistics):
            overall_acc = self.accuracy_stats(discriminant_type, test) # Prints statistics for classification algorithm
            print("{} Runtime: {} seconds".format(discriminant_type, time.time() - start_time))
            print("") # Need newline

        if (plot_predictions): # Plots our decision boundary vs. predictions for individual points
            predicted_data = []
            for i in range(0, test.shape[0]):
                predicted_data.append([test.iloc[i, 0], 
                                       test.iloc[i, 1],
                                       self.predictions[i]])

            df = pd.DataFrame(predicted_data, columns = ['x', 'y', 'label'])

        if (show_statistics):
            return overall_acc


    def accuracy_stats(self, dis_type, test_data):
        ''' 
        Prints the overall accuracy and classwise accuracy after a prediction procedure has finished
        Arguments:
        ----------
        dis_type: string
            - Specifies the type of discriminant function being used for the classfication
            - Options: "euclidean", "mahalanobis", and "quadratic"
        test_data: pandas dataframe
            - Dataframe containing the test data

        Returns:
        --------
        overall_acc: float
            - Overall accuracy of the given classifer on our testing dataset
        '''
        # Need:
        #   1. overall classification accuracy
        #   2. classwise accuracy - all classes
        #   3. run time - printed in classify function

        # Overall classification accuracy:
        true_labels = list(test_data.iloc[:, -1])

        overall_correct = 0
        class_wise_correct = [0] * len(self.labels_unique)
        class_wise_total = [0] * len(self.labels_unique)
        # For classwise accuracy, numbers will be stored in 

        for i in range(0, len(true_labels)):

            predict = self.predictions[i]

            if (true_labels[i] == predict):
                overall_correct += 1 # Add to correct count if they match

                # Add to class-wise accuracy
                class_wise_correct[self.labels_unique.index(predict)] += 1

            #class_wise_total[self.labels_unique.index(predict)] += 1
            class_wise_total[self.labels_unique.index(true_labels[i])] += 1

        # If no testing samples are predicted to be class A, then the "class_wise_correct"
        #   value for A will be zero. Thus, we set the class_wise_total to be 1 to avoid
        #   division-by-zero errors.
        for i in range(len(class_wise_total)):
            if (class_wise_total[i] == 0):
                class_wise_total[i] = 1

        overall_acc = overall_correct / len(true_labels)
        class_wise_accuracy = [(class_wise_correct[i] / class_wise_total[i]) \
                                for i in range(0, len(self.labels_unique))]

        # Insert accuracy values in to various class_0 accuracy lists
        if (dis_type == "euclidean"):
            self.class_0_acc_e.append(class_wise_accuracy[0])

        if (dis_type == "mahalanobis"):
            self.class_0_acc_m.append(class_wise_accuracy[0])

        if (dis_type == "quadratic"):
            self.class_0_acc_q.append(class_wise_accuracy[0])

        print("Overall Accuracy:    {acc:.5f}".format(acc = overall_acc))

        for i in range(0, len(self.labels_unique)):
            print("Classwise accuracy for \"{cl}\" class: {acc:.5f}"\
                    .format(cl = self.labels_unique[i], acc = class_wise_accuracy[i]))

        return overall_acc

    def plot_class_0_acc(self, prior_prob_vals):
        '''
        Plot class 0 accuracy for given different prior probabilities

        Arguments:
        ----------
        prior_prob_vals: list
            - List corresponding to prior probabilities that were used to generate class 0
                accuracies currently stored in the class

        Returns:
        --------
        No explicit return, but it does display the plot generated
        '''

        plt.plot(prior_prob_vals, self.class_0_acc_e, 'r', label = "E Fn")
        plt.plot(prior_prob_vals, self.class_0_acc_m, 'g', label = "M Fn")
        plt.plot(prior_prob_vals, self.class_0_acc_q, 'y', label = "Q Fn")

        plt.xlabel("Prior Probability of Class 0")
        #plt.ylabel("(# correct predictions for class 0) / (# predicted class 0)")
        plt.ylabel("(# correct predictions for class 0) / (# true class 0)")
        plt.title("Class 0 Accuracy vs. Prior Probability")
        
        plt.legend()
        plt.show()



