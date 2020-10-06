import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from clustering_model import split_data_by_class
from parametric_model import MD_classifier

def project_data(w, output_data):
    '''
    Projects the output_data onto the subspace give by w

    Arguments:
    ----------
    w: np.array
        - Subspace on which to project the output data
    output_data: pandas DataFrame
        - Data which to project

    Returns:
    --------
    pd_projected: pandas dataframe
        - Projected data
    '''

    output_labels = output_data.iloc[:,-1]
    output_data_np = output_data.iloc[:, 0:-1].to_numpy()
    projected_np_data = np.transpose(np.matmul(w, np.transpose(output_data_np)))
    
    pd_projected = pd.DataFrame(data = projected_np_data)

    pd_projected["label"] = output_labels

    return pd_projected

def case3_acc_for_pca(train_data, test_data):
    '''
    Arguments:
    ----------
    train_data: pandas dataframe
        - Training data for Case 3 model
    test_data: pandas dataframe
        - Data on which to test the Case 3 model

    Returns:
    --------
    No explicit return
    '''

    num_features = train_data.shape[1] - 1

    overall_acc = []

    # Keeps different number of dimensions for every iteration:
    for i in range(1, num_features + 1):

        pX_train, pX_test, error_rate = pca(train_data, test_data, num_dim = i, return_both = True)        

        model = MD_classifier.min_dist_classifier(pX_train)
        acc = model.classify(test_data = pX_test, discriminant_type = "quadratic", \
                            prior_probs = [0.25, 0.75], show_statistics = True, test_is_pd = True)

        overall_acc.append(acc)
        print("Representation error rate with {} dimensions = {:.5f}".format(i, error_rate))

    plt.plot(range(1, num_features + 1), overall_acc)

    plt.title("Case 3 Accuracy with PCA")
    plt.ylabel("Total Accuracy")
    plt.xlabel("Number of Eigenvectors Kept")
    plt.show()

def pca(data, output_data, num_dim = -1, tolerance = 0.15, return_both = False):
    '''
    Arguments:
    ----------
    data: pandas DataFrame
        - Will be data that we reduce
        - Labels must be left in, will be returned in final dataframe
    output_data: pandas DataFrame
        - Data that will be projected with computed principal components
        - Equivalent to testing data
    num_dim: positive int
        - Default: -1
        - If not left to default, this specifies the number of dimensions to keep in pca
        - Must be <= number of dimensions in the data
    tolerance: float [0, 1]
        - Specifies the maximum error that we will allow in our reduction
    return_both: bool, optional
        - Default: False
        - If true, returns a tuple of projected data and projected output_data (in that order)

    Returns:
    --------
    trimmed: pandas DataFrame
        - Output data projected onto reduced dimensions
    Return value may be tuple of projected data and projected output_data if return_both = True
    '''

    # Store names of columns
    labels = data.iloc[:,-1]

    # Calculate covariance matrix
    data_as_np = data.iloc[:, 0:-1].to_numpy() # Convert to numpy array

    cov_mat = np.cov(np.transpose(data_as_np)) # Get covariance matrix

    # Calculate eigenvalues for the cov matrix
    evals, evecs = np.linalg.eig(cov_mat)

    # Note: evecs are stored in columns corresponding to each eval
    #   - Access eigenvector corresponding to evals[i] at evecs[:,i]

    # Sort eigenvalues from largest to smallest
    order_eigen = np.flip(np.argsort(evals))
    ordered_evals = [evals[i] for i in order_eigen]

    # Choose eigenvalues to keep
    if (num_dim > -1): # Prioritize based on num_dim
        to_keep = order_eigen[0:num_dim]
        
        # Calculate eigenvectors for each eigenvalue we keep
        keep_evecs = [evecs[:,i] for i in to_keep]

        # Get error rate - eigenvalues not kept as specified
        error_rate = np.sum( [ordered_evals[num_dim:]] ) / len(order_eigen)

    else: # Else, we need to keep based on tolerance
        error_rate = 0
        num_keep = 0

        # Iterate backwards on ordered eigenvalues
        for i in np.arange(len(order_eigen) - 1, -1, -1):
            error_rate += ordered_evals[i] / len(order_eigen)
            
            # If our error rate is above tolerance, adjust it and break
            if (error_rate >= tolerance):
                error_rate -= ordered_evals[i] / len(order_eigen)
                break

            num_keep += 1 # Need to increment number of eigenvectors we want to keep

        keep_evecs = [evecs[:,i] for i in order_eigen[0:num_keep]]

    # Ensure that eigenvectors are normalized
    #   Note: The numpy algorithm automatically normalizes the eigenvectors

    # Project data onto basis vectors

    w = np.array(keep_evecs)

    if (return_both):
        return (project_data(w, data), project_data(w, output_data), error_rate)
    else:
        return (project_data(w, output_data), error_rate)

def fld(data, output_data, return_both = False):
    '''
    Arguments:
    ----------
    data: pandas dataframe
        - Data upon which the discriminant is calculated
        - Equivalent to training data
    output_data: pandas dataframe
        - Data that will be projected with computed discriminant
        - Equivalent to testing data
    return_both: bool, optional
        - Default: False
        - If true, returns a tuple of projected data and projected output_data (in that order)

    Returns:
    -------
    pd_projected: pandas dataframe
        - This is output data projected onto discriminant from data
    Return value may be tuple of projected data and projected output_data if return_both = True
    '''

    split_data = split_data_by_class(data)

    split_data = {l:df.iloc[:,0:-1] for l, df in split_data.items()}

    # Calculate m dictionary
    m = {int(l):df.shape[0] for l, df in split_data.items()}

    # Calculate classwise means vector
    classwise_mean = {l:[] for l, i in m.items()}

    for l, df in split_data.items(): # Iterate over all classes in split data
        
        for i in range(0, df.shape[1]): # Go over all columns in class data (exclude class)
            # Append mean for that given column to that label's mean vector
            classwise_mean[l].append( np.mean(df.iloc[:,i]) )

    classwise_mean = {l: np.array(m) for l, m in classwise_mean.items()}

    labels = split_data.keys()

    # Calculate S_b and S_w
    # S_w is just equal to (1 - n) * (sum of covariance matrices for each class)
    S_w = np.zeros(shape = (data.shape[1] - 1, data.shape[1] - 1))

    for l in labels: # Add all of the covariance matrices (element-wise)
        S_w = np.add(S_w, np.cov( np.transpose( split_data[l].to_numpy() ) ))

    S_w = np.multiply(len(labels) - 1, S_w) # Multiply by (n - 1) to turn this into a scatter matrix

    # Get the projection vector:
    w = np.matmul(np.linalg.inv(S_w), np.subtract(classwise_mean[1], classwise_mean[0]))
    w = w.reshape(1, data.shape[1] - 1)

    # Project and return the data
    if (return_both):
        return (project_data(w, data), project_data(w, output_data))
    else:
        return (project_data(w, output_data))

def fld_and_pca_hists(fld_data, pca_data):
    ''' 
    Plots FLD and PCA histograms for reduction of pima dataset onto one dimension
        - Note: both datasets MUST be 1D
        - Shows the plot immediately

    Arguments:
    ----------
    fld_data: pandas dataframe
        - Data derived from FLD (will receive FLD labeling in histogram)
        - Data assumed to have labels
    pca_data: pandas dataframe
        - Data derived from PCA (will receive PCA labeling in histogram)
        - Assumed to have labels

    Returns:
    --------
    No explicit return value
    '''

    fld_split = split_data_by_class(fld_data)
    pca_split = split_data_by_class(pca_data)

    # Shows a 2x1 fld vs. pca histogram

    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Derives all of the statistics for each separate class in the datasets
    fld_yes = fld_split[0.0].iloc[:,0]
    fld_no  = fld_split[1.0].iloc[:,0]

    fld_yes_mean = np.mean(fld_yes)
    fld_yes_var  = np.var(fld_yes)

    fld_no_mean = np.mean(fld_no)
    fld_no_var  = np.var(fld_no)

    pca_yes = pca_split[0.0].iloc[:,0]
    pca_no  = pca_split[1.0].iloc[:,0]

    pca_yes_mean = np.mean(pca_yes)
    pca_yes_var  = np.var(pca_yes)

    pca_no_mean = np.mean(pca_no)
    pca_no_var  = np.var(pca_no)

    fig.suptitle("Dimensionality Reduction Distributions")

    ax1.hist(fld_yes, color = 'g', alpha = 0.4, label = "Yes")
    ax1.hist(fld_no, color = 'r', alpha = 0.4, label = "No")

    # Setting the aspects for each plot
    ax1.title.set_text("FLD")
    ax1.set_xlabel("Projected Value")
    ax1.set_ylabel("Frequency")
    ax1.axvline(fld_yes_mean, c = "green")
    ax1.axvline(fld_no_mean, c = "red")
    ax1.text(x = -3, y = 50, s = "Var(Yes) = {:.3f}".format(fld_yes_var))
    ax1.text(x = -3, y = 47, s = "Mean(Yes) = {:.3f}".format(fld_yes_mean))
    ax1.text(x = -3, y = 43, s = "Var(No) = {:.3f}".format(fld_no_var))
    ax1.text(x = -3, y = 40, s = "Mean(No) = {:.3f}".format(fld_no_mean))
    ax1.legend()

    # Set the plot of PCA
    ax2.hist(pca_yes, color = 'g', alpha = 0.4, label = "Yes")
    ax2.hist(pca_no, color = 'r', alpha = 0.4, label = "No")

    ax2.title.set_text("PCA")
    ax2.set_xlabel("Projected Value")
    ax2.set_ylabel("Frequency")
    ax2.axvline(pca_yes_mean, c = "green")
    ax2.axvline(pca_no_mean, c = "red")
    ax2.text(x = -4, y = 50, s = "Var(Yes) = {:.3f}".format(pca_yes_var))
    ax2.text(x = -4, y = 47, s = "Mean(Yes) = {:.3f}".format(pca_yes_mean))
    ax2.text(x = -4, y = 43, s = "Var(No) = {:.3f}".format(pca_no_var))
    ax2.text(x = -4, y = 40, s = "Mean(No) = {:.3f}".format(pca_no_mean))
    ax2.legend()

    plt.show()

    

    



