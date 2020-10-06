# Reads all of the data and performs normalization
import pandas as pd
import numpy as np

def read_data_for_pima(file_name, export_fname):
    '''
    Reads in the training data and returns mu and std

    Arguments:
    ----------
    file_name: string
        - name of file with original testing data
    export_fname: string
        - Name of file to export normalized data to

    Returns:
    --------
    (mu, std): tuple of lists of floats
        - mu is mean of data, std is standard deviation
    '''


    pima_df = pd.DataFrame(columns = ['npreg', 'glu', 'bp', 'skin', 'bmi', 'ped', 'age', 'type'])
    with open(file_name) as f:
        for l in f:
            line = l.split()
            
            # Change from "Yes" and "No" to 0 and 1, respectively

            data_line = {'npreg': int(line[0]), 
                         'glu': int(line[1]), 
                         'bp': int(line[2]),
                         'skin': int(line[3]),
                         'bmi': float(line[4]),
                         'ped': float(line[5]),
                         'age': int(line[6]),
                         'type': 0 if (line[7] == 'Yes') else 1}
            
            pima_df = pima_df.append(data_line, ignore_index = True, sort = False)

    #Compute mean, variance:
    mu, std = mu_std_by_class(pima_df)

    # Need to normalize all of the columns (features)
    for i in range(0, len(pima_df.columns) - 1):

        column_i = pima_df.iloc[:, i] # Get ith col

        # Compute standard deviation of the ith column
        std_i = std[i]
        
        # Compute mean of the ith column
        mu_i = mu[i]

        # Normalize and set the columns
        pima_df.iloc[:, i] = [((x - mu_i) / std_i) for x in pima_df.iloc[:, i]]
    
    pima_df.to_csv("pima_data/" + export_fname, index = False)
    return (mu, std)

def process_testing_data(file_name, export_fname, mu, std):
    '''
    Processes the testing data given mu and std vectors

    Arguments:
    ----------
    file_name: string
        - name of file with original testing data
    export_fname: string
        - Name of file to export normalized data to
    mu: list
        - Vector of mu to normalize data on
    std: list
        - Vector of std to normalize data on

    Returns:
    --------
    No return value
    '''

    pima_df = pd.DataFrame(columns = ['npreg', 'glu', 'bp', 'skin', 'bmi', 'ped', 'age', 'type'])
    with open(file_name) as f:
        for l in f:
            line = l.split()

            data_line = {'npreg': int(line[0]), 
                         'glu': int(line[1]), 
                         'bp': int(line[2]),
                         'skin': int(line[3]),
                         'bmi': float(line[4]),
                         'ped': float(line[5]),
                         'age': int(line[6]),
                         'type': 0 if (line[7] == 'Yes') else 1}
            
            pima_df = pima_df.append(data_line, ignore_index = True, sort = False)
    
    # Need to normalize all of the columns (features)
    for i in range(0, len(pima_df.columns) - 1):

        column_i = pima_df.iloc[:, i] # Get ith col

        # Compute standard deviation of the ith column
        std_i = std[i]
        
        # Compute mean of the ith column
        mu_i = mu[i]

        # Normalize and set the columns
        pima_df.iloc[:, i] = [((x - mu_i) / std_i) for x in pima_df.iloc[:, i]]

    pima_df.to_csv("pima_data/" + export_fname, index = False)

def mu_std_by_class(pima_df):
    '''
    Returns mean and std vecs for given data

    Arguments:
    ----------
    pima_df: pandas dataframe
        - Data on which to compute a mean and standard deviation

    Returns:
    --------
    (mu, std): tuple of lists of floats
        - mu is mean of data, std is standard deviation
    '''

    mu = []
    std = []

    # Need to normalize all of the columns (features)
    for i in range(0, pima_df.shape[1] - 1):

        column_i = pima_df.iloc[:, i] # Get ith col

        # Compute standard deviation of the ith column
        std.append(np.std(column_i))
        
        # Compute mean of the ith column
        #mu_i = np.mean(column_i)
        mu.append(np.mean(column_i))

    return (mu, std)

if __name__ == "__main__":
    # Runs the reading of data for the Pima Datasets
    m, s = read_data_for_pima(file_name = "pima_data/pima.tr.txt", export_fname = "pima_train.csv")

    process_testing_data(file_name = "pima_data/pima.te.txt", export_fname = "pima_test.csv", \
        mu = m, std = s)
