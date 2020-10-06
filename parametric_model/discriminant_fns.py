import math
import numpy as np

def euclidean(x, mu_i, variance, prior_prob):
    '''
    Calculates the euclidean discriminant function for given input
    Arguments:
    ----------
    x : np.array (1D)
        - Data point to be predicted
    mu_i: np.array (1 x n size)
        - Mean vector for given features
        - No need to take transpose in this function *******
    variance: float
        - Variance for given data
    prior_prob: float
        - Prior probability for this class
    pp_equal: bool
        - True if prior probabilities are equal
        - Therefore, they are not included in calculation if equal
    '''
    additive = np.log(prior_prob)
        
    a = np.dot(mu_i, x) / variance
    b = np.dot(mu_i, mu_i) / (2 * variance)

    return (a - b + additive)


def mahalanobis(x, mu_i, cov_matrix_inv, prior_prob):
    '''
    Computes the mahalanobis distance metric for a given sample
    '''

    additive = np.log(prior_prob)

    cov_T = np.transpose(cov_matrix_inv)

    a = np.matmul(mu_i, cov_T)
    a = np.matmul(a, x)

    b = np.matmul(mu_i, cov_matrix_inv)
    b = np.matmul(b, np.transpose(mu_i))

    return (a - ((0.5) * b) + additive)

def quadratic(x, mu_i, cov_mat_i, prior_prob):

    additive = np.log(prior_prob)

    if (len(mu_i) == 1): # Calculate quadratic for case of data being 1 dimension
        a = float(- (1/2) * np.log(cov_mat_i))
        b = float(- ((x - mu_i[0]) ** 2) / (2 * cov_mat_i))
        c = 0
        d = 0

    else:
        # mu_i is one dimensional
        mu_i = np.array(mu_i)

        det_cov = np.linalg.det(cov_mat_i)

        cov_inv = np.linalg.inv(cov_mat_i)

        xT = np.transpose(x)

        # a, b, c, d correspond to components being added in end equation
        #   Note: every mu_i corresponds to (mu_i)^T in original equation
        #   Note: ''    x    ''          '' (x)^T    ''                ''
        a = (-0.5) * np.matmul(x, cov_inv)
        a = np.matmul(a, xT)

        b = np.matmul(mu_i, np.transpose(cov_inv))
        b = np.matmul(b, xT)

        c = (-0.5) * np.matmul(mu_i, cov_inv)
        c = np.matmul(c, np.transpose(mu_i))

        d = (-0.5) * np.log(det_cov)

    return ( a + b + c + d + additive)

def euclidean_decision_bd(mu, variance):
    '''
    Returns a function that gives a function of x value to determine decision boundary
    Return function assumes two dimensional data
    '''
    multiplier = -(mu[0][0] - mu[1][0]) / (mu[0][1] - mu[1][1])

    #C = (np.dot(mu[0], mu[0]) - np.dot(mu[1], mu[1])) / (2 * (variance ** 2))

    C = (np.dot(mu[0], mu[0]) - np.dot(mu[1], mu[1])) / (2 * (mu[0][1] - mu[1][1]))

    def decision_bd(x):
        return x * multiplier + C

    return decision_bd # Returns defined function

def mahalanobis_decision_bd(mu, cov_inv):
    '''
    Returns a function that gives a function of x value to determine decision boundary
    Return function assumes two dimensional data

    Arguments:
    ----------
    cov_inv: inverse of covariance matrix
    '''

    cov_inv_T = np.transpose(cov_inv)

    # First calculate constant
    a = np.matmul(mu[1], cov_inv)
    a = np.matmul(a, np.transpose(mu[1]))
    a = (a / 2)

    b = np.matmul(mu[0], cov_inv)
    b = np.matmul(b, np.transpose(mu[0]))
    b = (b / 2)

    # Now calculate multiplier
    alpha_vec = np.matmul(mu[0], cov_inv_T)
    beta_vec  = np.matmul(mu[1], cov_inv_T)

    multiplier = (alpha_vec[0] - beta_vec[0]) / (beta_vec[1] - alpha_vec[1])

    C = (a - b) / (beta_vec[1] - alpha_vec[1])

    def decision_bd(x):
        return x * multiplier + C

    return decision_bd

def quadratic_decision_bd(mu, cov_mat_list):

    # Compute the inverse of our matrices:
    cov_mat_inv_list = []
    for cov_mat in cov_mat_list:
        cov_mat_inv_list.append(np.linalg.inv(cov_mat))

    # Need our phi vector (2 x 1):
    phi_vec = np.matmul(mu[0], np.transpose(cov_mat_inv_list[0])) - \
                np.matmul(mu[1], np.transpose(cov_mat_inv_list[1]))

    # Need our gamma matrix:
    gamma_vec = cov_mat_inv_list[1] - cov_mat_inv_list[0]
    gamma_vec *= 0.5

    # First, calculate our constant:

    a_c = np.matmul(mu[1], cov_mat_inv_list[1])
    a_c = np.matmul(a_c, np.transpose(mu[1]))

    b_c = np.matmul(mu[0], cov_mat_inv_list[0])
    b_c = np.matmul(b_c, np.transpose(mu[0]))

    c_c = np.log( np.linalg.det(cov_mat_list[1]) )
    d_c = np.log( np.linalg.det(cov_mat_list[0]) )

    C = (0.5) * (a_c - b_c + c_c - d_c)


    def quad_bd(x):
        # Calculate the c value (in ax^2 + bx + c form)
        #c = (gamma_vec[0][0] + gamma_vec[1][0]) * (x ** 2) + phi_vec[0] * x + C
        c = (gamma_vec[0][0]) * (x ** 2) + phi_vec[0] * x + C

        # Then, calculate our b value (in ax^2 + bx + c form)
        #b = ((gamma_vec[0][0] + gamma_vec[0][1] + gamma_vec[1][0] + gamma_vec[1][1]) * x + phi_vec[1])
        b = ((gamma_vec[0][1] + gamma_vec[1][0]) * x + phi_vec[1])

        # Finally, our a value
        #a = (gamma_vec[0][1] + gamma_vec[1][1])
        a = gamma_vec[1][1]

        # Now we need to find roots of our equations:
        coeff_vec = [a, b, c]

        roots = np.roots(coeff_vec) # Get roots of the polynomial

        #print(roots)

        ret_roots = []

        # Need to remove the root if either one is complex (i.e. root doesn't exist in real numbers)
        for i in range(0, len(roots)):
            if (not (np.iscomplex(roots[i]))):
                ret_roots.append(roots[i])

        if (len(ret_roots) == 0):
            return []
        else:
            return [min(ret_roots)]

        #print("Trimmed:", ret_roots)

        # Return possibly modified list:
        #return [min_roots]
    
    return quad_bd
