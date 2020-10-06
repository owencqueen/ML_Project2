import pandas as pd
from MD_classifier import min_dist_classifier as classifier

# Run the model on synth dataset:
# ------------------------------

# Run the model on pima dataset:
# ------------------------------
pima_tr = pd.read_csv("pima_data/pima_train.csv")

model = classifier(pima_tr)

# We will run part B of Question 1 (equal prior probabilities):
model.classify( test_data = 'pima_data/pima_test.csv', discriminant_type = "euclidean", prior_probs = [0.25, 0.75])
model.classify( test_data = 'pima_data/pima_test.csv', discriminant_type = "mahalanobis", prior_probs = [0.25, 0.75])
model.classify( test_data = 'pima_data/pima_test.csv', discriminant_type = "quadratic", prior_probs = [0.25, 0.75])

