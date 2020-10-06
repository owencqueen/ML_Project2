# ML_Project2
Unsupervised Learning and Dimensionality Reduction
COSC 522, Fall 2020 
Owen Queen

**Report**: The report for this project is in [oqueen_proj2.pdf](https://github.com/owencqueen/ML_Project2/blob/main/oqueen_proj2.pdf).

All visuals are in the visuals folder.

## Running the Code
If you want to run the code used to normalize the datasets, run the script parse_data.py. 

The rest of the code to run all of the plots and tables shown on the report are in the Jupyter Notebook file [run_project2.ipynb](https://github.com/owencqueen/ML_Project2/blob/main/run_project2.ipynb). 

### Very Important Note before Running the Code
Both kMeans and Winner-Takes-All are based on randomly choosing initial cluster centers. Because of this, there is a probability that when running these models, the clustering procedure will result in very low accuracy, with the majority of samples being assigned to one cluster or the other. Please note that the tables I have included in the report are runs of the models when this did not occur. There is a possibility that this could occur while running the code for this project, so be aware that one run of the model may not necessarily align with the results shown on the report.

## Documentation
All the documentation for the functions has been done in the .py files themselves in the form of docstrings. Check these for more rigorous information about how to use the functions.

## General Structure/ Guide to Files:

1. bonus.py: Contains code for solving Bonus problem.
2. cluster_image.py: Contains code used to cluster the image.
3. clustering_model.py: Contains both kMeans and WTA clustering models.
4. dim_reduce.py: Contains all dimensionality reduction code.
5. flowersm.ppm: Image that we compress
6. kNN.py: Code for kNN (from Project 1).
7. oqueen_proj2.pdf: report
8. parametric_model: Directory that contains parametric models used in Cases 1 - 3.
9. parse_data.py: Script used to parse the .txt files to .csv files.
10. pima_data: Directory containing all Pima datasets.
11. run_project2.ipynb: Jupyter Notebook for running all of the code.
12. visuals: Folder of images included in the report.

