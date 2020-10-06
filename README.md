# ML_Project2
Unsupervised Learning and Dimensionality Reduction
COSC 522, Fall 2020 
Owen Queen

**Report**: The report for this project is in oqueen_proj2.pdf.

All visuals are in the visuals folder.

## Running the Code
If you want to run the code used to normalize the datasets, run the script parse_data.py. 

The rest of the code to run all of the plots and tables shown on the report are in the Jupyter Notebook file "run_project2.ipynb". 

### Very Important Note before Running the Code
Both kMeans and Winner-Takes-All are based on randomly choosing initial cluster centers. Because of this, there is a probability that when running these models, the clustering procedure will result in very low accuracy, with the majority of samples being assigned to one cluster or the other. Please note that the tables I have included in the report are runs of the models when this did not occur. There is a possibility that this could occur while running the code for this project, so be aware that one run of the model may not necessarily align with the results shown on the report.

## Documentation
All the documentation for the functions has been done in the .py files themselves in the form of docstrings. Check these for more rigorous information about how to use the functions.
