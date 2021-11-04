### In this project we will write a series of solvers for the following linear classifiers: 

1. Least squares
2. Regularized least squares
3. Hinge loss
4. Regularized hinge loss
5. Logistic regression
6. Adaptive learning rate for hinge


The input to your program is the data file and the train labels given in the datasets
posted course on the website. For example to run your project on the breast cancer
data with trainlabels.0 you would type

python project1.py traindata testdata

where traindata and testdata are in the same format as the
datasets *.train.0 and *.test.0 given on the course website. For
example see https://web.njit.edu/~usman/courses/cs675_fall20/bc.train.0
and https://web.njit.edu/~usman/courses/cs675_fall20/bc.test.0

The output of your project are just the predicted labels of each
datapoint in the test file. Each predicted label is either +1 or -1.
For each classifier you will output a separate file:

1. Least squares output is "least_squares_prediction"
2. Regularized least squares is "reg_least_squares_prediction"
3. Hinge loss is "hinge_prediction"
4. Regularized hinge loss is "regularized_hinge_prediction"
5. Logistic regression is "logistic_prediction"
6. Adaptive learning rate for hinge "adaptive_eta_hinge_prediction"

Use eta=0.001 and stop condition of .001. For the regularized versions
keep lambda=0.01.

To predict the test datapoints we use 0 as the cutoff for least squares and
hinge and 0.5 for logistic.