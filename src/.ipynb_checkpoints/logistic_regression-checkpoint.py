#============================================================#
#=============> Logistic Regression Classifier <=============#
#============================================================#

#=====> Import modules
# System tools
import os
import sys
sys.path.append(os.getcwd())

# Import teaching utils
import numpy as np
import utils.classifier_utils as clf_util

# Neural network 
from utils.neuralnetwork import NeuralNetwork

# Import sklearn metrics
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer

#=====> Define Functions
# > Load data 
def load_data():
    # Print info 
    print("[info] Loading data...")
    # Load data 
    X, y = fetch_openml("mnist_784", return_X_y=True)
    # Print info
    print("[info] Data loaded")
    
    return (X,y)

# > Prepare data
def prep_data(X, y):
    # Print info 
    print("[info] Processing data...")
    # Converting X and y to numpy arrays
    X = np.array(X)
    y = np.array(y)
    # Splitting data 
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=42,
                                                        train_size = 7500,
                                                        test_size = 2500)
    # Scaling the features
    X_train_scaled = X_train / 255
    X_test_scaled = X_test / 255
    
    return (X_train_scaled, X_test_scaled, y_train, y_test)

# > Train model
def train_model(X_train, y_train):
    # Print info
    print("[info] Training model...")
    # Initialyzing model
    clf = LogisticRegression(multi_class="multinomial")
    # Training model 
    clf = LogisticRegression(penalty="none",
                             tol=0.1,
                             solver="saga",
                             multi_class="multinomial").fit(X_train, y_train)
    return clf
    
# > Report
def report(clf, X_test, y_test):
    # Print info 
    print("[info] Reporting results...")
    # Predict classification of test data
    y_pred = clf.predict(X_test)
    # Get metrics
    report = metrics.classification_report(y_test, y_pred)
    # Print metrics
    print(report)
    # Save metrics
    outpath = os.path.join("output", "lr_report.txt")
    with open(outpath, "w") as f:
        f.write(report)

#=====> Define main()
def main():
    # Loading data
    X, y = load_data()
    # processing data
    X_train, X_test, y_train, y_test = prep_data(X, y)
    # Training model 
    clf = train_model(X_train, y_train)
    # Reporting data 
    report(clf, X_test, y_test)

# Run main() function from terminal only
if __name__ == "__main__":
    main()
