#=======================================================#
#=============> Neural Network Classifier <=============#
#=======================================================#

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
    
    # I don't really remember why I do this 
    # Is it so I can use a logistic regression as an activation function on categorical data with more than 1 categories? 
    y_train_binarized = LabelBinarizer().fit_transform(y_train)
    y_test_binarized = LabelBinarizer().fit_transform(y_test)
    
    return (X_train_scaled, X_test_scaled, y_train_binarized, y_test_binarized)

# > Train model
def train_model(X_train, y_train):
    # Print info
    print("[INFO] Training network...")
    # Defining shape of input
    input_shape = X_train.shape[1]
    # Initialyzing model
    nn = NeuralNetwork([input_shape, 64, 10]) 
    # Print info
    print(f"[INFO] {nn}")
    # Training model 
    nn.fit(X_train, y_train, epochs=150, displayUpdate=1) 
    
    return nn

# > Report
def report(nn, X_test, y_test):
    # Print info 
    print("[info] Reporting results...")
    # Predict classification of test data
    predictions = nn.predict(X_test)
    # Define prediction variable
    y_pred = predictions.argmax(axis=1) # I am not sure what the argmax does? 
    # Get metrics
    report = metrics.classification_report(y_test.argmax(axis=1), y_pred)
    # Print metrics
    print(report)
    # Save metrics
    outpath = os.path.join("output", "nn_report.txt")
    with open(outpath, "w") as f:
        f.write(report)

#=====> Define main()
def main():
    # Loading data
    X, y = load_data()
    # processing data
    X_train, X_test, y_train, y_test = prep_data(X, y)
    # Training model 
    nn = train_model(X_train, y_train)
    # Reporting data 
    report(nn, X_test, y_test)

# Run main() function from terminal only
if __name__ == "__main__":
    main()
