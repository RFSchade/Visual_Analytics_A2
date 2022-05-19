#============================================================#
#=============> Logistic Regression Classifier <=============#
#============================================================#

#=====> Import modules
# System tools
import os
import sys
import argparse
sys.path.append(os.getcwd())

# Data tools
import numpy as np
from tqdm import tqdm
import pandas as pd

# Cifar-10 data
from tensorflow.keras.datasets import cifar10

# Image manipulation tools
import cv2

# Import sklearn metrics
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer

#=====> Define Functions
# > Load mnist data 
def load_mnist():
    # Print info 
    print("[INFO] Loading data...")
    # Load data 
    X, y = fetch_openml("mnist_784", return_X_y=True)
    # Get label names 
    label_names = sorted(y.unique())
    # Print info
    print("[INFO] Data loaded")
    
    # Converting X and y to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Splitting data 
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=42,
                                                        test_size = 0.2)
    
    return (X_train, X_test, y_train, y_test, label_names)

# > Load cifar-10 data
def load_cifar():
    # Print info
    print("[INFO] loading data...")
    # Load data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # Initialize label names
    label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    # Print info
    print("[INFO] Data loaded")
    
    return (X_train, X_test, y_train, y_test, label_names)

# > Reshape data
def reshape_data(X):
    # Define empthy list 
    X_flat = []
    # For each image in the file...
    for image in tqdm(X):
        # Convert to grayscale
        gray = cv2.bitwise_not(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        # Flatten and append to list 
        X_flat.append(gray.flatten())
    # Convert to dataframe
    X_flat_df = pd.DataFrame(X_flat)
    
    return X_flat_df

# > Normalize data
def normalize(X_train, X_test):
    # Scaling the features
    X_train_scaled = X_train / 255
    X_test_scaled = X_test / 255
    
    return (X_train, X_test)

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
                             multi_class="multinomial").fit(X_train, y_train.ravel()) 
    
    return clf
    
# > Report
def report(clf, X_test, y_test, label_names, data):
    # Print info 
    print("[info] Reporting results...")
    # Predict classification of test data
    y_pred = clf.predict(X_test)
    # Get metrics
    report = metrics.classification_report(y_test, 
                                           y_pred,
                                           target_names=label_names)
    # Print metrics
    print(report)
    # Save metrics
    outpath = os.path.join("output", f"lr_report_{data}.txt")
    with open(outpath, "w") as f:
        f.write(report)

# > Create argument
def parse_args(): 
    # Initialize argparse
    ap = argparse.ArgumentParser()
    # Commandline parameters 
    ap.add_argument("-d", "--data", required=True, help="Data to fit the model to - either 'cifar' or 'mnist'")
    # Parse argument
    args = vars(ap.parse_args())
    # return list of argumnets 
    return args        
        
#=====> Define main()
def main():
    # Get argument
    args = parse_args()
    # fork depending on argument 
    if args["data"] == "cifar":
        # Load data
        X_train, X_test, y_train, y_test, label_names = load_cifar()
        # Print info
        print("[INFO] Reshaping data...")
        # Reshape data 
        X_train = reshape_data(X_train)
        X_test = reshape_data(X_test)
    else:
        # Load data
        X_train, X_test, y_train, y_test, label_names = load_mnist()
        
    # normalize data
    X_train, X_test = normalize(X_train, X_test)
    # Training model 
    clf = train_model(X_train, y_train)
    # Reporting data 
    report(clf, X_test, y_test, label_names, args["data"])
    
    # Print info 
    print("[INFO] Job complete")

# Run main() function from terminal only
if __name__ == "__main__":
    main()
