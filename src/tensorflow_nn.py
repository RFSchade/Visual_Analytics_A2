#=======================================================================#
#=============> Neural Network Classifier With Tensorflow <=============#
#=======================================================================#

#=====> Import modules 
# data tools
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function tools
import argparse
from tqdm import tqdm

# Image manipulation tools
import cv2

# sklearn tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# tf tools
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD

#=====> Define global variables 
# Number of epochs
EPOCHS = 150

#=====> Define functions
# > Load mnist data 
def load_mnist():
    # Print info 
    print("[info] Loading data...")
    # Load data 
    X, y = fetch_openml("mnist_784", return_X_y=True)
    # Get label names 
    label_names = sorted(y.unique())
    # Print info
    print("[info] Data loaded")
    
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

# > Normalize and binarize data
def prep_data(X_train, X_test, y_train, y_test):
    # Scaling the features
    X_train_scaled = X_train / 255
    X_test_scaled = X_test / 255
    # Binarize data 
    y_train_binarized = LabelBinarizer().fit_transform(y_train)
    y_test_binarized = LabelBinarizer().fit_transform(y_test)
    
    return (X_train_scaled, X_test_scaled, y_train_binarized, y_test_binarized)

# > Create model
def create_model(data_width):
    # Print info 
    print("[INFO] Initializing model")
    
    # define simple architecture 784x256x128x10
    model = Sequential()
    model.add(Dense(256, input_shape=(data_width,), activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(10, activation="softmax")) #softmax generalises LogReg for multiclass tasks
    
    # define the gradient descent
    sgd = SGD(0.01)
    # compile model
    model.compile(loss="categorical_crossentropy",
                  optimizer=sgd,
                  metrics=["accuracy"])
    
    # Print info
    print("[INFO] Model summary:")
    model.summary()
    
    return model

# > Evaluate model
def report(model, X_test, y_test, label_names, data):
    # Print info 
    print("[info] Reporting results...")
    # evaluate network
    predictions = model.predict(X_test, batch_size=32)
    # print classification report
    report = classification_report(y_test.argmax(axis=1), 
                                   predictions.argmax(axis=1), 
                                   target_names=label_names)
    # Print metrics
    print(report)
    # Save metrics
    outpath = os.path.join("output", f"tensorflow_report_{data}.txt")
    with open(outpath, "w") as f:
        f.write(report)
        
# > Plot history
def plot_history(H, epochs, data):
    plt.style.use("seaborn-colorblind")

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    # Saving image
    plt.savefig(os.path.join("output", f"history_img_{data}.png"))

# > Create argument
def parse_args(): 
    # Initialize argparse
    ap = argparse.ArgumentParser()
    # Commandline parameters 
    ap.add_argument("-d", "--data", required=True, help="Data to fit the model to - either 'cifar' of 'mnist'")
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
    # processing data
    X_train, X_test, y_train, y_test = prep_data(X_train, X_test, y_train, y_test)
    
    # Building model
    model = create_model(X_train.shape[1])
    # Train model with extra validation split
    history = model.fit(X_train, y_train,
                        validation_data = (X_test, y_test),
                        epochs = EPOCHS,
                        validation_split = 0.1,
                        batch_size = 32,
                        verbose = 1)
    # Evaluate model
    report(model, X_test, y_test, label_names, args["data"])
    # Plot history
    plot_history(history, EPOCHS, args["data"])
    
    # Print info 
    print("[INFO] Job complete")
    
# Run main() function from terminal only
if __name__ == "__main__":
    main()
