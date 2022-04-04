#=======================================================#
#=============> Neural Network Classifier <=============#
#=======================================================#

#=====> Import modules
# System tools
import os

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