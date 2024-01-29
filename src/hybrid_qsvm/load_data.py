import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import random

seed = 123

def load(datakey, gamma):
    if "MNIST" in datakey:
        c1 = int(datakey[-2])
        c2 = int(datakey[-1])
        print(f"loading dataset MNIST with classes '{c1}' and '{c2}'")
        X_train,X_test, Y_train, Y_test = load_MNIST(n=250, seed=seed, test_size=.4, n_comp=gamma, class0=c1, class1=c2)
    else:
        raise NotImplemented()
        print("loading dataset HEART")
        X_train,X_test, Y_train, Y_test = load_HEART(n=100, seed=123, test_size=.5, n_comp=3)
    return X_train,X_test, Y_train, Y_test  # data, labels


def load_MNIST(n=100, seed=123, test_size=.1, class0=4, class1=7, n_comp = 2):
    
    train_size = 1-test_size
    from tensorflow.keras.datasets import mnist
    (x_raw, y_raw), _ = mnist.load_data()
    ix = np.where((y_raw == class0) | (y_raw == class1)) 
    x_raw, y_raw =x_raw[ix], y_raw[ix]
    np.random.seed(seed)

    mask = np.hstack([np.random.choice(np.where(y_raw == l)[0], n, replace=False)
                          for l in np.unique(y_raw)])
    random.seed(seed)

    random.shuffle(mask)
    x_raw, y_raw =x_raw[mask], y_raw[mask]

    # Random splitting of dataset in training and test
    num_data = len(y_raw)
    num_train = int(train_size * num_data)
    np.random.seed(seed)

    index = np.random.permutation(range(num_data))

    # Training set
    X_train = x_raw[index[:num_train]]
    Y_train = y_raw[index[:num_train]]

    # Test set
    X_test = x_raw[index[num_train:]]
    Y_test = y_raw[index[num_train:]]

    ncol = x_raw.shape[1]*x_raw.shape[2]
    x_flat = X_train.reshape(-1,ncol)

    # Rename the columns
    feat_cols = ['pixel'+str(i) for i in range(x_flat.shape[1])]

    # construction of the pandas dataframe
    df_flat = pd.DataFrame(x_flat,columns=feat_cols)
    df_flat['Y'] = Y_train

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # Two principal components are considered
    pca = PCA(n_components=n_comp) 

    # Application of the pca to the dataset
    principalComponents = pca.fit_transform(x_flat) 
    
    total_var = 100 * np.sum(pca.explained_variance_ratio_)
    print(f'{total_var:.3}% of total variance is explained by 2 principal components')

    # We create a new dataset where the images are represented by a two-dimensional vector
    # that corresponds to the first two principal components
    # cols=['Component '+str(i+1) for i in range(principalComponents.shape[1])]
    # data_pca = pd.DataFrame(data = principalComponents, 
    #                         columns = cols)

    # Append the target variable to the new dataset
    # data_pca['Y'] = df_flat.iloc[:,-1:].to_numpy()


    # Extract the new feature as numpy array
    x_pca = principalComponents # data_pca[cols].to_numpy()

    MAX=np.max(x_pca)
    MIN=np.min(x_pca)

    # Rescaleing of the values of the features
    X_train = (x_pca-MIN)/(MAX-MIN)
    Y_train = (Y_train-class0)/(class1-class0)

    # We define our training set, that will be the input of our QML model
    # X_train = X.copy()
    # Y_train = 

    X_test = pca.transform(X_test.reshape(-1,ncol))
    X_test = (X_test-MIN)/(MAX-MIN)
    
    Y_test = (Y_test-class0)/(class1-class0)
    return X_train*np.pi/2, X_test*np.pi/2, Y_train*2-1 , Y_test*2-1

