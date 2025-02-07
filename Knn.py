# %%
#%pip install pywavelets
#%pip install aeon
#%pip install tslearn
#%pip install scikit-learn
#%pip install Pandas
#%pip install Numpy
#%pip install Scipy
#%pip install Tqdm
#%pip install feature-engine

# %%
import pandas as pd
import numpy as np
import pywt

from aeon.datasets import load_classification
from aeon.datasets.tsc_datasets import univariate_equal_length
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from aeon.classification.base import BaseClassifier

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from tslearn.piecewise import PiecewiseAggregateApproximation, SymbolicAggregateApproximation

from scipy.fftpack import fft
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")
# %%
def load_data(dataset):
    le = LabelEncoder()

    # Carregar conjunto de dados do repositório UCR
    X_train, y_train = load_classification(dataset, split="TRAIN")
    X_test, y_test = load_classification(dataset, split="test")

    # Formatar o conjunto de dados para 2D
    features_train = X_train.reshape(X_train.shape[0], -1)
    features_test = X_test.reshape(X_test.shape[0], -1)

    # Ajustar e transformar as labels alvo
    target_train = le.fit_transform(y_train)
    target_test = le.transform(y_test)

    return features_train, features_test, target_train, target_test
# %%
#Fast Fourier Transform
def apply_fft(X_train, X_test):
    # apply FFT to training data
    X_train_fft = np.abs(fft(X_train))

    # apply FFT to test data
    X_test_fft = np.abs(fft(X_test))

    return X_train_fft, X_test_fft

#Discrete Wavelets Transform
def apply_dwt(X):
    coeffs_cA, coeffs_cD = pywt.dwt(X, wavelet='db1', axis=1, mode='constant')
    X_dwt = np.hstack((coeffs_cA, coeffs_cD))

    return X_dwt

#Piecewise Aggregation Approximation
def apply_paa(X):
    n_paa_segments = int(X.shape[1] / 4)
    paa = PiecewiseAggregateApproximation(n_segments=n_paa_segments)
    X_paa_ = paa.inverse_transform(paa.fit_transform(X))
    X_paa = X_paa_.reshape(X_paa_.shape[0], -1)
    df_PAA = pd.DataFrame(X_paa)

    return df_PAA


#Função de transformação das strings do SAX em Int
def transform_sax(arr):
    result = []
    for row in arr:
        new_row = []
        for letter in row:
            if letter.isalpha():
                value = ord(letter.lower()) - 96
            else:
                value = 0
            new_row.append(value)
        result.append(new_row)
    return np.array(result)

#Symbolic Aggregation Approximation
def apply_sax(X):
    n_sax_symbols = int(X.shape[1] / 4)
    n_paa_segments = int(X.shape[1] / 4)
    sax = SymbolicAggregateApproximation(n_segments=n_paa_segments, alphabet_size_avg=n_sax_symbols)
    X_sax_ = sax.inverse_transform(sax.fit_transform(X))
    X_sax = X_sax_.reshape(X_sax_.shape[0], -1)
    df_SAX = pd.DataFrame(X_sax)

    return df_SAX


# %%
#Treinamento dos classificadores
#TSC
UMnn = KNeighborsTimeSeriesClassifier(n_neighbors=1, distance='euclidean')
accuracy_data = []
for dataset_name in univariate_equal_length:
    features_train, features_test, target_train, target_test = load_data(dataset_name)
    UMnn.fit(features_train, target_train)
    y_pred_nn = UMnn.predict(features_test)
    accuracy = accuracy_score(target_test, y_pred_nn)
        
    accuracy_data.append({'Dataset Name': dataset_name, 'Accuracy': accuracy})
    
    print(f"Acurácia {dataset_name}: {accuracy}")
    
accuracy_df_extf = pd.DataFrame(accuracy_data)



# %%
#FFT
UMnn = KNeighborsTimeSeriesClassifier(n_neighbors=1, distance='euclidean')
accuracy_data = []
for dataset_name in univariate_equal_length:
    features_train, features_test, target_train, target_test = load_data(dataset_name)
    X_train_fft, X_test_fft = apply_fft(features_train, features_test)
    UMnn.fit(X_train_fft, target_train)
    y_pred_nn = UMnn.predict(X_test_fft)
    accuracy = accuracy_score(target_test, y_pred_nn)
        
    accuracy_data.append({'Dataset Name': dataset_name, 'Accuracy': accuracy})
    
    print(f"Acurácia {dataset_name}: {accuracy}")
    
accuracy_df_fft = pd.DataFrame(accuracy_data)

# %%
#DWT
UMnn = KNeighborsTimeSeriesClassifier(n_neighbors=1, distance='euclidean')
accuracy_data = []
for dataset_name in univariate_equal_length:
    features_train, features_test, target_train, target_test = load_data(dataset_name)
    X_train_dwt = apply_dwt(features_train)
    X_test_dwt = apply_dwt(features_test)
    UMnn.fit(X_train_dwt, target_train)
    y_pred_nn = UMnn.predict(X_test_dwt)
    accuracy = accuracy_score(target_test, y_pred_nn)
        
    accuracy_data.append({'Dataset Name': dataset_name, 'Accuracy': accuracy})
    
    print(f"Acurácia {dataset_name}: {accuracy}")
    
accuracy_df_dwt = pd.DataFrame(accuracy_data)

# %%
#PAA
UMnn = KNeighborsTimeSeriesClassifier(n_neighbors=1, distance='euclidean')
accuracy_data = []
for dataset_name in univariate_equal_length:
    features_train, features_test, target_train, target_test = load_data(dataset_name)
    X_train_paa = apply_paa(features_train)
    X_test_paa = apply_paa(features_test)
    UMnn.fit(X_train_paa, target_train)
    y_pred_nn = UMnn.predict(X_test_paa)
    accuracy = accuracy_score(target_test, y_pred_nn)
        
    accuracy_data.append({'Dataset Name': dataset_name, 'Accuracy': accuracy})
    
    print(f"Acurácia {dataset_name}: {accuracy}")
    
accuracy_df_pax = pd.DataFrame(accuracy_data)

# %%
#SAX
UMnn = KNeighborsTimeSeriesClassifier(n_neighbors=1, distance='euclidean')
accuracy_data = []
for dataset_name in univariate_equal_length:
    features_train, features_test, target_train, target_test = load_data(dataset_name)
    X_train_sax = apply_sax(features_train)
    X_test_sax = apply_sax(features_test)
    UMnn.fit(X_train_sax, target_train)
    y_pred_nn = UMnn.predict(X_test_sax)
    accuracy = accuracy_score(target_test, y_pred_nn)
        
    accuracy_data.append({'Dataset Name': dataset_name, 'Accuracy': accuracy})
    
    print(f"Acurácia {dataset_name}: {accuracy}")
    
accuracy_df_sax = pd.DataFrame(accuracy_data)
# %%
