#%%
import pandas as pd
import numpy as np
from aeon.datasets import load_classification
from aeon.datasets.tsc_datasets import univariate_equal_length
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import warnings
warnings.filterwarnings("ignore")


# %%

def load_data(dataset):
    le = LabelEncoder()
    
    X_train, y_train = load_classification(dataset, split="TRAIN")
    X_test, y_test = load_classification(dataset, split="TEST")

    features_train = X_train.reshape(X_train.shape[0], -1)
    features_test = X_test.reshape(X_test.shape[0], -1)

    target_train = le.fit_transform(y_train)
    target_test = le.transform(y_test)

    #distribuição das classes
    class_counts_train = Counter(target_train)
    class_counts_test = Counter(target_test)

    def check_balance(class_counts):
        values = np.array(list(class_counts.values()))
        imbalance_ratio = values.max() / values.min() if values.min() > 0 else np.inf
        return imbalance_ratio <= 1.2 

    is_balanced_train = check_balance(class_counts_train)
    is_balanced_test = check_balance(class_counts_test)

    return features_train, features_test, target_train, target_test, is_balanced_train, is_balanced_test

accuracy_data = []

for dataset_name in univariate_equal_length:
    features_train, features_test, target_train, target_test, is_balanced_train, is_balanced_test = load_data(dataset_name)
        
    accuracy_data.append({
        'Dataset': dataset_name,
        'Balanced_Train': is_balanced_train,
        'Balanced_Test': is_balanced_test
    })

# Criar DataFrame
df_accuracy = pd.DataFrame(accuracy_data)

pdf_filename = "./dataset_balanceamento.pdf"

with PdfPages(pdf_filename) as pdf:
    for dataset_name in df_accuracy['Dataset']:
        _, _, target_train, target_test, _, _ = load_data(dataset_name)

        train_counts = Counter(target_train)
        test_counts = Counter(target_test)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Gráfico das classes treino
        axes[0].bar(train_counts.keys(), train_counts.values(), color='blue')
        axes[0].set_title(f"Distribuição dos dados de Treino - {dataset_name}")
        axes[0].set_xlabel("Classes")
        axes[0].set_ylabel("Quantidade")
        
        # Gráfico das classes no teste
        axes[1].bar(test_counts.keys(), test_counts.values(), color='red')
        axes[1].set_title(f"Distribuição dos dados de Teste - {dataset_name}")
        axes[1].set_xlabel("Classes")
        axes[1].set_ylabel("Quantidade")

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

pdf_filename

# %%
