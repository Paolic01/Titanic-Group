import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
train_set = pd.read_csv("/Users/pasqualepaolicelli/Downloads/Titanic-Group/dati/train.csv")
print(train_set.head())
test_set = pd.read_csv("/Users/pasqualepaolicelli/Downloads/Titanic-Group/dati/test.csv")
print(test_set.head())

## Check for missing values in the training set
'''missing_train = train_set.isnull().sum()
print("Missing values in training set:\n", missing_train)

## Check for missing values in the test set
missing_test = test_set.isnull().sum()
print("Missing values in test set:\n", missing_test)'''   

## Summary statistics of the training set
summary_train = train_set.describe()
print("Summary statistics of training set:\n", summary_train)

## Summary statistics of the test set
summary_test = test_set.describe()
print("Summary statistics of test set:\n", summary_test)

## Data types of each column in the training set
data_types_train = train_set.dtypes
print("Data types in training set:\n", data_types_train)


'''plt.figure(figsize=(10,5))
sns.heatmap(train_set.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values nel Training Set")
plt.show()

train_set.isnull().sum().sort_values(ascending=False)
test_set.isnull().sum().sort_values(ascending=False)'''

# Calcola i valori mancati sul train set
missing_values = train_set.isnull().sum() 
missing_percentage = (missing_values / len(train_set)) * 100

# Crea un DataFrame per visualizzare i risultati 

missing_data = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage}) 
print(missing_data)






# Calcola i valori mancati sul test set
missing_values = test_set.isnull().sum() 
missing_percentage = (missing_values / len(test_set)) * 100

# Crea un DataFrame per visualizzare i risultati 

missing_data = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage}) 
print(missing_data)

# PRIMA FASE: GESTIONE DEI MISSING VALUES PER LE VARIABILI NUMERICHE
# Imputazione della variabile Age con la mediana
train_set['Age'].fillna(train_set['Age'].median(), inplace=True)
test_set['Age'].fillna(test_set['Age'].median(), inplace=True)


# --- Ecco il codice che ti serve ---
# .notna() restituisce True se non è NaN, False se è NaN.
# .astype(int) converte True -> 1 e False -> 0.
train_set['Has_Cabin'] = train_set['Cabin'].notna().astype(int)

print("\n--- DOPO Has_Cabin ---")
print(train_set)

# Ora puoi eliminare la colonna 'Cabin' originale
train_set = train_set.drop('Cabin', axis=1)

print("\n--- FINALE dopo aver tolto cabin ---")
print(train_set)

##test set
test_set['Has_Cabin'] = test_set['Cabin'].notna().astype(int)

print("\n--- DOPO Has_Cabin ---")
print(test_set)

# Ora puoi eliminare la colonna 'Cabin' originale
test_set = test_set.drop('Cabin', axis=1)

print("\n--- FINALE ---")
print(test_set)

#SECONDA FASE: FEATURE ENGINEERING 

# Crea la nuova colonna 'FamilySize'
train_set['FamilySize'] = train_set['SibSp'] + train_set['Parch'] + 1

print("\n--- DOPO AVER CREATO FamilySize ---")
print(train_set)

test_set['FamilySize'] = test_set['SibSp'] + test_set['Parch'] + 1

print("\n--- DOPO AVER CREATO FamilySize ---")
print(test_set)

# Passenger 1: 1 (SibSp) + 0 (Parch) + 1 (se stesso) = 2
# Passenger 2: 0 (SibSp) + 0 (Parch) + 1 (se stesso) = 1 (viaggiava solo)
# Passenger 3: 2 (SibSp) + 1 (Parch) + 1 (se stesso) = 4

# Mappa 'male' a 0 e 'female' a 1
train_set['Sex'] = train_set['Sex'].map({'male': 0, 'female': 1})
test_set['Sex'] = test_set['Sex'].map({'male': 0, 'female': 1})

print("\n--- DOPO AVER MAPPATO Sex ---")
print(train_set)
print(test_set)

