import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
train_set = pd.read_csv("/Users/pasqualepaolicelli/Downloads/Titanic-Group/dati/train.csv")
##print(train_set.head())
test_set = pd.read_csv("/Users/pasqualepaolicelli/Downloads/Titanic-Group/dati/test.csv")
##print(test_set.head())

## Check for missing values in the training set
missing_train = train_set.isnull().sum()
##print("Missing values in training set:\n", missing_train)

## Check for missing values in the test set
missing_test = test_set.isnull().sum()
##print("Missing values in test set:\n", missing_test)

## Summary statistics of the training set
summary_train = train_set.describe()
##print("Summary statistics of training set:\n", summary_train)

## Summary statistics of the test set
summary_test = test_set.describe()
##print("Summary statistics of test set:\n", summary_test)

## Data types of each column in the training set
data_types_train = train_set.dtypes
##print("Data types in training set:\n", data_types_train)


##plt.figure(figsize=(10,5))
sns.heatmap(train_set.isnull(), cbar=False, cmap='viridis')
##plt.title("Missing Values nel Training Set")
##plt.show()

train_set.isnull().sum().sort_values(ascending=False)
test_set.isnull().sum().sort_values(ascending=False)

# Calcola i valori mancati sul train set
missing_values = train_set.isnull().sum() 
missing_percentage = (missing_values / len(train_set)) * 100

# Crea un DataFrame per visualizzare i risultati 

missing_data = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage}) 
##print(missing_data)






# Calcola i valori mancati sul test set
missing_values = test_set.isnull().sum() 
missing_percentage = (missing_values / len(test_set)) * 100

# Crea un DataFrame per visualizzare i risultati 

missing_data = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage}) 
##print(missing_data)

# PRIMA FASE: GESTIONE DEI MISSING VALUES PER LE VARIABILI NUMERICHE
# Imputazione della variabile Age con la mediana
train_set['Age'].fillna(train_set['Age'].median(), inplace=True)
test_set['Age'].fillna(test_set['Age'].median(), inplace=True)


# --- Ecco il codice che ti serve ---
# .notna() restituisce True se non è NaN, False se è NaN.
# .astype(int) converte True -> 1 e False -> 0.
train_set['Has_Cabin'] = train_set['Cabin'].notna().astype(int)

##print("\n--- DOPO Has_Cabin ---")
##print(train_set)

# Ora puoi eliminare la colonna 'Cabin' originale
train_set = train_set.drop('Cabin', axis=1)

##print("\n--- FINALE dopo aver tolto cabin ---")
##print(train_set)

##test set
test_set['Has_Cabin'] = test_set['Cabin'].notna().astype(int)

##print("\n--- DOPO Has_Cabin ---")
##print(test_set)

# Ora puoi eliminare la colonna 'Cabin' originale
test_set = test_set.drop('Cabin', axis=1)

##print("\n--- FINALE ---")
##print(test_set)

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


#####################

# Seleziona solo le colonne numeriche
corr_features = ['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'Has_Cabin']
corr_matrix = train_set[corr_features].corr()

# Visualizza la matrice numerica
print(corr_matrix.round(2))

# Visualizza graficamente con heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matrice di Correlazione - Titanic Dataset')
plt.show()

####################


train_set = pd.get_dummies(train_set, columns=['Embarked'], drop_first=True)
test_set = pd.get_dummies(test_set, columns=['Embarked'], drop_first=True)

print("\n--- DOPO One-Hot Encoding di Embarked ---")
print(train_set.head())
print("\n--- DOPO One-Hot Encoding di Embarked ---")
print(test_set.head())



# 🔹 Predizioni sul test set
test_features = test_set.drop(columns=['PassengerId', 'Name', 'Ticket'])

# Assicurati che le colonne nel test_set siano le stesse di X_train
test_data_ready = test_features.reindex(columns=X_train.columns, fill_value=0)

test_pred = model.predict(test_features)   # test_features = X_test preprocessato
test_prob = model.predict_proba(test_features)[:,1]


from sklearn.impute import SimpleImputer
import pandas as pd

# Copia di sicurezza
X_train = X_train.copy()
X_valid = X_valid.copy()

# Imputer per colonne numeriche (Age, Fare, ecc.)
num_imputer = SimpleImputer(strategy='median')

# Imputer per colonne categoriche (Embarked, ecc.)
cat_imputer = SimpleImputer(strategy='most_frequent')

# Identifica colonne numeriche e categoriche
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X_train.select_dtypes(include=['object']).columns

# Applica gli imputer
X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
X_valid[num_cols] = num_imputer.transform(X_valid[num_cols])

if len(cat_cols) > 0:
    X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
    X_valid[cat_cols] = cat_imputer.transform(X_valid[cat_cols])


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_valid[num_cols] = scaler.transform(X_valid[num_cols])


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

y_pred = model.predict(X_valid)
print("Accuratezza:", accuracy_score(y_valid, y_pred))
print(classification_report(y_valid, y_pred))


importances = rf_model.feature_importances_
feature_names = X_train.columns
coefficients = pd.DataFrame({'Feature': feature_names, 'Importanza': importances})
coefficients = coefficients.sort_values(by='Importanza', ascending=False)