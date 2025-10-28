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



# Definisci le variabili di input e il target per il dataset di addestramento.
X = train_set.drop(columns=['Survived', 'PassengerId', 'Name', 'Ticket'])
y = train_set['Survived']

#Per una maggiore precisione divido ulteriormente il set di addestramento in sottoinsieeme di addestramento e di validazione
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


#Addestramento del modello di regressione logistica
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

model = LogisticRegression(max_iter=1000)  # max_iter per garantire la convergenza
model.fit(X_train, y_train)

y_pred = model.predict(X_valid)

#print("Accuratezza:", accuracy_score(y_valid, y_pred))
#print(classification_report(y_valid, y_pred))


# Un esempio con 'Age' e 'Survived'
sns.scatterplot(data=train_set, x='Age', y='Survived', hue='Sex', alpha=0.6)
plt.title('Sopravvivenza in base all\'Età e al Sesso')
plt.xlabel('Età')
plt.ylabel('Sopravvivenza')
#plt.show()
# Calcola la matrice di confusione
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_valid, y_pred)

# Visualizza la matrice di confusione
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non Sopravvissuti', 'Sopravvissuti'],
            yticklabels=['Non Sopravvissuti', 'Sopravvissuti'])
plt.title('Matrice di Confusione')
plt.xlabel('Predizioni')
plt.ylabel('Reali')
plt.show()


#CURVA DI ROC
from sklearn.metrics import roc_curve, auc

# Calcola le probabilità di previsione
y_pred_prob = model.predict_proba(X_valid)[:, 1]

# Calcola i dati per la curva ROC
fpr, tpr, _ = roc_curve(y_valid, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Traccia la curva ROC
plt.plot(fpr, tpr, label='AUC = {:.2f}'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'k--')  # Linea di riferimento
plt.title('Curva ROC')
plt.xlabel('Tasso di Falsi Positivi')
plt.ylabel('Tasso di Veri Positivi')
plt.legend(loc='best')
plt.show()


from sklearn.impute import SimpleImputer

train_set['Fare'].fillna(train_set['Fare'].median(), inplace=True)
test_set['Fare'].fillna(test_set['Fare'].median(), inplace=True)

# Rimuovere colonne non necessarie
test_data_ready = test_set.drop(columns=['PassengerId', 'Name', 'Ticket'])

# Allinea le colonne
test_data_ready = test_data_ready.reindex(columns=X_train.columns, fill_value=0)

print(test_data_ready.isnull().sum())

from sklearn.impute import SimpleImputer

# Create an imputer that fills missing values with the mean
imputer = SimpleImputer(strategy='mean')

# Assuming test_data_ready contains only numerical columns
test_data_ready = imputer.fit_transform(test_data_ready)

# Fare previsioni
predictions = model.predict(test_data_ready)

# Aggiungere le previsioni al dataset di test
test_set['Survived'] = predictions
print(test_set[ 'Survived'])


# Visualizza la distribuzione delle previsioni di sopravvivenza
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
sns.countplot(data=test_set, x='Survived', palette='coolwarm')
plt.title('Distribuzione delle Previsioni di Sopravvivenza')
plt.xlabel('Sopravvivenza (0 = No, 1 = Sì)')
plt.ylabel('Numero di Passeggeri')
plt.show()


# Visualizza la distribuzione delle previsioni di sopravvivenza in base al sesso
plt.figure(figsize=(7,5))
sns.countplot(data=test_set, x='Sex', hue='Survived', palette='coolwarm')
plt.title('Previsioni di Sopravvivenza per Sesso')
plt.xlabel('Sesso (0 = Uomo, 1 = Donna)')
plt.ylabel('Conteggio')
plt.legend(title='Sopravvivenza', labels=['Non Sopravvissuto', 'Sopravvissuto'])
plt.show()

# Visualizza la distribuzione delle previsioni di sopravvivenza in base alla classe di viaggio
plt.figure(figsize=(7,5))
sns.countplot(data=test_set, x='Pclass', hue='Survived', palette='coolwarm')
plt.title('Previsioni di Sopravvivenza per Classe di Viaggio')
plt.xlabel('Classe (1 = Prima, 3 = Terza)')
plt.ylabel('Conteggio')
plt.legend(title='Sopravvivenza', labels=['Non Sopravvissuto', 'Sopravvissuto'])
plt.show()




# Visualizza l'importanza delle feature
'''plt.figure(figsize=(8,5))
sns.barplot(data=coefficients, x='Importanza', y='Feature', palette='viridis')
plt.title('Importanza delle Feature nel Modello di Regressione Logistica')
plt.xlabel('Peso Assoluto del Coefficiente')
plt.ylabel('Feature')
plt.show()'''



# prova con il RANDOM FOREST

'''from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

rf_model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=6, 
    random_state=42
)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_valid)

print("Accuratezza Random Forest:", accuracy_score(y_valid, y_pred_rf))
print(classification_report(y_valid, y_pred_rf))


# Visualizza l'importanza delle feature per Random Forest
importances = rf_model.feature_importances_
feature_names = X_train.columns
coefficients = pd.DataFrame({'Feature': feature_names, 'Importanza': importances})
coefficients = coefficients.sort_values(by='Importanza', ascending=False)
plt.figure(figsize=(8,5))
sns.barplot(data=coefficients, x='Importanza', y='Feature', palette='viridis')
plt.title('Importanza delle Feature nel Modello Random Forest')
plt.xlabel('Importanza')
plt.ylabel('Feature')
plt.show()'''



# Prova con XGBoost
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_valid)

print("Accuratezza XGBoost:", accuracy_score(y_valid, y_pred_xgb))
print(classification_report(y_valid, y_pred_xgb))



# Visualizza l'importanza delle feature per XGBoost 
importances = xgb_model.feature_importances_
feature_names = X_train.columns
coefficients = pd.DataFrame({'Feature': feature_names, 'Importanza': importances})
coefficients = coefficients.sort_values(by='Importanza', ascending=False)
plt.figure(figsize=(8,5))
sns.barplot(data=coefficients, x='Importanza', y='Feature', palette='viridis')
plt.title('Importanza delle Feature nel Modello XGBoost')
plt.xlabel('Importanza')
plt.ylabel('Feature')
plt.show()




