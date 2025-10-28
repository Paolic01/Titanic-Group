import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report


train_set = pd.read_csv("/Users/ludovicamontanaro/progetto_titanic /Titanic-Group/dati/train.csv")
print(train_set.head())
test_set = pd.read_csv("/Users/ludovicamontanaro/progetto_titanic /Titanic-Group/dati/test.csv")
print(test_set.head())

## Check for missing values in the training set 
missing_train = train_set.isnull().sum()
print("Missing values in training set:\n", missing_train)

## Check for missing values in the test set
missing_test = test_set.isnull().sum()
print("Missing values in test set:\n", missing_test)

## Summary statistics del training set
summary_train = train_set.describe()
print("Summary statistics of training set:\n", summary_train)

## Summary statistics del test set
summary_test = test_set.describe()
print("Summary statistics of test set:\n", summary_test)

## Data types of each column in the training set
data_types_train = train_set.dtypes
print("Data types in training set:\n", data_types_train)

## Which are the missing values in the training set?
plt.figure(figsize=(10,5))
sns.heatmap(train_set.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values nel Training Set")
plt.show()

train_set.isnull().sum().sort_values(ascending=False)
test_set.isnull().sum().sort_values(ascending=False)

# Percentage of missing values in the training set
missing_values = train_set.isnull().sum() 
missing_percentage = (missing_values / len(train_set)) * 100
missing_data = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage}) 
print(missing_data)

# Percentage of missing values in the testing set
missing_values = test_set.isnull().sum() 
missing_percentage = (missing_values / len(test_set)) * 100
missing_data = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage}) 
print(missing_data)

#.              PREPROCESSING

# MISSING VALUES

# Replacing missing age and fare values ​​with the median
train_set['Age'].fillna(train_set['Age'].median(), inplace=True)
test_set['Age'].fillna(test_set['Age'].median(), inplace=True)

train_set['Fare'].fillna(train_set['Fare'].median(), inplace=True)
test_set['Fare'].fillna(test_set['Fare'].median(), inplace=True)

# FEATURE ENGINEERING 

# Creation of the new feature: 'Has_Cabin'
# .notna() restituisce True se non è NaN, False se è NaN.
# .astype(int) converte True -> 1 e False -> 0.
train_set['Has_Cabin'] = train_set['Cabin'].notna().astype(int)
test_set['Has_Cabin'] = test_set['Cabin'].notna().astype(int)

# Now we can delete the oringinal column 'Cabin'
train_set = train_set.drop('Cabin', axis=1)
test_set = test_set.drop('Cabin', axis=1)

# Creation of the new colomn 'FamilySize'
# Passenger 1: 1 (SibSp) + 0 (Parch) + 1 (se stesso) = 2
# Passenger 2: 0 (SibSp) + 0 (Parch) + 1 (se stesso) = 1 (viaggiava solo)
# Passenger 3: 2 (SibSp) + 1 (Parch) + 1 (se stesso) = 4
train_set['FamilySize'] = train_set['SibSp'] + train_set['Parch'] + 1
test_set['FamilySize'] = test_set['SibSp'] + test_set['Parch'] + 1

# Mapping 'male' to 0 and 'female' to 1
train_set['Sex'] = train_set['Sex'].map({'male': 0, 'female': 1})
test_set['Sex'] = test_set['Sex'].map({'male': 0, 'female': 1})

# Trasforming categorical variables into binaries
train_set = pd.get_dummies(train_set, columns=['Embarked'], drop_first=True)
test_set = pd.get_dummies(test_set, columns=['Embarked'], drop_first=True)


# CORRELATION MATRIX
# We are looking for numerical columns
corr_features = ['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'Has_Cabin']
corr_matrix = train_set[corr_features].corr()
print(corr_matrix.round(2))

# Graphic visualization with heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matrice di Correlazione - Titanic Dataset')
plt.show()

# Definition of the input variables and target for the training dataset
X = train_set.drop(columns=['Survived', 'PassengerId', 'Name', 'Ticket'])
y = train_set['Survived']

# In order to achieve a better precision we divided the training set into training and validation set
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# TRAINING OF THE LOGISTIC REGRESSION MODEL 
model = LogisticRegression(max_iter=1000)  # max_iter per garantire la convergenza
model.fit(X_train, y_train)
y_pred = model.predict(X_valid)

print("Accuratezza:", accuracy_score(y_valid, y_pred))
print(classification_report(y_valid, y_pred))

# Scatter plot Age vs. Survival 
sns.scatterplot(data=train_set, x='Age', y='Survived', hue='Sex', alpha=0.6)
plt.title('Sopravvivenza in base all\'Età e al Sesso')
plt.xlabel('Età')
plt.ylabel('Sopravvivenza')
plt.show()

# Confusion matrix
conf_matrix = confusion_matrix(y_valid, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non Sopravvissuti', 'Sopravvissuti'],
            yticklabels=['Non Sopravvissuti', 'Sopravvissuti'])
plt.title('Matrice di Confusione')
plt.xlabel('Predizioni')
plt.ylabel('Reali')
plt.show()


# ROC Curve
y_pred_prob = model.predict_proba(X_valid)[:, 1]
fpr, tpr, _ = roc_curve(y_valid, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='AUC = {:.2f}'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'k--')  # Linea di riferimento
plt.title('Curva ROC')
plt.xlabel('Tasso di Falsi Positivi')
plt.ylabel('Tasso di Veri Positivi')
plt.legend(loc='best')
plt.show()

# PREDICTION ON THE TEST SET
# Eliminstion of the unnecessary columns 
test_data_ready = test_set.drop(columns=['PassengerId', 'Name', 'Ticket'])

# Allineation of the columns of the train set and the test data set
test_data_ready = test_data_ready.reindex(columns=X_train.columns, fill_value=0)

# Are there other missing values?
print(test_data_ready.isnull().sum())

# Creation of an imputer that fills eventually missing values with the mean
imputer = SimpleImputer(strategy='mean')

# Assuming test_data_ready contains only numerical columns
test_data_ready = imputer.fit_transform(test_data_ready)

# Doing predictions
predictions = model.predict(test_data_ready)

# Adding predictions to the testing set
test_set['Survived'] = predictions
print(test_set[ 'Survived'])

# Visualizza la distribuzione delle previsioni di sopravvivenza
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
#plt.figure(figsize=(8,5))
#sns.barplot(data=coefficients, x='Importanza', y='Feature', palette='viridis')
#plt.title('Importanza delle Feature nel Modello di Regressione Logistica')
#plt.xlabel('Peso Assoluto del Coefficiente')
#plt.ylabel('Feature')
#plt.show()



# TRAINING THE MODEL USING RANDOM FOREST 
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
plt.show()

# Confusion matrix
conf_matrix = confusion_matrix(y_valid, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non Sopravvissuti', 'Sopravvissuti'],
            yticklabels=['Non Sopravvissuti', 'Sopravvissuti'])
plt.title('Matrice di Confusione')
plt.xlabel('Predizioni')
plt.ylabel('Reali')
plt.show()


# ROC Curve
y_pred_prob = rf_model.predict_proba(X_valid)[:, 1]
fpr, tpr, _ = roc_curve(y_valid, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='AUC = {:.2f}'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'k--')  # Linea di riferimento
plt.title('Curva ROC')
plt.xlabel('Tasso di Falsi Positivi')
plt.ylabel('Tasso di Veri Positivi')
plt.legend(loc='best')
plt.show()


# TRAINING OF THE MODEL WITH THE XGBoost
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

# Confusion matrix
conf_matrix = confusion_matrix(y_valid, y_pred_xgb)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non Sopravvissuti', 'Sopravvissuti'],
            yticklabels=['Non Sopravvissuti', 'Sopravvissuti'])
plt.title('Matrice di Confusione')
plt.xlabel('Predizioni')
plt.ylabel('Reali')
plt.show()


# ROC Curve
y_pred_prob = xgb_model.predict_proba(X_valid)[:, 1]
fpr, tpr, _ = roc_curve(y_valid, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='AUC = {:.2f}'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'k--')  # Linea di riferimento
plt.title('Curva ROC')
plt.xlabel('Tasso di Falsi Positivi')
plt.ylabel('Tasso di Veri Positivi')
plt.legend(loc='best')
plt.show()




