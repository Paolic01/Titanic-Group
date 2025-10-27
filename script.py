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

