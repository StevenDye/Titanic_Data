import numpy as np
import pandas as pd
from sklearn import svm


# READ TRAINING FILE
df = pd.read_csv('input/train.csv')
original_train_df = pd.DataFrame.copy(df)# Makes a copy of the original before we turn it into numbers
df_features = ['Pclass', 'Sex', 'SibSp', 'Parch']


# PREPROCESSING
# Age
#df_men = df[df.Sex=='male']
#df_men['Age'].fillna(value=df_men['Age'].median(), inplace=True)
#
#df_women = df[df.Sex=='female']
#df_women['Age'].fillna(value=df_women['Age'].median(), inplace=True)
#
#df = pd.concat([df_men,df_women])

# Fare
#df['Fare'].fillna(value=df.Fare.mean(), inplace=True)

'''Even though the Age and Fare features has been preprocessed, Kaggle says the model is still
 more accurate without these features. Will look into imporoving the preprocessing.'''


X = df[df_features]
y = df['Survived']

# One Hot Encoding
X = pd.get_dummies(X, columns=['Sex'], drop_first=True)


# DEFINING MODEL AND FIT
titanic_model_svm = svm.SVC().fit(X,y)


# PREDICTIONS FROM TRAIN DATA
accuracy = titanic_model_svm.score(X, y)
print('SVM Train Accuracy: {}'.format(accuracy))


# READING TEST DATA
test = pd.read_csv('input/test.csv')


# PREPROCESSING TEST DATA
# Age
#test_men = test[test.Sex=='male']
#test_men['Age'].fillna(value=test_men['Age'].median(), inplace=True)
#
#test_women = test[test.Sex=='female']
#test_women['Age'].fillna(value=test_women['Age'].median(), inplace=True)
#
#test = pd.concat([test_men,test_women])

# Fare
#test_X['Fare'].fillna(value=test_X.Fare.mean(), inplace=True)

test_X = test[df_features]

# One Hot Encoding
test_X = pd.get_dummies(test_X, columns=['Sex'], drop_first=True)

# PREDICTIONS FROM TEST DATA
predicted_survival_svm = titanic_model_svm.predict(test_X)


# OUTPUT TO CSV
titanic_submission_svm = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predicted_survival_svm})
titanic_submission_svm.to_csv('output/titanic_submission_svm.csv', index=False)
