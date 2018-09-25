import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score


# READ TRAINING FILE
df = pd.read_csv('input/train.csv')


# PREPROCESSING
df['Fare'].fillna(value=df.Fare.median(), inplace=True)
df['Age'].fillna(value=df.Age.median(), inplace=True)
''' Fare and Age should contirbute greatly to the accuracy of the model, but for some reason makes
	it worse. For now they have been droped from the features'''
df_features = ['Pclass', 'Sex', 'SibSp', 'Parch']
X = df[df_features]
X = pd.get_dummies(X, columns=['Sex'], drop_first=True)
y = df['Survived']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)


# DEFINING MODEL AND FIT
titanic_model_svm = svm.SVC().fit(X,y)
scores_svm = cross_val_score(titanic_model_svm, X_train, y_train, scoring='neg_mean_absolute_error')
print(scores_svm)

# PREDICTIONS FROM TRAIN DATA
accuracy = titanic_model_svm.score(X_train, y_train)
print('SVM Accuracy: {}'.format(accuracy))


# PREDICTIONS FROM TEST DATA
test = pd.read_csv('input/test.csv')
test_X = test[df_features]
test_X = pd.get_dummies(test_X, columns=['Sex'], drop_first=True)
#test_X['Age'].fillna(value=test_X.Age.median(), inplace=True)
#test_X['Fare'].fillna(value=test_X.Fare.median(), inplace=True)

predicted_survival_svm = titanic_model_svm.predict(test_X)


# OUTPUT TO CSV
titanic_submission_svm = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predicted_survival_svm})
titanic_submission_svm.to_csv('output/titanic_submission_svm.csv', index=False)
