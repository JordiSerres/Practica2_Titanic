# -*- coding: utf-8 -*-
"""
PRÀCTICA 2 - Cicle i tipologia de vida de les dades
KAGGLE - Titanic: Machine Learning from Disaster
Author = Jordi Serres
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import cross_val_score
from sklearn import tree
from io import StringIO


# Llegir train dataset
#path = "C:/Users/Jordi/MU Data Science/Tipologia i cicle de vida/Practica 2/Kaggle_Titanic/"
#path = "G:/MU Data Science/Tipologia i cicle de vida/PRAC2/"
path = "E:/Kaggle_Titanic/"
filename = "train.csv"
df = pd.read_csv(path+filename)

# Visualització de les dades
df.describe()
df[df.columns].isnull().sum()
# hi ha un 38.38% de supervivents. Hi ha 177 valors nuls al camp "Age", 687 al camp "Cabin" i 2 a "Embarked"
df_age_nulls = df[df['Age'].isnull()]
df_age_nulls.describe() # Descripció registres amb edat nula. hi ha un 29.38% de supervivents
df['Age'].describe()
max_age = df['Age'].max()
df['Age'].hist(bins=int(max_age))
# Hi ha molta diferència en la probabilitat de sobreviure entre homes i dones  
print(pd.crosstab(df['Sex'], df['Survived'], rownames=['Sex'], colnames=['Survived']))


# Preparació de dades per tenir els camps numèrics 
df['Sex'].replace({'male': 0, 'female': 1}, inplace=True)
# Transformem 'Embarked' en n vairables 'Dummy'
df['Embarked'].astype('category')
df = pd.get_dummies(df, columns=['Embarked'])
# Creem una variable nova que indica si el passatger estava en cabina o no 
df['With Cabin'] = df['Cabin'].isnull()
df['With Cabin'] = df['With Cabin'] * 1
# Substituïm els valors nuls de l'edat per la mitjana
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Partició del dataset de train en conjunt de train (75%) i test (25%)
ratio = 0.75
df['is_train'] = np.random.uniform(0, 1, len(df)) <= ratio
train, test = df[df['is_train']==True], df[df['is_train']==False]

# Models d'arbre de decisió i Random Forest
clf_0 = DecisionTreeClassifier(max_depth=4)
clf_1 = RandomForestClassifier(n_jobs=1, random_state=0)
clf_2 = RandomForestClassifier(n_jobs=10, random_state=0)

# provarem el model amb 2 conjunts de variables
features_1 = ['Pclass','Sex', 'Age', 'SibSp','Parch',  'With Cabin', 'Fare']
features_2 = ['Pclass','Sex', 'Age', 'SibSp','Parch',  'With Cabin', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']

# Train the Classifier to take the training features and learn how they relate
# to the training y (the species)
y_train = train['Survived']
clf_0.fit(train[features_2], y_train)
clf_1.fit(train[features_1], y_train)        
clf_2.fit(train[features_2], y_train)


# Càlcul de l'accuracy de train i de test dels diferents models
# Tree decision
y_test = test['Survived']
preds_0 = clf_0.predict(test[features_2])
test['Survived_0'] = preds_0
print("clf_0 Train Accuracy  : ", clf_0.score(train[features_2], y_train))
print("clf_0 Test Accuracy  : ",clf_0.score(test[features_2], y_test))
print(pd.crosstab(test['Survived'], preds_0, rownames=['Real'], colnames=['Predicted']))
# Generem el graf de l'arbre de decisió. El camp "Sex" és el camp més determinant
out = StringIO()
filename = 'arbre_decisio_deep_4.dot'
tree.export_graphviz(clf_0, out_file=path+filename,feature_names=features_2,
                           class_names=['Dead', 'Survived'],  
                           filled=True, rounded=True )
'''
tree.export_graphviz(clf_0, out_file='clf_0_deep_4_bis.dot',feature_names=features_2,
                           class_names=['Dead', 'Survived'],  
                           filled=True, rounded=True )
'''
# es pot visualitzar des de http://webgraphviz.com/ 

# Random Forest
# Create predictions with different models
preds_1 = clf_1.predict(test[features_1])
preds_2 = clf_2.predict(test[features_2])

print("clf_1 Train Accuracy : ", accuracy_score(y_train, clf_1.predict(train[features_1])))
print("clf_1 Test Accuracy  : ", accuracy_score(y_test, preds_1))
print("Importància de les variables:", list(zip(train[features_1], clf_1.feature_importances_)))
print(pd.crosstab(test['Survived'], preds_1, rownames=['Real'], colnames=['Predicted']))

print("clf_2 Train Accuracy: ", accuracy_score(y_train, clf_2.predict(train[features_2])))
print("clf_2 Test Accuracy : ", accuracy_score(y_test, preds_2))
print("Importància de les variables:", list(zip(train[features_2], clf_2.feature_importances_)))
print(pd.crosstab(test['Survived'], preds_2, rownames=['Real'], colnames=['Predicted']))

pd.crosstab(train['Age'], train['Survived'], rownames=['Ages'], colnames=['Supervised'])
pd.crosstab(df['Sex'], df['Survived'])
pd.crosstab(train['Sex'], train['Survived'])

# Fem cross validation dels diferents models per evitar sobreentrenament
scores_0 = cross_val_score(clf_0, df[features_2], df['Survived'], cv=10, scoring='accuracy')
print("clf_0 cv=10 Accuracy: %0.5f (+/- %0.5f)" % (scores_0.mean(), scores_0.std() * 2))
scores_1 = cross_val_score(clf_1, df[features_1], df['Survived'], cv=10, scoring='accuracy')
print("clf_1 cv=10 Accuracy: %0.5f (+/- %0.5f)" % (scores_1.mean(), scores_1.std() * 2))
scores_2 = cross_val_score(clf_2, df[features_2], df['Survived'], cv=10, scoring='accuracy')
print("clf_2 cv=10 Accuracy: %0.5f (+/- %0.5f)" % (scores_2.mean(), scores_2.std() * 2)) 

# Provem el model Support Vector Machine
from sklearn import svm    
clf_svm_3 = svm.SVC(kernel='linear', C=1)
scores_svm = cross_val_score(clf_svm_3, df[features_2], df['Survived'], cv=10, scoring='accuracy')
clf_svm_3.fit(train[features_2], y_train)
preds_3 = clf_svm_3.predict(test[features_2])
print("Train Accuracy :: ", accuracy_score(y_train, clf_svm_3.predict(train[features_2])))
print("Test Accuracy  :: ", accuracy_score(y_test, preds_3))
print(pd.crosstab(test['Survived'], preds_3, rownames=['Real'], colnames=['Predicted']))
scores_svm_3 = cross_val_score(clf_svm_3, df[features_2], df['Survived'], cv=10, scoring='accuracy')
print("clf_svm_3 cv=10 Accuracy: %0.5f (+/- %0.5f)" % (scores_1.mean(), scores_1.std() * 2))

# Fem prediccions del dataset de test de la prova de Kaggle. Entrenem els models 
# amb el # 100% del conjunt de train
y = df['Survived']
clf_0.fit(df[features_2], y)
clf_1.fit(df[features_1], y)
clf_2.fit(df[features_2], y)
clf_svm_3.fit(df[features_2], y)


# Llegir el dataset de test, preparar les dades i fer les prediccions i escriure el fitxer de "submission" de kaggle
filename = "test.csv"
df_test = pd.read_csv(path+filename)
df_test[df_test.columns].isnull().sum()
df_test['Sex'].replace({'male': 0, 'female': 1}, inplace=True)
df_test['Embarked'].astype('category')
df_test['With Cabin'] = df_test['Cabin'].isnull()
df_test['With Cabin'] = df_test['With Cabin'] * 1
df_test['Age'].fillna(df_test['Age'].mean(), inplace=True)
df_test = pd.get_dummies(df_test, columns=['Embarked'])
df_test['Fare'].fillna(df_test['Fare'].mean(), inplace=True)
df_test[df_test.columns].isnull().sum()


predict_test_2 = clf_2.predict(df_test[features_2])
df_test['Survived_2'] = predict_test_2 
df_submission = df_test[['PassengerId', 'Survived_2']]
df_submission.rename(columns={'Survived_2': 'Survived'}, inplace=True)
filename = "submission_clf_2_20180611_def.csv"
df_submission.to_csv(path+filename, sep=',', index=False)

predict_test_1 = clf_1.predict(df_test[features_1])
df_test['Survived_1'] = predict_test_1
df_submission = df_test[['PassengerId', 'Survived_1']]
df_submission.rename(columns={'Survived_1': 'Survived'}, inplace=True)      
filename = "submission_clf_1_20180610_def.csv"
df_submission.to_csv(path+filename, sep=',', index=False)

predict_test_3 = clf_svm_3.predict(df_test[features_2])
df_test['Survived_3'] = predict_test_3
df_submission = df_test[['PassengerId', 'Survived_3']]
df_submission.rename(columns={'Survived_3': 'Survived'}, inplace=True)
filename = "submission_clf_svm_3_20180611_def.csv"
df_submission.to_csv(path+filename, sep=',', index=False)

# El resultat a Kaggle del clf_1 té una accuracy de 0.73205742 (153 de 209)
# El resultat a Kaggle del clf_2 té una accuracy de 0.77033493 (161 de 209)
# El resultat a Kaggle del clf_svm_3 té una accuracy de 0.76555024 (160 de 209)

# Mirem de millorar el classificador Random Forest variant el paràmetre 
# "n_estimators" (a clf_1 i clf_2 hem utilitzat el valor per defecte = 10)

estimators_col = range(1,100)
accuracy_estimator = pd.DataFrame(columns = ['Estimators', 'Train Accuracy', 'Test Accuracy'])
for estimators in estimators_col:
    clf = RandomForestClassifier(n_estimators=estimators, n_jobs=4, random_state=0)
    clf.fit(train[features_2], y_train)
    accuracy_train =  accuracy_score(y_train, clf.predict(train[features_2]))
    accuracy_test = accuracy_score(y_test, clf.predict(test[features_2]))
    row = pd.DataFrame([[estimators, accuracy_train, accuracy_test]], columns=['Estimators', 'Train Accuracy', 'Test Accuracy'])
    accuracy_estimator = accuracy_estimator.append(row, ignore_index=True)

accuracy_estimator.plot(x='Estimators', y='Train Accuracy', style='')
accuracy_estimator.plot(x='Estimators', y='Test Accuracy', style='')
print("estimadors amb millor accuracy test:", accuracy_estimator['Test Accuracy'].argmax())
# Surt que l'òptim és al voltant dels 30

# provem amb n_estimators = 30
clf = RandomForestClassifier(n_estimators=30, n_jobs=4, random_state=0)
clf.fit(df[features_2], y)
#clf.fit(df[features_2], y_train)
predict_test = clf.predict(df_test[features_2])
df_test['Survived_30_estimators'] = predict_test
df_submission = df_test[['PassengerId', 'Survived_30_estimators']]
df_submission.rename(columns={'Survived_30_estimators': 'Survived'}, inplace=True)
filename = "submission_clf_30_20180611_def.csv"
df_submission.to_csv(path+filename, sep=',', index=False)

# El resultat a Kaggle del clf_2 amb 30 estimadors no millora l'accuracy:0.75119 (157 de 209)

# Millores: Tractament dels valors nuls en funció d'altres variables
# Utilitzar la informació que no hem utilitzat: 'Name' i Ticket'
# provar altres mètodes












