import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sb
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn import tree  
from sklearn.svm import SVC  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.naive_bayes import GaussianNB  
from sklearn.model_selection import GridSearchCV, cross_val_score ,train_test_split 
from sklearn.model_selection import RandomizedSearchCV 
from sklearn import metrics 
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, Perceptron
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import scipy.stats as ss
from numpy.random import randint
import csv
import time


#### Data Read from CSV
df = pd.read_csv('creditdataset.csv')
newdf = pd.read_pickle('curateddataset.pkl')
df_reduced = pd.read_pickle('reduceddataset.pkl')
df_reducedmon = pd.read_pickle('reducedmondataset.pkl')
datasets = [newdf, df_reduced]
dsname = ['Curated Data', 'Reduced Data']
dftab = pd.DataFrame(columns = ['Classifier','Dataset', 'Training Score', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'])
i = 0
print 'Entering Loop'
start_time = time.clock()
print 'Time at start of loop : ', time.clock()
for x in datasets:
	train_start_time = time.clock()
	train, test = train_test_split(x)
	y_train = train['Y']
	x_train = train.drop(['Y'],axis = 1)
	y_test = test['Y']
	x_test = test.drop(['Y'],axis = 1)
	# #Decision Tree

	# tree_parameters_dt = {"criterion": ["gini","entropy"], 'max_depth': np.linspace(1,32,32), 'min_samples_split': np.linspace(0.1,1,32), 'min_samples_leaf': np.linspace(0.1,.5,32)}
# 	dt = tree.DecisionTreeClassifier()
# 	randomsearch = RandomizedSearchCV(dt, tree_parameters_dt, cv = 6, refit = True ,n_iter = 300, scoring = metrics.make_scorer(metrics.roc_auc_score)  )
# 	randomsearch.fit(x_train,y_train)
# 	best_models = pd.DataFrame(randomsearch.cv_results_)
# 	best_models = best_models.sort_values(by=['rank_test_score'])
# 	best_models.to_csv("Metrics/Metrics_DecisionTree.csv")
# 	print randomsearch.scorer_
# 	best_model = randomsearch.best_estimator_
# 	predicted = pd.DataFrame(best_model.predict(x_test))
# 	fpr, tpr, thres = metrics.roc_curve(y_test,predicted)
# 	print "Test Data AUC " +dsname[i]+  " : ",metrics.auc(fpr,tpr)
# 	tree_name = dsname[i] + ".dot"
# 	tree.export_graphviz(best_model,tree_name)
# 	
	i = i + 1
	tree_parameters_etc = {"criterion": ["gini","entropy"], 'max_depth': np.linspace(1,100,5), 'max_features': ['auto', 'sqrt'], 'min_samples_leaf': np.linspace(0.1,.5,32), 'min_samples_split': np.linspace(0.1,1,32), 'n_estimators': range(50,1000,50)}
	etc = ExtraTreesClassifier()
	randomsearch = RandomizedSearchCV(etc, tree_parameters_etc, cv = 6, n_iter = 200, scoring = metrics.make_scorer(metrics.roc_auc_score))
	randomsearch.fit(x_train,y_train)
	best_models = pd.DataFrame(randomsearch.cv_results_)
	best_models = best_models.sort_values(by=['rank_test_score'])
	best_models.to_csv("Metrics/Metrics_ExtraTreesClassifier.csv")
	print randomsearch.scorer_
	best_model = randomsearch.best_estimator_
	predicted = pd.DataFrame(best_model.predict(x_test))
	fpr, tpr, thres = metrics.roc_curve(y_test,predicted)
	print "Test Data AUC etc" +dsname[i]+  " : ",metrics.auc(fpr,tpr)
	train_end_time = time.clock()
	print 'Train time for ' + dsname[i]+ ' : ',train_start_time - train_end_time
	
	
print 'Exiting Loop'
end_time = time.clock()
print "Total loop time : ", start_time - end_time
# 
# 
# tree_parameters_rf = {"criterion": ["gini","entropy"], 'max_depth': np.linspace(1,100,5), 'max_features': ['auto', 'sqrt'], 'min_samples_leaf': np.linspace(0.1,.5,32), 'min_samples_split': np.linspace(0.1,1,32), 'n_estimators': np.linspace(10,1000,20)}
# rf = RandomForestClassifier()
# randomsearch = RandomizedSearchCV(rf, tree_parameters_rf, cv = 6, n_iter = 200, scoring = 'roc_auc' )
# randomsearch.fit(x_train,y_train)
# best_models = pd.DataFrame(randomsearch.cv_results_)
# best_models = best_models.sort_values(by=['rank_test_score'])
# best_models.to_csv("Metrics/Metrics_RandomForestClassifier.csv")
