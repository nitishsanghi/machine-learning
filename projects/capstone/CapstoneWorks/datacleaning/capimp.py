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
from sklearn import metrics  
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, Perceptron
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import scipy.stats as ss


#### Data Read from CSV
df = pd.read_csv('creditdataset.csv')
newdf = pd.read_pickle('curateddataset.pkl')
df_reduced = pd.read_pickle('reduceddataset.pkl')
df_reducedmon = pd.read_pickle('reducedmondataset.pkl')

datasets = [newdf, df_reduced]
dsname = ['Curated Data', 'Reduced Data']
dftab = pd.DataFrame(columns = ['Classifier','Dataset', 'Training Score', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'])
i = 0
colors = ['r','k','g','b']
for x in datasets:
	train, test = train_test_split(x)
	y_train = train['Y']
	x_train = train.drop(['Y'],axis = 1)
	y_test = test['Y']
	x_test = test.drop(['Y'],axis = 1)
# 	
# 	tree_para = {'criterion': ['gini','entropy'], 'max_depth': [5,10,20,40,100]}
# 	dt = GridSearchCV(tree.DecisionTreeClassifier(), tree_para, cv = 5)
# 	dt = dt.fit(x_train,y_train)
# 	predicted = pd.DataFrame(dt.predict(x_test))
# 	fprtest, tprtest, thres = metrics.roc_curve(y_test,predicted)
# 	dftab = dftab.append({'Classifier': 'DT','Dataset': x,'Training Score': dt.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1), 'AUC': metrics.auc(fprtest,tprtest)}, ignore_index=True)
# 	print "Decision Tree Complete"
# 
# 	tree_para = {'criterion': ['gini','entropy'], 'max_depth': [5,10,20,40,100],'n_estimators': [10,100,1000]}
# 	rf = GridSearchCV(RandomForestClassifier(),tree_para, cv = 5)
# 	rf.fit(x_train,y_train)
# 	predicted = pd.DataFrame(rf.predict(x_test))
# 	fprtest, tprtest, thres = metrics.roc_curve(y_test,predicted)
# 	dftab = dftab.append({'Classifier': 'RF','Datasets': x,'Training Score': rf.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1), 'AUC': metrics.auc(fprtest,tprtest)}, ignore_index=True)
# 	print "Random Forest Complete"
	
	# #Decision Tree
	parameters = np.linspace(1,32,32)
	for y in parameters:
		dt = tree.DecisionTreeClassifier(max_depth = y, cv= 5)
		dt = dt.fit(x_train,y_train)
		predicted = pd.DataFrame(dt.predict(x_test))
		fprtest, tprtest, thres = metrics.roc_curve(y_test,predicted)
		dftab = dftab.append({'Classifier': 'DT','Dataset': dsname[i] ,'Training Score': dt.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1), 'AUC': metrics.auc(fprtest,tprtest)}, ignore_index=True)
	# tree_parameters = {'max_depth': np.linspace(1,32,32, endpoint = True), 'min_samples_split' : np.linspace(0.1, 1.0, 10, endpoint= True), 'min_samples_leaf': np.linspace(0.1,0.5,5, endpoint = True) }
# 	dt = GridSearchCV(tree.DecisionTreeClassifier(), tree_parameters,cv = 5)
# 	dt = dt.fit(x_train,y_train)
# 	predicted = pd.DataFrame(dt.predict(x_test))
# 	fprtest, tprtest, thres = metrics.roc_curve(y_test,predicted)
# 	dftab = dftab.append({'Classifier': 'DT','Dataset': dsname[i] ,'Training Score': dt.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1), 'AUC': metrics.auc(fprtest,tprtest)}, ignore_index=True)




	# Random Forest
# 	rf = RandomForestClassifier(bootstrap=True, criterion='gini', n_estimators=1000)
# 	rf.fit(x_train,y_train)
# 	predicted = pd.DataFrame(rf.predict(x_test))
# 	fprtest, tprtest, thres = metrics.roc_curve(y_test,predicted)
# 	dftab = dftab.append({'Classifier': 'RF','Dataset': dsname[i] ,'Training Score': rf.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1), 'AUC': metrics.auc(fprtest,tprtest)}, ignore_index=True)
# 
# 	Extra Trees Classifier
# 	etc = ExtraTreesClassifier()
# 	etc = etc.fit(x_train,y_train)
# 	predicted = pd.DataFrame(etc.predict(x_test))
# 	fprtest, tprtest, thres = metrics.roc_curve(y_test,predicted)
# 	dftab = dftab.append({'Classifier': 'ETC','Dataset': dsname[i] ,'Training Score': etc.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1), 'AUC': metrics.auc(fprtest,tprtest)}, ignore_index=True)
	filename = "DecisionTreeMetrics/AUC.png"
	n = len(dftab['AUC'])-32
	plt.figure(1)
	plt.plot(range(0,32,1),dftab.iloc[n:,7],label = '%s  AUC' % dsname[i],color = colors[i+2])
	plt.xlabel("Max_Depth")
	plt.ylabel("AUC")
	plt.title("Metrics Decision Tree: Max_Depth vs AUC")
	plt.legend()
	plt.grid(True)
	plt.savefig(filename)
	filename = "DecisionTreeMetrics/Precision.png"
	plt.figure(2)
	plt.plot(range(0,32,1),dftab.iloc[n:,4],label = '%s Precision' % dsname[i],color = colors[i])
	plt.xlabel("Max_Depth")
	plt.ylabel("Precision")
	plt.title("Metrics Decision Tree: Max_Depth vs Precision")
	plt.legend()
	plt.grid(True)
	plt.savefig(filename)
	
	
	i = i + 1
	#print len(dftab["AUC"])
#print dftab.sort_values(by='AUC', ascending=False)
#print dftab
f = open("metrics/Classifier Stats Tuning.csv",'w+')
f.write(str(dftab))
f.close()