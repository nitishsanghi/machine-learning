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

datasets = [newdf]
dsname = ['Raw Data', 'Curated Data', 'Reduced Data', 'Ex-Reduced Data']
dftab = pd.DataFrame(columns = ['Classifier','Dataset', 'Training Score', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'])
i = 0
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
# # 	print "Random Forest Complete"
# 		#Logistic Regression
# 	lr = LogisticRegression()
# 	lr = lr.fit(x_train,y_train)
# 	predicted = pd.DataFrame(lr.predict(x_test))
# 	fprtest, tprtest, thres = metrics.roc_curve(y_test,predicted)
# 	dftab = dftab.append({'Classifier': 'LR','Dataset': dsname[i] ,'Training Score': lr.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1), 'AUC': metrics.auc(fprtest,tprtest)}, ignore_index=True)

	# #Decision Tree
	dt = tree.DecisionTreeClassifier(max_depth = 5, min_samples_split = 0.24516129032258066, criterion = 'entropy', min_samples_leaf =  0.12580645161290324)
	dt = dt.fit(x_train,y_train)
	predicted = pd.DataFrame(dt.predict(x_test))
	fprtest, tprtest, thres = metrics.roc_curve(y_test,predicted)
	dftab = dftab.append({'Classifier': 'DT','Dataset': dsname[i] ,'Training Score': dt.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1), 'AUC': metrics.auc(fprtest,tprtest)}, ignore_index=True)
# 
# 	# #Random Forest
# 	rf = RandomForestClassifier(bootstrap=True, criterion='gini', n_estimators=1000)
# 	rf.fit(x_train,y_train)
# 	predicted = pd.DataFrame(rf.predict(x_test))
# 	fprtest, tprtest, thres = metrics.roc_curve(y_test,predicted)
# 	dftab = dftab.append({'Classifier': 'RF','Dataset': dsname[i] ,'Training Score': rf.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1), 'AUC': metrics.auc(fprtest,tprtest)}, ignore_index=True)
# 
# 	# #Support Vector Machine
# 	svm = SVC()
# 	svm = svm.fit(x_train,y_train)
# 	predicted = pd.DataFrame(svm.predict(x_test))
# 	fprtest, tprtest, thres = metrics.roc_curve(y_test,predicted)
# 	dftab = dftab.append({'Classifier': 'SVM','Dataset': dsname[i] ,'Training Score': svm.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1), 'AUC': metrics.auc(fprtest,tprtest)}, ignore_index=True)
#  
# 	# #KNN
# 	knn = KNeighborsClassifier(n_neighbors = 5)
# 	knn = knn.fit(x_train,y_train)
# 	predicted = pd.DataFrame(knn.predict(x_test))
# 	fprtest, tprtest, thres = metrics.roc_curve(y_test,predicted)
# 	dftab = dftab.append({'Classifier': 'KNN','Dataset': dsname[i] ,'Training Score': knn.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1), 'AUC': metrics.auc(fprtest,tprtest)}, ignore_index=True)
# 
# 	# #Two Class Bayes
# 	bm = GaussianNB()
# 	bm = bm.fit(x_train,y_train)
# 	predicted = pd.DataFrame(bm.predict(x_test))
# 	fprtest, tprtest, thres = metrics.roc_curve(y_test,predicted)
# 	dftab = dftab.append({'Classifier': 'Bayes','Dataset': dsname[i] ,'Training Score': bm.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1), 'AUC': metrics.auc(fprtest,tprtest)}, ignore_index=True)
# 
# 	# #Stochastic Gradient Descent
# 	sdg = SGDClassifier()
# 	sdg = sdg.fit(x_train,y_train)
# 	predicted = pd.DataFrame(sdg.predict(x_test))
# 	fprtest, tprtest, thres = metrics.roc_curve(y_test,predicted)
# 	dftab = dftab.append({'Classifier': 'SGD','Dataset': dsname[i] ,'Training Score': sdg.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1), 'AUC': metrics.auc(fprtest,tprtest)}, ignore_index=True)
# 
# 	# #Perceptron
# 	p = Perceptron()
# 	p = p.fit(x_train,y_train)
# 	predicted = pd.DataFrame(p.predict(x_test))
# 	fprtest, tprtest, thres = metrics.roc_curve(y_test,predicted)
# 	dftab = dftab.append({'Classifier': 'Perceptron','Dataset': dsname[i] ,'Training Score': p.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1), 'AUC': metrics.auc(fprtest,tprtest)}, ignore_index=True)
# 
# 	# #Passive Aggressive Classifier
# 	pac = PassiveAggressiveClassifier()
# 	pac = pac.fit(x_train,y_train)
# 	predicted = pd.DataFrame(pac.predict(x_test))
# 	fprtest, tprtest, thres = metrics.roc_curve(y_test,predicted)
# 	dftab = dftab.append({'Classifier': 'PAC','Dataset': dsname[i] ,'Training Score': pac.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1), 'AUC': metrics.auc(fprtest,tprtest)}, ignore_index=True)
# 
# 	# #Linear Discriminant Analysis
# 	lda = LinearDiscriminantAnalysis()
# 	lda = lda.fit(x_train,y_train)
# 	predicted = pd.DataFrame(lda.predict(x_test))
# 	fprtest, tprtest, thres = metrics.roc_curve(y_test,predicted)
# 	dftab = dftab.append({'Classifier': 'LDA','Dataset': dsname[i] ,'Training Score': lda.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1), 'AUC': metrics.auc(fprtest,tprtest)}, ignore_index=True)
#  
# 	# #Quadratic Discriminant Analysis
# 	qda = QuadraticDiscriminantAnalysis()
# 	qda = qda.fit(x_train,y_train)
# 	predicted = pd.DataFrame(qda.predict(x_test))
# 	fprtest, tprtest, thres = metrics.roc_curve(y_test,predicted)
# 	dftab = dftab.append({'Classifier': 'QDA','Dataset': dsname[i] ,'Training Score': qda.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1), 'AUC': metrics.auc(fprtest,tprtest)}, ignore_index=True)
# 
# 	# #Gradient Boosting Classifier
# 	gbc = GradientBoostingClassifier()
# 	gbc = gbc.fit(x_train,y_train)
# 	predicted = pd.DataFrame(gbc.predict(x_test))
# 	fprtest, tprtest, thres = metrics.roc_curve(y_test,predicted)
# 	dftab = dftab.append({'Classifier': 'GBC','Dataset': dsname[i] ,'Training Score': gbc.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1), 'AUC': metrics.auc(fprtest,tprtest)}, ignore_index=True)
# 
# 	# #AdaBoost Classifier
# 	abc = AdaBoostClassifier()
# 	abc = abc.fit(x_train,y_train)
# 	predicted = pd.DataFrame(abc.predict(x_test))
# 	fprtest, tprtest, thres = metrics.roc_curve(y_test,predicted)
# 	dftab = dftab.append({'Classifier': 'ABC','Dataset': dsname[i] ,'Training Score': abc.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1), 'AUC': metrics.auc(fprtest,tprtest)}, ignore_index=True)

	# #Extra Trees Classifier
	etc = ExtraTreesClassifier(min_samples_leaf = .1, n_estimators = 750, max_features = 'auto', criterion = 'gini', min_samples_split = 0.70967741935483875, max_depth = 25.75)
	etc = etc.fit(x_train,y_train)
	predicted = pd.DataFrame(etc.predict(x_test))
	fprtest, tprtest, thres = metrics.roc_curve(y_test,predicted)
	fprtrain, tprtrain, threstrain = metrics.roc_curve(y_train,pd.DataFrame(etc.predict(x_train)))
	print "AUC ETC : ", metrics.auc(fprtrain,tprtrain)
	dftab = dftab.append({'Classifier': 'ETC','Dataset': dsname[i] ,'Training Score': etc.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1), 'AUC': metrics.auc(fprtest,tprtest)}, ignore_index=True)

	# Multi-layer Perceptron Classifier
# 	mlpc = MLPClassifier(solver='lbfgs', alpha=1e-5,  hidden_layer_sizes=(5, 2), random_state=1)
# 	mlpc = mlpc.fit(x_train,y_train)
# 	predicted = pd.DataFrame(mlpc.predict(x_test))
# 	fprtest, tprtest, thres = metrics.roc_curve(y_test,predicted)
# 	dftab = dftab.append({'Classifier': 'MLPC','Dataset': dsname[i] ,'Training Score': mlpc.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1), 'AUC': metrics.auc(fprtest,tprtest)}, ignore_index=True)
    #i = i + 1
# 	print dftab.sort_values(by='Accuracy', ascending=False)

print dftab
f = open("Classifier Stats.txt",'w+')
f.write(str(dftab))
f.close()