import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn import tree  
from sklearn.svm import SVC  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.naive_bayes import GaussianNB  
from sklearn.cross_validation import cross_val_score ,train_test_split 
from sklearn import metrics  
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, Perceptron
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier



#### Data Read from CSV
df = pd.read_csv('creditdataset.csv')

#### Data Exploration
plt.hist(df['Y']) # Histogram 
plt
#### Repayment Data ####
covar_matrix = PCA(n_components = 6)
covar_matrix.fit(df.iloc[:,16:23])
variance = covar_matrix.explained_variance_ratio_
var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)
print "Variance Ratio X17 - X23 : ", var #[ 66.8  78.1  84.   88.7  92.6  96.3]

###### PCA because they are correlated and dimensional reduction always helps
# pca = PCA(n_components = 4)
# pc = pca.fit_transform(df.iloc[:,17:23])
# prep = pd.DataFrame(pc,columns=['pc1','pc2','pc3','pc4'])

#### Due Bill Data ####
covar_matrix = PCA(n_components = 6)
covar_matrix.fit(df.iloc[:,11:17])
variance = covar_matrix.explained_variance_ratio_
var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)
print "Variance Ratio X12 - X17 : ",var

###### PCA because they are correlated and dimensional reduction always helps
pca = PCA(n_components = 2)
pc = pca.fit_transform(df.iloc[:,11:17])
pdue = pd.DataFrame(pc,columns=['dpc1','dpc2'])



#### Payment Delay Data ####
covar_matrix = PCA(n_components = 6)
covar_matrix.fit(df.iloc[:,5:11])
variance = covar_matrix.explained_variance_ratio_
var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)
print "Variance Ratio X6 - X11 : ",var

###### PCA because they are correlated and dimensional reduction always helps
pca = PCA(n_components = 4)
pc = pca.fit_transform(df.iloc[:,5:11])
plap = pd.DataFrame(pc,columns=['lpc1','lpc2','lpc3','lpc4'])

#### Person Non-monetary ####
covar_matrix = PCA(n_components = 4)
covar_matrix.fit(df.iloc[:,1:5])
variance = covar_matrix.explained_variance_ratio_
var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)
print "Variance Ratio X2 - X5 : ",var


df_reduced = pd.concat([df['X1'],plap,pdue,df['X18'],df['X19'],df['X20'],df['X21'],df['X22'],df['X23'],df['Y']],axis=1)

covar_matrix = PCA(n_components = 13)
covar_matrix.fit(df_reduced.iloc[:,0:13])
variance = covar_matrix.explained_variance_ratio_
var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)
print "Variance Ratio Dimensionally reduced data : ",var

#### Plots #####

plt.show()
# correlation = df.corr()
# plt.figure(num = 1, figsize=(14,10))
# sb.heatmap(correlation)
# sb.pairplot(df.iloc[:,12:15].sample(frac=.5))
# sb.pairplot(df.iloc[:,6:9].sample(frac=.5))
#sb.pairplot(pdue)
#sb.pairplot(prep)

# df_reduced = pd.concat([df.iloc[:,0:5],pdue,prep,df['Y']],axis=1)
# #plt.show()
# 

 
train, test = train_test_split(df_reduced)

#print train.shape, test.shape

y_train = train['Y']
x_train = train.drop(['Y'],axis = 1)
y_test = test['Y']
x_test = test.drop(['Y'],axis = 1)


rf = RandomForestClassifier()
rf.fit(x_train,y_train)
print "features sorted by their score:"
print sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), x_train), reverse=True) 


dftab = pd.DataFrame(columns = ['Classifier', 'Training Score', 'Accuracy', 'Precision', 'Recall', 'F1'])

#Logistic Regression
lr = LogisticRegression()
lr = lr.fit(x_train,y_train)
predicted = pd.DataFrame(lr.predict(x_test))
dftab = dftab.append({'Classifier': 'LR','Training Score': lr.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1)}, ignore_index=True)
# print "Logistic Regression Train Data Score : ", lr.score(x_train,y_train)
# print "Logistic Regression Test Data Score : ",metrics.accuracy_score(y_test,predicted)
# print "Logistic Regression Precision Score : ", metrics.precision_score(y_test,predicted,pos_label=1)
# print "Logistic Regression Recall Score : ", metrics.recall_score(y_test,predicted,pos_label=1)
# print "Logistic Regression F1 Score : ", metrics.f1_score(y_test, predicted, pos_label=1)
# print "Confusion Matrix : ",metrics.confusion_matrix(y_test,predicted)
# 
# #Decision Tree
dt = tree.DecisionTreeClassifier(max_depth = 5)
dt = dt.fit(x_train,y_train)
predicted = pd.DataFrame(dt.predict(x_test))
dftab = dftab.append({'Classifier': 'DT','Training Score': dt.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1)}, ignore_index=True)
# print "Decision Tree Train Data Score : ", dt.score(x_train,y_train)
# print "Decision Tree Test Data Score : ", metrics.accuracy_score(y_test,predicted)
# print "Decision Tree Precision Score : ", metrics.precision_score(y_test,predicted,pos_label=1)
# print "Decision Tree Recall Score : ", metrics.recall_score(y_test,predicted,pos_label=1)
# print "Decision Tree F1 Score : ", metrics.f1_score(y_test, predicted, pos_label=1)
# 
# #Random Forest
rf = RandomForestClassifier(bootstrap=True, criterion='gini', n_estimators=1000)
rf.fit(x_train,y_train)
predicted = pd.DataFrame(rf.predict(x_test))
dftab = dftab.append({'Classifier': 'RF','Training Score': rf.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1)}, ignore_index=True)
# print "Random Forest Train Score : ", rf.score(x_train,y_train)
# print "Random Forest Test Data Score : ", metrics.accuracy_score(y_test,predicted)
# print "Random Forest Precision Score : ", metrics.precision_score(y_test,predicted,pos_label=1)
# print "Random Forest Recall Score : ", metrics.recall_score(y_test,predicted,pos_label=1)
# print "Random Forest F1 Score : ", metrics.f1_score(y_test, predicted, pos_label=1)

# #Support Vector Machine
svm = SVC()
svm = svm.fit(x_train,y_train)
predicted = pd.DataFrame(svm.predict(x_test))
dftab = dftab.append({'Classifier': 'SVM','Training Score': svm.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1)}, ignore_index=True)
# print "Support Vector Machine Train Score : ", svm.score(x_train,y_train)
# print "Support Vector Machine Test Data Score : ", metrics.accuracy_score(y_test,predicted)
# print "Support Vector Machine Precision Score : ", metrics.precision_score(y_test,predicted,pos_label=1)
# print "Support Vector Machine Recall Score : ", metrics.recall_score(y_test,predicted,pos_label=1)
# print "Support Vector Machine F1 Score : ", metrics.f1_score(y_test, predicted, pos_label=1)
 
# #KNN
knn = KNeighborsClassifier(n_neighbors = 5)
knn = knn.fit(x_train,y_train)
predicted = pd.DataFrame(knn.predict(x_test))
dftab = dftab.append({'Classifier': 'KNN','Training Score': knn.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1)}, ignore_index=True)
# print "KNN Train Score : ", knn.score(x_train,y_train)
# print "KNN Test Data Score : ", metrics.accuracy_score(y_test,predicted)
# print "KNN Precision Score : ", metrics.precision_score(y_test,predicted,pos_label=1)
# print "KNN ecall Score : ", metrics.recall_score(y_test,predicted,pos_label=1)
# print "KNN F1 Score : ", metrics.f1_score(y_test, predicted, pos_label=1)

# #Two Class Bayes
bm = GaussianNB()
bm = bm.fit(x_train,y_train)
predicted = pd.DataFrame(bm.predict(x_test))
dftab = dftab.append({'Classifier': 'Bayes','Training Score': bm.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1)}, ignore_index=True)
# print "Bayes Train Score : ", bm.score(x_train,y_train)
# print "Bayes Test Data Score : ", metrics.accuracy_score(y_test,predicted)
# print "Bayes Precision Score : ", metrics.precision_score(y_test,predicted,pos_label=1)
# print "Bayes Recall Score : ", metrics.recall_score(y_test,predicted,pos_label=1)
# print "Bayes F1 Score : ", metrics.f1_score(y_test, predicted, pos_label=1)

# #Stochastic Gradient Descent
sdg = SGDClassifier()
sdg = sdg.fit(x_train,y_train)
predicted = pd.DataFrame(sdg.predict(x_test))
dftab = dftab.append({'Classifier': 'SGD','Training Score': sdg.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1)}, ignore_index=True)
# print "Train Score : ", sdg.score(x_train,y_train)
# print "Test Data Score : ", metrics.accuracy_score(y_test,predicted)
# print "Precision Score : ", metrics.precision_score(y_test,predicted,pos_label=1)
# print "Recall Score : ", metrics.recall_score(y_test,predicted,pos_label=1)
# print "F1 Score : ", metrics.f1_score(y_test, predicted, pos_label=1)

# #Perceptron
p = Perceptron()
p = p.fit(x_train,y_train)
predicted = pd.DataFrame(p.predict(x_test))
dftab = dftab.append({'Classifier': 'Perceptron','Training Score': p.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1)}, ignore_index=True)
# print "Perceptron Train Score : ", p.score(x_train,y_train)
# print "Perceptron Test Data Score : ", metrics.accuracy_score(y_test,predicted)
# print "Perceptron Precision Score : ", metrics.precision_score(y_test,predicted,pos_label=1)
# print "Perceptron Recall Score : ", metrics.recall_score(y_test,predicted,pos_label=1)
# print "Perceptron F1 Score : ", metrics.f1_score(y_test, predicted, pos_label=1)

# #Passive Aggressive Classifier
pac = PassiveAggressiveClassifier()
pac = pac.fit(x_train,y_train)
predicted = pd.DataFrame(pac.predict(x_test))
dftab = dftab.append({'Classifier': 'PAC','Training Score': pac.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1)}, ignore_index=True)
# print "Passive Aggressive Classifier Train Score : ", pac.score(x_train,y_train)
# print "Passive Aggressive Classifier Test Data Score : ", metrics.accuracy_score(y_test,predicted)
# print "Passive Aggressive Classifier Precision Score : ", metrics.precision_score(y_test,predicted,pos_label=1)
# print "Passive Aggressive Classifier Recall Score : ", metrics.recall_score(y_test,predicted,pos_label=1)
# print "Passive Aggressive Classifier F1 Score : ", metrics.f1_score(y_test, predicted, pos_label=1)
 
# #Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis()
lda = lda.fit(x_train,y_train)
predicted = pd.DataFrame(lda.predict(x_test))
dftab = dftab.append({'Classifier': 'LDA','Training Score': lda.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1)}, ignore_index=True)
# print "Linear Discriminant Analysis Train Score : ", lda.score(x_train,y_train)
# print "Linear Discriminant Analysis Test Data Score : ", metrics.accuracy_score(y_test,predicted)
# print "Linear Discriminant Analysis Precision Score : ", metrics.precision_score(y_test,predicted,pos_label=1)
# print "Linear Discriminant Analysis Recall Score : ", metrics.recall_score(y_test,predicted,pos_label=1)
# print "Linear Discriminant Analysis F1 Score : ", metrics.f1_score(y_test, predicted, pos_label=1)
 
# #Quadratic Discriminant Analysis
qda = QuadraticDiscriminantAnalysis()
qda = qda.fit(x_train,y_train)
predicted = pd.DataFrame(qda.predict(x_test))
dftab = dftab.append({'Classifier': 'QDA','Training Score': qda.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1)}, ignore_index=True)
# print "Quadratic Discriminant Analysis Train Score : ", qda.score(x_train,y_train)
# print "Quadratic Discriminant Analysis Test Data Score : ", metrics.accuracy_score(y_test,predicted)
# print "Quadratic Discriminant Analysis Precision Score : ", metrics.precision_score(y_test,predicted,pos_label=1)
# print "Quadratic Discriminant Analysis Recall Score : ", metrics.recall_score(y_test,predicted,pos_label=1)
# print "Quadratic Discriminant Analysis F1 Score : ", metrics.f1_score(y_test, predicted, pos_label=1)

# #Gradient Boosting Classifier
gbc = GradientBoostingClassifier()
gbc = gbc.fit(x_train,y_train)
predicted = pd.DataFrame(gbc.predict(x_test))
dftab = dftab.append({'Classifier': 'GBC','Training Score': gbc.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1)}, ignore_index=True)
# print "Gradient Boosting Classifier Train Score : ", gbc.score(x_train,y_train)
# print "Gradient Boosting Classifier Test Data Score : ", metrics.accuracy_score(y_test,predicted)
# print "Gradient Boosting Classifier Precision Score : ", metrics.precision_score(y_test,predicted,pos_label=1)
# print "Gradient Boosting Classifier Recall Score : ", metrics.recall_score(y_test,predicted,pos_label=1)
# print "Gradient Boosting Classifier F1 Score : ", metrics.f1_score(y_test, predicted, pos_label=1)

# #AdaBoost Classifier
abc = AdaBoostClassifier()
abc = abc.fit(x_train,y_train)
predicted = pd.DataFrame(abc.predict(x_test))
dftab = dftab.append({'Classifier': 'ABC','Training Score': abc.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1)}, ignore_index=True)
# print "AdaBoost Classifier Train Score : ", abc.score(x_train,y_train)
# print "AdaBoost Classifier Test Data Score : ", metrics.accuracy_score(y_test,predicted)
# print "AdaBoost Classifier Precision Score : ", metrics.precision_score(y_test,predicted,pos_label=1)
# print "AdaBoost Classifier Recall Score : ", metrics.recall_score(y_test,predicted,pos_label=1)
# print "AdaBoost Classifier F1 Score : ", metrics.f1_score(y_test, predicted, pos_label=1)

# #Extra Trees Classifier
etc = ExtraTreesClassifier()
etc = etc.fit(x_train,y_train)
predicted = pd.DataFrame(etc.predict(x_test))
dftab = dftab.append({'Classifier': 'ETC','Training Score': etc.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1)}, ignore_index=True)
# print "Extra Trees Classifier Train Score : ", etc.score(x_train,y_train)
# print "Extra Trees Classifier Test Data Score : ", metrics.accuracy_score(y_test,predicted)
# print "Extra Trees Classifier Precision Score : ", metrics.precision_score(y_test,predicted,pos_label=1)
# print "Extra Trees Classifier Recall Score : ", metrics.recall_score(y_test,predicted,pos_label=1)
# print "Extra Trees Classifier F1 Score : ", metrics.f1_score(y_test, predicted, pos_label=1)

# #Multi-layer Perceptron Classifier
mlpc = MLPClassifier(solver='lbfgs', alpha=1e-5,  hidden_layer_sizes=(5, 2), random_state=1)
mlpc = mlpc.fit(x_train,y_train)
predicted = pd.DataFrame(mlpc.predict(x_test))
dftab = dftab.append({'Classifier': 'MLPC','Training Score': mlpc.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1)}, ignore_index=True)
# print "Multi-layer Perceptron Classifier Train Score : ", mlpc.score(x_train,y_train)
# print "Multi-layer Perceptron Classifier Test Data Score : ", metrics.accuracy_score(y_test,predicted)
# print "Multi-layer Perceptron Classifier Precision Score : ", metrics.precision_score(y_test,predicted,pos_label=1)
# print "Multi-layer Perceptron Classifier Recall Score : ", metrics.recall_score(y_test,predicted,pos_label=1)
# print "Multi-layer Perceptron Classifier F1 Score : ", metrics.f1_score(y_test, predicted, pos_label=1)

print dftab.sort_values(by='Accuracy', ascending=False)


#  = .fit(x_train,y_train)
# predicted = pd.DataFrame(.predict(x_test))
# print "Train Score : ", .score(x_train,y_train)
# print "Test Data Score : ", metrics.accuracy_score(y_test,predicted)
# print "Precision Score : ", metrics.precision_score(y_test,predicted,pos_label=1)
# print "Recall Score : ", metrics.recall_score(y_test,predicted,pos_label=1)
# print "F1 Score : ", metrics.f1_score(y_test, predicted, pos_label=1)

#/Users/nitishsanghi/documents/projects/courses/machine-learning-master/projects/capstone/CapstoneWorks
