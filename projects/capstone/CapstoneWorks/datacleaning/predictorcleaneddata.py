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
from sklearn.model_selection import cross_val_score ,train_test_split, GridSearchCV 
from sklearn import metrics  
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, Perceptron
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import scipy.stats as ss

#### Data Read from CSV
df = pd.read_csv('creditdataset.csv')
categorical = ['X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','Y']
continuous = ['X1','X12','X13','X14','X15','X16','X17','X18','X19','X20','X21','X22','X23',]
list_col_headers = list(df.columns.values)

#### Raw Data Statistics ####
##Continuous##
num_instances = len(df)
columns =["Variable", "Maximum", "Minimum", "Mean", "Median", "25% Quartile", "75% Quartile"]
statsarray = []
for x in continuous:
	statsarray.append([x, np.round(max(df[x]),2), np.round(min(df[x]),2), np.round(np.mean(df[x]),2), np.round(np.median(df[x]),2), np.round(np.percentile(df[x],25)),np.round(np.percentile(df[x],75))])

arraystats = np.array(statsarray).T.tolist()
stats = pd.DataFrame(arraystats,columns)
writer = pd.ExcelWriter("Continuous Raw Data Basic Statistics.xlsx")
stats.to_excel(writer)
writer.save()

##Categorical##
columns =["Variable", "Maximum", "Minimum", "Median"]
statsarray = []
for x in categorical:
	statsarray.append([x, max(df[x]), min(df[x]), np.median(df[x])])

arraystats = np.array(statsarray).T.tolist()
statscat = pd.DataFrame(arraystats,columns)
writer = pd.ExcelWriter("Categorical Raw Data Basic Statistics.xlsx")
statscat.to_excel(writer)
writer.save()

## Raw Data Size ##
print "Total # of Instances : ",num_instances
print "Sum of target class variable Y : ", len(df[df['Y']==1])
print " # of 0 target class variable Y : ", len(df[df['Y']==0])
print "Legitimate instances : ",  len(df[df['Y']==0])+np.sum(df['Y'])

## Bad Categorical Variables ##
baddata = []

baddata.append(df.index[df['X3']>4].tolist())
print "X3 incongruent data # : ", len(baddata[0])

baddata.append(df.index[df['X1']<df['X12']].tolist())
print "Points Approved Credit < Credit Bill Due X12  :",len(df[df['X1']<df['X12']])
baddata.append(df.index[df['X1']<df['X13']].tolist())
print "Points Approved Credit < Credit Bill Due X13  :", len(df[df['X1']<df['X13']])
baddata.append(df.index[df['X1']<df['X14']].tolist())
print "Points Approved Credit < Credit Bill Due X14  :", len(df[df['X1']<df['X14']])
baddata.append(df.index[df['X1']<df['X15']].tolist())
print "Points Approved Credit < Credit Bill Due X15  :", len(df[df['X1']<df['X15']])
baddata.append(df.index[df['X1']<df['X16']].tolist())
print "Points Approved Credit < Credit Bill Due X16  :", len(df[df['X1']<df['X16']])
baddata.append(df.index[df['X1']<df['X17']].tolist())
print "Points Approved Credit < Credit Bill Due X17  :", len(df[df['X1']<df['X17']])
print "# Incongruent Variables ", len(baddata)
baddata = list(set([item for sublist in baddata for item in sublist]))
print "Total # bad points :  ", len(baddata)

#### Removing Outliers ####
listoutliers = []
for x in continuous:
	#print " \n ",x
	#print "# Rows to be deleted : ",len(df.index[df[x]>np.float(stats.iloc[6,continuous.index(x)])+1.5*(np.float(stats.iloc[6,continuous.index(x)])-np.float(stats.iloc[5,continuous.index(x)]))].tolist())
	listoutliers.append(df.index[df[x]>np.float(stats.iloc[6,continuous.index(x)])+1.5*(np.float(stats.iloc[6,continuous.index(x)])-np.float(stats.iloc[5,continuous.index(x)]))].tolist())
	# #print "# Rows small : ",len(df.index[df[x]<np.float(stats.iloc[5,continuous.index(x)])-1.5*(np.float(stats.iloc[6,continuous.index(x)])-np.float(stats.iloc[5,continuous.index(x)]))].tolist())
	listoutliers.append(df.index[df[x]<np.float(stats.iloc[5,continuous.index(x)])-1.5*(np.float(stats.iloc[6,continuous.index(x)])-np.float(stats.iloc[5,continuous.index(x)]))].tolist())
listoutliers = list(set([item for sublist in listoutliers for item in sublist]))
print "# of Outliers : ",len(listoutliers)
dataset1 = list(set([item for sublist in [listoutliers,baddata] for item in sublist]))
print "# of instances to remove : ",len(dataset1)

#### Visualizations Raw Data ####

## Histogram Binary Target Classes##
plt.figure(1,figsize =[10,10])
plt.hist(df['Y'], [-.3,.3,.7,1.3], histtype ='bar', color = 'green')
plt.xlabel("Will customer default on credit bill payment?")
plt.xticks(np.arange(0,2,1),['No == 0','Yes == 1'])
plt.ylabel("Frequency")
plt.title('Raw Data Histogram : Binary Target Classes Y')
plt.grid()
plt.savefig('Raw Target Variable Y.png')

## Histogram Approved Credit Amount##
plt.figure(2,figsize =[10,10])
plt.hist(df['X1'], histtype ='bar', color = 'blue')
plt.xlabel("Approved Credit Amount X1")
plt.ylabel("Frequency")
plt.title('Raw Data Histogram :Approved Credit Amount X1')
plt.grid()
plt.savefig('Raw Approved Credit Amount Histogram.png')

## Histogram April Credit Bill Amount Due##
plt.figure(3,figsize =[10,10])
plt.hist(df['X12'], histtype ='bar', color = 'red', align = 'right')
plt.xlabel("April Credit Bill Amount Due X12")
plt.ylabel("Frequency")
plt.title('Raw Data Histogram : April Credit Bill Amount Due X12')
plt.grid()
plt.savefig('Raw April Credit Bill Due.png')

## Histogram April Credit Repayment Amount##
plt.figure(4,figsize =[10,10])
plt.hist(df['X18'], histtype ='bar', color = 'red')
plt.xlabel("April Credit Repayment Amount X18")
plt.ylabel("Frequency")
plt.title('Raw Data Histogram : April Credit Repayment Amount X18')
plt.grid()
plt.savefig('Raw April Credit Repayment Amount.png')

## Histogram Educational Level##
plt.figure(5,figsize =[10,10])
plt.hist(df['X3'], histtype ='bar', color = 'red', align = 'mid')
plt.xlabel("Educational Level X3")
plt.xticks(np.arange(0,7,1),['0','Graduate School','University','High School','Others'],rotation='vertical')
plt.ylabel("Frequency")
plt.title('Raw Data Histogram : Educational Level X3')
plt.grid()
plt.savefig('Raw Educational Level.png')

## Boxplot Credit Approved ##
plt.figure(6,figsize =[10,10])
plt.boxplot(df['X1'])
plt.xlabel("Feature  X1")
plt.ylabel("Approved Credit Amount",rotation = 90)
plt.title('Raw Data Boxplot : Approved Credit Amount X1')
plt.text(1.15,150000,"Mean", horizontalalignment='center',verticalalignment='top',multialignment='center')
plt.text(1.1,240000,"75% Quartile", horizontalalignment='left',verticalalignment='center',multialignment='center')
plt.text(1.1,50000,"25% Quartile", horizontalalignment='left',verticalalignment='center',multialignment='center')
plt.text(.9,140000,"IQR", rotation = 90,horizontalalignment='right',verticalalignment='center',multialignment='center')
plt.plot([.835,.835],[0,525000],'r-')
plt.plot([.8,.87],[525000,525000],'r-')
plt.text(.75,600000,"1.5 x IQR \n Upper Limit", horizontalalignment='left',verticalalignment='center',multialignment='center')
plt.text(.8,240000,"1.5 x IQR", rotation = 90,horizontalalignment='right',verticalalignment='center',multialignment='center')
plt.plot([1.1,1.1],[525000,1000000],'b-')
plt.plot([1.07,1.13],[525000,525000],'b-')
plt.plot([1.07,1.13],[1000000,1000000],'b-')
plt.text(1.07,780000,"Outliers", rotation = 90,horizontalalignment='right',verticalalignment='center',multialignment='center')
plt.grid()
plt.savefig('Raw Approved Credit Amount Boxplot.png')

#### Removing Incongruent Data Points only ####
dataset2 = pd.DataFrame()
dataset2 = df.drop(baddata).reset_index()

#### Removing Incongruent and Outliers Data Points ####
newdf = pd.DataFrame()
newdf = df.drop(dataset1).reset_index()

#### Curated Data Statistics ####
##Continuous##
num_instances = len(newdf)
columns =["Variable", "Maximum", "Minimum", "Mean", "Median", "25% Quartile", "75% Quartile"]
statsarray = []
for x in continuous:
	statsarray.append([x, np.round(max(newdf[x]),2), np.round(min(newdf[x]),2), np.round(np.mean(newdf[x]),2), np.round(np.median(newdf[x]),2), np.round(np.percentile(newdf[x],25)),np.round(np.percentile(newdf[x],75))])

arraystats = np.array(statsarray).T.tolist()
stats = pd.DataFrame(arraystats,columns)
writer = pd.ExcelWriter("Continuous Curated Data Basic Statistics.xlsx")
stats.to_excel(writer)
writer.save()

##Categorical##
columns =["Variable", "Maximum", "Minimum", "Median"]
statsarray = []
for x in categorical:
	statsarray.append([x, max(newdf[x]), min(newdf[x]), np.median(newdf[x])])

arraystats = np.array(statsarray).T.tolist()
statscat = pd.DataFrame(arraystats,columns)
writer = pd.ExcelWriter("Categorical Curated Data Basic Statistics.xlsx")
statscat.to_excel(writer)
writer.save()

#### Visualizations Curated Data ####

## Histogram Binary Target Classes##
plt.figure(7,figsize =[10,10])
plt.hist(newdf['Y'], [-.3,.3,.7,1.3], histtype ='bar', color = 'green')
plt.xlabel("Will customer default on credit bill payment?")
plt.xticks(np.arange(0,2,1),['No == 0','Yes == 1'])
plt.ylabel("Frequency")
plt.title('Curated Data Histogram : Binary Target Classes Y')
plt.grid()
plt.savefig('Curated  Target Variable Y.png')

## Histogram Approved Credit Amount##
plt.figure(8,figsize =[10,10])
plt.hist(newdf['X1'], histtype ='bar', color = 'blue')
plt.xlabel("Approved Credit Amount X1")
plt.ylabel("Frequency")
plt.title('Curated  Data Histogram :Approved Credit Amount X1')
plt.grid()
plt.savefig('Curated  Approved Credit Amount Histogram.png')

## Histogram April Credit Bill Amount Due##
plt.figure(9,figsize =[10,10])
plt.hist(newdf['X12'], histtype ='bar', color = 'red', align = 'right')
plt.xlabel("April Credit Bill Amount Due X12")
plt.ylabel("Frequency")
plt.title('Curated  Data Histogram : April Credit Bill Amount Due X12')
plt.grid()
plt.savefig('Curated  April Credit Bill Due.png')

## Histogram April Credit Repayment Amount##
plt.figure(10,figsize =[10,10])
plt.hist(newdf['X18'], histtype ='bar', color = 'red')
plt.xlabel("April Credit Repayment Amount X18")
plt.ylabel("Frequency")
plt.title('Curated  Data Histogram : April Credit Repayment Amount X18')
plt.grid()
plt.savefig('Curated April Credit Repayment Amount.png')

## Histogram Educational Level##
plt.figure(11,figsize =[10,10])
plt.hist(newdf['X3'], histtype ='bar', color = 'red', align = 'mid')
plt.xlabel("Educational Level X3")
plt.xticks(np.arange(0,7,1),['0','Graduate School','University','High School','Others'],rotation='vertical')
plt.ylabel("Frequency")
plt.title('Curated  Data Histogram : Educational Level X3')
plt.grid()
plt.savefig('Curated  Educational Level.png')

## Boxplot Credit Approved ##
plt.figure(12,figsize =[10,10])
plt.boxplot(newdf['X1'])
plt.xlabel("Feature  X1")
plt.ylabel("Approved Credit Amount",rotation = 90)
plt.title('Curated  Data Boxplot : Approved Credit Amount X1')
plt.text(1.15,110000,"Mean", horizontalalignment='center',verticalalignment='top',multialignment='center')
plt.text(1.1,200000,"75% Quartile", horizontalalignment='left',verticalalignment='center',multialignment='center')
plt.text(1.1,50000,"25% Quartile", horizontalalignment='left',verticalalignment='center',multialignment='center')
plt.text(.9,110000,"IQR", rotation = 90,horizontalalignment='right',verticalalignment='center',multialignment='center')
plt.grid()
plt.savefig('Curated  Approved Credit Amount Boxplot.png')


#### Dimensionality Reduction ####

## Repayment Data Covariance and PCA ##
covar_matrix = PCA()
covar_matrix.fit(newdf.iloc[:,17:23])
variance = covar_matrix.explained_variance_ratio_*100
var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3))
print "Variance % X18 - X23 : ", [np.round(x,2) for x in variance]

pca = PCA(n_components = 1)
pc = pca.fit_transform(newdf.iloc[:,17:23])
prep = pd.DataFrame(pc,columns=['Credit Repayment'])



## Due Bill Data Covariance and PCA ##
covar_matrix = PCA(n_components = 6)
covar_matrix.fit(newdf.iloc[:,11:17])
variance = covar_matrix.explained_variance_ratio_*100
var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)
print "Variance % X12 - X17 : ",[np.round(x,2) for x in variance]

pca = PCA(n_components = 1)
pc = pca.fit_transform(newdf.iloc[:,11:17])
pdue = pd.DataFrame(pc,columns=['Credit Bill Due'])

####  Continuous Monetary Variables Due and Payback Covariance and PCA ####
covar_matrix = PCA(n_components = 12)
covar_matrix.fit(newdf.iloc[:,11:23])
variance = covar_matrix.explained_variance_ratio_*100
var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)
print "Variance % X12 - X23 : ",[np.round(x,2) for x in variance]

pca = PCA(n_components = 1)
pc = pca.fit_transform(newdf.iloc[:,11:23])
mone = pd.DataFrame(pc,columns=['Monetary Variable'])


## Payment Delay Data Covariance and PCA ##
covar_matrix = PCA(n_components = 6)
covar_matrix.fit(newdf.iloc[:,5:11])
variance = covar_matrix.explained_variance_ratio_*100
var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)
print "Variance % X6 - X11 : ",[np.round(x,2) for x in variance]

pca = PCA(n_components = 1)
pc = pca.fit_transform(newdf.iloc[:,5:11])
plap = pd.DataFrame(pc,columns=['Late Payment'])

## Person Non-monetary Data Covariance and PCA ##
covar_matrix = PCA(n_components = 4)
covar_matrix.fit(newdf.iloc[:,1:5])
variance = covar_matrix.explained_variance_ratio_*100
var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)
print "Variance % X2 - X5 : ",[np.round(x,2) for x in variance]

pca = PCA(n_components = 1)
pnm = pca.fit_transform(newdf.iloc[:,1:5])
pnm = pd.DataFrame(pnm,columns=['Non-monetary Personal'])

#### Dimensionally Reduced Data Set ####

## 
df_reduced = pd.concat([newdf['X1'],plap,pdue,prep,newdf['Y']],axis=1)
df_reducedmon = pd.concat([newdf['X1'],plap,mone,newdf['Y']],axis=1)

### Heat Maps #####
correlation = df.corr()
plt.figure(num = 13, figsize=(14,10))
sb.heatmap(correlation)
plt.xlabel("Raw Attributes")
plt.xlabel("Raw Attributes")
plt.title('Heat Map Raw Dataset')
plt.xticks(rotation = 90)
plt.yticks(rotation = 0)
plt.savefig('Heat Map Raw Dataset.png')

correlation = newdf.corr()
plt.figure(num = 14, figsize=(14,10))
sb.heatmap(correlation)
plt.xlabel("Curated Attributes")
plt.xlabel("Curated Attributes")
plt.title('Heat Map Curated Dataset')
plt.xticks(rotation = 90)
plt.yticks(rotation = 0)
plt.savefig('Heat Map Curated Dataset.png')

correlation = df_reduced.corr()
plt.figure(num = 15, figsize=(14,10))
sb.heatmap(correlation)
plt.xlabel("4 dimensional Attributes")
plt.xlabel("4 dimensional Attributes")
plt.title('Heat Map Dimensionally Reduced Dataset')
plt.xticks(rotation = 0)
plt.yticks(rotation = 0)
plt.savefig('Heat Map Dimensionally Reduced Dataset.png')

correlation = df_reducedmon.corr()
plt.figure(num = 16, figsize=(14,10))
sb.heatmap(correlation)
plt.xlabel("3 Dimensional Attributes")
plt.xlabel("3 Dimensional Attributes")
plt.title('Heat Map Raw Dataset')
plt.xticks(rotation = 0)
plt.yticks(rotation = 0)
plt.savefig('Heat Map Extreme Dimensionality Reduction Dataset.png')


# sb.pairplot(df_reduced.iloc[:,:].sample(frac=.5))
# 
# plt.savefig('Correlation plots 4 Dimensional Attributes Dataset.png')
# 
# sb.pairplot(df_reducedmon.iloc[:,:].sample(frac=.5))
# sb.figure(num = 18, figsize=(16,16))
# 
# 
# plt.savefig('Correlation plots 3 Dimensional Attributes Dataset.png')


 
train, test = train_test_split(df_reducedmon)

print train.shape, test.shape

y_train = train['Y']
x_train = train.drop(['Y'],axis = 1)
y_test = test['Y']
x_test = test.drop(['Y'],axis = 1)


rf = RandomForestClassifier()
rf.fit(x_train,y_train)
print "features sorted by their score:"
print sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), x_train), reverse=True) 


# #/Users/nitishsanghi/documents/projects/courses/machine-learning-master/projects/capstone/CapstoneWorks

dftab = pd.DataFrame(columns = ['Classifier', 'Training Score', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'])

# Logistic Regression
# lr = LogisticRegression()
# lr = lr.fit(x_train,y_train)
# predicted = pd.DataFrame(lr.predict(x_test))
# fpr ,tpr, thres = metrics.roc_curve(y_test,predicted)
# dftab = dftab.append({'Classifier': 'LR','Training Score': lr.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1), 'AUC': metrics.auc(fpr,tpr)}, ignore_index=True)
# 
#  
# #Decision Tree
# dt = tree.DecisionTreeClassifier(max_depth = 5)
# dt = dt.fit(x_train,y_train)
# predicted = pd.DataFrame(dt.predict(x_test))
# fpr ,tpr, thres = metrics.roc_curve(y_test,predicted)
# dftab = dftab.append({'Classifier': 'DT','Training Score': dt.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1), 'AUC': metrics.auc(fpr,tpr)}, ignore_index=True)
 
#Random Forest
rocauc = pd.DataFrame()


parameters = np.linspace(1,100,100)
for x in parameters:
	rf = RandomForestClassifier(n_estimators = int(x))
	rf.fit(x_train,y_train)
	predicted = pd.DataFrame(rf.predict(x_test))
	fpr ,tpr, thres = metrics.roc_curve(y_train,rf.predict(x_train))
	rocauc['AUC Train'] = metrics.auc(fpr,tpr)
	fpr ,tpr, thres = metrics.roc_curve(y_test,predicted)
	rocauc['AUC Test'] = metrics.auc(fpr,tpr)
	dftab = dftab.append({'Classifier': 'RF','Training Score': rf.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1), 'AUC': metrics.auc(fpr,tpr)}, ignore_index=True)

plt.plot(rocauc["AUC Train"])
plt.plot(rocauc["AUC Test"])
plt.show()

# Support Vector Machine
# svm = SVC()
# svm = svm.fit(x_train,y_train)
# predicted = pd.DataFrame(svm.predict(x_test))
# fpr ,tpr, thres = metrics.roc_curve(y_test,predicted)
# dftab = dftab.append({'Classifier': 'SVM','Training Score': svm.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1), 'AUC': metrics.auc(fpr,tpr)}, ignore_index=True)
#  
# KNN
# knn = KNeighborsClassifier(n_neighbors = 5)
# knn = knn.fit(x_train,y_train)
# predicted = pd.DataFrame(knn.predict(x_test))
# fpr ,tpr, thres = metrics.roc_curve(y_test,predicted)
# dftab = dftab.append({'Classifier': 'KNN','Training Score': knn.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1), 'AUC': metrics.auc(fpr,tpr)}, ignore_index=True)
# 
# Two Class Bayes
# bm = GaussianNB()
# bm = bm.fit(x_train,y_train)
# predicted = pd.DataFrame(bm.predict(x_test))
# fpr ,tpr, thres = metrics.roc_curve(y_test,predicted)
# dftab = dftab.append({'Classifier': 'Bayes','Training Score': bm.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1), 'AUC': metrics.auc(fpr,tpr)}, ignore_index=True)
# 
# Stochastic Gradient Descent
# sdg = SGDClassifier()
# sdg = sdg.fit(x_train,y_train)
# predicted = pd.DataFrame(sdg.predict(x_test))
# fpr ,tpr, thres = metrics.roc_curve(y_test,predicted)
# dftab = dftab.append({'Classifier': 'SGD','Training Score': sdg.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1), 'AUC': metrics.auc(fpr,tpr)}, ignore_index=True)
# 
# Perceptron
# p = Perceptron()
# p = p.fit(x_train,y_train)
# predicted = pd.DataFrame(p.predict(x_test))
# fpr ,tpr, thres = metrics.roc_curve(y_test,predicted)
# dftab = dftab.append({'Classifier': 'Perceptron','Training Score': p.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1), 'AUC': metrics.auc(fpr,tpr)}, ignore_index=True)
# 
# Passive Aggressive Classifier
# pac = PassiveAggressiveClassifier()
# pac = pac.fit(x_train,y_train)
# predicted = pd.DataFrame(pac.predict(x_test))
# fpr ,tpr, thres = metrics.roc_curve(y_test,predicted)
# dftab = dftab.append({'Classifier': 'PAC','Training Score': pac.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1), 'AUC': metrics.auc(fpr,tpr)}, ignore_index=True)
#  
# Linear Discriminant Analysis
# lda = LinearDiscriminantAnalysis()
# lda = lda.fit(x_train,y_train)
# predicted = pd.DataFrame(lda.predict(x_test))
# fpr ,tpr, thres = metrics.roc_curve(y_test,predicted)
# dftab = dftab.append({'Classifier': 'LDA','Training Score': lda.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1),'AUC': metrics.auc(fpr,tpr)}, ignore_index=True)
#  
# Quadratic Discriminant Analysis
# qda = QuadraticDiscriminantAnalysis()
# qda = qda.fit(x_train,y_train)
# predicted = pd.DataFrame(qda.predict(x_test))
# fpr ,tpr, thres = metrics.roc_curve(y_test,predicted)
# dftab = dftab.append({'Classifier': 'QDA','Training Score': qda.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1),'AUC': metrics.auc(fpr,tpr)}, ignore_index=True)
# 
# Gradient Boosting Classifier
# gbc = GradientBoostingClassifier()
# gbc = gbc.fit(x_train,y_train)
# predicted = pd.DataFrame(gbc.predict(x_test))
# fpr ,tpr, thres = metrics.roc_curve(y_test,predicted)
# dftab = dftab.append({'Classifier': 'GBC','Training Score': gbc.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1), 'AUC': metrics.auc(fpr,tpr)}, ignore_index=True)
# 
# AdaBoost Classifier
# abc = AdaBoostClassifier()
# abc = abc.fit(x_train,y_train)
# predicted = pd.DataFrame(abc.predict(x_test))
# fpr ,tpr, thres = metrics.roc_curve(y_test,predicted)
# dftab = dftab.append({'Classifier': 'ABC','Training Score': abc.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1),'AUC': metrics.auc(fpr,tpr)}, ignore_index=True)
# 
# Extra Trees Classifier
# etc = ExtraTreesClassifier()
# etc = etc.fit(x_train,y_train)
# predicted = pd.DataFrame(etc.predict(x_test))
# fpr ,tpr, thres = metrics.roc_curve(y_test,predicted)
# dftab = dftab.append({'Classifier': 'ETC','Training Score': etc.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1),'AUC': metrics.auc(fpr,tpr)}, ignore_index=True)
# 
# Multi-layer Perceptron Classifier
# mlpc = MLPClassifier(solver='lbfgs', alpha=1e-5,  hidden_layer_sizes=(5, 2), random_state=1)
# mlpc = mlpc.fit(x_train,y_train)
# predicted = pd.DataFrame(mlpc.predict(x_test))
# fpr ,tpr, thres = metrics.roc_curve(y_test,predicted)
# dftab = dftab.append({'Classifier': 'MLPC','Training Score': mlpc.score(x_train,y_train),'Accuracy': metrics.accuracy_score(y_test,predicted),'Precision':metrics.precision_score(y_test,predicted,pos_label=1),'Recall': metrics.recall_score(y_test,predicted,pos_label=1),'F1': metrics.f1_score(y_test, predicted, pos_label=1),'AUC': metrics.auc(fpr,tpr)}, ignore_index=True)

#print dftab.sort_values(by='AUC', ascending=False)


#  = .fit(x_train,y_train)
# predicted = pd.DataFrame(.predict(x_test))
# print "Train Score : ", .score(x_train,y_train)
# print "Test Data Score : ", metrics.accuracy_score(y_test,predicted)
# print "Precision Score : ", metrics.precision_score(y_test,predicted,pos_label=1)
# print "Recall Score : ", metrics.recall_score(y_test,predicted,pos_label=1)
# print "F1 Score : ", metrics.f1_score(y_test, predicted, pos_label=1)

#/Users/nitishsanghi/documents/projects/courses/machine-learning-master/projects/capstone/CapstoneWorks
