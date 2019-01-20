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
from sklearn.model_selection import cross_val_score ,train_test_split 
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

Baddata = []

print "Sum of target class variable Y", np.sum(df['Y'])

Baddata.append(df.index[df['X1']<df['X12']].tolist())
print len(df[df['X1']<df['X12']])
Baddata.append(df.index[df['X1']<df['X13']].tolist())
print len(df[df['X1']<df['X13']])
Baddata.append(df.index[df['X1']<df['X14']].tolist())
print len(df[df['X1']<df['X14']])
Baddata.append(df.index[df['X1']<df['X15']].tolist())
print len(df[df['X1']<df['X15']])
Baddata.append(df.index[df['X1']<df['X16']].tolist())
print len(df[df['X1']<df['X16']])
Baddata.append(df.index[df['X1']<df['X17']].tolist())
print len(df[df['X1']<df['X17']])

Baddata = list(set([item for sublist in Baddata for item in sublist]))
print "# bad data points",len(Baddata)

#### Statistics ####
columns =["Variable", "Maximum", "Minimum", "Mean", "Median", "25% Quartile", "75% Quartile", "Variance", "Standard Deviation", "Skewness", "Kurtosis"]
statsarray = []
for x in continuous:
	statsarray.append([x, np.round(max(df[x]),2), np.round(min(df[x]),2), np.round(np.mean(df[x]),2), np.round(np.median(df[x]),2), np.round(np.percentile(df[x],25)),np.round(np.percentile(df[x],75)), np.round(np.var(df[x]),2), np.round(np.std(df[x]),2), np.round(ss.skew(df[x]),2), np.round(ss.kurtosis(df[x]),2)])

arraystats = np.array(statsarray).T.tolist()
stats = pd.DataFrame(arraystats,columns)
writer = pd.ExcelWriter("Continuous Data Basic Statistics.xlsx")
stats.to_excel(writer)
writer.save()

#print df['X1'].quantile([0.25,0.5,0.75])

# plt.subplot(3,1,1)
# plt.boxplot(df['X1'])
# plt.subplot(3,1,2)
# plt.boxplot([df['X12'],df['X13'],df['X14'],df['X15'],df['X16'],df['X17']])
# plt.subplot(3,1,3)
# plt.boxplot([df['X18'],df['X19'],df['X20'],df['X21'],df['X22'],df['X23']]) 
# plt.figure(1)
# plt.subplot(3,1,1)
# plt.hist(df['X1'],bins = 10)
# plt.subplot(3,1,2)
# plt.hist(df['X12'],bins = 10)
# plt.subplot(3,1,3)
# plt.hist(df['X18'],bins = 10) 


#plt.show()


#### Removing Outliers ####
listoutliers = []
for x in continuous:
	#print " \n "
	#print "# Rows to be deleted : ",len(df.index[df[x]>np.float(stats.iloc[6,continuous.index(x)])+1.5*(np.float(stats.iloc[6,continuous.index(x)])-np.float(stats.iloc[5,continuous.index(x)]))].tolist())
	listoutliers.append(df.index[df[x]>np.float(stats.iloc[6,continuous.index(x)])+1.5*(np.float(stats.iloc[6,continuous.index(x)])-np.float(stats.iloc[5,continuous.index(x)]))].tolist())
	#print "# Rows small : ",len(df.index[df[x]<np.float(stats.iloc[5,continuous.index(x)])-1.5*(np.float(stats.iloc[6,continuous.index(x)])-np.float(stats.iloc[5,continuous.index(x)]))].tolist())
	listoutliers.append(df.index[df[x]<np.float(stats.iloc[5,continuous.index(x)])-1.5*(np.float(stats.iloc[6,continuous.index(x)])-np.float(stats.iloc[5,continuous.index(x)]))].tolist())

listoutliers = list(set([item for sublist in listoutliers for item in sublist]))
print "# Outliers : ", len(listoutliers)
listoutliers = list(set([item for sublist in [listoutliers,Baddata] for item in sublist]))
print "# Outliers + Bad Data : ", len(listoutliers)
#print "Deleted a lot",len(listoutliers)

newdf = pd.DataFrame()
newdf = df.drop(listoutliers)
print "New df # : ", len(newdf)

columns =["Variable", "Maximum", "Minimum", "Mean", "Median", "25% Quartile", "75% Quartile", "Variance", "Standard Deviation", "Skewness", "Kurtosis"]
statsarray2 = []
for x in continuous:
	statsarray2.append([x, np.round(max(newdf[x]),2), np.round(min(newdf[x]),2), np.round(np.mean(newdf[x]),2), np.round(np.median(newdf[x]),2), np.round(np.percentile(newdf[x],25)),np.round(np.percentile(newdf[x],75)), np.round(np.var(newdf[x]),2), np.round(np.std(newdf[x]),2), np.round(ss.skew(newdf[x]),2), np.round(ss.kurtosis(newdf[x]),2)])

arraystats2 = np.array(statsarray2).T.tolist()
stats2 = pd.DataFrame(arraystats2,columns)
writer = pd.ExcelWriter("Continuous Data Basic Statistics 2.xlsx")
stats2.to_excel(writer)
writer.save()

print "New df # : ", len(newdf)


# plt.figure(2)
# plt.subplot(3,1,1)
# plt.hist(newdf['X1'],bins = 10)
# plt.subplot(3,1,2)
# plt.hist(newdf['X12'],bins = 10)
# plt.subplot(3,1,3)
# plt.hist(newdf['X18'],bins = 10) 
# plt.show()
# 
# plt.figure(3)
# plt.subplot(3,1,1)
# plt.boxplot(newdf['X1'])
# plt.subplot(3,1,2)
# plt.boxplot([newdf['X12'],newdf['X13'],newdf['X14'],newdf['X15'],newdf['X16'],newdf['X17']])
# plt.subplot(3,1,3)
# plt.boxplot([newdf['X18'],newdf['X19'],newdf['X20'],newdf['X21'],newdf['X22'],newdf['X23']])
# plt.show()
# 
# for x in categorical:
# 	plt.figure(4)
# 	plt.hist(newdf[x])
# 	plt.title(x)
# 	plt.show()
	


#### Data Exploration
#hist, bin_edges = np.histogram(df['Y']+1) # Histogram 
#plt.hist(df['Y'])
# plt.xlabel('Default Classes Yes = 1 and No = 0')
# plt.ylabel('Frequency')
# plt.title('Histogram of binary class instances')
# #plt.xticks(np.arange(0,4,step = 1),("",'No','Yes'))
#plt.savefig("Class Histogram",format = 'png')
#print "Max value : ", max(df['Y']),'  ',"Min value : ",min(df['Y'])


#### Repayment Data ####
covar_matrix = PCA(n_components = 6)
covar_matrix.fit(newdf.iloc[:,16:23])
variance = covar_matrix.explained_variance_ratio_
var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)
print "Variance Ratio X17 - X23 : ", var #[ 66.8  78.1  84.   88.7  92.6  96.3]

print "New df # : ", len(newdf)
##### PCA because they are correlated and dimensional reduction always helps
pca = PCA(n_components = 1)
pc = pca.fit_transform(newdf.iloc[:,17:23])
prep=newdf['X17']
print "printing prep",prep
print "printing pc",pc
prep['0'] = pc
#prep = pd.DataFrame(pc,columns=['pc1'])

#### Due Bill Data ####
covar_matrix = PCA(n_components = 6)
covar_matrix.fit(newdf.iloc[:,11:17])
variance = covar_matrix.explained_variance_ratio_
var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)
print "Variance Ratio X12 - X17 : ",var
print "New df # : ", len(newdf)
###### PCA because they are correlated and dimensional reduction always helps
# pca = PCA(n_components = 2)
# pc = pca.fit_transform(df.iloc[:,11:17])
# pdue = pd.DataFrame(pc,columns=['dpc1','dpc2'])

pca = PCA(n_components = 2)
pc = pca.fit_transform(newdf.iloc[:,11:17])
pdue = pd.DataFrame(pc,columns=['dpc1','dpc2'])
print "New df # : ", len(newdf)

#### Payment Delay Data ####
covar_matrix = PCA(n_components = 6)
covar_matrix.fit(newdf.iloc[:,5:11])
variance = covar_matrix.explained_variance_ratio_
var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)
print "Variance Ratio X6 - X11 : ",var

###### PCA because they are correlated and dimensional reduction always helps
pca = PCA(n_components = 4)
pc = pca.fit_transform(newdf.iloc[:,5:11])
plap = pd.DataFrame(pc,columns=['lpc1','lpc2','lpc3','lpc4'])

#### Person Non-monetary ####
covar_matrix = PCA(n_components = 4)
covar_matrix.fit(newdf.iloc[:,1:5])
variance = covar_matrix.explained_variance_ratio_
var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)
print "Variance Ratio X2 - X5 : ",var
print "New df # : ", len(newdf)

print "New df # : ", plap.iloc[:10,:5]
print "New df # : ", pdue.iloc[:10,:5]
print "New df # : ", prep

print "New df # : ", len(newdf['Y'])
print "New df # : ", len(newdf['X1'])
print newdf.iloc[:10,:5]
df_reduced = pd.concat([newdf['X1'],plap,pdue,prep,newdf['Y']],axis=1)
print df_reduced.iloc[:10,:5]

# covar_matrix = PCA(n_components = 2)
# covar_matrix.fit(df_reduced.iloc[:,0:2])
# variance = covar_matrix.explained_variance_ratio_
# var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)
# print "Variance Ratio Dimensionally reduced data : ",var

#### Plots #####

#plt.show()
# correlation = newdf.corr()
# plt.figure(num = 5, figsize=(14,10))
# sb.heatmap(correlation)
# sb.pairplot(newdf.iloc[:,12:15].sample(frac=.5))
# sb.pairplot(newdf.iloc[:,6:9].sample(frac=.5))
# sb.pairplot(pdue)
# sb.pairplot(prep)

# df_reduced = pd.concat([df.iloc[:,0:5],pdue,prep,df['Y']],axis=1)
plt.show()
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


#/Users/nitishsanghi/documents/projects/courses/machine-learning-master/projects/capstone/CapstoneWorks
