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
from sklearn.model_selection import cross_val_score ,train_test_split 
from sklearn import metrics  
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, Perceptron
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import scipy.stats as ss
matplotlib.use('Agg')

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
writer = pd.ExcelWriter("Statistics/Continuous Raw Data Basic Statistics.xlsx")
stats.to_excel(writer)
writer.save()

##Categorical##
columns =["Variable", "Maximum", "Minimum", "Median"]
statsarray = []
for x in categorical:
	statsarray.append([x, max(df[x]), min(df[x]), np.median(df[x])])

arraystats = np.array(statsarray).T.tolist()
statscat = pd.DataFrame(arraystats,columns)
writer = pd.ExcelWriter("Statistics/Categorical Raw Data Basic Statistics.xlsx")
statscat.to_excel(writer)
writer.save()

## Raw Data Size ##
# print "Total # of Instances : ",num_instances
# print "Sum of target class variable Y : ", len(df[df['Y']==1])
# print " # of 0 target class variable Y : ", len(df[df['Y']==0])
# print "Legitimate instances : ",  len(df[df['Y']==0])+np.sum(df['Y'])

## Bad Categorical Variables ##
baddata = []

baddata.append(df.index[df['X3']>4].tolist())
#print "X3 incongruent data # : ", len(baddata[0])
baddata.append(df.index[df['X1']<df['X12']].tolist())
#print "Points Approved Credit < Credit Bill Due X12  :",len(df[df['X1']<df['X12']])
baddata.append(df.index[df['X1']<df['X13']].tolist())
#print "Points Approved Credit < Credit Bill Due X13  :", len(df[df['X1']<df['X13']])
baddata.append(df.index[df['X1']<df['X14']].tolist())
#print "Points Approved Credit < Credit Bill Due X14  :", len(df[df['X1']<df['X14']])
baddata.append(df.index[df['X1']<df['X15']].tolist())
#print "Points Approved Credit < Credit Bill Due X15  :", len(df[df['X1']<df['X15']])
baddata.append(df.index[df['X1']<df['X16']].tolist())
#print "Points Approved Credit < Credit Bill Due X16  :", len(df[df['X1']<df['X16']])
baddata.append(df.index[df['X1']<df['X17']].tolist())
#print "Points Approved Credit < Credit Bill Due X17  :", len(df[df['X1']<df['X17']])
#print "# Incongruent Variables ", len(baddata)
baddata = list(set([item for sublist in baddata for item in sublist]))
#print "Total # bad points :  ", len(baddata)

#### Removing Outliers ####
listoutliers = []
for x in continuous:
	#print " \n ",x
	#print "# Rows to be deleted : ",len(df.index[df[x]>np.float(stats.iloc[6,continuous.index(x)])+1.5*(np.float(stats.iloc[6,continuous.index(x)])-np.float(stats.iloc[5,continuous.index(x)]))].tolist())
	listoutliers.append(df.index[df[x]>np.float(stats.iloc[6,continuous.index(x)])+1.5*(np.float(stats.iloc[6,continuous.index(x)])-np.float(stats.iloc[5,continuous.index(x)]))].tolist())
	# #print "# Rows small : ",len(df.index[df[x]<np.float(stats.iloc[5,continuous.index(x)])-1.5*(np.float(stats.iloc[6,continuous.index(x)])-np.float(stats.iloc[5,continuous.index(x)]))].tolist())
	listoutliers.append(df.index[df[x]<np.float(stats.iloc[5,continuous.index(x)])-1.5*(np.float(stats.iloc[6,continuous.index(x)])-np.float(stats.iloc[5,continuous.index(x)]))].tolist())
listoutliers = list(set([item for sublist in listoutliers for item in sublist]))
#print "# of Outliers : ",len(listoutliers)
listoutliers = list(set([item for sublist in [listoutliers,baddata] for item in sublist]))
#print "# of instances to remove : ",len(listoutliers)

#### Visualizations Raw Data ####

## Histogram Binary Target Classes##
plt.figure(1)
plt.hist(df['Y'], [-.3,.3,.7,1.3], histtype ='bar', color = 'green')
plt.xlabel("Will customer default on credit bill payment?")
plt.xticks(np.arange(0,2,1),['No == 0','Yes == 1'])
plt.ylabel("Frequency")
plt.title('Raw Data Histogram : Binary Target Classes')
plt.grid()
plt.savefig('Histograms/Raw Target Variable Y.png')

## Histogram Approved Credit Amount##
plt.figure(2)
plt.hist(df['X1'], histtype ='bar', color = 'blue')
plt.xlabel("Approved Credit Amount")
plt.ylabel("Frequency")
plt.title('Raw Data Histogram :Approved Credit Amount')
plt.grid()
plt.savefig('Histograms/Raw Approved Credit Amount Histogram.png')

## Histogram April Credit Bill Amount Due##
plt.figure(3)
plt.hist(df['X12'], histtype ='bar', color = 'red', align = 'right')
plt.xlabel("April Credit Bill Amount Due")
plt.ylabel("Frequency")
plt.title('Raw Data Histogram : April Credit Bill Amount Due')
plt.grid()
plt.savefig('Histograms/Raw April Credit Bill Due.png')

## Histogram April Credit Repayment Amount##
plt.figure(4)
plt.hist(df['X18'], histtype ='bar', color = 'red')
plt.xlabel("April Credit Repayment Amount")
plt.ylabel("Frequency")
plt.title('Raw Data Histogram : April Credit Repayment Amount')
plt.grid()
plt.savefig('Histograms/Raw April Credit Repayment Amount.png')

## Histogram Educational Level##
plt.figure(5)
plt.hist(df['X3'], histtype ='bar', color = 'red', align = 'mid')
plt.xlabel("Educational Level")
plt.xticks(np.arange(0,7,1),['0','Graduate School','University','High School','Others'],rotation='vertical')
plt.ylabel("Frequency")
plt.title('Raw Data Histogram : Educational Level')
plt.grid()
plt.savefig('Histograms/Raw Educational Level.png')

## Boxplot Credit Approved ##
plt.figure(6)
plt.boxplot(df['X1'])
plt.xlabel("Feature  X1")
plt.ylabel("Approved Credit Amount",rotation = 90)
plt.title('Raw Data Boxplot : Approved Credit Amount')
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
plt.savefig('Boxplots/Raw Approved Credit Amount Boxplot.png')


#### Removing Incongruent and Outliers Data Points ####
newdf = pd.DataFrame()
newdf = df.drop(listoutliers).reset_index()

#### Curated Data Statistics ####
##Continuous##
num_instances = len(newdf)
columns =["Variable", "Maximum", "Minimum", "Mean", "Median", "25% Quartile", "75% Quartile"]
statsarray = []
for x in continuous:
	statsarray.append([x, np.round(max(newdf[x]),2), np.round(min(newdf[x]),2), np.round(np.mean(newdf[x]),2), np.round(np.median(newdf[x]),2), np.round(np.percentile(newdf[x],25)),np.round(np.percentile(newdf[x],75))])

arraystats = np.array(statsarray).T.tolist()
stats = pd.DataFrame(arraystats,columns)
writer = pd.ExcelWriter("Statistics/Continuous Curated Data Basic Statistics.xlsx")
stats.to_excel(writer)
writer.save()

##Categorical##
columns =["Variable", "Maximum", "Minimum", "Median"]
statsarray = []
for x in categorical:
	statsarray.append([x, max(newdf[x]), min(newdf[x]), np.median(newdf[x])])

arraystats = np.array(statsarray).T.tolist()
statscat = pd.DataFrame(arraystats,columns)
writer = pd.ExcelWriter("Statistics/Categorical Curated Data Basic Statistics.xlsx")
statscat.to_excel(writer)
writer.save()

newdf = newdf.drop('index',axis = 1)
#### Visualizations Curated Data ####

## Histogram Binary Target Classes##
plt.figure(7)
plt.hist(newdf['Y'], [-.3,.3,.7,1.3], histtype ='bar', color = 'green')
plt.xlabel("Will customer default on credit bill payment?")
plt.xticks(np.arange(0,2,1),['No == 0','Yes == 1'])
plt.ylabel("Frequency")
plt.title('Curated Data Histogram : Binary Target Classes')
plt.grid()
plt.savefig('Histograms/Curated  Target Variable Y.png')

## Histogram Approved Credit Amount##
plt.figure(8)
plt.hist(newdf['X1'], histtype ='bar', color = 'blue')
plt.xlabel("Approved Credit Amount")
plt.ylabel("Frequency")
plt.title('Curated  Data Histogram :Approved Credit Amount')
plt.grid()
plt.savefig('Histograms/Curated  Approved Credit Amount Histogram.png')

## Histogram April Credit Bill Amount Due##
plt.figure(9)
plt.hist(newdf['X12'], histtype ='bar', color = 'red', align = 'right')
plt.xlabel("April Credit Bill Amount Due")
plt.ylabel("Frequency")
plt.title('Curated  Data Histogram : April Credit Bill Amount Due')
plt.grid()
plt.savefig('Histograms/Curated  April Credit Bill Due.png')

## Histogram April Credit Repayment Amount##
plt.figure(10)
plt.hist(newdf['X18'], histtype ='bar', color = 'red')
plt.xlabel("April Credit Repayment Amount")
plt.ylabel("Frequency")
plt.title('Curated  Data Histogram : April Credit Repayment Amount')
plt.grid()
plt.savefig('Histograms/Raw April Credit Repayment Amount.png')

## Histogram Educational Level##
plt.figure(11)
plt.hist(newdf['X3'], histtype ='bar', color = 'red', align = 'mid')
plt.xlabel("Educational Level")
plt.xticks(np.arange(0,7,1),['0','Graduate School','University','High School','Others'],rotation='vertical')
plt.ylabel("Frequency")
plt.title('Curated  Data Histogram : Educational Level')
plt.grid()
plt.savefig('Histograms/Curated  Educational Level.png')

## Boxplot Credit Approved ##
plt.figure(12)
plt.boxplot(newdf['X1'])
plt.xlabel("Feature  X1")
plt.ylabel("Approved Credit Amount",rotation = 90)
plt.title('Curated  Data Boxplot : Approved Credit Amount')
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
plt.savefig('Boxplots/Curated  Approved Credit Amount Boxplot.png')


#### Repayment Data ####
covar_matrix = PCA()
#print newdf.iloc[1,17:23]
covar_matrix.fit(newdf.iloc[:,17:23])
variance = covar_matrix.explained_variance_ratio_*100
var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3))
print "Variance Ratio X18 - X23 : ",[np.round(x,2) for x in variance]
plt.figure(13)
plt.plot(range(1,7,1),var*100,linestyle = "-",marker = 'o')
plt.xlabel("Principle Components Attributes X18 - X23")
plt.ylabel("Cumulative Explained Variance %")
plt.title("PCA: Variance Analysis")
plt.grid()
plt.savefig('Dimensionality Reduction/Explained Variance X18 - X23.png')

#### Covariance Using Numpy ####
# Covariance = np.cov(newdf.iloc[:,17:23])
# f = open("Cov_17_23.txt", 'w+')
# f.write(str(Covariance))
# f.close()


##### PCA because they are correlated and dimensional reduction always helps
pca = PCA(n_components = 6)
pc = pca.fit_transform(newdf.iloc[:,17:23])
prep = pd.DataFrame(pc,columns=['pc1','pc2','pc3','pc4','pc5','pc6'])
# f = open("CovPCA_17_23.txt",'w+')
# f.write(str(pca.get_covariance()))
# f.close()

#### Due Bill Data ####
covar_matrix = PCA(n_components = 6)
#print newdf.iloc[1,11:17]
covar_matrix.fit(newdf.iloc[:,11:17])
variance = covar_matrix.explained_variance_ratio_*100
var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)
print "Variance Ratio X12 - X17 : ",[np.round(x,2) for x in variance]
plt.figure(14)
plt.plot(range(1,7,1),var,linestyle = "-",marker = 'o')
plt.xlabel("Principle Components Attributes X12 - X17")
plt.ylabel("Cumulative Explained Variance %")
plt.title("PCA: Variance Analysis")
plt.grid()
plt.savefig('Dimensionality Reduction/Explained Variance X12 - X17.png')

##### PCA because they are correlated and dimensional reduction always helps
pca = PCA(n_components = 2)
pc = pca.fit_transform(newdf.iloc[:,11:17])
pdue = pd.DataFrame(pc,columns=['dpc1','dpc2'])

####  Continuous Monetary Variables Due and Payback ####
covar_matrix = PCA(n_components = 12)
covar_matrix.fit(newdf.iloc[:,11:23])
variance = covar_matrix.explained_variance_ratio_*100
var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)
print "Variance Ratio X12 - X23 : ",[np.round(x,2) for x in variance]

##### PCA because they are correlated and dimensional reduction always helps
pca = PCA(n_components = 1)
pc = pca.fit_transform(newdf.iloc[:,11:24])
mone = pd.DataFrame(pc,columns=['monetary'])


#### Payment Delay Data ####
covar_matrix = PCA(n_components = 6)
#print newdf.iloc[1,5:11]
covar_matrix.fit(newdf.iloc[:,5:11])
variance = covar_matrix.explained_variance_ratio_*100
var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)
print "Variance Ratio X6 - X11 : ",[np.round(x,2) for x in variance]
plt.figure(15)
plt.plot(range(1,7,1),var,linestyle = "-",marker = 'o')
plt.xlabel("Principle Components Attributes X6 - X11")
plt.ylabel("Cumulative Explained Variance %")
plt.title("PCA: Variance Analysis")
plt.grid()
plt.savefig('Dimensionality Reduction/Explained Variance X6 - X11.png')

###### PCA because they are correlated and dimensional reduction always helps
pca = PCA(n_components = 5)
pc = pca.fit_transform(newdf.iloc[:,5:11])
plap = pd.DataFrame(pc,columns=['lpc1','lpc2','lpc3','lpc4','lpc5'])

#### Person Non-monetary ####
covar_matrix = PCA(n_components = 4)
#print newdf.iloc[1,1:5]
covar_matrix.fit(newdf.iloc[:,1:5])
variance = covar_matrix.explained_variance_ratio_*100
var=np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3)*100)
print "Variance Ratio X2 - X5 : ",[np.round(x,2) for x in variance]
plt.figure(16)
plt.plot(range(1,5,1),var,linestyle = "-",marker = 'o')
plt.xlabel("Principle Components Attributes X2 - X5")
plt.ylabel("Cumulative Explained Variance %")
plt.title("PCA: Variance Analysis")
plt.grid()
plt.savefig('Dimensionality Reduction/Explained Variance X2 - X5.png')

###### PCA because they are correlated and dimensional reduction always helps
pca = PCA(n_components = 1)
pnm = pca.fit_transform(newdf.iloc[:,1:5])
pnm = pd.DataFrame(pnm,columns=['pnm'])

#### Dimensionally Reduced Data Set ####
df_reduced = pd.concat([newdf['X1'],pnm,plap,pdue,prep,newdf['Y']],axis=1)
df_reducedmon = pd.concat([newdf['X1'],pnm,plap,mone,newdf['Y']],axis=1)

newdf.to_pickle("curateddataset.pkl")
df_reduced.to_pickle("reduceddataset.pkl")
df_reduced.to_pickle("reducedmondataset.pkl")

#print df_reduced
### Heat Maps #####
correlation = df.corr()
plt.figure(num = 13, figsize=(14,10))
sb.heatmap(correlation)
plt.xticks(rotation = 90)
plt.yticks(rotation = 0)
plt.savefig('Heatmaps/Heat Map Raw Dataset.png')
correlation = newdf.corr()
plt.figure(num = 14, figsize=(14,10))
sb.heatmap(correlation)
plt.xticks(rotation = 90)
plt.yticks(rotation = 0)
plt.savefig('Heatmaps/Heat Map Curated Dataset.png')
correlation = df_reduced.corr()
plt.figure(num = 15, figsize=(14,10))
sb.heatmap(correlation)
plt.xticks(rotation = 90)
plt.yticks(rotation = 0)
plt.savefig('Heatmaps/Heat Map Dimensionally Reduced Dataset.png')

correlation = df_reducedmon.corr()
plt.figure(num = 16, figsize=(14,10))
sb.heatmap(correlation)
plt.xticks(rotation = 90)
plt.yticks(rotation = 0)
plt.savefig('Heatmaps/Heat Map Extreme Dimensionality Reduction Dataset.png')
sb.pairplot(df_reducedmon.iloc[:,:].sample(frac=.5))

#plt.show()
f = open("Statistics/Variance Random Forest.txt",'w+')
datasets = [df, newdf, df_reduced, df_reducedmon]
for x in datasets:
	train, test = train_test_split(x)

	y_train = train['Y']
	x_train = train.drop(['Y'],axis = 1)
	y_test = test['Y']
	x_test = test.drop(['Y'],axis = 1)



	rf = RandomForestClassifier()
	rf.fit(x_train,y_train)
	print "features sorted by their score:"
	print sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), x_train), reverse=True) 
	g = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), x_train), reverse=True) 
	f.write(str(g))
f.close()

# #/Users/nitishsanghi/documents/projects/courses/machine-learning-master/projects/capstone/CapstoneWorks
