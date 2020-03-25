import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # para leer datos
import sklearn.ensemble # para el random forest
import sklearn.model_selection # para split train-test
import sklearn.metrics # para calcular el f1-score
from scipy.io import arff
#%matplotlib inline

names = [
'X1 net profit / total assets',
'X2 total liabilities / total assets',
'X3 working capital / total assets',
'X4 current assets / short-term liabilities',
'X5 [(cash + short-term securities + receivables - short-term liabilities) / (operating expenses - depreciation)] * 365',
'X6 retained earnings / total assets',
'X7 EBIT / total assets',
'X8 book value of equity / total liabilities',
'X9 sales / total assets',
'X10 equity / total assets',
'X11 (gross profit + extraordinary items + financial expenses) / total assets',
'X12 gross profit / short-term liabilities',
'X13 (gross profit + depreciation) / sales',
'X14 (gross profit + interest) / total assets',
'X15 (total liabilities * 365) / (gross profit + depreciation)',
'X16 (gross profit + depreciation) / total liabilities',
'X17 total assets / total liabilities',
'X18 gross profit / total assets',
'X19 gross profit / sales',
'X20 (inventory * 365) / sales',
'X21 sales (n) / sales (n-1)',
'X22 profit on operating activities / total assets',
'X23 net profit / sales',
'X24 gross profit (in 3 years) / total assets',
'X25 (equity - share capital) / total assets',
'X26 (net profit + depreciation) / total liabilities',
'X27 profit on operating activities / financial expenses',
'X28 working capital / fixed assets',
'X29 logarithm of total assets',
'X30 (total liabilities - cash) / sales',
'X31 (gross profit + interest) / sales',
'X32 (current liabilities * 365) / cost of products sold',
'X33 operating expenses / short-term liabilities',
'X34 operating expenses / total liabilities',
'X35 profit on sales / total assets',
'X36 total sales / total assets',
'X37 (current assets - inventories) / long-term liabilities',
'X38 constant capital / total assets',
'X39 profit on sales / sales',
'X40 (current assets - inventory - receivables) / short-term liabilities',
'X41 total liabilities / ((profit on operating activities + depreciation) * (12/365))',
'X42 profit on operating activities / sales',
'X43 rotation receivables + inventory turnover in days',
'X44 (receivables * 365) / sales',
'X45 net profit / inventory',
'X46 (current assets - inventory) / short-term liabilities',
'X47 (inventory * 365) / cost of products sold',
'X48 EBITDA (profit on operating activities - depreciation) / total assets',
'X49 EBITDA (profit on operating activities - depreciation) / sales',
'X50 current assets / total liabilities',
'X51 short-term liabilities / total assets',
'X52 (short-term liabilities * 365) / cost of products sold)',
'X53 equity / fixed assets',
'X54 constant capital / fixed assets',
'X55 working capital',
'X56 (sales - cost of products sold) / sales',
'X57 (current assets - inventory - short-term liabilities) / (sales - gross profit - depreciation)',
'X58 total costs /total sales',
'X59 long-term liabilities / equity',
'X60 sales / inventory',
'X61 sales / receivables',
'X62 (short-term liabilities *365) / sales',
'X63 sales / short-term liabilities',
'X64 sales / fixed assets','class']

data1 = arff.loadarff('1year.arff')
data1 = pd.DataFrame(data1[0])
data2 = arff.loadarff('2year.arff')
data2 = pd.DataFrame(data2[0])
data3 = arff.loadarff('3year.arff')
data3 = pd.DataFrame(data3[0])
data4 = arff.loadarff('4year.arff')
data4 = pd.DataFrame(data4[0])
data5 = arff.loadarff('5year.arff')
data5 = pd.DataFrame(data5[0])
data = pd.concat([data1, data2, data3, data4, data5])
data.columns = names
#df.head()
data = data.dropna()

purchasebin = np.ones(len(data), dtype = int)
ii = np.array(data['class'] == b'0')
purchasebin[ii] = 0
data['Target'] = purchasebin

data = data.drop(['class'],axis = 1)
# Crea un dataframe con los predictores
predictors = list(data.keys())
predictors.remove('Target')
#predictors.remove('Unnamed: 0')
#print(predictors, np.shape(np.array(predictors)))

X_data, x_val, y_data, y_val = sklearn.model_selection.train_test_split(
                                    data[predictors], data['Target'], test_size = 0.8)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
                                    X_data, y_data, test_size = 0.6)

n_trees = np.arange(1,10,1)
f1_train = []
f1_test = []
feature_importance = np.zeros((len(n_trees), len(predictors)))
#feature_importance.shape
for i, n_tree in enumerate(n_trees):
    clf = sklearn.ensemble.RandomForestClassifier(n_estimators = n_tree, max_features = 'sqrt')
    clf.fit(X_train, y_train)
    f1_train.append(sklearn.metrics.f1_score(y_train, clf.predict(X_train)))
    f1_test.append(sklearn.metrics.f1_score(y_test, clf.predict(X_test)))
    feature_importance[i, :] = clf.feature_importances_

tree_max = np.argmax(f1_test) + 1

# Grafica los features mas importantes
clf = sklearn.ensemble.RandomForestClassifier(n_estimators = tree_max, max_features = 'sqrt')
clf.fit(X_train, y_train)
f1 = sklearn.metrics.f1_score(y_val, clf.predict(x_val))
avg_importance = np.average(feature_importance, axis=0)
a = pd.Series(avg_importance, index=predictors)
#print(a)
a.nlargest().plot(kind='barh')
plt.xlabel('Average Feature Importance')
plt.title(r'$M = $ %0.0f' %tree_max + r'$f_1 = $ %0.4f' %f1)
plt.savefig('features.png')
plt.show()

