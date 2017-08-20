import pandas as pd
import numpy as np
from catboost import Pool, CatBoostClassifier, cv, CatboostIpythonWidget
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


train_df = pd.read_csv("C:\\Users\\kkumasau\\Downloads\\205e1808-6-dataset\\train.csv")
test = pd.read_csv("C:\\Users\\kkumasau\\Downloads\\205e1808-6-dataset\\test.csv")

train_df.head()

train_df.describe()
#drop columns with more than two nan values
train_df = train_df.dropna(axis=0, thresh=8)

#separete positive and negative examples
positive = train_df[train_df.click == 1]
negative = train_df[train_df.click == 0]

#Drop duplicates from positive and negative examples
positive.drop_duplicates(inplace = True, subset = ['datetime', 'siteid', 'offerid', 'category', 'merchant', 'countrycode', 'browserid', 'devid'])
negative.drop_duplicates(inplace = True, subset = ['datetime', 'siteid', 'offerid', 'category', 'merchant', 'countrycode', 'browserid', 'devid'])

#merge positive and negative examples
train = pd.concat([positive,negative])

#remove duplicates from the combined set
train.drop_duplicates(inplace = True, keep = False, subset = ['datetime', 'siteid', 'offerid', 'category', 'merchant', 'countrycode', 'browserid', 'devid'])

train = train.reset_index(drop=True)


# check missing values per column
train.isnull().sum(axis=0)/train.shape[0]

# impute missing values

train['siteid'].fillna(-999, inplace=True)
test['siteid'].fillna(-999, inplace=True)

train['browserid'].fillna("None", inplace=True)
test['browserid'].fillna("None", inplace=True)

train['devid'].fillna("None", inplace=True)
test['devid'].fillna("None", inplace=True)

# set datatime
train['datetime'] = pd.to_datetime(train['datetime'])
test['datetime'] = pd.to_datetime(test['datetime'])

# create datetime variable
train['tweekday'] = train['datetime'].dt.weekday
train['thour'] = train['datetime'].dt.hour
train['tminute'] = train['datetime'].dt.minute

test['tweekday'] = test['datetime'].dt.weekday
test['thour'] = test['datetime'].dt.hour
test['tminute'] = test['datetime'].dt.minute

#categorical columns
cols = ['siteid','offerid','category','merchant']

for x in cols:
    train[x] = train[x].astype('object')
    test[x] = test[x].astype('object')
   

cols_to_use = list(set(train.columns) - set(['ID','datetime','click']))

# modeling on sampled (1e6) rows
rows = np.random.choice(train.index.values, 1000000)
sampled_train = train.loc[rows]
#sampled_train = train



trainX = sampled_train[cols_to_use]
trainY = sampled_train['click']
trainX.dtypes
cat_cols = np.where(trainX.dtypes == np.object)[0]

X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size = 0.5)

#auto_stop_pval is used to prevent overfitting
model = CatBoostClassifier(depth=10, learning_rate=0.1, auto_stop_pval = 0.01, iterations=150, verbose=True )

model.fit(X_train
          ,y_train
          ,cat_features=cat_cols
          ,eval_set = (X_test, y_test)
          ,use_best_model = True
         )

model.score(X_train, y_train)
p = model.predict(X_test)

#groundtruth for precision and recall calculations
gt = y_test.reset_index(drop=True)

tp=0
tn=0
fp=0
fn=0
for i in range (len(p)):
    if (p[i] == 1 and gt[i]==1):
        tp+=1
    elif (p[i]==1 and gt[i]==0):
        fp+=1
    elif (p[i] ==0 and gt[i] == 1):
        fn+=1
    else:
        tn+=1
        
precision = tp/(tp+fp)
recall = tp/(tp+fn)
print (precision)
print(recall)
pred = model.predict_proba(test[cols_to_use], verbose=True)[:,1]

sub = pd.DataFrame({'ID':test['ID'],'click':pred})
sub.to_csv('C:\\Users\\kkumasau\\Downloads\\205e1808-6-dataset\\cb_sub1.csv',index=False)