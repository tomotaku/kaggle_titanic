import pandas as pd 
import numpy as np
from pandas import DataFrame,Series

train = pd.read_csv("D:/kaggle/titanic/train.csv")

from sklearn.ensemble import RandomForestRegressor
def set_missing_ages(df):
    """
    随机森林填补年龄空缺
    """
    age_df = df[['Age','Fare','Parch','SibSp','Pclass']]
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    y = known_age[:,0]
    x = known_age[:,1::]
    rfr = RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
    rfr.fit(x,y)
    predictages = rfr.predict(unknown_age[:,1::])
    df.loc[(df.Age.isnull()),'Age'] = predictages
    return df,rfr

def set_Cabin_type(df):
    """
    设置舱位属性，分为有无两种
    """
    df.loc[(df.Cabin.notnull()),'Cabin'] = 'Yes'
    df.loc[(df.Cabin.isnull()),'Cabin'] = 'No'
    return df
train,rfr = set_missing_ages(train)
train = set_Cabin_type(train)

#对类目型的特征因子化
#get_dummies将取值展开成属性，属性符合的为1，不符为0
dummies_Cabin = pd.get_dummies(train['Cabin'],prefix = 'Cabin')
dummies_Embarked = pd.get_dummies(train['Embarked'],prefix = 'Embarked')
dummies_Sex = pd.get_dummies(train['Sex'],prefix = 'Sex')
dummies_Pclass = pd.get_dummies(train['Pclass'],prefix = 'Pclass')
df = pd.concat([train,dummies_Cabin,dummies_Embarked,dummies_Pclass,dummies_Sex],axis = 1)
df.drop(['Pclass','Embarked','Sex','Pclass','Ticket','Name'],axis = 1,inplace = True)

#将差距过大的age和fare属性进行收敛化
import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df['Age'].values,reshape(-1,1),age_scale_param)
df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1,1), age_scale_param)
fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1,1))#给定-1指这个值根据其他的值推算出来
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1,1),fare_scale_param)

#构造模型
from sklearn import linear_model
train_df = df.filter(regex = 'Survived|Age._*|SibSp|Fare._*|Cabin._*|Embarked._*|Sex._*|Pclass._*')
train_np = train_df.as_matrix()
y = train_np[:,0]
X = train_np[:,1:]
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X,y)

#将测试集也做之前一样的处理
data_test = pd.read_csv("D:/kaggle/titanic/test.csv")
data_test.loc[(data_test.Fare.isnull()),'Fare'] = 0
tmp_df = data_test[['Age','Fare','SibSp','Pclass','Parch']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()
X = null_age[:,1:]
predictedAges = rfr.predict(X)
data_test.loc[(data_test.Age.isnull()),'Age'] = predictedAges

data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'],prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'],prefix='Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'],prefix='Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'],prefix='Pclass')

df_test = pd.concat([data_test,dummies_Cabin,dummies_Embarked,dummies_Pclass,dummies_Sex],axis = 1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'],axis=1,inplace=True)
df_test['Age_Scaled']=scaler.fit_transform(df_test['Age'].values.reshape(-1,1),age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].values.reshape(-1,1), fare_scale_param)

#做出相关的预测并输出到.csv中
test = df_test.filter(regex='Age_.*|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("D:/kaggle/logistic_regression_predictions.csv", index=False)

#做交叉验证
from sklearn import cross_validation
clf = linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6)
all_data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
X = all_data.as_matrix()[:,1:]
y = all_data.as_matrix()[:,0]
print(cross_validation.cross_val_score(clf,X,y,cv=5))
