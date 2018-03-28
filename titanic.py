# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pandas import DataFrame,Series

#%%
data_train = pd.read_csv("D:/kaggle/titanic/train.csv")

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#有中文出现的情况，需要u'内容'
fig = plt.figure()
#新建绘画窗口,独立显示绘画的图片
fig.set(alpha=0.2)
#%%
"""
plt.subplot2grid((2,3),(0,0))
data_train.Survived.value_counts().plot(kind='bar')#Returns object containing counts of unique values.
plt.title(u"rescue(1)")
plt.ylabel(u"num_p")

plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind='bar')
plt.title(u"level")
plt.ylabel(u"num_p")

plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived,data_train.Age)
plt.ylabel("age")
plt.grid(b=True,which='major',axis='y')#网格
plt.title(u"age_survived")

plt.subplot2grid((2,3),(1,0),colspan=2)
data_train.Age[data_train.Pclass==1].plot(kind='kde')
data_train.Age[data_train.Pclass==2].plot(kind='kde')
data_train.Age[data_train.Pclass==3].plot(kind='kde')
plt.legend((u'1st', u'2nd',u'3rd'),loc='best') # sets our legend for our graph.

plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind='bar')
"""
#%%
"""
S_0 = data_train.Pclass[data_train.Survived==0].value_counts()
S_1 = data_train.Pclass[data_train.Survived==1].value_counts()
df = pd.DataFrame({'saved':S_1,'died':S_0})
df.plot(kind='bar',stacked=True)
"""
#%%
S_male = data_train.Survived[data_train.Sex =='male'].value_counts()
S_female = data_train.Survived[data_train.Sex == 'female'].value_counts()
gender = pd.DataFrame({'male':S_male,'female':S_female})
gender.plot(kind='bar',stacked=True)
#%%
from sklearn.ensemble import RandomForestRegressor

def set_missing_ages(df):
    age_df = df[['Age','Fare','Parch','SibSp','Pclass']]
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    y = known_age[:,0]
    x = known_age[:,1::]
    rfr = RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)#n...：树个数，n jobs 处理的任务数量 -1是最大化
    rfr.fit(x,y)#建立模型
    predictedAges = rfr.predict(unknown_age[:,1::])
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges
    return df,rfr

def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()),'Cabin'] = 'Yes'
    df.loc[(df.Cabin.isnull()),'Cabin'] = 'No'
    return df

data_train,rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)
#print(data_train.head())

#%%
dummies_Cabin = pd.get_dummies(data_train['Cabin'],prefix = 'Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'],prefix = 'Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'],prefix = 'Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'],prefix = 'Pclass')
df = pd.concat([data_train,dummies_Cabin,dummies_Embarked,dummies_Pclass,dummies_Sex],axis = 1)
df.drop(['Pclass','Embarked','Sex','Pclass','Ticket','Name'],axis = 1,inplace = True)
df
#%%
import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df['Age'].values.reshape(-1,1))
df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1,1), age_scale_param)
fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1,1))
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1,1),fare_scale_param)
df
#%%
from sklearn import linear_model
train_df = df.filter(regex = 'Survived|Age._*|SibSp|Fare._*|Cabin._*|Embarked._*|Sex._*|Pclass._*')
train_np = train_df.as_matrix()
y = train_np[:,0]
X = train_np[:,1:]
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X,y)
#%%
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
df_test
#%%
test = df_test.filter(regex='Age_.*|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("D:/kaggle/logistic_regression_predictions.csv", index=False)
#%%
#pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(clf.coef_.T)})
from sklearn import cross_validation
clf = linear_model.LogisticRegression(C=1.0,penalty='l1',tol=1e-6)
all_data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
X = all_data.as_matrix()[:,1:]
y = all_data.as_matrix()[:,0]
print(cross_validation.cross_val_score(clf,X,y,cv=5))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve

# 用sklearn的learning_curve得到training_score和cv_score，使用matplotlib画出learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, 
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    """
    画出data在某模型上的learning curve.
    参数解释
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"训练样本数")
        plt.ylabel(u"得分")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"交叉验证集上得分")

        plt.legend(loc="best")

        plt.draw()
        plt.show()
        plt.gca().invert_yaxis()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff

plot_learning_curve(clf, u"学习曲线", X, y)