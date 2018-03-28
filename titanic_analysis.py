import pandas as pd
import numpy as np
#from pandas import DataFrame,Series
#%%
data_train = pd.read_csv("D:/kaggle/titanic/train.csv")

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
fig = plt.figure()
#新建绘画窗口,独立显示绘画的图片
fig.set(alpha=0.2)

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

S_0 = data_train.Pclass[data_train.Survived==0].value_counts()
S_1 = data_train.Pclass[data_train.Survived==1].value_counts()
df = pd.DataFrame({'saved':S_1,'died':S_0})
df.plot(kind='bar',stacked=True)

S_male = data_train.Survived[data_train.Sex =='male'].value_counts()
S_female = data_train.Survived[data_train.Sex == 'female'].value_counts()
gender = pd.DataFrame({'male':S_male,'female':S_female})
gender.plot(kind='bar',stacked=True)
