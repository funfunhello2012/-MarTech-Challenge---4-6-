# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 14:02:43 2021

@author: funfun2
"""

import pandas as pd
train=pd.read_csv("F:\\data\\FraudDetect\\train.csv")
pd.set_option('display.max_columns', None)
test1=pd.read_csv("F:\\data\\FraudDetect\\test1.csv")
labels=train['label']
features=train.drop(['label'],axis=1)
## 合并训练集测试集
df=pd.concat([features,test1],axis=0)
df=df.iloc[:,1:]
## 缺失值处理
df['lan'].fillna(df['lan'].mode()[0],inplace=True)
df['osv'].fillna(df['osv'].mode()[0],inplace=True)

df.info()
df.isnull().sum()

# 查看唯一值的个数
for feature in df.columns:
    print(feature,df[feature].nunique())
    
train['os'].value_counts()
train['carrier'].value_counts()

remove_list=['os','sid']
col=df.columns.tolist()
for f in remove_list:
    col.remove(f)
col


# 特征筛选
df=df[col]
## 对fea_hash进行编码,如果长度大于16为异常值设为0，否则保留原始值
df['fea_hash'].value_counts()
temp=df['fea_hash'].apply(lambda x:len(str(x)))
temp.value_counts()
df['fea_hash']=df['fea_hash'].apply(lambda x :0 if len(str(x))>16 else int(x))
## 针对fea1_hash进行编码
df['fea1_hash']=df['fea1_hash'].apply(lambda x :0 if len(str(x))>16 else int(x))

## 对object类型进行处理，数据规范化
df['lan'].value_counts()
df['osv'].value_counts()
df['version'].value_counts()
from sklearn.preprocessing import LabelEncoder
obj_list=['lan','osv','version']
for obj in obj_list:
    #labelencode 标签编码对字符串类型进行数值转换
    le=LabelEncoder()
    df[obj]=le.fit_transform(df[obj])

## 还原数据集
features=df.iloc[0:len(train),:]
test1 = df.iloc[len(train):,:]

## 分类模型训练
#lgb经验参数
import lightgbm as lgb
clf=lgb.LGBMClassifier(num_leaves=2**5-1,reg_alpha=0.25,reg_lambda=0.25,objective='binary',
                       max_depth=-1,learning_rate=0.005,min_child_samples=3,random_state=2021,
                       n_estimators=2000,subsample=1,colsample_bytree=1,)   

clf.fit(features,labels)
result=clf.predict(test1)
# xgb经验参数
import xgboost as xgb
xgb=xgb.XGBClassifier(max_depth=6,learning_rate=0.05,n_estimators=2000,
                      objective='binary:logistic',silent=True,
                      subsample=0.8,colsample_bytree=0.8,min_child_samples=3,
                      eval_metric='auc',reg_lambda=0.5)
xgb.fit(features,labels)
result=xgb.predict(test1)




## 提交结果
temp=pd.read_csv("F:\\data\\FraudDetect\\test1.csv")
a=pd.DataFrame(temp['sid'])
a['label']=result
a.to_csv("F:\\data\\FraudDetect\\baseline2.csv",index=False)