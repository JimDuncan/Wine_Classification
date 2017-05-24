import numpy as np
import pandas as pd
from itertools import izip
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def get_data():
    #grab red & white wine data from csv
    df_red = pd.read_csv('winequality-red.csv',delimiter=';')
    df_white = pd.read_csv('winequality-white.csv',delimiter=';')
    #clean data and create dataframe that includes both red & white info
    col_names = df_red.columns.tolist()
    col_names = [col.replace(' ','_') for col in col_names]
    df_red.columns = col_names
    df_white.columns = col_names
    df_red['Wine_Type']='Red'
    df_white['Wine_Type']='White'
    df_total=pd.concat([df_red,df_white])
    df_total.index=range(len(df_total))

    #write df_total to csv
    df_total.to_csv('data/Wine_Quality.csv',sep=',')

def random_forrest(train,cv,test):
    #instantiate random forrest classifier,fit and predict
    rf = RandomForestClassifier(criterion='gini',n_estimators=20,max_features =1,random_state=42)
    rf.fit(train.loc[:,:'quality'].values,train.loc[:,'Wine_Type'].values)
    rf_predicted= rf.predict(test.loc[:,:'quality'].values)
    #determine accuracy of predicitions
    rf_proba =rf.predict_proba(test.loc[:,:'quality'].values)[:,1]
    accuracy_tuple_rf = (rf_proba,rf_predicted,test.loc[:,'Wine_Type'].values)

    return accuracy_tuple_rf

def logistic_regression(train,cv,test):
    #instantiate logistic regression classifier,fit and predict
    lr = LogisticRegression(random_state=42)
    lr.fit(train.loc[:,:'quality'].values,train.loc[:,'Wine_Type'].values)
    lr_predicted=lr.predict(test.loc[:,:'quality'].values)
    lr_proba = lr.predict_proba(test.loc[:,:'quality'].values)[:,1]
    accuracy_tuple_lr = (lr_proba,lr_predicted,test.loc[:,'Wine_Type'].values)
    return accuracy_tuple_lr

def support_vector_machine(train,cv,test):
    #instantiate support_vector_machine classifier,fit and predict
    svc = SVC(random_state=42,probability=True)
    svc.fit(train.loc[:,:'quality'].values,train.loc[:,'Wine_Type'].values)
    svc_predicted=svc.predict(test.loc[:,:'quality'].values)
    svc_proba = svc.predict_proba(test.loc[:,:'quality'].values)[:,1]
    accuracy_tuple_svc = (svc_proba,svc_predicted,test.loc[:,'Wine_Type'].values)
    return accuracy_tuple_svc

def plot_ROC(tp_ar,fp_ar,ml_type):

    if ml_type == 'RandomForrest':
        color = 'darkorange'
    elif ml_type =='LogisticRegression':
        color = 'red'
    else:
        color = 'purple'

    #area under each classifier's respective curve
    roc_auc = auc(fp_ar, tp_ar)

    #plotting ROC curve
    plt.plot(fp_ar, tp_ar,color=color,lw=3,linestyle='-',label=ml_type+'(area = {0:0.4f})'.format(roc_auc) )
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Sensitivity, Recall)")
    plt.title("ROC plot -- Prediciton of Wine Color")


if __name__=="__main__":
    df_total = pd.read_csv('Wine_Quality.csv')
    df_total.drop('Unnamed: 0',inplace=True,axis=1)
    print df_total.head()
    #create train/cv/test split by randomly sampling without replacement df_total
    train,cv,test = np.split(df_total.sample(frac=1),[int(round(.6*len(df_total))),int(round(.8*len(df_total)))])
    #randomforrest classifier
    accuracy_tuple_rf=random_forrest(train,cv,test)
    #Classifying White wine as '1' and 'Red' wine as '0'
    binary_wine =[1 if wine=='White' else 0 for wine in accuracy_tuple_rf[2]]
    fp_ar,tp_ar,_ = roc_curve(binary_wine,accuracy_tuple_rf[0])
    plot_ROC(tp_ar,fp_ar,'RandomForrest')
    #logistic regression classifier
    accuracy_tuple_lr=logistic_regression(train,cv,test)
    fp_ar,tp_ar,_ = roc_curve(binary_wine,accuracy_tuple_lr[0])
    plot_ROC(tp_ar,fp_ar,'LogisticRegression')
    #support vector classifier
    accuracy_tuple_svc=support_vector_machine(train,cv,test)
    fp_ar,tp_ar,_ = roc_curve(binary_wine,accuracy_tuple_svc[0])
    plot_ROC(tp_ar,fp_ar,'SupportVectorMachine')

    plt.legend(loc="lower right")
    plt.savefig("ROC_Curve")
