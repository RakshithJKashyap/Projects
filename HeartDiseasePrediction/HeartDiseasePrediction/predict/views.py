from django.shortcuts import render
from django.http import HttpResponse, response
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Create your views here.
def model(input_data):
    heart_data=pd.read_csv("heart.csv")
    print("hello")
    X = heart_data[['age','sex','cp','trestbps','chol','fbs','restecg','thalach']]
    print(X)

    print('hi')
    Y = heart_data['target']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    input_data_as_numpy_array= np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    return model.predict(input_data_reshaped)


    
def predict_health(request):
    if request.method=="GET":
        
        return render(request,'predict/index.html',{})#templete here
    else:
        print("else")
        rage=int(request.POST.get('age'))
        rsex=float(request.POST.get('sex'))
        rcp=float(request.POST.get('cp'))
        rtrestbps=float(request.POST.get('trestbps'))
        rchol=float(request.POST.get('chol'))
        rfbs=float(request.POST.get('fbs'))
        rrestecg=float(request.POST.get('restecg'))
        rthalach=float(request.POST.get('thalach'))
        input_data = np.zeros(8)
        input_data[0] = rage
        input_data[1] = rsex
        input_data[2] = rcp
        input_data[3] = rtrestbps
        input_data[4] = rchol
        input_data[5] = rfbs
        input_data[6] = rrestecg
        input_data[7] = rthalach
        prd=model(input_data)
        print(prd)
        if(prd==0):
            text="The Person does not have a Heart Disease"
        else:
            text="The Person has Heart Disease"
        data = {
            'age' : rage,
            'sex' : rsex,
            'cp' : rcp,
            'trestbps' : rtrestbps,
            'chol' : rchol,
            'fbs' : rfbs,
            'restecg' : rrestecg,
            'thalach' : rthalach,
            'price':text
                
         }
        return render(request,'predict/index.html',{'data':data})