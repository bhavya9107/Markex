from flask import Flask,render_template,request,url_for,redirect,flash,jsonify,send_file
from werkzeug import secure_filename

UPLOAD_FOLDER = '/Desktop/Markex-Bhavya'

app=Flask(__name__ , static_url_path='')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

import random, string
import httplib2
import json
from flask import make_response
import requests
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt, mpld3
import seaborn as sns

def clean(data):

        data['job'] = pd.Categorical(data['job'])
        data['marital']=pd.Categorical(data['marital'])
        data['education']=pd.Categorical(data['education'])
        data['default']=pd.Categorical(data['default'])
        data['housing']=pd.Categorical(data['housing'])
        data['month']=pd.Categorical(data['month'])
        data['poutcome']=pd.Categorical(data['poutcome'])
        data['loan']=pd.Categorical(data['loan'])

        data['marital']=data['marital'].cat.codes
        data['education']=data['education'].cat.codes
        data['default']=data['default'].cat.codes
        data['housing']=data['housing'].cat.codes
        data['loan']=data['loan'].cat.codes
        data['month']=data['month'].cat.codes
        data['poutcome']=data['poutcome'].cat.codes
        data['job']=data['job'].cat.codes

        data['pdays'] = preprocessing.scale(data['pdays'])
        data['age'] = preprocessing.scale(data['age'])

        return data

source = pd.read_excel('bank-full.xlsx')
source=clean(source)
source['y']=pd.Categorical(source['y'])
source['y']=source['y'].cat.codes

X_train = pd.read_csv('x_train_output.csv')
y_train = pd.read_csv('file_output.csv')

csv = y_train
model = rfc(random_state=22,n_jobs=-1)
print ('training the machine.....')
model.fit(X_train, y_train)
print ('successfully trained')

@app.route('/')
def first():
	return render_template('index.html')

@app.route('/temp', methods=['GET', 'POST'])
def plot_func():
	# precessing function
	if request.method == 'POST':
		data_file = request.files['pic']
		data = pd.read_csv(secure_filename(data_file.filename))
        data1=data.copy()
        data = clean(data)
        data.drop('contact', inplace=True, axis=1)
        pred = model.predict(data.drop('y', axis=1))
        pred=[pred[i][1] for i in range(len(pred))]
        data['y']=pd.Categorical(data['y'])
        data['y']=data['y'].cat.codes

        print('printing classification report......\n')
        print(classification_report(data['y'], pred))

        print('\nprinting accuracy.....\t')
        accuracy=accuracy_score(data['y'], pred)

        print(accuracy)
        print('printing confusion_matrix.....\n')
        print(confusion_matrix(data['y'], pred))
 
        one = np.bincount(pred)[1]
        total = np.bincount(pred)[0]+one
        print('printing yes %..... \t')
        yesp = one*100/total
        print(yesp)

        fig1 = plt.figure()
        fig1 = sns.distplot(data['month']).get_figure()
        plot1 = mpld3.fig_to_html(fig1)
        fig2 = plt.figure()
        fig2 = sns.distplot(data1['age']).get_figure()
        plot2 = mpld3.fig_to_html(fig2)

        data1.drop('y',axis=1)
        data1['prediction'] = pd.Series(pred,np.arange(len(pred)))
        csv = data1.to_csv('output.csv')

        return render_template('result.html', accuracy=accuracy, yesp=yesp, p1=plot1, p2=plot2)


@app.route('/note')
def note():
	return render_template('note.html')

@app.route('/output')
def output():
        return send_file('output.csv', attachment_filename="results.csv")

if __name__=='__main__':
	app.secret_key="bhavya curieo intern vatsal dusad"
	app.debug= False
	app.run(host="0.0.0.0",port=5555)