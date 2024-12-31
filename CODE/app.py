from flask import Flask, render_template,request, flash, url_for, redirect,session
import mysql.connector
import pandas as pd
import pymysql
import matplotlib.pyplot as plt
#import necessary libraries
import numpy as np # linear algebra
import pandas as pd
from sklearn.metrics import accuracy_score
import re
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import train_test_split
app=Flask(__name__)
app.config['SECRET_KEY']="twitter"

mydb=pymysql.connect(host="localhost", user="root", password="",port=3306, database="heart_dd")
mycursor=mydb.cursor()


@app.route('/')
def index():
    return render_template('index.html')
@app.route('/contact', methods=['POST','GET'])
def contact():
    if request.method=="POST":
        email=request.form['email']
        session['email'] = email
        password=request.form['password']

        if email=="admin@gmail.com" and password=="admin":
            flash("Welcome admin","success")
            return render_template('Admin.html')
        else:
            sql="select count(*) from user_registration where email='%s' and password='%s'"%(email,password)
            x=pd.read_sql_query(sql,mydb)
            count=x.values[0][0]
            if count==0:
                flash("Invalid data entered","warning")
                return render_template('contact.html')
            else:
                session['email']=email
                flash("Welcome Admin","success")
                return render_template('Customer.html')
    return render_template('contact.html')


@app.route('/register', methods=['POST','GET'])
def register():

    if request.method=='POST':
        fname=request.form['firstname']
        lname=request.form['lastname']
        email=request.form['email']
        password=request.form['password']
        mobile=request.form['mobile']
        sql="select email from user_registration where  email='%s'" %(email)
        mycursor.execute(sql)
        data=mycursor.fetchall()
        mydb.commit()
        data=[j for i in data for j in i]
        if data ==[]:
            sql="insert into user_registration(first_name,last_name,email,password,mobile_number) values(%s,%s,%s,%s,%s)"
            val=(fname,lname,email,password,mobile)
            print(val)
            print(sql)
            mycursor.execute(sql,val)
            mydb.commit()
            flash("Account created Successully", "success")
            return render_template('contact.html')
        flash("Details already Exists","warning")
        return render_template('register.html')
    return render_template('register.html')

@app.route('/profile')
def profile():
    email = session['email']
    print(session['email'])

    sql="select * from user_registration where Email='%s' "%(email)
    mycursor.execute(sql)
    data=mycursor.fetchall()
    mydb.commit()
    print(data)
    print(type(data))
    # Create a DataFrame from the given data
    userdata = pd.DataFrame(data, columns=['id','first_name','last_name','email','password','mobile_number'])

    # Display the DataFrame
    print(userdata)
    return render_template('profile.html', columns=userdata.columns.values, rows=userdata.values.tolist())
@app.route('/all_users')
def all_users():
    sql="select * from user_registration"
    mycursor.execute(sql)
    data=mycursor.fetchall()
    mydb.commit()
    print(data)
    print(type(data))
    # Create a DataFrame from the given data
    userdata = pd.DataFrame(data, columns=['id','first_name','last_name','email','password','mobile_number'])

    # Display the DataFrame
    print(userdata)
    return render_template('all_users.html', columns=userdata.columns.values, rows=userdata.values.tolist())

@app.route('/load',methods=['POST','GET'])
def load():
    global df,dataset
    if request.method=="POST":
        file=request.files['file']
        df=pd.read_csv(file)
        dataset=df.head(100)
        msg='Data Loaded Successfully'
        return render_template('load.html',msg=msg)
    return render_template('load.html')




@app.route('/model', methods=["POST","GET"])
def model():
    global X_train, X_test, y_train, y_test,df,acc_lstm,acc_dt
    df = pd.read_csv(r'heart_deasease.csv')
    
    X = df.iloc[:, :-1]
    y = df.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
    if request.method=="POST":
        global model
        s=int(request.form['algo'])
        if s==0:
            return render_template('model.html',msg="Choose an algorithm")
        elif s == 1:
          
            rf = RandomForestClassifier()
            rf.fit(X_train,y_train)
            y_pred = rf.predict(X_test)
            acc_rf = accuracy_score(y_test,y_pred)*100
            msg = 'The accuracy obtained by RandomForestClassifier is ' + str(acc_rf) + str('%')
            return render_template('model.html', msg=msg)
        elif s == 2:
            
            xg = XGBClassifier()
            xg.fit(X_train,y_train)
            y_pred = xg.predict(X_test)
            acc_xg = accuracy_score(y_test,y_pred)*100
            msg = 'The accuracy obtained by XGBClassifier is ' + str(acc_xg) + str('%')
            return render_template('model.html', msg=msg)
        elif s == 3:
            
            gb = GradientBoostingClassifier()
            gb.fit(X_train,y_train)
            y_pred = gb.predict(X_test)
            acc_gb = accuracy_score(y_test,y_pred)*100
            msg = 'The accuracy obtained by GradientBoostingClassifier is ' + str(acc_gb) + str('%')
            return render_template('model.html', msg=msg)
        elif s ==4:
            from sklearn.ensemble import VotingClassifier
            from sklearn.datasets import make_classification
            clf1=RandomForestClassifier()
            clf2=GradientBoostingClassifier()
            hybrid_model = VotingClassifier(estimators=[('dtc',clf1),('lr',clf2)] , voting='hard')
            hybrid_model.fit(X_train,y_train)
            y_pred=hybrid_model.predict(X_test)
            acc_hyb=accuracy_score(y_test,y_pred)
            acc_hyb=acc_hyb*100
            msg = 'The accuracy obtained by Hybrid Model is ' + str(acc_hyb) + str('%')
            return render_template('model.html', msg=msg)

        

    return render_template("model.html")


@app.route('/pradiction', methods=['POST','GET'])
def pradiction():
    global  X_train, X_test, y_train, y_test,X,y
    df = pd.read_csv(r'heart_deasease.csv')
    
    X = df.iloc[:, :-1]
    y = df.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
    if request.method=='POST':
        #Accepts all values
        age=request.form['age']
        sex = request.form['sex']
        cp = request.form['cp']
        trestbps = request.form['trestbps']
        chol = request.form['chol']
        fbs = request.form['fbs']
        restecg = request.form['restecg']
        thalach = request.form['thalach']
        exang = request.form['exang']
        oldpeak = request.form['oldpeak']
        slope = request.form['slope']
        ca = request.form['ca']
        thal = request.form['thal']
        

        all_obj_vals=[float(age),float(sex),float(cp),float(trestbps),float(chol),float(fbs),float(restecg),float(thalach),float(exang),float(oldpeak),float(slope),float(ca),float(thal)]
        dtc = RandomForestClassifier()
        model = dtc.fit(X_train,y_train)

        pred = model.predict([all_obj_vals])
        print(pred)
        type(pred)

        if pred == 0:
            msg = "The Prediction result is No  Heart Disease."
        elif pred == 1:
            msg = "The Prediction result is   Heart Disease."
        
        
        email = session['email']
        print(session['email'])
        print("***************")
        sql = "INSERT INTO test(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal,pred,email) values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        val = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal,msg,email)

        dc = mycursor.execute(sql, val)
        mydb.commit()
        return render_template('pradiction.html', msg=msg, dc=dc)
    return render_template('pradiction.html')



@app.route('/history')
def history():
    email = session['email']
    sql = "SELECT * FROM test WHERE email='%s'"%(email)
    data = pd.read_sql_query(sql,mydb)
    
    return render_template('history.html',cols=data.columns.values,rows = data.values.tolist()) 



@app.route('/graph')
def graph():
   
    return render_template('graph.html')

if __name__=='__main__':
    app.run(debug=True)