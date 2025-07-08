
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
import streamlit as st

# pip install streamlit

df = pd.read_csv('Telco_Customer_churn.csv')
st.subheader('Telco data')
st.dataframe(df.head())

f1 = plt.figure(figsize=(4,3))
sns.countplot(x='Churn', hue='gender', data=df)

st.pyplot(f1)

nav = st.sidebar.radio("Select Countplot Feature",["gender","PaymentMethod","InternetService"])

f1 = plt.figure(figsize=(3,3))
sns.countplot(x='Churn', hue='gender', data=df)

st.pyplot(f1)
    
# st.write(df.columns)
df1 = df[['gender', 'PaymentMethod', 'MonthlyCharges','tenure', 'Churn']].copy()
# st.dataframe(df1.head(3))
# st.sidebar.multiselect('Select ')

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df1['gender'] = lb.fit_transform(df1['gender'])
df1['PaymentMethod'] = lb.fit_transform(df1['PaymentMethod'])
df1['Churn'] = lb.fit_transform(df1['Churn'])

st.subheader('Data after LabelEncoding')
st.dataframe(df1.head(5))

# Select dependent and independent features
x = df1.iloc[:,:-1] # df1.drop('Churn',axis=1) 
y = df1.iloc[:,-1]  # df1['Churn']
st.write(x.shape,y.shape)
# st.write(type(x),type(y))

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)

classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN', 'SVM', 'Random Forest','DecisionTree','Logistic Regression')
)


def select_param(clf_name):
    params = {}
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
        ker = st.sidebar.selectbox('Select kernel',('linear','rbf'))
        params['kernel'] = ker
    elif clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    elif clf_name=='DecisionTree':
        max_depth = st.sidebar.slider('max_depth', 2, 15)      # max depth of each tree
        params['max_depth'] = max_depth
        criteria = st.sidebar.selectbox('Criteria', ('gini','entropy'))  # criteria for splitting
        params['criteria'] = criteria
    elif clf_name=='Random Forest':
        max_depth = st.sidebar.slider('max_depth', 2, 15)         # max depth of the tree
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)  # number of trees
        params['n_estimators'] = n_estimators
    else:
        None
    return params

params = select_param(classifier_name)

def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(kernel=params['kernel'],C=params['C'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    elif clf_name == 'DecisionTree':
        clf = DecisionTreeClassifier(criterion=params['criteria'],max_depth=params['max_depth'])
    elif clf_name=='Random Forest':
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'], random_state=1234)
    else:
        clf = LogisticRegression(max_iter=10000)
    return clf

model = get_classifier(classifier_name, params)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test,y_pred)



st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)
st.write(cm)
st.write(f'Classification Report\n',classification_report(y_test,y_pred))

# f2 = plt.figure(figsize=(2,2))
# sns.heatmap(cm,annot=True)
# st.pyplot(f2)


# 1) To run Streamlit web app in Browser, open the terminal and write the following
# streamlit run app.py
# 2) To stop the running server, in the terminal
# Press Ctrl + C 
