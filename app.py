# Import Statements
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv('C:\\Users\\ADMIN\\Downloads\\diabetes_prediction-master\\diabetes.csv')

# Streamlit Application Title and Sidebar
st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())

# Prepare Data for Training
X = df.drop(['Outcome'], axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# User Input Function
def user_input():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    blood_pressure = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skin_thickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0, 67, 20)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
    age = st.sidebar.slider('Age', 21, 88, 33)

    user_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    return pd.DataFrame(user_data, index=[0])

# Get User Data
user_data = user_input()
st.subheader('Patient Data')
st.write(user_data)

# Train the Model
model = RandomForestClassifier()
model.fit(X_train, y_train)
user_result = model.predict(user_data)

# Visualization Section
st.title('Visualized Patient Report')
color = 'blue' if user_result[0] == 0 else 'red'

# Plotting Function
def plot_comparison(x_col, y_col, user_val):
    fig = plt.figure()
    sns.scatterplot(x='Age', y=y_col, data=df, hue='Outcome', palette='Greens')
    sns.scatterplot(x=user_data['Age'], y=user_val, s=150, color=color)
    plt.title(f'0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig)

# Visualizations
st.header('Pregnancy Count Graph (Others vs Yours)')
plot_comparison('Age', 'Pregnancies', user_data['Pregnancies'])

st.header('Glucose Value Graph (Others vs Yours)')
plot_comparison('Age', 'Glucose', user_data['Glucose'])

st.header('Blood Pressure Value Graph (Others vs Yours)')
plot_comparison('Age', 'BloodPressure', user_data['BloodPressure'])

st.header('Skin Thickness Value Graph (Others vs Yours)')
plot_comparison('Age', 'SkinThickness', user_data['SkinThickness'])

st.header('Insulin Value Graph (Others vs Yours)')
plot_comparison('Age', 'Insulin', user_data['Insulin'])

st.header('BMI Value Graph (Others vs Yours)')
plot_comparison('Age', 'BMI', user_data['BMI'])

st.header('DPF Value Graph (Others vs Yours)')
plot_comparison('Age', 'DiabetesPedigreeFunction', user_data['DiabetesPedigreeFunction'])

# Output Result
st.subheader('Your Report:')
output = 'You are not Diabetic' if user_result[0] == 0 else 'You are Diabetic'
st.title(output)

# Accuracy Display
st.subheader('Accuracy:')
st.write(f"{accuracy_score(y_test, model.predict(X_test)) * 100:.2f}%")
