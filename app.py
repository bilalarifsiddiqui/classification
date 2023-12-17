# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load the dataset
data = pd.read_csv('d:\Tools-Project\smoke_detection_iot.csv')  # Update with the correct path

# Preprocess the data
data = data.sample(n=1000)
x = data.iloc[:, 0:13]
y = data.iloc[:, -1]
data_scaled = RobustScaler()
x_scaled = data_scaled.fit_transform(x)
xtrain, xtest, ytrain, ytest = train_test_split(x_scaled, y, test_size=0.25, random_state=101)

# Function to train and evaluate models
def train_and_evaluate_model(model, xtrain, ytrain, xtest, ytest, model_name):
    model.fit(xtrain, ytrain)
    y_pred = model.predict(xtest)
    
    accuracy = accuracy_score(ytest, y_pred)
    precision = precision_score(ytest, y_pred)
    recall = recall_score(ytest, y_pred)
    
    confusion_matrix = metrics.confusion_matrix(ytest, y_pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])

    # Display metrics
    st.subheader(f"{model_name} Metrics:")
    st.write(f"Accuracy: {accuracy:.2%}")
    st.write(f"Precision: {precision:.2%}")
    st.write(f"Recall: {recall:.2%}")

    # Display precision-recall curve
    st.subheader("Precision-Recall Curve")
    precision, recall, _ = metrics.precision_recall_curve(ytest, y_pred)
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    st.pyplot()

    # Display confusion matrix
    st.subheader("Confusion Matrix")
    st.pyplot(cm_display.plot())

# Streamlit app
st.title("Machine Learning Model Comparison")

# Model comparison
models = {
    'SVM': SVC(kernel='linear'),
    'KNN': KNeighborsClassifier(n_neighbors=7),
    'AdaBoost': AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=0),
    'XGBoost': xgb.XGBClassifier(random_state=1, learning_rate=0.01)
}

selected_model = st.selectbox("Select a Model", list(models.keys()))

if st.button("Train and Evaluate"):
    st.subheader(f"Training and Evaluating {selected_model}...")
    train_and_evaluate_model(models[selected_model], xtrain, ytrain, xtest, ytest, selected_model)

# Additional Visualizations
st.sidebar.header("Exploratory Data Analysis")

# Sidebar for selecting the type of plot
selected_plot = st.sidebar.selectbox("Select a Plot Type", ["Correlation Heatmap", "Box Plot",
                                                            "Histogram - Feature Distribution", "Scatter Plot - Feature vs Target"])

# Plot based on user selection
if selected_plot == "Correlation Heatmap":
    st.sidebar.subheader("Correlation Heatmap")
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    st.sidebar.pyplot()

elif selected_plot == "Box Plot":
    selected_feature_box = st.sidebar.selectbox("Select a Feature for Box Plot", x.columns)
    st.sidebar.subheader(f"Box Plot for {selected_feature_box}")
    fig, ax = plt.subplots()
    
    sns.boxplot(x=data[selected_feature_box], y=data['Fire Alarm'], ax=ax)
    plt.title(f'Box Plot of {selected_feature_box} against Count')
    plt.xlabel(selected_feature_box)
    plt.ylabel('Count')
    st.sidebar.pyplot(fig)

elif selected_plot == "Histogram - Feature Distribution":
    st.subheader("Histogram - Feature Distribution")
    selected_feature = st.sidebar.selectbox("Select a Feature for Histogram", x.columns)
    sns.hist(data[selected_feature], bins=30)
    sns.pyplot()

elif selected_plot == "Scatter Plot - Feature vs Target":
    st.subheader("Scatter Plot - Feature vs Target")
    selected_scatter_feature = st.selectbox("Select a Feature for Scatter Plot", x.columns)
    fig, ax = plt.subplots()
    ax.scatter(x[selected_scatter_feature], y, alpha=0.5)
    ax.set_xlabel(selected_scatter_feature)
    ax.set_ylabel("Fire Alarm")
    st.pyplot(fig)



rf = joblib.load('./smoke-detection-rf.joblib')
sc = joblib.load('./standardscaler.joblib')

temp = st.text_input('Temperature[C]', value=20.0)
hum = st.text_input('Humidity[%]', value=57.36)
tvoc = st.text_input('TVOC[ppb]', value=0)
eco2 = st.text_input('eCO2[ppm]', value=400)
h2 = st.text_input('Raw H2', value=12306)
eth = st.text_input('Raw Ethanol', value=18520)
press = st.text_input('Pressure[hPa]', value=939.735)

if st.button('Predict'):
     data_input = np.array([[temp,hum,tvoc,eco2,h2,eth,press]])
     predict = rf.predict(sc.transform(data_input))
     if predict == 0:
          st.write('Prediction Fire Alarm = OFF')
     else:
          st.write('Prediction Fire Alarm = ON')
