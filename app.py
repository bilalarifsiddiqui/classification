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
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier 
import xgboost as xgb
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score


title_html = """
    <style>
        .title {
            color: #FF0000; /* Red color */
            font-size: 3em; /* Adjust the font size as needed */
        }
    </style>
    <h1 class="title">Smoke Detection FireAlarm</h1>
"""

# Render the styled title using markdown
st.markdown(title_html, unsafe_allow_html=True)



# Load the dataset
data = pd.read_csv('smoke_detection_iot.csv')  # Update with the correct path

# Preprocess the data
data = data.sample(n=1000)
data.drop(['Unnamed: 0','UTC','CNT','PM1.0','PM2.5','NC0.5','NC1.0','NC2.5'], axis=1, inplace=True)

x = data.iloc[:, 0:13]
y = data.iloc[:, -1]
data_scaled = RobustScaler()
x_scaled = data_scaled.fit_transform(x)
xtrain, xtest, ytrain, ytest = train_test_split(x_scaled, y, test_size=0.25, random_state=101)

# Function to train and evaluate models
def train_and_evaluate_model(model, xtrain, ytrain, xtest, ytest, model_name):
    # Initialize progress bar
    progress_bar = st.progress(0)

    # Train the model
    model.fit(xtrain, ytrain)

    # Update progress bar
    progress_bar.progress(50)  # Adjust the progress value based on your training steps

    # Make predictions
    y_pred = model.predict(xtest)

    # Calculate metrics
    accuracy = accuracy_score(ytest, y_pred)
    precision = precision_score(ytest, y_pred)
    recall = recall_score(ytest, y_pred)

    # Display metrics
    st.subheader(f"{model_name} Metrics:")
    st.write(f"Accuracy: {accuracy:.2%}")
    st.write(f"Precision: {precision:.2%}")
    st.write(f"Recall: {recall:.2%}")

    # Update progress bar
    progress_bar.progress(100)

    # Create a dataframe for plotting
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall'],
        'Value': [accuracy, precision, recall]
    })

    # Plot metrics
    st.bar_chart(metrics_df.set_index('Metric'), width=400, height=300)



# Streamlit app

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Additional Visualizations
st.sidebar.header("Different Methods for Data")


def screen_zero():
    # Display the DataFrame loaded from the CSV file
    st.header("DataFrame Loaded from CSV")
    st.dataframe(data)

def screen_one():
    # Create models with tuned hyperparameters
    models = {
        'SVM': SVC(),
        'KNN': KNeighborsClassifier(),
        'AdaBoost': AdaBoostClassifier(),
        'XGBoost': xgb.XGBClassifier(),
        'Random Forest': rf_model 
    }

    st.header("Models")
    selected_model = st.selectbox("Select a Model", list(models.keys()))

    # Display sliders based on the selected model
    if selected_model == 'SVM':
        svm_kernel = st.slider("SVM Kernel (linear)", 0.0, 1.0, 0.5)
        models['SVM'].kernel = 'linear' if svm_kernel < 0.5 else 'rbf'
    elif selected_model == 'KNN':
        knn_neighbors = st.slider("KNN Number of Neighbors", 1, 20, 7)
        models['KNN'].n_neighbors = knn_neighbors
    elif selected_model == 'AdaBoost':
        adaboost_n_estimators = st.slider("AdaBoost Number of Estimators", 1, 100, 50)
        models['AdaBoost'].n_estimators = adaboost_n_estimators
    elif selected_model == 'XGBoost':
        xgboost_learning_rate = st.slider("XGBoost Learning Rate", 0.01, 1.0, 0.01)
        models['XGBoost'].learning_rate = xgboost_learning_rate

    if st.button("Train and Evaluate"):
        st.subheader(f"Training and Evaluating {selected_model}...")
        train_and_evaluate_model(models[selected_model], xtrain, ytrain, xtest, ytest, selected_model)

def screen_two():
    st.header("Visualization")
    # Sidebar for selecting the type of plot
    selected_plot = st.selectbox("Select a Plot Type", ["Correlation Heatmap", "Box Plot",
                                                                "Histogram - Feature Distribution", "Scatter Plot - Feature vs Target", "Pie Chart"])

    # Plot based on user selection
    if selected_plot == "Correlation Heatmap":
        st.subheader("Correlation Heatmap")
        corr_matrix = data.corr()

        # Set the size of the figure
        fig, ax = plt.subplots(figsize=(18, 20))

        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)

        # Use st.pyplot() without setting 'deprecation.showPyplotGlobalUse' to False
        st.pyplot(fig)

    elif selected_plot == "Box Plot":
        selected_feature_box = st.selectbox("Select a Feature for Box Plot", x.columns)
        st.subheader(f"Box Plot for {selected_feature_box}")
        fig, ax = plt.subplots()
        
        sns.boxplot(x=data[selected_feature_box], y=data['Fire Alarm'], ax=ax)
        plt.title(f'Box Plot of {selected_feature_box} against Count')
        plt.xlabel(selected_feature_box)
        plt.ylabel('Count')
        st.pyplot(fig)

    elif selected_plot == "Histogram - Feature Distribution":
        st.subheader("Histogram - Feature Distribution")
        fig, ax = plt.subplots()
        selected_feature = st.selectbox("Select a Feature for Histogram", x.columns)
        sns.histplot(data[selected_feature], bins=30)
        st.pyplot(fig)

    elif selected_plot == "Scatter Plot - Feature vs Target":
        st.subheader("Scatter Plot - Feature vs Target")
        selected_scatter_feature = st.selectbox("Select a Feature for Scatter Plot", x.columns)
        fig, ax = plt.subplots()
        ax.scatter(x[selected_scatter_feature], y, alpha=0.5)
        ax.set_xlabel(selected_scatter_feature)
        ax.set_ylabel("Fire Alarm")
        st.pyplot(fig)
    elif selected_plot =="Pie Chart":
        fig, ax = plt.subplots()
        data['Fire Alarm'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, ax=ax)
        ax.set_title("Distribution of Fire Alarm")

        st.pyplot(fig)

def screen_three():
    st.header("Prediction using Dynamic Values")
    rf = joblib.load('smoke-detection-rf.joblib')
    sc = joblib.load('standardscaler.joblib')

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


# Create radio buttons for screen selection
selected_screen = st.sidebar.radio("Select a Screen", ["Data","Models", "Visualization", "Prediction using Dynamic Values"])

# Based on the selected radio button, display the corresponding screen
if selected_screen == "Data":
    screen_zero()
elif selected_screen == "Models":
    screen_one()
elif selected_screen == "Visualization":
    screen_two()
elif selected_screen == "Prediction using Dynamic Values":
    screen_three()

# footer 
st.sidebar.markdown("---")

st.sidebar.title("Team Members")

# Team members
team_members = [
    {"name": "Bilal Arif"},
    {"name": "Yasir Arfat"},
    {"name": "Asif Rasheed"},
]

# Display team members
for member in team_members:
    st.sidebar.write(f"**{member['name']}**")

