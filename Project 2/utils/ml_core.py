import pandas as pd
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

@st.cache_data
def load_and_prepare_data():
    iris_obj = load_iris()
    
    # Create pandas DataFrame
    df = pd.DataFrame(data=iris_obj.data, columns=iris_obj.feature_names)
    df['species_id'] = iris_obj.target
    
    # Map species names
    species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    df['species'] = df['species_id'].map(species_map)
    
    return iris_obj, df

def get_train_test_split(iris_data):
    X = iris_data.data
    y = iris_data.target
    
    # Split the dataset into training (80%) and testing (20%) sets
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train_raw, X_test_raw, y_train, y_test

def train_model_and_scale(X_train_raw, X_test_raw, y_train, k_val):
    # Apply Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    # Fit KNN model using chosen K from sidebar
    knn_model = KNeighborsClassifier(n_neighbors=k_val)
    knn_model.fit(X_train, y_train)
    
    return scaler, X_train, X_test, knn_model
