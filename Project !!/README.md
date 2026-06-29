# Data Classification Using AI – Iris Classifier 🌸

An interactive, beginner-friendly machine learning dashboard built in Python using **Streamlit** to classify Iris flowers into three species: **Setosa**, **Versicolor**, or **Virginica** using the **K-Nearest Neighbors (KNN)** algorithm.

This project is structured step-by-step to explain core concepts like data preprocessing, standard scaling, train-test splits, evaluation metrics, and hyperparameter tuning in an accessible way.

---

## 🚀 How to Run the App Locally

### Prerequisites
Make sure you have **Python 3.8+** installed.

### 1. Install Dependencies
Navigate to the project folder and install the required scientific and visualization libraries:
```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit App
Run the following command in your terminal to start the local development server:
```bash
streamlit run app.py
```
This will automatically open the application in your default web browser (typically at `http://localhost:8501`).

---

## 📁 Project Folder Structure
* `app.py`: Main dashboard application containing the entire machine learning pipeline and user interface.
* `requirements.txt`: List of required Python packages (`streamlit`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`).
* `README.md`: Explanatory project documentation (this file).

---

## 🧬 Understanding the ML Pipeline

This project implements a complete supervised machine learning pipeline divided into the following key sections:

### 1. Import Libraries
We import Python's scientific computing libraries:
* `pandas` & `numpy` for data manipulation.
* `scikit-learn` for machine learning utilities, model training, and metrics.
* `matplotlib` & `seaborn` for generating charts.
* `streamlit` for the interactive UI dashboard.

### 2. Load Dataset
We load the classic **Iris Dataset** directly from `sklearn.datasets`. The dataset consists of:
* **150 samples** of Iris flowers.
* **4 features**: Sepal Length, Sepal Width, Petal Length, Petal Width (all measured in cm).
* **3 target classes**: Iris-Setosa (0), Iris-Versicolor (1), Iris-Virginica (2).

### 3. Exploratory Data Display
Before training any model, we perform Exploratory Data Analysis (EDA):
* View basic dataset shape and columns.
* Show the raw data table.
* Generate a **Class Distribution Bar Chart** to ensure the classes are balanced (50 samples per species).
* Draw a **Feature Relationship Scatter Plot** (Petal Length vs. Petal Width) to visualize class clustering.

### 4. Preprocessing & Feature Scaling
To feed the features to the KNN model, we first separate the predictors ($X$) and targets ($y$). We then apply **Standardization (StandardScaler)**:

$$z = \frac{x - \mu}{\sigma}$$

#### ⚠️ Why Feature Scaling is Essential for KNN:
KNN relies entirely on **Euclidean distance** to find the closest data points. If one feature has a much larger range of values than another (e.g. 1000m vs 1cm), it will completely dominate the distance calculation. Scaling ensures all features contribute equally to the distance computation.

### 5. Train-Test Split
To evaluate our model's accuracy on unseen data, we divide the dataset using `train_test_split`:
* **80% Training Set (120 samples)**: Used to fit the model.
* **20% Testing Set (30 samples)**: Held out to evaluate the final model performance.
* **Stratified Split**: Ensures the train and test sets have equal distributions of the three species.
* **Random State (42)**: Set to secure reproducibility across runs.

### 6. Model Training (K-Nearest Neighbors)
We train a `KNeighborsClassifier` from scikit-learn. KNN makes predictions by locating the $K$ training samples nearest to a query point and performing a majority vote. The default value is set to **$K=5$** but can be changed dynamically in the sidebar.

### 7. Evaluation Metrics
Once predictions are generated on the test set, we assess performance using:
* **Accuracy Score**: The ratio of correct predictions to total test samples.
* **F1-Score (Macro)**: Harmonic mean of precision and recall, summarizing performance across all three classes.
* **Confusion Matrix**: A table showing the actual vs. predicted labels to identify exact misclassifications.
* **Classification Report**: Shows precision, recall, and f1-score per individual species.

### 8. Visualization
We plot:
* A **Confusion Matrix Heatmap** mapping true vs. predicted species.
* A **Hyperparameter Tuning Curve** plotting model accuracy for K values from 1 to 15 to find the optimal $K$.

### 9. Custom Input Prediction
Users can adjust sliders in the sidebar representing a flower's sepal and petal measurements. The app automatically scales these inputs and runs them through the trained KNN classifier to output the predicted species and confidence percentages in real-time.
