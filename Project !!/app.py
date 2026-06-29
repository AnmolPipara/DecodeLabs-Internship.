import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

# Set page layout and aesthetics
st.set_page_config(
    page_title="Data Classification Using AI – Iris Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;500;700&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        color: #1E293B;
    }
    
    /* Main title styling */
    .main-title {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 50%, #4D96FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #64748B;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Cards and Glassmorphism */
    .premium-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px -2px rgba(50, 50, 93, 0.05), 0 2px 10px -2px rgba(0, 0, 0, 0.02);
        transition: transform 0.2s ease;
    }
    
    .premium-card:hover {
        transform: translateY(-2px);
    }
    
    /* KPI Metric styling */
    .kpi-container {
        display: flex;
        justify-content: space-around;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .kpi-card {
        flex: 1;
        background: white;
        border-left: 5px solid #4D96FF;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin: 0;
    }
    
    .kpi-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }
    
    /* Dynamic Badges for predicted species */
    .prediction-badge {
        font-size: 2rem;
        font-weight: 800;
        text-align: center;
        padding: 1.5rem;
        border-radius: 16px;
        color: white;
        margin-top: 1rem;
        animation: pulse 2s infinite;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    .badge-setosa {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        border: 2px solid #34D399;
    }
    
    .badge-versicolor {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
        border: 2px solid #FBBF24;
    }
    
    .badge-virginica {
        background: linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%);
        border: 2px solid #A78BFA;
    }
    
    /* Codeblock custom styling */
    .stCodeBlock {
        border-radius: 10px;
        border: 1px solid #E2E8F0;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("<h1 class='main-title'>Data Classification Using AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>🌸 An Interactive K-Nearest Neighbors (KNN) Iris Flower Classifier 🌸</p>", unsafe_allow_html=True)

# ==========================================
# 2. LOAD DATASET
# ==========================================
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

iris_data, raw_df = load_and_prepare_data()

# Sidebar for Model Configuration & Custom Predictions
st.sidebar.markdown("### ⚙️ KNN Configurations")
k_val = st.sidebar.slider(
    "Choose Hyperparameter K (Number of Neighbors)", 
    min_value=1, 
    max_value=15, 
    value=5, 
    step=2,
    help="Number of nearest neighbors to consult for the classification decision."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧬 Custom Flower Predictor")
st.sidebar.write("Input measurements for an unknown Iris flower:")

custom_sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
custom_sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)
custom_petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.35, 0.1)
custom_petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.3, 0.1)

# Tabs navigation
tab_explorer, tab_preprocess, tab_model, tab_prediction = st.tabs([
    "📊 Dataset Explorer", 
    "⚙️ Preprocessing & Split", 
    "🧠 Model Training & K-Analysis", 
    "🔮 Custom Prediction"
])

# ==========================================
# 3. EXPLORATORY DATA DISPLAY
# ==========================================
with tab_explorer:
    st.markdown("## 📊 Exploratory Data Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class='premium-card'>
            <h3>Dataset Overview</h3>
            <p>The <b>Iris flower dataset</b> is a classic dataset in machine learning, introduced by the British statistician Ronald Fisher in 1936.</p>
            <ul>
                <li><b>Total Samples:</b> 150 (50 for each of the three species)</li>
                <li><b>Features:</b> 4 physical measurements (sepal length, sepal width, petal length, petal width)</li>
                <li><b>Target Classes:</b> 3 species (Iris Setosa, Iris Versicolor, Iris Virginica)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Display dataset dimensions using KPI metrics
        st.markdown(f"""
        <div class='kpi-container'>
            <div class='kpi-card' style='border-left-color: #FF6B6B;'>
                <p class='kpi-value'>{raw_df.shape[0]}</p>
                <p class='kpi-label'>Total Rows</p>
            </div>
            <div class='kpi-card' style='border-left-color: #FF8E53;'>
                <p class='kpi-value'>{raw_df.shape[1] - 2}</p>
                <p class='kpi-label'>Features</p>
            </div>
            <div class='kpi-card' style='border-left-color: #4D96FF;'>
                <p class='kpi-value'>3</p>
                <p class='kpi-label'>Target Classes</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("#### 📄 First 5 Rows of the Dataset")
        st.dataframe(raw_df.head(), use_container_width=True)
        
    with col2:
        st.markdown("""
        <div class='premium-card'>
            <h3>Feature & Target Information</h3>
            <p><b>Feature Names (Predictors):</b></p>
            <code>sepal length (cm)</code>, <code>sepal width (cm)</code>, <code>petal length (cm)</code>, <code>petal width (cm)</code>
            <br/><br/>
            <p><b>Class Names (Targets):</b></p>
            <span style='color:#10B981; font-weight:bold;'>0: Iris Setosa</span> | 
            <span style='color:#F59E0B; font-weight:bold;'>1: Iris Versicolor</span> | 
            <span style='color:#8B5CF6; font-weight:bold;'>2: Iris Virginica</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Summary statistics
        st.write("#### 📈 Feature Descriptive Statistics")
        st.dataframe(raw_df.iloc[:, :4].describe(), use_container_width=True)

    st.markdown("---")
    st.write("### 🎨 Beginner-Friendly Visualizations")
    
    col_viz1, col_viz2 = st.columns([1, 1.2])
    
    with col_viz1:
        st.write("#### 1. Class Distribution of Iris Species")
        fig, ax = plt.subplots(figsize=(6, 4))
        # Custom elegant palette
        colors = ['#10B981', '#F59E0B', '#8B5CF6']
        sns.countplot(data=raw_df, x='species', palette=colors, ax=ax)
        ax.set_title("Balanced Class Distribution (50 samples each)", fontsize=11, fontweight='bold', pad=12)
        ax.set_xlabel("Species", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        sns.despine()
        st.pyplot(fig)
        st.info("💡 A balanced dataset ensures the model doesn't develop bias towards any particular species during training.")
        
    with col_viz2:
        st.write("#### 2. Feature Relationships (Petal Length vs Petal Width)")
        # We can draw a scatter plot showing species separation based on petal measurements
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.scatterplot(
            data=raw_df, 
            x='petal length (cm)', 
            y='petal width (cm)', 
            hue='species', 
            palette=colors, 
            style='species',
            s=80, 
            ax=ax
        )
        ax.set_title("Petal Length vs Width (Clear linear separation of Setosa)", fontsize=11, fontweight='bold', pad=12)
        ax.set_xlabel("Petal Length (cm)", fontsize=9)
        ax.set_ylabel("Petal Width (cm)", fontsize=9)
        ax.legend(title="Species", frameon=True, facecolor='white', edgecolor='#E2E8F0')
        sns.despine()
        st.pyplot(fig)
        st.success("📝 **Observation:** Iris Setosa (green) forms a completely distinct cluster, while Versicolor (orange) and Virginica (purple) have a slight overlap. Petal dimensions are highly discriminative features!")

# ==========================================
# 4. PREPROCESSING
# ==========================================
# Separate features X and labels y
X = iris_data.data
y = iris_data.target

# ==========================================
# 5. TRAIN-TEST SPLIT
# ==========================================
# Split the dataset into training (80%) and testing (20%) sets
# Shuffling enabled (default), random_state=42 for reproducibility
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Apply Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

with tab_preprocess:
    st.markdown("## ⚙️ Data Preprocessing & Scaling Pipeline")
    
    col_pre1, col_pre2 = st.columns([1.2, 1])
    
    with col_pre1:
        st.markdown("""
        <div class='premium-card' style='border-left: 5px solid #FF8E53;'>
            <h3>🧠 Why Feature Scaling is Crucial for KNN</h3>
            <p><b>K-Nearest Neighbors (KNN)</b> is a distance-based algorithm. It calculates the Euclidean distance between data points to find neighbors:</p>
            <p style='text-align: center;'><code>Distance = &radic;((x<sub>1</sub> - y<sub>1</sub>)<sup>2</sup> + (x<sub>2</sub> - y<sub>2</sub>)<sup>2</sup> + ...)</code></p>
            <p>If one feature has values ranging from 0 to 1000 (e.g., area) and another ranges from 0 to 1 (e.g., probability), the feature with larger magnitudes will completely dominate the distance calculations. Feature scaling centers and normalizes the attributes so they contribute equally.</p>
            <p><b>StandardScaler</b> scales the data to have a mean (&mu;) of <b>0</b> and a standard deviation (&sigma;) of <b>1</b> using the formula:</p>
            <p style='text-align: center; font-size: 1.15rem; font-weight: bold;'>z = (x - &mu;) / &sigma;</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_pre2:
        st.markdown("""
        <div class='premium-card' style='border-left: 5px solid #4D96FF;'>
            <h3>✂️ Train-Test Split (80/20 Ratio)</h3>
            <p>To evaluate model performance objectively, we split the 150 samples into:</p>
            <ul>
                <li><b>Training Set (80%):</b> 120 samples (used to train the classifier)</li>
                <li><b>Testing Set (20%):</b> 30 samples (held out to evaluate predictions)</li>
            </ul>
            <p>We use <code>stratify=y</code> to ensure training and testing splits have exactly the same proportion of target classes, and set <code>random_state=42</code> for reproducible splits.</p>
        </div>
        """, unsafe_allow_html=True)
        
    # Visualizing Scaled vs Unscaled Data
    st.write("### 🔍 Before vs. After Scaling Visualization")
    
    # Create simple dataframes for visualization
    unscaled_sample = pd.DataFrame(X_train_raw, columns=iris_data.feature_names)
    scaled_sample = pd.DataFrame(X_train, columns=iris_data.feature_names)
    
    col_plot1, col_plot2 = st.columns([1, 1])
    
    with col_plot1:
        st.write("#### Original Unscaled Features (Sepal Length vs Sepal Width)")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.scatterplot(data=unscaled_sample, x='sepal length (cm)', y='sepal width (cm)', color='#38BDF8', alpha=0.8, ax=ax)
        ax.set_title("Mean & Range vary by feature", fontsize=10)
        sns.despine()
        st.pyplot(fig)
        
    with col_plot2:
        st.write("#### Scaled Features (Standardized)")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.scatterplot(data=scaled_sample, x='sepal length (cm)', y='sepal width (cm)', color='#818CF8', alpha=0.8, ax=ax)
        ax.set_title("Mean = 0, Std Dev = 1 for all features", fontsize=10)
        sns.despine()
        st.pyplot(fig)

# ==========================================
# 6. MODEL TRAINING & 7. PREDICTION
# ==========================================
# Fit KNN model using chosen K from sidebar
knn_model = KNeighborsClassifier(n_neighbors=k_val)
knn_model.fit(X_train, y_train)

# Predict on test set
y_pred = knn_model.predict(X_test)

# ==========================================
# 8. EVALUATION
# ==========================================
acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')
conf_mat = confusion_matrix(y_test, y_pred)
class_rep = classification_report(y_test, y_pred, target_names=iris_data.target_names)

with tab_model:
    st.markdown("## 🧠 KNN Model Training & Evaluation")
    st.write(f"The KNN model is fitted using **K = {k_val}** neighbors.")
    
    # Display primary evaluation metrics using high-quality cards
    st.markdown(f"""
    <div class='kpi-container'>
        <div class='kpi-card' style='border-left-color: #10B981;'>
            <p class='kpi-value'>{acc * 100:.1f}%</p>
            <p class='kpi-label'>Test Accuracy</p>
        </div>
        <div class='kpi-card' style='border-left-color: #8B5CF6;'>
            <p class='kpi-value'>{f1_macro:.4f}</p>
            <p class='kpi-label'>F1 Score (Macro)</p>
        </div>
        <div class='kpi-card' style='border-left-color: #F59E0B;'>
            <p class='kpi-value'>{k_val}</p>
            <p class='kpi-label'>Value of K</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_eval1, col_eval2 = st.columns([1, 1])
    
    # ==========================================
    # 9. VISUALIZATION
    # ==========================================
    with col_eval1:
        st.write("#### Confusion Matrix Heatmap")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        # Custom color map matching palette
        sns.heatmap(
            conf_mat, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            xticklabels=iris_data.target_names, 
            yticklabels=iris_data.target_names,
            cbar=False,
            ax=ax,
            annot_kws={"size": 12, "weight": "bold"}
        )
        ax.set_title("Predictions vs Actual Classes", fontsize=11, fontweight='bold', pad=12)
        ax.set_xlabel("Predicted Label", fontsize=9, fontweight='bold')
        ax.set_ylabel("True Label", fontsize=9, fontweight='bold')
        st.pyplot(fig)
        st.info("🔍 **Understanding Confusion Matrix:** The diagonal squares represent correct predictions. Off-diagonal values indicate misclassifications.")
        
    with col_eval2:
        st.write("#### Text-based Classification Report")
        st.markdown("Detailed precision, recall, and f1-score per class:")
        st.code(class_rep, language='text')
        
    st.markdown("---")
    
    # Nice-to-have: Comparison of K values
    st.write("### 📈 Hyperparameter Tuning: Analysis of K-Values")
    st.markdown("Let's analyze how the value of **K** (number of neighbors) impacts model accuracy. We train multiple KNN models with different K values ($k=1, 3, 5, 7, 9, 11, 13, 15$) and plot their respective test accuracies.")
    
    # Calculate accuracies for different K values
    k_range = [1, 3, 5, 7, 9, 11, 13, 15]
    accuracies = []
    
    for k in k_range:
        temp_model = KNeighborsClassifier(n_neighbors=k)
        temp_model.fit(X_train, y_train)
        temp_pred = temp_model.predict(X_test)
        accuracies.append(accuracy_score(y_test, temp_pred))
        
    # Plot accuracy curve
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(k_range, accuracies, marker='o', linestyle='-', color='#4D96FF', markersize=8, linewidth=2)
    ax.set_title("Model Accuracy vs K-Neighbors Value", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Value of K (Neighbors)", fontsize=10)
    ax.set_ylabel("Testing Accuracy Score", fontsize=10)
    ax.set_xticks(k_range)
    ax.set_ylim(0.8, 1.02)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Highlight selected K value
    if k_val in k_range:
        idx = k_range.index(k_val)
        ax.scatter(k_val, accuracies[idx], color='#FF6B6B', s=200, zorder=5, label=f"Selected K={k_val}")
        ax.legend()
        
    sns.despine()
    st.pyplot(fig)
    
    st.markdown("""
    💡 **Key Concept (Bias-Variance Tradeoff):** 
    - A **low K value (e.g., K=1)** makes the decision boundary highly complex and sensitive to noise (overfitting).
    - A **high K value (e.g., K=15)** makes the decision boundary smoother, but may overlook local patterns (underfitting).
    """)

# ==========================================
# 10. OPTIONAL CUSTOM INPUT PREDICTION
# ==========================================
with tab_prediction:
    st.markdown("## 🔮 Real-Time Species Classifier")
    st.write("Adjust the features on the **sidebar menu** to see the classifier predict the flower species instantly.")
    
    # Create input vector
    input_vector = np.array([[custom_sepal_length, custom_sepal_width, custom_petal_length, custom_petal_width]])
    
    # Fit scaling transform
    input_scaled = scaler.transform(input_vector)
    
    # Predict species index and probabilities
    pred_class_idx = knn_model.predict(input_scaled)[0]
    pred_probabilities = knn_model.predict_proba(input_scaled)[0]
    predicted_species = iris_data.target_names[pred_class_idx]
    
    # Show input values summary in a table
    st.write("#### Inputs Received:")
    input_df = pd.DataFrame({
        "Sepal Length (cm)": [custom_sepal_length],
        "Sepal Width (cm)": [custom_sepal_width],
        "Petal Length (cm)": [custom_petal_length],
        "Petal Width (cm)": [custom_petal_width]
    })
    st.dataframe(input_df, hide_index=True)
    
    # Display premium result banner
    badge_style = "badge-setosa"
    species_desc = "Typically characterized by short, narrow petals and wider sepals."
    flower_icon = "🌸"
    
    if predicted_species == "versicolor":
        badge_style = "badge-versicolor"
        species_desc = "Typically characterized by moderate lengths and widths of both sepals and petals."
        flower_icon = "🌺"
    elif predicted_species == "virginica":
        badge_style = "badge-virginica"
        species_desc = "Typically characterized by long, wide petals and larger sepals."
        flower_icon = "🌹"
        
    st.markdown(f"""
    <div class='prediction-badge {badge_style}'>
        Predicted Species: {predicted_species.upper()} {flower_icon}
    </div>
    <div style='text-align: center; margin-top: 10px; color: #64748B; font-size: 1.1rem; font-style: italic;'>
        "{species_desc}"
    </div>
    """, unsafe_allow_html=True)
    
    # Visualizing prediction confidence
    st.write("#### Prediction Confidence Probabilities")
    fig, ax = plt.subplots(figsize=(7, 2.5))
    probs_df = pd.DataFrame({
        "Species": [name.upper() for name in iris_data.target_names],
        "Probability": pred_probabilities
    })
    sns.barplot(data=probs_df, x='Probability', y='Species', palette=colors, ax=ax)
    ax.set_xlim(0, 1.05)
    ax.set_title("Probability Distribution among Classes", fontsize=10, fontweight='bold')
    ax.set_xlabel("Confidence / Probability Score", fontsize=8)
    ax.set_ylabel("Species", fontsize=8)
    sns.despine()
    st.pyplot(fig)
    
    # Explain how the decision was made based on distance
    st.markdown("### 🧮 Understanding the prediction:")
    st.write(f"The model calculates distances from your input features to all 120 samples in the scaled training dataset. It then identifies the **{k_val} nearest samples**. The species of these {k_val} samples vote to determine the final prediction.")
