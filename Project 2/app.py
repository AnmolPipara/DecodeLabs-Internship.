import streamlit as st

# ==========================================
# 1. SETUP & CUSTOM STYLING
# ==========================================
from utils.style import set_page_config, apply_custom_styles, render_header
set_page_config()
apply_custom_styles()
render_header()

# ==========================================
# 2. IMPORT MODULES
# ==========================================
from utils.ml_core import load_and_prepare_data, get_train_test_split, train_model_and_scale
from views.explorer import render_explorer_tab
from views.preprocess import render_preprocess_tab
from views.model import render_model_tab
from views.prediction import render_prediction_tab

# ==========================================
# 3. LOAD DATA & SIDEBAR CONFIG
# ==========================================
iris_data, raw_df = load_and_prepare_data()

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

# ==========================================
# 4. PREPARE DATA & TRAIN MODEL
# ==========================================
X_train_raw, X_test_raw, y_train, y_test = get_train_test_split(iris_data)
scaler, X_train, X_test, knn_model = train_model_and_scale(X_train_raw, X_test_raw, y_train, k_val)
y_pred = knn_model.predict(X_test)

# ==========================================
# 5. RENDER TABS
# ==========================================
tab_explorer, tab_preprocess, tab_model, tab_prediction = st.tabs([
    "📊 Dataset Explorer", 
    "⚙️ Preprocessing & Split", 
    "🧠 Model Training & K-Analysis", 
    "🔮 Custom Prediction"
])

with tab_explorer:
    render_explorer_tab(raw_df)
    
with tab_preprocess:
    render_preprocess_tab(iris_data, X_train_raw, X_train)
    
with tab_model:
    render_model_tab(knn_model, X_train, y_train, X_test, y_test, y_pred, k_val, iris_data)
    
with tab_prediction:
    render_prediction_tab(knn_model, scaler, custom_sepal_length, custom_sepal_width, custom_petal_length, custom_petal_width, iris_data, k_val)
