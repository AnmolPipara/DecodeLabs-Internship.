import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def render_preprocess_tab(iris_data, X_train_raw, X_train):
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
