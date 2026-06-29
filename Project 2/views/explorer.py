import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def render_explorer_tab(raw_df):
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
        colors = ['#10B981', '#F59E0B', '#8B5CF6']
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
