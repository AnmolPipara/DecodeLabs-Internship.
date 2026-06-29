import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def render_prediction_tab(knn_model, scaler, custom_sepal_length, custom_sepal_width, custom_petal_length, custom_petal_width, iris_data, k_val):
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
    colors = ['#10B981', '#F59E0B', '#8B5CF6']
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
