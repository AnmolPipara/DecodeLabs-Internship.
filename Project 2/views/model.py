import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.neighbors import KNeighborsClassifier

def render_model_tab(knn_model, X_train, y_train, X_test, y_test, y_pred, k_val, iris_data):
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    conf_mat = confusion_matrix(y_test, y_pred)
    class_rep = classification_report(y_test, y_pred, target_names=iris_data.target_names)

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
    
    with col_eval1:
        st.write("#### Confusion Matrix Heatmap")
        fig, ax = plt.subplots(figsize=(6, 4.5))
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
    
    # Analysis of K values
    st.write("### 📈 Hyperparameter Tuning: Analysis of K-Values")
    st.markdown("Let's analyze how the value of **K** (number of neighbors) impacts model accuracy. We train multiple KNN models with different K values ($k=1, 3, 5, 7, 9, 11, 13, 15$) and plot their respective test accuracies.")
    
    k_range = [1, 3, 5, 7, 9, 11, 13, 15]
    accuracies = []
    
    for k in k_range:
        temp_model = KNeighborsClassifier(n_neighbors=k)
        temp_model.fit(X_train, y_train)
        temp_pred = temp_model.predict(X_test)
        accuracies.append(accuracy_score(y_test, temp_pred))
        
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(k_range, accuracies, marker='o', linestyle='-', color='#4D96FF', markersize=8, linewidth=2)
    ax.set_title("Model Accuracy vs K-Neighbors Value", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Value of K (Neighbors)", fontsize=10)
    ax.set_ylabel("Testing Accuracy Score", fontsize=10)
    ax.set_xticks(k_range)
    ax.set_ylim(0.8, 1.02)
    ax.grid(True, linestyle='--', alpha=0.5)
    
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
