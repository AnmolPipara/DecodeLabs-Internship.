import streamlit as st

def set_page_config():
    st.set_page_config(
        page_title="Data Classification Using AI – Iris Classifier",
        page_icon="🌸",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def apply_custom_styles():
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

def render_header():
    st.markdown("<h1 class='main-title'>Data Classification Using AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>🌸 An Interactive K-Nearest Neighbors (KNN) Iris Flower Classifier 🌸</p>", unsafe_allow_html=True)
