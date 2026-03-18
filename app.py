import streamlit as st
from modules import numpy_ops, plotly_ops, pytorch_ops, ai_expert

st.set_page_config(page_title="Data Science Showcase", layout="wide")

st.title("🔬 Data Science Showcase - Módulos Avançados")
st.markdown("Demonstração de habilidades com **NumPy**, **Plotly**, **PyTorch** e **IA local** (Hermes).")

# Menu lateral
modulo = st.sidebar.selectbox(
    "Escolha o módulo",
    ["NumPy - Análise Numérica", "Plotly - Visualização", "PyTorch - Redes Neurais", "IA Expert - Matemática/Estatística"]
)

if modulo == "NumPy - Análise Numérica":
    numpy_ops.numpy_demo()
elif modulo == "Plotly - Visualização":
    plotly_ops.plotly_demo()
elif modulo == "PyTorch - Redes Neurais":
    pytorch_ops.pytorch_demo()
elif modulo == "IA Expert - Matemática/Estatística":
    ai_expert.ai_expert()

    