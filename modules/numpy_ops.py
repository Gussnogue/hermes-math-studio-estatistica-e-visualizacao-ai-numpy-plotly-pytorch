import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from io import StringIO

def numpy_demo():
    st.header("🧮 Módulo NumPy - Análise Numérica Avançada")
    st.markdown("Realize operações estatísticas, álgebra linear e transformações em seus dados.")

    # Upload de dados
    uploaded_file = st.file_uploader("Carregue um arquivo CSV (opcional)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Prévia dos dados:", df.head())
        data = df.select_dtypes(include=[np.number]).values
    else:
        # Dados de exemplo
        st.info("Usando dados aleatórios de exemplo (distribuição normal).")
        data = np.random.randn(100, 5)

    st.subheader("Dados de entrada")
    st.write(data[:5, :])

    # Seleção de operações
    op = st.selectbox("Escolha a operação", [
        "Estatísticas Descritivas",
        "Matriz de Correlação",
        "PCA (Análise de Componentes Principais)",
        "FFT (Transformada Rápida de Fourier)",
        "Regressão Linear (Mínimos Quadrados)"
    ])

    if st.button("Executar"):
        with st.spinner("Processando..."):
            if op == "Estatísticas Descritivas":
                st.subheader("Resultados")
                st.write("Média por coluna:", data.mean(axis=0))
                st.write("Desvio padrão:", data.std(axis=0))
                st.write("Mínimo:", data.min(axis=0))
                st.write("Máximo:", data.max(axis=0))
                # Histograma
                fig, ax = plt.subplots()
                ax.hist(data.flatten(), bins=30)
                ax.set_title("Histograma dos dados")
                st.pyplot(fig)

            elif op == "Matriz de Correlação":
                corr = np.corrcoef(data.T)
                st.write("Matriz de correlação:")
                st.write(corr)
                # Heatmap
                fig, ax = plt.subplots()
                cax = ax.matshow(corr, cmap='coolwarm')
                fig.colorbar(cax)
                st.pyplot(fig)

            elif op == "PCA (Análise de Componentes Principais)":
                # Centralizar
                data_centered = data - data.mean(axis=0)
                cov = np.cov(data_centered.T)
                eigvals, eigvecs = np.linalg.eig(cov)
                # Ordenar
                idx = np.argsort(eigvals)[::-1]
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:, idx]
                st.write("Autovalores:", eigvals)
                st.write("Variância explicada (primeiros 2 componentes):", eigvals[:2].sum() / eigvals.sum())
                # Projeção
                proj = data_centered @ eigvecs[:, :2]
                fig, ax = plt.subplots()
                ax.scatter(proj[:, 0], proj[:, 1], alpha=0.6)
                ax.set_xlabel("Componente 1")
                ax.set_ylabel("Componente 2")
                ax.set_title("Projeção PCA")
                st.pyplot(fig)

            elif op == "FFT (Transformada Rápida de Fourier)":
                # Pega primeira coluna como sinal
                sinal = data[:, 0]
                fft = np.fft.fft(sinal)
                freq = np.fft.fftfreq(len(sinal))
                st.write("FFT (primeiros 10 coeficientes):", fft[:10])
                fig, ax = plt.subplots(2, 1)
                ax[0].plot(sinal)
                ax[0].set_title("Sinal original")
                ax[1].plot(freq[:len(freq)//2], np.abs(fft)[:len(freq)//2])
                ax[1].set_title("Espectro de frequência")
                st.pyplot(fig)

            elif op == "Regressão Linear (Mínimos Quadrados)":
                # Simples: y = X * beta + erro
                # Pega as duas primeiras colunas: X e y
                X = data[:, 0].reshape(-1, 1)
                y = data[:, 1]
                # Adiciona intercepto
                X_design = np.hstack([np.ones((X.shape[0], 1)), X])
                beta = np.linalg.pinv(X_design.T @ X_design) @ X_design.T @ y
                st.write("Coeficientes (intercepto, inclinação):", beta)
                # Plot
                fig, ax = plt.subplots()
                ax.scatter(X, y, label='Dados')
                ax.plot(X, X_design @ beta, color='red', label='Regressão')
                ax.legend()
                st.pyplot(fig)

                