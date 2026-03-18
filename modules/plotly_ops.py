import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st

def plotly_demo():
    st.header("📊 Módulo Plotly - Visualizações Interativas")
    st.markdown("Crie gráficos interativos com seus dados ou com datasets de exemplo.")

    # Upload ou dados de exemplo
    uploaded_file = st.file_uploader("Carregue um CSV (opcional)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Prévia:", df.head())
    else:
        # Dados de exemplo
        dataset_option = st.selectbox("Escolha um dataset de exemplo", [
            "Iris", "Gapminder", "Stocks", "Superfície 3D"
        ])
        if dataset_option == "Iris":
            df = px.data.iris()
        elif dataset_option == "Gapminder":
            df = px.data.gapminder().query("year==2007")
        elif dataset_option == "Stocks":
            df = px.data.stocks()
        else:
            # Superfície 3D
            x = np.linspace(-5,5,50)
            y = np.linspace(-5,5,50)
            X, Y = np.meshgrid(x, y)
            Z = np.sin(np.sqrt(X**2+Y**2))
            # Não é DataFrame, trataremos separadamente
            df = None

    # Seleção do tipo de gráfico
    chart_type = st.selectbox("Tipo de gráfico", [
        "Dispersão", "Linha", "Barra", "Histograma", "Mapa coroplético", "Superfície 3D"
    ])

    if chart_type == "Superfície 3D" and df is None:
        # Usar dados da superfície
        x = np.linspace(-5,5,50)
        y = np.linspace(-5,5,50)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(np.sqrt(X**2+Y**2))
        fig = go.Figure(data=[go.Surface(z=Z, x=x, y=y, colorscale='Viridis')])
        fig.update_layout(title="Superfície 3D", scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
        st.plotly_chart(fig)
        return

    if df is None:
        st.error("Para este gráfico, escolha um dataset de exemplo ou faça upload.")
        return

    # Configurações comuns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if not numeric_cols:
        st.error("Dataset não contém colunas numéricas.")
        return

    if chart_type == "Dispersão":
        x = st.selectbox("Eixo X", numeric_cols, index=0)
        y = st.selectbox("Eixo Y", numeric_cols, index=min(1, len(numeric_cols)-1))
        color = st.selectbox("Cor (opcional)", ["Nenhuma"] + df.columns.tolist())
        if color != "Nenhuma":
            fig = px.scatter(df, x=x, y=y, color=color, title=f"Dispersão de {x} vs {y}")
        else:
            fig = px.scatter(df, x=x, y=y, title=f"Dispersão de {x} vs {y}")
        st.plotly_chart(fig)

    elif chart_type == "Linha":
        x = st.selectbox("Eixo X", df.columns)
        y = st.multiselect("Eixo Y (séries)", numeric_cols, default=numeric_cols[:2])
        if y:
            fig = px.line(df, x=x, y=y, title="Séries temporais")
            st.plotly_chart(fig)

    elif chart_type == "Barra":
        x = st.selectbox("Eixo X (categorias)", df.columns)
        y = st.selectbox("Eixo Y (valores)", numeric_cols)
        fig = px.bar(df, x=x, y=y, title=f"Barra de {y} por {x}")
        st.plotly_chart(fig)

    elif chart_type == "Histograma":
        col = st.selectbox("Coluna", numeric_cols)
        nbins = st.slider("Número de bins", 5, 100, 30)
        fig = px.histogram(df, x=col, nbins=nbins, title=f"Histograma de {col}")
        st.plotly_chart(fig)

    elif chart_type == "Mapa coroplético":
        if 'iso_alpha' in df.columns and 'gdpPercap' in df.columns:
            fig = px.choropleth(df, locations="iso_alpha", color="gdpPercap",
                                 hover_name="country", title="PIB per capita")
            st.plotly_chart(fig)
        else:
            st.warning("Dataset não tem colunas 'iso_alpha' e 'gdpPercap'. Usando dados de exemplo.")
            df = px.data.gapminder().query("year==2007")
            fig = px.choropleth(df, locations="iso_alpha", color="gdpPercap",
                                 hover_name="country", title="PIB per capita (exemplo)")
            st.plotly_chart(fig)

            