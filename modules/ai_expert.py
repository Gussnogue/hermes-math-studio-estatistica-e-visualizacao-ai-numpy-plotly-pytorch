import streamlit as st
import requests
import json

def ai_expert():
    st.header("🎓 Assistente Especialista em Matemática e Estatística")
    st.markdown("Faça perguntas sobre conceitos matemáticos e estatísticos. O sistema usa IA local (Hermes) para gerar explicações com fontes confiáveis.")

    # Configuração do LM Studio
    lm_url = st.sidebar.text_input("URL do LM Studio", value="http://localhost:1234/v1/chat/completions")
    model_name = st.sidebar.text_input("Modelo", value="hermes-3-llama-3.2-3b")

    pergunta = st.text_area("Sua pergunta:", placeholder="Ex: Explique o teorema de Bayes com um exemplo prático. Cite fontes.")

    if st.button("Consultar"):
        if not pergunta:
            st.warning("Digite uma pergunta.")
        else:
            with st.spinner("Consultando o especialista..."):
                # Prompt que pede explicação com fontes
                prompt = f"""
                Você é um professor universitário de matemática e estatística. Responda à seguinte pergunta de forma clara, didática e precisa. 
                Inclua referências a fontes confiáveis (livros, artigos, autores) quando apropriado.
                Pergunta: {pergunta}
                """
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": "Você é um especialista em matemática e estatística."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1000
                }
                try:
                    response = requests.post(lm_url, json=payload)
                    response.raise_for_status()
                    resposta = response.json()["choices"][0]["message"]["content"]
                    st.markdown("### Resposta")
                    st.write(resposta)
                except Exception as e:
                    st.error(f"Erro na consulta: {e}")

                    