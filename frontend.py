"""
==================================================================
ETAPA C — Interface Web (Streamlit)
==================================================================

Interface simples que:
1. Busca o /schema no backend e monta o formulário automaticamente
   (não precisa hardcodar os campos -- se você trocar o dataset/
   modelo na Etapa A, o formulário se adapta sozinho).
2. Envia os dados para /predict.
3. Envia a predição para /explain (Gemini) e mostra a explicação.

RODAR:
    pip install streamlit requests
    streamlit run frontend.py
"""

import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Agente Preditivo Especialista", page_icon="🩺")
st.title("🩺 Agente Preditivo Especialista")
st.caption("Predição de diagnóstico (Breast Cancer Wisconsin) com explicação em linguagem natural (Gemini)")
st.warning(
    "⚠️ Esta ferramenta é um trabalho acadêmico de Machine Learning. "
    "NÃO substitui diagnóstico médico. Qualquer decisão clínica deve ser "
    "feita por um profissional de saúde.",
    icon="⚠️",
)


@st.cache_data(ttl=60)
def buscar_schema():
    resp = requests.get(f"{BACKEND_URL}/schema", timeout=10)
    resp.raise_for_status()
    return resp.json()["campos"]


try:
    campos = buscar_schema()
except Exception as e:
    st.error(
        f"Não foi possível conectar ao backend em {BACKEND_URL}. "
        f"Verifique se ele está rodando (`uvicorn backend:app --reload --port 8000`).\n\nErro: {e}"
    )
    st.stop()

st.subheader("Medidas da amostra (exame de imagem)")

dados_form = {}
colunas_layout = st.columns(2)

for i, campo in enumerate(campos):
    container = colunas_layout[i % 2]
    nome = campo["nome"]

    if campo["tipo"] == "categorico":
        dados_form[nome] = container.selectbox(nome, campo["opcoes"])
    else:
        dados_form[nome] = container.number_input(
            nome, value=1.0, step=0.01, format="%.4f"
        )

if st.button("🔍 Predizer e explicar", type="primary", use_container_width=True):
    with st.spinner("Consultando o modelo..."):
        try:
            resp_pred = requests.post(
                f"{BACKEND_URL}/predict", json={"dados": dados_form}, timeout=15
            )
            resp_pred.raise_for_status()
            resultado = resp_pred.json()
        except Exception as e:
            st.error(f"Erro ao chamar /predict: {e}")
            st.stop()

    predicao = resultado["predicao"]
    probabilidade = resultado["probabilidade_malignidade"]

    st.divider()
    st.subheader("📊 Resultado bruto do modelo")

    col_a, col_b = st.columns(2)
    with col_a:
        if predicao == "Benigno":
            st.success(f"Predição: **{predicao}**")
        else:
            st.error(f"Predição: **{predicao}**")
    with col_b:
        st.metric("Probabilidade de malignidade", f"{probabilidade:.1%}")

    st.subheader("🤖 Explicação do agente")
    with st.spinner("O agente está interpretando o resultado..."):
        try:
            resp_exp = requests.post(
                f"{BACKEND_URL}/explain",
                json={
                    "dados": dados_form,
                    "predicao": predicao,
                    "probabilidade": probabilidade,
                },
                timeout=30,
            )
            resp_exp.raise_for_status()
            explicacao = resp_exp.json()["explicacao"]
            st.info(explicacao)
        except Exception as e:
            st.error(f"Erro ao chamar /explain: {e}")
