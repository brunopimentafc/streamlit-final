# ===============================================
# app.py  (dentro da pasta streamlit_app/)
# Aplicativo Streamlit para escoragem de clientes
# usando modelo de Regressão Logística (Sklearn)
# ===============================================

import streamlit as st
import pandas as pd

from score_functions import (
    load_scoring_model,
    score_base_clientes,
)

# ------------------------------------------------
# Configuração da página
# ------------------------------------------------
st.set_page_config(
    page_title="Credit Scoring - Regressão Logística",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Credit Scoring - Regressão Logística")
st.write(
    """
Este aplicativo aplica um modelo de **credit scoring** treinado em regressão logística.
Faça o upload de um arquivo **CSV** com a base de clientes para calcular o **score de inadimplência (`mau`)**.
"""
)

# Colunas que o modelo espera receber (sem a variável alvo)
REQUIRED_COLS = [
    "sexo",
    "posse_de_veiculo",
    "posse_de_imovel",
    "qtd_filhos",
    "tipo_renda",
    "educacao",
    "estado_civil",
    "tipo_residencia",
    "idade",
    "tempo_emprego",
    "qt_pessoas_residencia",
    "renda",
]

# ------------------------------------------------
# Carregar modelo (com cache)
# ------------------------------------------------
@st.cache_resource
def carregar_modelo():
    return load_scoring_model("model/model_final.pkl")


try:
    modelo = carregar_modelo()
except FileNotFoundError as e:
    st.error(
        f"❌ Não encontrei o arquivo de modelo: `{e}`\n\n"
        "Verifique se o arquivo **model_final.pkl** está dentro da pasta **model/**."
    )
    st.stop()

# ------------------------------------------------
# Seção de upload do CSV
# ------------------------------------------------
st.header("📁 Upload da base de clientes")

st.write(
    """
Envie um arquivo **CSV** com as seguintes colunas obrigatórias:

- `sexo`, `posse_de_veiculo`, `posse_de_imovel`, `qtd_filhos`  
- `tipo_renda`, `educacao`, `estado_civil`, `tipo_residencia`  
- `idade`, `tempo_emprego`, `qt_pessoas_residencia`, `renda`
"""
)

arquivo = st.file_uploader(
    "Selecione um arquivo CSV",
    type=["csv"],
    help="O arquivo deve estar em formato CSV (separado por vírgula).",
)

if arquivo is not None:
    # --------------------------------------------
    # 1) Ler CSV
    # --------------------------------------------
    try:
        df_clientes = pd.read_csv(arquivo)
    except Exception as e:
        st.error(f"Erro ao ler o CSV: {e}")
        st.stop()

    st.success(f"Arquivo carregado com sucesso! ({len(df_clientes)} linhas)")
    st.subheader("👀 Prévia da base enviada:")
    st.dataframe(df_clientes.head(), use_container_width=True)

    # --------------------------------------------
    # 2) Validar colunas obrigatórias
    # --------------------------------------------
    missing = [col for col in REQUIRED_COLS if col not in df_clientes.columns]

    if missing:
        st.error(
            "⚠️ O arquivo enviado está faltando as seguintes colunas obrigatórias:\n\n"
            + ", ".join(f"`{c}`" for c in missing)
        )
        st.stop()

    # --------------------------------------------
    # 3) Aplicar modelo para gerar scores
    # --------------------------------------------
    with st.spinner("Calculando scores de inadimplência..."):
        df_scores = score_base_clientes(df_clientes[REQUIRED_COLS], modelo)

    # --------------------------------------------
    # 4) Mostrar resultados
    # --------------------------------------------
    st.header("📊 Resultados")

    st.write("Prévia dos resultados com score (primeiras linhas):")
    st.dataframe(df_scores.head(), use_container_width=True)

    # --------------------------------------------
    # 5) Download do CSV com scores
    # --------------------------------------------
    st.header("📥 Download dos scores")

    csv_scores = df_scores.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="⬇️ Baixar CSV com scores",
        data=csv_scores,
        file_name="scores_clientes.csv",
        mime="text/csv",
    )

else:
    st.info("⬆️ Faça upload de um arquivo CSV para começar.")
