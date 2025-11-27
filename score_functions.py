# ===============================================
# score_functions.py
# Funções de escoragem e carregamento do modelo
# ===============================================

import pandas as pd
import numpy as np
import pickle
import os

# -------------------------------------------
# 1) Carregar modelo treinado (.pkl)
# -------------------------------------------

def load_scoring_model(path_model="model/model_final.pkl"):
    if not os.path.exists(path_model):
        raise FileNotFoundError(f"⚠️ Modelo não encontrado: {path_model}")
    with open(path_model, "rb") as f:
        modelo = pickle.load(f)
    return modelo

# -------------------------------------------
# 2) Pré-processamento necessário antes do score
# -------------------------------------------

def preprocess_clientes(df):
    df = df.copy()

    # Criar coluna estrutural usada no treino
    df["tempo_emprego_zero"] = (df["tempo_emprego"] == 0).astype(int)
    df.loc[df["tempo_emprego"] == 0, "tempo_emprego"] = np.nan

    return df

# -------------------------------------------
# 3) Gerar o score / probabilidade + classe
# -------------------------------------------

def score_base_clientes(df_clientes, modelo):
    df = preprocess_clientes(df_clientes)

    # Aplica o pipeline + modelo
    prob = modelo.predict_proba(df)[:, 1]  # classe 1 = mau

    df_out = df.copy()
    df_out["score_inadimplencia"] = prob
    df_out["classe_prevista"] = (prob >= 0.5).astype(int)

    return df_out

# -------------------------------------------
# 4) Gerar template CSV para upload no app
# -------------------------------------------

def gerar_template_csv(df_treino):
    colunas = df_treino.drop(columns=["mau"]).columns.tolist()
    exemplo = pd.DataFrame({col: [] for col in colunas})
    exemplo.to_csv("template_upload.csv", index=False)
    return exemplo
