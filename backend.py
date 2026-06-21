"""
==================================================================
ETAPA B — Backend + Agente Inteligente (FastAPI + Gemini)
Dataset: Breast Cancer Wisconsin (Diagnostic) Data Set
==================================================================

Carrega os artefatos exportados pela Etapa A:
    modelo_melhor.joblib, scaler.joblib, colunas_features.joblib,
    encoders.joblib

Endpoints:
    GET  /schema   -> descreve as features esperadas (nome, tipo,
                       categorias possíveis) para o frontend montar
                       o formulário dinamicamente.
    POST /predict  -> recebe os valores das variáveis, roda o modelo
                       e devolve a predição bruta.
    POST /explain  -> recebe a predição (+ os dados de entrada) e
                       pede ao Gemini que explique o resultado em
                       linguagem natural, de forma fundamentada.

RODAR:
    pip install fastapi uvicorn joblib pandas requests
    export GEMINI_API_KEY="sua_chave_aqui"
    uvicorn backend:app --reload --port 8000
"""

import os
import joblib
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-3.1-flash-lite"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
)

ARTIFACTS_DIR = "."

SYSTEM_PROMPT = """Você é um assistente que explica, de forma clara e em
português, o resultado de um modelo estatístico de machine learning treinado
sobre medidas de exames de imagem (o dataset Breast Cancer Wisconsin
Diagnostic), que classifica uma amostra como BENIGNA ou MALIGNA com base em
características morfológicas de células (raio, textura, perímetro, área,
etc.).

REGRAS OBRIGATÓRIAS:
1. Use SOMENTE as informações fornecidas no contexto (predição, probabilidade
   e dados de entrada). NUNCA invente números, fatos, estatísticas ou
   recomendações que não possam ser inferidos diretamente dos dados
   fornecidos.
2. Se não houver dados suficientes para justificar uma afirmação, diga
   explicitamente que essa informação não está disponível, em vez de supor.
3. Este resultado é puramente estatístico/educacional e NÃO é um diagnóstico
   médico. Você deve declarar isso claramente em toda resposta e recomendar
   que qualquer decisão clínica seja tomada por um médico, com exames
   complementares.
4. NUNCA dê conselhos de tratamento, prognóstico, ou afirme probabilidade de
   sobrevida. Mantenha o foco em explicar o que o modelo encontrou nos dados,
   não no que isso significa clinicamente para a pessoa.
5. Seja objetivo: 1 a 3 parágrafos curtos. Explique o resultado e, se
   identificável a partir dos dados de entrada, quais medidas parecem ter
   mais relação com o resultado, evitando jargão técnico de ML ("classe",
   "features", "score").
6. Nunca afirme certeza absoluta. O modelo é probabilístico e pode errar
   (falsos positivos e falsos negativos existem).
"""

app = FastAPI(title="Agente Preditivo Especialista - Etapa B")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# CARREGAMENTO DOS ARTEFATOS (uma vez, na inicialização)
# ------------------------------------------------------------------
try:
    modelo = joblib.load(f"{ARTIFACTS_DIR}/modelo_melhor.joblib")
    scaler = joblib.load(f"{ARTIFACTS_DIR}/scaler.joblib")
    colunas_features = joblib.load(f"{ARTIFACTS_DIR}/colunas_features.joblib")
    encoders = joblib.load(f"{ARTIFACTS_DIR}/encoders.joblib")
except FileNotFoundError as e:
    raise RuntimeError(
        "Artefatos não encontrados. Rode o script da Etapa A primeiro "
        "para gerar modelo_melhor.joblib, scaler.joblib, "
        "colunas_features.joblib e encoders.joblib na mesma pasta do backend."
    ) from e


# ------------------------------------------------------------------
# SCHEMAS (Pydantic)
# ------------------------------------------------------------------
class PredictRequest(BaseModel):
    dados: Dict[str, Any]   # ex: {"age": 17, "studytime": 2, "sex": "F", ...}


class ExplainRequest(BaseModel):
    dados: Dict[str, Any]
    predicao: str            # "Aprovado" ou "Reprovado"
    probabilidade: float     # 0.0 a 1.0


# ------------------------------------------------------------------
# /schema — descreve as features para o frontend
# ------------------------------------------------------------------
@app.get("/schema")
def schema():
    campos = []
    for col in colunas_features:
        if col in encoders:
            campos.append({
                "nome": col,
                "tipo": "categorico",
                "opcoes": list(encoders[col].classes_),
            })
        else:
            campos.append({
                "nome": col,
                "tipo": "numerico",
            })
    return {"campos": campos}


# ------------------------------------------------------------------
# /predict — roda o modelo
# ------------------------------------------------------------------
def _codificar_entrada(dados: Dict[str, Any]) -> pd.DataFrame:
    linha = {}
    for col in colunas_features:
        if col not in dados:
            raise HTTPException(
                status_code=400,
                detail=f"Campo obrigatório ausente: '{col}'"
            )
        valor = dados[col]
        if col in encoders:
            le = encoders[col]
            if valor not in le.classes_:
                raise HTTPException(
                    status_code=400,
                    detail=f"Valor inválido para '{col}': '{valor}'. "
                           f"Opções válidas: {list(le.classes_)}"
                )
            valor = le.transform([valor])[0]
        linha[col] = valor

    df_linha = pd.DataFrame([linha])[colunas_features]  # garante a ordem certa
    return df_linha


@app.post("/predict")
def predict(req: PredictRequest):
    df_linha = _codificar_entrada(req.dados)
    X_scaled = scaler.transform(df_linha)

    pred = modelo.predict(X_scaled)[0]

    # nem todo classificador tem predict_proba (mas KNN, MLP e NB têm)
    if hasattr(modelo, "predict_proba"):
        proba = modelo.predict_proba(X_scaled)[0]
        probabilidade = float(proba[1])  # probabilidade da classe "Maligno" (1)
    else:
        probabilidade = float(pred)

    resultado = "Maligno" if pred == 1 else "Benigno"

    return {
        "predicao": resultado,
        "probabilidade_malignidade": round(probabilidade, 4),
    }


# ------------------------------------------------------------------
# /explain — chama o Gemini para traduzir o resultado
# ------------------------------------------------------------------
@app.post("/explain")
def explain(req: ExplainRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY não configurada no ambiente do servidor."
        )

    contexto = (
        f"Predição do modelo: {req.predicao}\n"
        f"Probabilidade estimada de malignidade: {req.probabilidade:.1%}\n"
        f"Medidas de entrada da amostra: {req.dados}\n\n"
        "Explique este resultado para o usuário final, seguindo estritamente "
        "as regras do seu papel."
    )

    body = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": contexto}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 500},
    }

    resp = requests.post(GEMINI_URL, json=body, timeout=30)
    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Erro ao chamar Gemini: {resp.status_code} - {resp.text}"
        )

    data = resp.json()
    try:
        texto = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise HTTPException(
            status_code=502,
            detail=f"Resposta inesperada do Gemini: {data}"
        )

    return {"explicacao": texto}


@app.get("/")
def health():
    return {"status": "ok", "modelo_carregado": type(modelo).__name__}
