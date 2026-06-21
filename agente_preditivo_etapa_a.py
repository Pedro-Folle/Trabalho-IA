"""
==================================================================
ETAPA A — Preparação e Modelagem de Dados (Machine Learning)
Dataset: Breast Cancer Wisconsin (Diagnostic) Data Set
Fonte: https://www.kaggle.com/code/vikasukani/breast-cancer-prediction-using-machine-learning
Arquivo esperado: data.csv (separador ',')
==================================================================

ESTRUTURA TÍPICA DO data.csv:
    id, diagnosis (M/B), radius_mean, texture_mean, perimeter_mean,
    area_mean, smoothness_mean, compactness_mean, concavity_mean,
    concave points_mean, symmetry_mean, fractal_dimension_mean,
    ... (versões _se e _worst das mesmas 10 medidas) ...,
    Unnamed: 32 (coluna vazia que algumas versões do CSV trazem)

ADAPTAÇÃO DE TAREFAS NESTE DATASET:
    - Classificação (KNN, MLP, Naive Bayes) -> 'diagnosis' (M=Maligno / B=Benigno)
      Este é o uso clássico e consagrado do dataset.
    - Regressão Linear Múltipla -> não existe alvo contínuo "natural" para
      esse domínio (o objetivo é diagnóstico, não uma nota). Para cumprir a
      etapa de regressão, prevemos 'radius_mean' (raio médio do tumor) a
      partir das demais variáveis -- é uma medida física contínua e
      realista de se estimar.

O QUE ESTE SCRIPT FAZ:
1. Carrega e pré-processa os dados (remove colunas inúteis, codifica
   'diagnosis' em 0/1).
2. Gera os gráficos exploratórios pedidos (correlação, boxplot,
   frequência) com Seaborn.
3. Treina:
   - Regressão Linear Múltipla  -> prediz radius_mean (contínua)
   - KNN, MLP e Naive Bayes     -> predizem 'diagnosis' (classificação)
4. Compara os classificadores via acurácia, sensibilidade,
   especificidade e precisão.
5. Exporta o melhor classificador (.joblib) + o scaler + a lista de
   colunas usadas + os encoders, para reuso no backend (Etapa B).

DEPENDÊNCIAS:
    pip install pandas numpy matplotlib seaborn scikit-learn joblib
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # gera os PNGs sem precisar de display
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    r2_score, mean_squared_error,
    accuracy_score, precision_score, recall_score, confusion_matrix
)

RANDOM_STATE = 42
CSV_PATH = "data.csv"   # ajuste o caminho se necessário
OUTPUT_DIR = "."        # onde salvar gráficos e modelo

COLUNA_DIAGNOSTICO = "diagnosis"
ALVO_REGRESSAO = "radius_mean"


# ------------------------------------------------------------------
# 1. CARGA E PRÉ-PROCESSAMENTO
# ------------------------------------------------------------------
def carregar_dados(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")

    # Remove colunas que não são features (id, e a coluna vazia 'Unnamed: 32'
    # que algumas exportações desse CSV trazem)
    colunas_para_remover = [c for c in ["id", "Unnamed: 32"] if c in df.columns]
    if colunas_para_remover:
        df = df.drop(columns=colunas_para_remover)
        print(f"Colunas removidas (não são features): {colunas_para_remover}")

    print("Valores nulos por coluna:\n", df.isnull().sum().sum(), "valores nulos no total")
    return df


def preprocessar(df: pd.DataFrame):
    df = df.copy()

    # Codifica diagnosis: B (benigno) -> 0, M (maligno) -> 1
    encoders = {}
    le = LabelEncoder()
    df[COLUNA_DIAGNOSTICO] = le.fit_transform(df[COLUNA_DIAGNOSTICO])
    encoders[COLUNA_DIAGNOSTICO] = le

    print(f"'{COLUNA_DIAGNOSTICO}' codificado: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    return df, encoders


# ------------------------------------------------------------------
# 2. ANÁLISE EXPLORATÓRIA (SEABORN)
# ------------------------------------------------------------------
def gerar_graficos(df: pd.DataFrame, output_dir: str = "."):
    sns.set_theme(style="whitegrid")

    # --- Gráfico de correlação (heatmap) ---
    plt.figure(figsize=(16, 12))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
    plt.title("Mapa de correlação entre variáveis")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/grafico_correlacao.png", dpi=150)
    plt.close()

    # --- Box Plot: radius_mean por diagnóstico ---
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=df, x=COLUNA_DIAGNOSTICO, y=ALVO_REGRESSAO,
        hue=COLUNA_DIAGNOSTICO, palette="viridis", legend=False
    )
    plt.title("Distribuição do raio médio do tumor por diagnóstico")
    plt.xlabel("Diagnóstico (0 = Benigno, 1 = Maligno)")
    plt.ylabel("Raio médio (radius_mean)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/grafico_boxplot_radius_por_diagnostico.png", dpi=150)
    plt.close()

    # --- Gráfico de frequência: distribuição de diagnósticos ---
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x=COLUNA_DIAGNOSTICO, hue=COLUNA_DIAGNOSTICO,
                   palette="viridis", legend=False)
    plt.title("Frequência de diagnósticos (0 = Benigno, 1 = Maligno)")
    plt.xlabel("Diagnóstico")
    plt.ylabel("Quantidade de casos")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/grafico_frequencia_diagnostico.png", dpi=150)
    plt.close()

    print(f"Gráficos salvos em: {output_dir}/grafico_*.png")


# ------------------------------------------------------------------
# 3. REGRESSÃO LINEAR MÚLTIPLA (prediz radius_mean)
# ------------------------------------------------------------------
def treinar_regressao_linear(df: pd.DataFrame):
    X = df.drop(columns=[ALVO_REGRESSAO, COLUNA_DIAGNOSTICO])
    y = df[ALVO_REGRESSAO]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    modelo = LinearRegression()
    modelo.fit(X_train_s, y_train)
    y_pred = modelo.predict(X_test_s)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n=== Regressão Linear Múltipla (prediz {ALVO_REGRESSAO}) ===")
    print(f"R²:   {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")

    return {"modelo": modelo, "scaler": scaler, "r2": r2, "rmse": rmse}


# ------------------------------------------------------------------
# 4. CLASSIFICADORES (predizem 'diagnosis'): KNN, MLP, Naive Bayes
# ------------------------------------------------------------------
def calcular_metricas(y_test, y_pred, nome_modelo: str):
    acuracia = accuracy_score(y_test, y_pred)
    precisao = precision_score(y_test, y_pred, zero_division=0)
    sensibilidade = recall_score(y_test, y_pred, zero_division=0)  # recall = sensibilidade

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"\n=== {nome_modelo} ===")
    print(f"Acurácia:      {acuracia:.4f}")
    print(f"Precisão:      {precisao:.4f}")
    print(f"Sensibilidade: {sensibilidade:.4f}")
    print(f"Especificidade:{especificidade:.4f}")

    return {
        "nome": nome_modelo,
        "acuracia": acuracia,
        "precisao": precisao,
        "sensibilidade": sensibilidade,
        "especificidade": especificidade,
    }


def treinar_classificadores(df: pd.DataFrame):
    X = df.drop(columns=[COLUNA_DIAGNOSTICO])
    y = df[COLUNA_DIAGNOSTICO]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    resultados = []
    modelos_treinados = {}

    # --- KNN ---
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_s, y_train)
    pred_knn = knn.predict(X_test_s)
    resultados.append(calcular_metricas(y_test, pred_knn, "KNN"))
    modelos_treinados["KNN"] = knn

    # --- MLP ---
    mlp = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    mlp.fit(X_train_s, y_train)
    pred_mlp = mlp.predict(X_test_s)
    resultados.append(calcular_metricas(y_test, pred_mlp, "MLP (Multi-Layer Perceptron)"))
    modelos_treinados["MLP"] = mlp

    # --- Naive Bayes ---
    nb = GaussianNB()
    nb.fit(X_train_s, y_train)
    pred_nb = nb.predict(X_test_s)
    resultados.append(calcular_metricas(y_test, pred_nb, "Naive Bayes"))
    modelos_treinados["Naive Bayes"] = nb

    return resultados, modelos_treinados, scaler, list(X.columns)


# ------------------------------------------------------------------
# 5. COMPARAÇÃO E EXPORT DO MELHOR MODELO
# ------------------------------------------------------------------
def comparar_e_exportar(resultados, modelos_treinados, scaler, colunas, encoders, output_dir="."):
    tabela = pd.DataFrame(resultados).sort_values("acuracia", ascending=False)
    print("\n=== Comparação final dos classificadores ===")
    print(tabela.to_string(index=False))

    melhor_nome = tabela.iloc[0]["nome"]
    chave = "MLP" if "MLP" in melhor_nome else melhor_nome
    melhor_modelo = modelos_treinados[chave]

    joblib.dump(melhor_modelo, f"{output_dir}/modelo_melhor.joblib")
    joblib.dump(scaler, f"{output_dir}/scaler.joblib")
    joblib.dump(colunas, f"{output_dir}/colunas_features.joblib")
    joblib.dump(encoders, f"{output_dir}/encoders.joblib")

    print(f"\nMelhor modelo: {melhor_nome}")
    print(f"Exportado para: {output_dir}/modelo_melhor.joblib")
    print(f"Scaler exportado para: {output_dir}/scaler.joblib")
    print(f"Lista de colunas exportada para: {output_dir}/colunas_features.joblib")
    print(f"Encoders exportados para: {output_dir}/encoders.joblib")

    return tabela, melhor_nome


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
if __name__ == "__main__":
    df_raw = carregar_dados(CSV_PATH)
    df, encoders = preprocessar(df_raw)

    gerar_graficos(df, OUTPUT_DIR)

    resultado_regressao = treinar_regressao_linear(df)

    resultados_clf, modelos, scaler, colunas = treinar_classificadores(df)

    tabela_final, melhor = comparar_e_exportar(
        resultados_clf, modelos, scaler, colunas, encoders, OUTPUT_DIR
    )

    print("\n==================================================")
    print("RESUMO GERAL")
    print("==================================================")
    print(f"Regressão Linear ({ALVO_REGRESSAO}) -> R²={resultado_regressao['r2']:.4f}  "
          f"RMSE={resultado_regressao['rmse']:.4f}")
    print(f"Melhor classificador               -> {melhor}")
    print("==================================================")
