# Agente Preditivo Especialista — Breast Cancer Wisconsin

Projeto que une Machine Learning (Etapa A), Backend + Agente IA (Etapa B) e
Interface Web (Etapa C) para predizer diagnóstico (Maligno/Benigno) a partir
de medidas de exames de imagem, com explicação em linguagem natural via
Gemini.

**Dataset:** [Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/code/vikasukani/breast-cancer-prediction-using-machine-learning)

---

## 📁 Arquivos do projeto

| Arquivo | Função |
|---|---|
| `agente_preditivo_etapa_a.py` | Treina os modelos e gera os `.joblib` |
| `backend.py` | API FastAPI (`/schema`, `/predict`, `/explain`) |
| `frontend.py` | Interface Streamlit |
| `data.csv` | Dataset (você baixa do Kaggle) |

Os 4 primeiros precisam estar **na mesma pasta**. A Etapa A vai gerar, nessa
mesma pasta:

```
modelo_melhor.joblib
scaler.joblib
colunas_features.joblib
encoders.joblib
grafico_correlacao.png
grafico_boxplot_radius_por_diagnostico.png
grafico_frequencia_diagnostico.png
```

---

## ✅ Pré-requisitos

- Python 3.10+ instalado
- Uma chave de API do Gemini (gratuita): [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

Verifique sua versão do Python:

```powershell
python --version
```

---

## 1️⃣ Baixar o dataset

1. Acesse o link do Kaggle acima.
2. Baixe o arquivo `data.csv` (botão de download na aba "Data" do notebook/dataset).
3. Coloque o `data.csv` na mesma pasta dos scripts (`agente_preditivo_etapa_a.py`, `backend.py`, `frontend.py`).

---

## 2️⃣ Instalar as dependências

Abra o PowerShell **na pasta do projeto** (`cd` até lá) e rode:

```powershell
python -m pip install pandas numpy matplotlib seaborn scikit-learn joblib fastapi uvicorn requests streamlit
```

> ⚠️ Confira a grafia: é **joblib** (com "b" no final), não "jobli" ou variações.

---

## 3️⃣ Etapa A — Treinar os modelos

```powershell
python agente_preditivo_etapa_a.py
```

**O que deve acontecer:** o terminal vai imprimir o carregamento do dataset,
as métricas de cada modelo (KNN, MLP, Naive Bayes) e da Regressão Linear, e
no final uma tabela comparando os classificadores + qual foi exportado como
"melhor modelo".

**Você só deve seguir para a Etapa B depois que este passo rodar sem erro**
e os 4 arquivos `.joblib` aparecerem na pasta.

---

## 4️⃣ Etapa B — Rodar o backend (FastAPI)

### 4.1. Configurar a chave do Gemini (nessa mesma sessão do terminal)

No **PowerShell** (não use `export`, isso é sintaxe de Linux/bash):

```powershell
$env:GEMINI_API_KEY="sua_chave_aqui"
```

> Essa variável vale só para o terminal atual. Se você fechar e abrir um novo
> terminal, precisa rodar esse comando de novo antes de iniciar o backend.

### 4.2. Iniciar o servidor

```powershell
python -m uvicorn backend:app --reload --port 8000
```

> Usamos `python -m uvicorn` em vez de só `uvicorn` para evitar o erro de
> "comando não reconhecido" quando o Windows não tem o executável no PATH.

**O que deve acontecer:** o terminal mostra `Application startup complete.`
e fica esperando requisições. **Deixe esse terminal aberto** — é o backend
rodando.

### 4.3. Testar rapidamente (opcional)

Com o backend rodando, abra no navegador:

```
http://127.0.0.1:8000/docs
```

Você verá a documentação interativa (Swagger). Pode testar `/schema` e
`/predict` direto por ali, sem precisar do frontend.

---

## 5️⃣ Etapa C — Rodar o frontend (Streamlit)

Abra um **segundo terminal** (deixe o backend rodando no primeiro), navegue
até a mesma pasta do projeto e rode:

```powershell
python -m streamlit run frontend.py
```

Isso deve abrir automaticamente uma aba no navegador em
`http://localhost:8501`. Se não abrir, copie esse endereço manualmente.

---

## 🔁 Resumo do fluxo (2 terminais simultâneos)

```
Terminal 1 (backend)                 Terminal 2 (frontend)
─────────────────────                ─────────────────────
$env:GEMINI_API_KEY="AQ.Ab8RN6K8pqqvCKE7-bs7DW2b_kYk_AfMgnEjZHw1CVUJi_SISg"
python -m uvicorn backend:app
  --reload --port 8000        ──►    python -m streamlit run frontend.py
(deixe rodando)                      (deixe rodando, abre no navegador)
```

A Etapa A (treino) só precisa ser rodada **uma vez** (ou de novo, se você
quiser re-treinar com outros parâmetros). Backend e frontend dependem dos
arquivos `.joblib` gerados por ela, mas não precisam que ela continue
rodando.

---

## 🛠️ Problemas comuns e soluções

### `streamlit : O termo 'streamlit' não é reconhecido...`
Use `python -m streamlit run frontend.py` em vez de `streamlit run frontend.py`.

### `export : O termo 'export' não é reconhecido...`
Você está numa sessão do PowerShell, não do bash. Use:
```powershell
$env:GEMINI_API_KEY="sua_chave_aqui"
```

### `FileNotFoundError: ... modelo_melhor.joblib`
A Etapa A ainda não foi rodada com sucesso nessa pasta. Rode
`python agente_preditivo_etapa_a.py` primeiro e confirme que os `.joblib`
apareceram na pasta antes de iniciar o backend.

### `ERROR: Could not find a version that satisfies the requirement jobli`
Erro de digitação — o nome correto do pacote é `joblib` (com "b" no final).

### `Erro ao chamar /explain: 502 Server Error`
O backend conseguiu rodar, mas a chamada ao Gemini falhou. Para ver o motivo
exato, abra `http://127.0.0.1:8000/docs`, teste o `POST /explain` ali, e
olhe o campo `"detail"` na resposta. Os motivos mais comuns:
- **`429 RESOURCE_EXHAUSTED` / `limit: 0`**: cota da API esgotada ou não
  habilitada para esse modelo no seu projeto. Verifique em
  [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
  se a chave tem cota disponível, ou troque `GEMINI_MODEL` no `backend.py`
  para `"gemini-flash-lite-latest"`.
- **`GEMINI_API_KEY não configurada`**: você esqueceu de rodar o
  `$env:GEMINI_API_KEY=...` antes de iniciar o `uvicorn`, ou abriu um
  terminal novo sem repetir o comando.
- **`400` ou `403`**: chave inválida ou copiada incorretamente — gere uma
  nova no AI Studio.

### O frontend abre mas mostra "Não foi possível conectar ao backend"
Confirme que o Terminal 1 (backend) ainda está rodando e mostrando
`Application startup complete.`. Se você fechou esse terminal, o frontend
perde a conexão — abra de novo com `python -m uvicorn backend:app --reload --port 8000`.

---

## 🔄 Re-treinando com outro modelo ou dataset

Se você quiser trocar o dataset depois (ex: ajustar `data.csv` ou trocar de
projeto completamente):

1. Rode a Etapa A de novo — ela sobrescreve os `.joblib`.
2. Reinicie o backend (Ctrl+C no terminal e rode de novo).
3. O frontend **não precisa de nenhuma mudança de código**, porque os campos
   do formulário são lidos automaticamente do `/schema` do backend.

---

## 📔 Diário de Bordo de Contribuições

### Integrante 1: Pedro Folle
-  Análise exploratória do dataset Breast Cancer Wisconsin; tratamento de valores ausentes e normalização dos dados; configuração inicial do ambiente Python com dependências.
-  Implementação e treinamento dos modelos de classificação (KNN, MLP, Naive Bayes); comparação de desempenho; exportação dos artefatos (modelos e scaler) em `.joblib`.
-  Testes unitários e validação dos modelos; documentação do código da Etapa A; revisão e correção de bugs no treinamento.

### Integrante 2: Maicon Klitzke
-  Desenvolvimento da API FastAPI com endpoints `/schema`, `/predict` e `/explain`; integração com Gemini API para explicações em linguagem natural.
-  Criação da interface Streamlit com formulário dinâmico; conexão entre frontend e backend; testes de integração end-to-end.
- Implementação de tratamento de erros e validação de entrada; ajustes de UI/UX no frontend; documentação completa do README e guia de uso.

