# Detecção de Fraude em Cartões de Crédito

Este projeto desenvolve uma solução de Machine Learning para identificar transações fraudulentas em cartões de crédito usando dados anonimizados. O objetivo é construir um modelo robusto capaz de realizar predições em tempo real via API e processar grandes volumes de dados em lote, garantindo implantação em ambiente de produção.

A detecção de fraude representa um desafio importante devido ao desbalanceamento significativo dos dados—apenas uma pequena fração das transações é fraudulenta. Assim, a seleção apropriada de métricas de avaliação e a metodologia de desenvolvimento são fundamentais para o sucesso da solução.

## Fonte dos Dados
O conjunto de dados deste projeto é o `creditcard.csv`, disponível no [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Ele contém transações de dois dias, com 492 fraudes entre 284.807 transações.

As características (`V1` a `V28`) resultam de uma Transformação de Componentes Principais (PCA) para proteger a confidencialidade. As características não transformadas são `Time` (tempo desde a primeira transação), `Amount` (valor da transação) e `Class` (indica fraude (1) ou não fraude (0)).

## Como Usar

Existem duas maneiras principais de interagir com este projeto: usando Docker (recomendado para a API) ou executando os scripts Python manualmente.

### 1. Configuração do Ambiente Local (Manual)

Se preferir não usar Docker, você pode configurar um ambiente virtual Python.

```bash
# Instale as dependências
pip install -r requirements.txt
```

### 2. Usando com Docker (Recomendado para a API)

A forma mais simples de executar a API é usando Docker e Docker Compose. Isso garante que o ambiente seja consistente e todas as dependências sejam gerenciadas automaticamente.

```bash
# A partir da raiz do projeto, construa a imagem e inicie o contêiner
docker-compose up --build
```

Este comando irá:
1.  Construir a imagem Docker com base no `Dockerfile`.
2.  Iniciar um contêiner que executa a API FastAPI.
3.  A API estará acessível em `http://localhost:8000`. Você pode ver a documentação interativa em `http://localhost:8000/docs`.

### 3. Executando os Componentes Manualmente

Os principais componentes do pipeline de MLOps podem ser executados individualmente através da linha de comando.

#### a. Pipeline de Treinamento Completo

Executa todo o fluxo: **pré-processamento** de dados, **treinamento** do modelo e **avaliação**.

```bash
python -m src.app.train_pipeline --config config.yaml
```

#### b. Avaliação de um Modelo Específico

Avalia um modelo já treinado usando os dados de teste definidos no `config.yaml`.

```bash
python -m src.app.evaluate --model-path "runs/train1/model.pkl"
```
*   `--model-path`: Caminho para o arquivo do modelo (`.pkl`) que você deseja avaliar.

#### c. Predição em Lote (Batch Prediction)

Usa um modelo treinado para fazer predições em um novo conjunto de dados.

```bash
python -m src.app.predict --model-path "runs/train1/model.pkl" --input-data "data/raw/new_transactions.csv"
```
*   `--model-path`: Caminho para o modelo treinado.
*   `--input-data`: Caminho para o arquivo CSV com as novas transações a serem classificadas.

#### d. Detecção de Desvio de Dados (Data Drift)

Compara um conjunto de dados atual com um de referência para detectar se houve uma mudança estatística significativa (drift).

```bash
python -m src.app.detect_drift --reference "data/processed/train_features.csv" --current "data/raw/production_features_batch.csv"
```
*   `--reference`: Dados de referência (geralmente, os dados de treinamento).
*   `--current`: Novos dados (geralmente, dados recentes de produção).

### 4. Usando a API REST (via Docker)

Uma vez que a API está rodando com Docker, você pode usar os seguintes endpoints:

#### `POST /batch-predict`

Executa predições em lote.

**Exemplo com cURL:**
```bash
curl -X 'POST' \
  'http://localhost:8000/batch-predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model_path": "runs/train1/model.pkl",
  "input_data_path": "data/raw/new_transactions.csv"
}'
```
**Resposta Esperada:**
```json
{
  "message": "Previsões em lote concluídas com sucesso.",
  "output_file": "runs/train1/predict_1/predictions.csv"
}
```

#### `POST /check-drift`

Verifica a existência de desvio de dados.

**Exemplo com cURL:**
```bash
curl -X 'POST' \
  'http://localhost:8000/check-drift' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "reference_path": "data/processed/train_features.csv",
  "current_path": "data/raw/production_features_batch.csv",
  "report_path": "runs/drift_report.json",
  "alpha": 0.01
}'
```
**Resposta Esperada:**
```json
{
  "message": "Detecção de desvio concluída.",
  "results": {
    "drift_detected": true,
    "alpha": 0.01,
    "drifted_features_count": 5,
    "drifted_features_list": ["V1", "V4", "V10", "V11", "V12"],
    "feature_details": {
        "V1": {"type": "numerical", "test": "KS", "statistic": 0.08, "...": "..."}
    }
  }
}
```

## Estrutura do Projeto

A organização do projeto segue as melhores práticas para desenvolvimento de soluções de Machine Learning, visando modularidade, reprodutibilidade e facilidade de manutenção.

```
.
├── data/
│   ├── raw/                  # Dados brutos (ex: creditcard.csv).
│   └── processed/            # Dados após pré-processamento (ex: train_features.csv).
├── notebooks/                # Notebooks para análise exploratória e experimentação.
├── runs/                     # Diretório para salvar os resultados de cada execução (modelos, métricas, predições).
├── src/
│   ├── app/                  # Scripts para executar o pipeline e a API.
│   │   ├── main.py           # Ponto de entrada da API FastAPI.
│   │   ├── train_pipeline.py # Orquestra o pipeline de treinamento completo.
│   │   ├── evaluate.py       # Script para avaliação de modelos.
│   │   ├── predict.py        # Script para predições em lote.
│   │   └── detect_drift.py   # Script para detecção de desvio de dados.
│   ├── data/                 # Módulos para manipulação e processamento de dados.
│   │   └── process_data.py
│   ├── features/             # Módulos para engenharia de features (se necessário).
│   │   └── build_features.py
│   ├── models/               # Módulos para treinamento, avaliação e predição de modelos.
│   │   ├── train_model.py    # Lógica de treinamento.
│   │   ├── evaluate_model.py # Lógica de avaliação.
│   │   └── predict_model.py  # Lógica de predição (usada pelos scripts do app).
│   └── utils/                # Funções utilitárias (ex: salvar/carregar modelos).
├── tests/                    # Testes unitários e de integração.
├── config.yaml               # Arquivo de configuração central do projeto.
├── requirements.txt          # Dependências do projeto Python.
├── Dockerfile                # Define a imagem Docker para a aplicação.
├── docker-compose.yml        # Orquestra os serviços Docker.
└── README.md                 # Este arquivo.
```