# Detecção de Fraude em Cartões de Crédito

## Visão Geral do Projeto

Este projeto tem como objetivo desenvolver uma solução de Machine Learning para identificar transações fraudulentas em cartões de crédito. Utilizando um conjunto de dados anonimizado, o foco é construir um modelo que seja capaz de realizar predições tanto em tempo real quanto em grandes volumes de dados (batch), garantindo a robustez e a capacidade de implantação em um ambiente de produção.

A detecção de fraude é um problema desafiador devido à natureza altamente desbalanceada dos dados (um número muito pequeno de transações é fraudulento). Por isso, a escolha de métricas de avaliação e a metodologia de desenvolvimento são cruciais para o sucesso do projeto.

## Fonte dos Dados

O conjunto de dados utilizado neste projeto é o `creditcard.csv`, proveniente de uma base de dados de transações com cartão de crédito do [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Ele contém transações que ocorreram em dois dias, onde temos 492 fraudes de um total de 284.807 transações.

As características (`V1` a `V28`) são resultado de uma Transformação de Componentes Principais (PCA), devido a questões de confidencialidade. As únicas características que não foram transformadas são `Time` (tempo decorrido entre a primeira transação e a transação atual), `Amount` (valor da transação) e `Class` (variável alvo, indicando fraude (1) ou não fraude (0)).

## Estrutura do Projeto

A organização do projeto segue as melhores práticas para desenvolvimento de soluções de Machine Learning, visando modularidade, reprodutibilidade e facilidade de manutenção:

```
.
├── data/
│   ├── raw/                  # Dados brutos (ex: creditcard.csv)
│   └── processed/            # Dados após pré-processamento
├── notebooks/
│   ├── analise_exploratoria.ipynb  # Notebooks para análise exploratória de dados (EDA)
│   └── testes.ipynb                # Notebooks para testes e experimentação
├── src/
│   ├── features/             # Módulos para engenharia de features
│   │   └── build_features.py
│   ├── models/               # Módulos para treinamento e avaliação de modelos
│   │   ├── train_model.py    # Exemplo de script de treinamento
│   │   └── predict_model.py  # Exemplo de script de predição
│   └── app/                  # Módulos para a API de predição online
│       └── main.py
├── requirements.txt          # Dependências do projeto
├── readme.ipynb              # Guia detalhado de desenvolvimento do projeto
└── README.md                 # Este arquivo
```


### TODO:
 * Reler toda a parte de processamento novamente pra ser se faz sentido.
 * Implementar um rolê que acha os melhores valores para o RandomForest.
 * Finalizar todo o resto.