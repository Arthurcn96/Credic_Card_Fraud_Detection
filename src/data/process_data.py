import pandas as pd
import numpy as np
import os
import logging
from ..features.build_features import select_features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run(config: dict) -> None:
    """
    Carrega, pré-processa e balanceia o conjunto de dados de fraude de cartão de crédito
    usando as configurações fornecidas.

    Args:
        config (dict): Dicionário de configuração carregado do config.yaml.
    """
    # Configurar logging
    logger = logging.getLogger("process_data")
    
    logger.info("--- Iniciando Etapa: Pré-processamento de Dados ---")

    # Extrair configurações do dicionário
    input_path = config['data']['raw_data_path']
    output_dir = config['data']['processed_data_dir']
    test_data_ratio = config['preprocessing']['test_data_ratio']
    feature_selection = config['features']['feature_selection']
    top_n_features = config['features']['top_n_features']

    # Garante que o diretório de saída exista
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Diretório de saída criado em: {output_dir}")

    # Carregar os dados
    try:
        df = pd.read_csv(input_path)
        logger.info(f"Dados carregados de {input_path}. Shape: {df.shape}")
    except FileNotFoundError:
        logger.error(f"Erro: Arquivo não encontrado em {input_path}")
        return

    # Verificar dados ausentes
    if df.isnull().sum().any():
        logger.warning("Dados ausentes encontrados. Preenchendo com a média.")
        df.fillna(df.mean(), inplace=True)

    # ---- Utilizar Feature Engineering com dados PCA dificilmente é necessário ----
    if feature_selection != 'all':
        df = select_features(df, feature_selection, top_n_features)
        logger.info(f"Features selecionadas usando estratégia: {feature_selection}")
    
    logger.info(f"Shape após feature engineering: {df.shape}")

    # Separar features (X) e alvo (y)
    X = df.drop(['id', 'Class'], axis=1, errors='ignore')
    y = df['Class']

    # Dividir em dados de treino e teste (YAML) de forma estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_data_ratio, random_state=42, stratify=y
    )
    logger.info(f"Dados divididos em treino ({X_train.shape}) e teste ({X_test.shape}).")


    #Random Forest não precisa de escalonamento
    # Árvores de decisão fazem divisões por comparações de valores, não por distâncias
    # Remover StandardScaler (desnecessario manter o mesmo peso na modelagem)
    # Cada feature é avaliada independentemente
    # A escala não afeta as divisões das árvores
    #TODO: Balancear os dados apenas no treino pra aumentar a performance do modelo

    # Salvar os dados de treino e teste processados
    X_train.to_csv(os.path.join(output_dir, 'train_processed.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'test_processed.csv'), index=False)
    logger.info(f"Dados de treino processados salvos em: {output_dir}")
    logger.info(f"Dados de teste processados salvos em: {output_dir}")
    
    y_train.to_csv(os.path.join(output_dir, 'train_processed_target.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'test_processed_target.csv'), index=False)
    logger.info(f"Target de treino processados salvos em: {output_dir}")
    logger.info(f"Target de teste processados salvos em: {output_dir}")

    logger.info("--- Etapa de Pré-processamento Concluída ---\n")