import pandas as pd
import numpy as np
import os
import logging
import joblib
import yaml
from sklearn.ensemble import RandomForestClassifier
from ..utils.path_manager import get_next_version_dir

def run(config: dict) -> str:
    """
    Treina o modelo de machine learning, salva os artefatos em um diretório de 'run'
    e retorna o caminho do modelo.
    
    Args:
        config: Dicionário de configuração do config.yaml
    
    Returns:
        str: O caminho para o arquivo de modelo treinado (.pkl).
    """
    logger = logging.getLogger(__name__)
    logger.info("--- Iniciando Etapa: Treinamento do Modelo ---")
    
    # Carregar dados de treino
    train_features_path = config['data']['train_features_path']
    train_target_path = config['data']['train_target_path']
    
    logger.info(f"Carregando features de treino de: {train_features_path}")
    logger.info(f"Carregando target de treino de: {train_target_path}")
    
    X_train = pd.read_csv(train_features_path)
    y_train = pd.read_csv(train_target_path).squeeze()
    
    logger.info(f"Dados de treino carregados. Shape: {X_train.shape}")
    
    # Treinar modelo
    model_type = config['training']['model_type']
    params = config['training']['params']
    
    if model_type == 'RandomForest':
        logger.info(f"Treinando RandomForest com parâmetros: {params}")
        model = RandomForestClassifier(**params, n_jobs=-1)
        model.fit(X_train, y_train)
        logger.info("Modelo treinado com sucesso!")
    else:
        raise ValueError(f"Tipo de modelo '{model_type}' não suportado.")
    
    # Gerenciamento de Artefatos do Run
    run_dir = get_next_version_dir(prefix='train')
    logger.info(f"Diretório do run criado em: {run_dir}")

    # Salvar modelo
    model_path = os.path.join(run_dir, 'model.pkl')
    joblib.dump(model, model_path)
    logger.info(f"Modelo salvo em: {model_path}")
    
    # Salvar hiperparâmetros (args.yaml)
    args_path = os.path.join(run_dir, 'args.yaml')
    with open(args_path, 'w') as f:
        yaml.dump(config['training'], f)
    logger.info(f"Hiperparâmetros salvos em: {args_path}")
    
    logger.info("--- Etapa de Treinamento Concluída ---\n")

    return model_path