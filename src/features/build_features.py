import pandas as pd
import numpy as np
import logging

def select_features(df: pd.DataFrame, feature_selection: str = 'all', top_n_features: int = 5) -> pd.DataFrame:
    """
    Seleciona features baseado na estratégia configurada.
    
    Args:
        df: DataFrame com todas as features
        feature_selection: 'all' (todas), 'top_correlated' (apenas top N)
        top_n_features: Número de features mais correlacionadas a selecionar (padrão: 5)
    
    Returns:
        DataFrame com features selecionadas
    """
    # Configurar logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"Iniciando seleção de features. Método: {feature_selection}")
    if feature_selection == 'top_correlated':
        logger.info(f"Selecionando top {top_n_features} features mais correlacionadas.")
    logger.info(f"Shape original do DataFrame: {df.shape}")
    
    # TODO: Remover a ideia do feature = all, uma condição anterior já cobre isso
    if feature_selection == 'all':
        logger.info("Selecionando todas as features.")
        return df
    
    elif feature_selection == 'top_correlated':
        # Top N features mais correlacionadas (se Class estiver presente)
        if 'Class' in df.columns:
            logger.info(f"Calculando correlação com a variável 'Class' para seleção das top {top_n_features} features.")
            corr = df.corr()['Class'].abs().sort_values(ascending=False)
            # top_n_features + 1 porque Class está incluída na correlação
            top_features = corr.head(top_n_features + 1).index.tolist()
            top_features.remove('Class')  # Remove Class da lista
            logger.info(f"Top {top_n_features} features mais correlacionadas: {top_features}")
            
            # Manter Time e Amount também
            keep_cols = ['Time', 'Amount'] + top_features
            # Garantir que as colunas existam no DataFrame
            available_cols = [col for col in keep_cols if col in df.columns]
            # Garantir que avisamos sobre colunas faltantes
            missing_cols = [col for col in keep_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Colunas não encontradas no DataFrame: {missing_cols}")
            
            selected_df = df[available_cols + ['Class']] if 'Class' in df.columns else df[available_cols]
            logger.info(f"Shape após seleção de features: {selected_df.shape}")
            logger.info(f"Features selecionadas: {list(selected_df.columns)}")
            return selected_df
        else:
            logger.warning("Coluna 'Class' não encontrada. Retornando todas as features.")
            return df
    
    logger.warning(f"Método de seleção '{feature_selection}' não reconhecido. Retornando todas as features.")
    return df

def create_time_features(df):
    """Cria features de tempo (hora do dia) a partir da coluna 'Time' em segundos."""
    if 'Time' not in df.columns:
        raise ValueError("A coluna 'Time' não existe no DataFrame.")
    df_copy = df.copy()
    # Converte segundos para horas e pega o módulo 24 para ter a hora do dia
    df_copy['Time_hour'] = (df_copy['Time'] / 3600) % 24
    return df_copy