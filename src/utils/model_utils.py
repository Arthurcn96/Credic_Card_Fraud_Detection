import joblib
import logging
import os

logger = logging.getLogger(__name__)

def load_model_from_pkl(model_path: str):
    """
    Carrega um modelo de machine learning de um arquivo .pkl.
    """
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Arquivo do modelo não encontrado: {model_path}")
        
        logger.info(f"Carregando modelo de: {model_path}")
        model = joblib.load(model_path)
        logger.info("Modelo carregado com sucesso.")
    
        return model
    
    except FileNotFoundError:
        logger.error(f"Erro: O arquivo do modelo não foi encontrado em {model_path}", exc_info=True)
    
        raise
    
    except Exception as e:
        logger.error(f"Erro inesperado ao carregar o modelo de {model_path}: {e}", exc_info=True)
        raise
