import logging
import os
import pandas as pd
import argparse

from src.utils.model_utils import load_model_from_pkl
from src.utils.path_manager import get_next_version_dir

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_batch_predictions(model_path: str, input_data_path: str):
    """
    Carrega um modelo, realiza predições em um conjunto de dados e salva os resultados.

    Args:
        model_path (str): Caminho para o arquivo do modelo treinado (.pkl).
        input_data_path (str): Caminho para o arquivo de dados de entrada (CSV).
    """
    try:
        # Carregar o modelo
        model = load_model_from_pkl(model_path)

        # Carregar os dados de entrada
        logger.info(f"Carregando dados de entrada de: {input_data_path}")
        if not os.path.exists(input_data_path):
            raise FileNotFoundError(f"Arquivo de dados de entrada não encontrado: {input_data_path}")
        
        input_df = pd.read_csv(input_data_path)
        logger.info("Dados de entrada carregados com sucesso.")

        # Realizar predições e obter probabilidades
        logger.info("Realizando predições no conjunto de dados de entrada...")
        predictions = model.predict(input_df)
        probabilities = model.predict_proba(input_df)
        logger.info("Predições realizadas com sucesso.")

        # Mapear predições numéricas para strings "FRAUDE" ou "Normal"
        status_predicao = ['FRAUDE' if p == 1 else 'Normal' for p in predictions]

        # Probabilidade da classe predita (certeza da predição)
        probabilidade_predita = probabilities.max(axis=1)

        # Salvar as predições no novo formato
        output_df = pd.DataFrame({
            'status_predicao': status_predicao,
            'predicao_raw': predictions,
            'probabilidade': probabilidade_predita
        })
        
        # Gerar diretório de saída dentro do diretório do modelo
        model_run_dir = os.path.dirname(model_path) # Ex: runs/train1
        output_dir = get_next_version_dir(base_dir=model_run_dir, prefix='predict')
        output_data_path = os.path.join(output_dir, "predictions.csv")

        output_df.to_csv(output_data_path, index=False)
        logger.info(f"Predições salvas com sucesso em: {output_data_path}")

        return output_data_path

    except FileNotFoundError as e:
        logger.error(f"Erro de arquivo não encontrado: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Ocorreu um erro inesperado durante a execução do batch de predições: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Executa predições em batch em um conjunto de dados.")
    parser.add_argument("--model-path", type=str, required=True, help="Caminho para o arquivo do modelo .pkl.")
    parser.add_argument("--input-data", type=str, required=True, help="Caminho para o arquivo CSV de dados de entrada.")
    
    args = parser.parse_args()

    run_batch_predictions(
        model_path=args.model_path,
        input_data_path=args.input_data
    )
