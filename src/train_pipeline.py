import yaml
import argparse
import logging
from .data import process_data
from .models import train_model, evaluate_model

def run_pipeline(config_path: str) -> None:
    """
    Executa o pipeline de ponta a ponta para detecção de fraude.
    Args:
        config_path (str): Caminho para o arquivo de configuração YAML.
    """
    logger = logging.getLogger(__name__)
    logger.info("--- Iniciando Pipeline de Detecção de Fraude ---")
    
    # 1. Carregar configuração
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuração carregada de: {config_path}")
    except FileNotFoundError:
        logger.error(f"Erro: Arquivo de configuração não encontrado em {config_path}")
        return
    except yaml.YAMLError as e:
        logger.error(f"Erro ao carregar o arquivo YAML: {e}")
        return

    # --- Executando as etapas do Pipeline em sequência ---

    # 2. Pré-processamento dos dados
    process_data.run(config)

    # 3. Treinamento do Modelo
    model_path = train_model.run(config)

    # 4. Avaliação do Modelo
    evaluate_model.run(config, model_path)
    # Se o modelo atual é melhor que o anterior for, ele poderia ser 'promovido' ou registrado.
    
    logger.info("\n--- Pipeline Concluído ---")

def main():
    """
    Função principal para executar o pipeline via linha de comando.
    """
    #Faz o loggin printar no tela as coisas
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Executa o pipeline de detecção de fraude.")
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml',
        help='Caminho para o arquivo de configuração YAML.'
    )
    args = parser.parse_args()
    
    run_pipeline(args.config)

if __name__ == '__main__':
    main()
