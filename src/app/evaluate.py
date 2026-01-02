import argparse
import yaml
import logging
from src.models import evaluate_model

def main():
    """
    Função principal para executar a avaliação de um modelo específico via linha de comando.
    
    Este script serve como um ponto de entrada que carrega a configuração e o modelo
    especificado e, em seguida, chama a lógica de avaliação principal.
    """
    # Configuração básica de logging para exibir as saídas no console
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Configuração do parser de argumentos da linha de comando
    parser = argparse.ArgumentParser(
        description="Executa a etapa de avaliação para um modelo treinado específico."
    )
    parser.add_argument(
        '-m','--model-path',
        type=str, 
        required=True,
        help='Caminho do modelo treinado (.pkl). Ex: runs/train1/model.pkl'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml',
        help='Caminho para o arquivo de configuração YAML. Padrão: config.yaml'
    )
    args = parser.parse_args()
    
    # Carregar o arquivo de configuração
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuração carregada de: {args.config}")
    except FileNotFoundError:
        logger.error(f"Erro: Arquivo de configuração não encontrado em {args.config}")
        return
    except yaml.YAMLError as e:
        logger.error(f"Erro ao carregar o arquivo YAML: {e}")
        return

    # Chamar a função principal de avaliação do modelo
    try:
        evaluate_model.run(config=config, model_path=args.model_path)
    except Exception as e:
        logger.error(f"Ocorreu um erro durante a execução da avaliação: {e}", exc_info=True)

    logger.info("\n--- Script de Avaliação Concluído ---")

if __name__ == '__main__':
    main()
