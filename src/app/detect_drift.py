import pandas as pd
import argparse
import logging
import sys
import json
from pathlib import Path
from scipy.stats import ks_2samp, chi2_contingency

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_drift(reference_path: str, current_path: str, report_path: str, alpha: float = 0.01) -> None:
    """
    Detecta desvio de dados entre dois datasets usando testes estatísticos (KS e Qui-quadrado).

    Args:
        reference_path (str): Caminho para o arquivo CSV de referência.
        current_path (str): Caminho para o arquivo CSV atual.
        report_path (str): Caminho para salvar o relatório JSON de desvio.
        alpha (float): Nível de significância para os testes estatísticos.
    """
    logger.info("="*60)
    logger.info("DETECÇÃO DE DESVIO DE DADOS (SCIPY)")
    logger.info("="*60)
    logger.info(f"Dados de Referência: {reference_path}")
    logger.info(f"Dados Atuais: {current_path}")
    logger.info(f"Nível de Significância (alpha): {alpha}")

    # --- Carregar Dados ---
    try:
        reference_data = pd.read_csv(reference_path)
        current_data = pd.read_csv(current_path)
        logger.info("Dados carregados com sucesso.")
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        sys.exit(1)
    
    # --- Preparar Dados ---
    if 'Class' in reference_data.columns:
        reference_data = reference_data.drop(columns=['Class'])
    if 'Class' in current_data.columns:
        current_data = current_data.drop(columns=['Class'])

    common_cols = list(set(reference_data.columns) & set(current_data.columns))
    if not common_cols:
        logger.error("Nenhuma coluna comum encontrada.")
        sys.exit(1)
    
    logger.info(f"Analisando {len(common_cols)} features comuns.")

    # --- Análise de Drift ---
    drift_results = {
        'drift_detected': False,
        'alpha': alpha,
        'drifted_features_count': 0,
        'drifted_features_list': [],
        'feature_details': {}
    }

    # Identificar colunas numéricas e categóricas (com base nos dados de referência)
    numerical_cols = reference_data.select_dtypes(include=['number']).columns
    categorical_cols = reference_data.select_dtypes(exclude=['number']).columns

    # 1. Drift em Features Numéricas (Teste KS)
    for col in numerical_cols:
        if col not in common_cols:
            continue
        
        # Ensure there are enough samples for KS test
        if len(reference_data[col].dropna()) < 2 or len(current_data[col].dropna()) < 2:
            logger.warning(f"Feature '{col}' pulada: não há amostras suficientes para o teste KS.")
            continue

        ks_stat, p_value = ks_2samp(reference_data[col].dropna(), current_data[col].dropna())
        is_drifted = p_value < alpha
        drift_results['feature_details'][col] = {
            'type': 'numerical', 'test': 'KS',
            'statistic': float(ks_stat), 'p_value': float(p_value), 'drifted': bool(is_drifted)
        }
        if is_drifted:
            drift_results['drifted_features_list'].append(col)

    # 2. Drift em Features Categóricas (Teste Qui-quadrado)
    for col in categorical_cols:
        if col not in common_cols:
            continue
            
        # Contagem de valores para cada categoria
        ref_counts = reference_data[col].value_counts()
        current_counts = current_data[col].value_counts()
        
        # Juntar as contagens para criar a tabela de contingência
        contingency_table = pd.concat([ref_counts, current_counts], axis=1).fillna(0)
        contingency_table.columns = ['reference', 'current']

        # Ensure there are enough samples and degrees of freedom for Chi-squared test
        # chi2_contingency requires at least 2x2 table and no expected frequencies to be 0
        if contingency_table.shape[0] < 2 or contingency_table.sum().sum() < 2:
             logger.warning(f"Feature '{col}' pulada: não há dados suficientes ou categorias (>1) para o teste Qui-quadrado.")
             continue
        if (contingency_table == 0).any().any():
             logger.warning(f"Feature '{col}' pulada: tabela de contingência com valores zero. Não é adequado para o teste Qui-quadrado.")
             continue

        chi2, p_value, _, _ = chi2_contingency(contingency_table.values)
        is_drifted = p_value < alpha
        drift_results['feature_details'][col] = {
            'type': 'categorical', 'test': 'Chi-squared',
            'statistic': float(chi2), 'p_value': float(p_value), 'drifted': bool(is_drifted)
        }
        if is_drifted:
            drift_results['drifted_features_list'].append(col)

    # --- Sumarizar e Salvar Relatório ---
    drifted_count = len(drift_results['drifted_features_list'])
    if drifted_count > 0:
        drift_results['drift_detected'] = True
        drift_results['drifted_features_count'] = drifted_count

    logger.info("\n" + "="*60)
    logger.info("RESULTADO DA ANÁLISE")
    logger.info("="*60)
    logger.info(f"Total de features analisadas: {len(common_cols)}")
    logger.info(f"Total de features com desvio: {drifted_count}")
    
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(drift_results, f, indent=4)
    logger.info(f"Relatório de desvio salvo em: {report_path.absolute()}")

    return drift_results


def main():
    parser = argparse.ArgumentParser(
        description="Detecta desvio de dados entre dois datasets usando testes estatísticos (Scipy).",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--reference', type=str, required=True, help='Caminho para o CSV de referência.')
    parser.add_argument('--current', type=str, required=True, help='Caminho para o CSV atual.')
    parser.add_argument('--report_path', type=str, default='drift_report.json', help='Caminho para salvar o relatório JSON.')
    parser.add_argument('--alpha', type=float, default=0.05, help='Nível de significância (p-value threshold).')
    
    args = parser.parse_args()
    
    results = detect_drift(args.reference, args.current, args.report_path, args.alpha)

    # --- Sair com status apropriado para uso em CLI ---
    if results['drift_detected']:
        logger.warning("\nDesvio de dados DETECTADO!")
        sys.exit(1)
    else:
        logger.info("\nNenhum desvio de dados significativo detectado.")
        sys.exit(0)


if __name__ == '__main__':
    main()
