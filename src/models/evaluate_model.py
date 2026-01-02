import pandas as pd
import numpy as np
import os
import logging
import joblib 
import yaml
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.model_utils import load_model_from_pkl
def run(config: dict, model_path: str):
    """
    Avalia o modelo treinado usando os dados de teste e salva os resultados.

    Args:
        config (dict): Dicionário de configuração carregado do config.yaml.
        model_path (str): Caminho para o arquivo de modelo treinado (.pkl).
    """
    logger = logging.getLogger(__name__)
    logger.info("--- Iniciando Etapa: Avaliação do Modelo ---")

    # Determinar o diretório do run a partir do caminho do modelo
    run_dir = os.path.dirname(model_path)

    # Carregar modelo treinado
    try:
        model = load_model_from_pkl(model_path)
    except (FileNotFoundError, Exception) as e: # Catch both specific FileNotFoundError and generic Exception from utility
        logger.error(f"Erro ao carregar o modelo: {e}")
        return

    # Carregar dados de teste
    test_features_path = config['data']['test_features_path']
    test_target_path = config['data']['test_target_path']
    
    try:
        X_test = pd.read_csv(test_features_path)
        y_test = pd.read_csv(test_target_path).squeeze()
        logger.info(f"Dados de teste carregados. Shape: {X_test.shape}")
    except FileNotFoundError as e:
        logger.error(f"Erro ao carregar dados de teste: {e}")
        return

    # Realizar previsões
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Gerar e logar métricas de avaliação
    logger.info("\n--- Relatório de Classificação ---")
    report = classification_report(y_test, y_pred)
    logger.info(f"\n{report}")

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    logger.info(f"ROC AUC Score: {roc_auc:.4f}")

    # Salvar métricas em um arquivo YAML
    metrics = {
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'roc_auc_score': roc_auc
    }
    metrics_path = os.path.join(run_dir, 'metrics.yaml')
    with open(metrics_path, 'w') as f:
        yaml.dump(metrics, f, default_flow_style=False)
    logger.info(f"Métricas de avaliação salvas em: {metrics_path}")

    # --- Geração de Gráficos ---

    # 1. Matriz de Confusão (Valores Absolutos)
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Não Fraude', 'Fraude'], yticklabels=['Não Fraude', 'Fraude'])
    plt.title('Matriz de Confusão (Valores Absolutos)')
    plt.xlabel('Previsão')
    plt.ylabel('Verdadeiro')
    confusion_matrix_path = os.path.join(run_dir, "confusion_matrix.png")
    plt.savefig(confusion_matrix_path)
    plt.close()
    logger.info(f"Matriz de Confusão salva em: {confusion_matrix_path}")

    # 2. Matriz de Confusão (Normalizada)
    plt.figure(figsize=(8, 6))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', xticklabels=['Não Fraude', 'Fraude'], yticklabels=['Não Fraude', 'Fraude'])
    plt.title('Matriz de Confusão (Normalizada)')
    plt.xlabel('Previsão')
    plt.ylabel('Verdadeiro')
    confusion_matrix_norm_path = os.path.join(run_dir, "confusion_matrix_normalized.png")
    plt.savefig(confusion_matrix_norm_path)
    plt.close()
    logger.info(f"Matriz de Confusão Normalizada salva em: {confusion_matrix_norm_path}")

    # 3. Curva ROC
    plt.figure(figsize=(8, 6))
    roc_display = RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title('Curva ROC')
    roc_curve_path = os.path.join(run_dir, "roc_curve.png")
    plt.savefig(roc_curve_path)
    plt.close()
    logger.info(f"Curva ROC salva em: {roc_curve_path}")

    # 4. Curva Precision-Recall
    plt.figure(figsize=(8, 6))
    pr_display = PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
    plt.title('Curva Precision-Recall')
    pr_curve_path = os.path.join(run_dir, "precision_recall_curve.png")
    plt.savefig(pr_curve_path)
    plt.close()
    logger.info(f"Curva Precision-Recall salva em: {pr_curve_path}")

    logger.info("--- Etapa de Avaliação do Modelo Concluída ---\n")
