from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from pathlib import Path
import os
import sys

# Import refactored functions
from src.app.predict import run_batch_predictions
from src.app.detect_drift import detect_drift

# Para esta configuração inicial, elas são executadas de forma síncrona.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="API de Detecção de Fraudes em Cartões de Crédito",
    description="API para previsões em lote e detecção de desvio de dados para modelos de fraude em cartões de crédito.",
    version="0.1.0",
)

class BatchPredictRequest(BaseModel):
    model_path: str = "runs/train1/model.pkl"
    input_data_path: str = "data/raw/new_transactions.csv"

class DriftCheckRequest(BaseModel):
    reference_path: str = "data/processed/train_features.csv"
    current_path: str = "data/raw/production_features_batch.csv"
    report_path: str = "runs/drift_report.json" # Saída padrão para o relatório
    alpha: float = 0.01 # Nível de significância para a detecção de desvio

@app.get("/")
async def read_root():
    return {"message": "Bem-vindo à API de Detecção de Fraudes em Cartões de Crédito!"}

@app.post("/batch-predict")
async def batch_predict(request: BatchPredictRequest):
    """
    Executa previsões em lote usando um modelo e dados de entrada especificados.
    """
    logger.info(f"Requisição de previsão em lote recebida: {request}")
    try:
        output_file_path = run_batch_predictions(
            model_path=request.model_path,
            input_data_path=request.input_data_path
        )
        return {"message": "Previsões em lote concluídas com sucesso.", "output_file": output_file_path}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Arquivo não encontrado: {e}")
    except Exception as e:
        logger.error(f"Erro durante a previsão em lote: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro interno do servidor: {e}")

@app.post("/check-drift")
async def check_drift(request: DriftCheckRequest):
    """
    Verifica desvio de dados entre conjuntos de dados de referência e atuais.
    """
    logger.info(f"Requisição de verificação de desvio recebida: {request}")
    try:
        # Chama a função detect_drift refatorada
        drift_results = detect_drift(
            reference_path=request.reference_path,
            current_path=request.current_path,
            report_path=request.report_path,
            alpha=request.alpha
        )
        return {"message": "Detecção de desvio concluída.", "results": drift_results}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Arquivo não encontrado: {e}")
    except Exception as e:
        logger.error(f"Erro durante a detecção de desvio: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro interno do servidor: {e}")
