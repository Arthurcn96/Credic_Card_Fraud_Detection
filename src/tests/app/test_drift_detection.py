import pytest
import pandas as pd
import numpy as np
import subprocess
import os

# Caminho para o script de detecção de desvio
DRIFT_DETECTION_SCRIPT = os.path.join(os.path.dirname(__file__), '..', 'app', 'detect_drift.py')

@pytest.fixture
def create_drift_test_data(tmp_path):
    """
    Cria dados de referência e dados atuais (com ou sem desvio) para os testes.
    Retorna os caminhos para os arquivos CSV criados.
    """
    ref_path = tmp_path / "reference.csv"
    current_no_drift_path = tmp_path / "current_no_drift.csv"
    current_with_drift_path = tmp_path / "current_with_drift.csv"

    # Dados de referência
    ref_data = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, 100),
        'feature_2': np.random.normal(10, 2, 100),
        'Class': np.random.randint(0, 2, 100)
    })
    ref_data.to_csv(ref_path, index=False)

    # Dados atuais sem desvio (iguais aos de referência, excluindo 'Class')
    ref_data.drop(columns=['Class']).to_csv(current_no_drift_path, index=False)

    # Dados atuais COM desvio (modificando feature_1 significativamente)
    drifted_data = pd.DataFrame({
        'feature_1': np.random.normal(5, 1, 100), # Desvio significativo na média
        'feature_2': np.random.normal(10, 2, 100),
        'Class': np.random.randint(0, 2, 100)
    })
    drifted_data.to_csv(current_with_drift_path, index=False)

    return {
        "reference": str(ref_path),
        "current_no_drift": str(current_no_drift_path),
        "current_with_drift": str(current_with_drift_path),
        "report_no_drift": str(tmp_path / "report_no_drift.html"),
        "report_with_drift": str(tmp_path / "report_with_drift.html")
    }

def test_no_data_drift_detected(create_drift_test_data):
    """
    Testa se o script detect_drift.py não detecta desvio quando os dados são semelhantes.
    """
    data_paths = create_drift_test_data
    
    command = [
        'python', DRIFT_DETECTION_SCRIPT,
        '--reference', data_paths["reference"],
        '--current', data_paths["current_no_drift"],
        '--report_path', data_paths["report_no_drift"]
    ]
    
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    
    assert result.returncode == 0, f"Esperado status de sucesso (0), mas obteve {result.returncode}. Erro: {result.stderr}"
    assert "Nenhum desvio de dados significativo detectado." in result.stdout
    assert os.path.exists(data_paths["report_no_drift"])

def test_data_drift_detected(create_drift_test_data):
    """
    Testa se o script detect_drift.py detecta desvio quando os dados são significativamente diferentes.
    """
    data_paths = create_drift_test_data
    
    command = [
        'python', DRIFT_DETECTION_SCRIPT,
        '--reference', data_paths["reference"],
        '--current', data_paths["current_with_drift"],
        '--report_path', data_paths["report_with_drift"]
    ]
    
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    
    assert result.returncode == 1, f"Esperado status de erro (1) devido a desvio, mas obteve {result.returncode}. Saída: {result.stdout}"
    assert "Desvio de dados DETECTADO!" in result.stderr # evidently logs to stderr by default
    assert os.path.exists(data_paths["report_with_drift"])