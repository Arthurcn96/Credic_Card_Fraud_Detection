import os
import pytest
import joblib
from sklearn.linear_model import LogisticRegression  # Um modelo de exemplo

from src.utils.model_utils import load_model_from_pkl

def test_load_model_from_pkl_success(tmp_path):
    """
    Testa o carregamento bem-sucedido de um modelo a partir de um arquivo .pkl.
    """
    # Arrange
    base_dir = tmp_path
    model_path = os.path.join(base_dir, "test_model.pkl")
    
    # Cria um objeto de modelo simples
    dummy_model = LogisticRegression()
    
    # Salva o modelo no caminho temporário
    joblib.dump(dummy_model, model_path)
    
    # Act
    loaded_model = load_model_from_pkl(model_path)
    
    # Assert
    assert loaded_model is not None
    # Verifica se o modelo carregado é do tipo esperado
    assert isinstance(loaded_model, LogisticRegression)

def test_load_model_from_pkl_file_not_found():
    """
    Testa se a função levanta FileNotFoundError para um caminho inexistente.
    """
    # Arrange
    non_existent_path = "caminho/que/nao/existe/model.pkl"
    
    # Act & Assert
    # Verifica se a exceção correta é levantada pelo pytest
    with pytest.raises(FileNotFoundError):
        load_model_from_pkl(non_existent_path)

def test_load_model_from_pkl_corrupted_file(tmp_path):
    """
    Testa como a função se comporta com um arquivo .pkl inválido/corrompido.
    """
    # Arrange
    base_dir = tmp_path
    corrupted_file_path = os.path.join(base_dir, "corrupted.pkl")
    
    # Cria um arquivo que não é um pkl válido
    with open(corrupted_file_path, "w") as f:
        f.write("isto nao e um modelo")
        
    # Act & Assert
    # joblib.load pode levantar diferentes exceções (ex: UnpicklingError),
    # então pegamos uma exceção genérica para garantir que a função falha.
    with pytest.raises(Exception):
        load_model_from_pkl(corrupted_file_path)

