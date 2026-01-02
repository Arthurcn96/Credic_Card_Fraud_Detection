import os
import pytest
from src.utils.path_manager import get_next_version_dir

def test_get_next_version_dir_first_run(tmp_path):
    """
    Testa se a primeira versão ('run1') é criada quando o diretório base está vazio.
    tmp_path é uma fixture do pytest que cria um diretório temporário.
    """
    # Arrange
    base_dir = tmp_path
    prefix = "run"

    # Act
    next_dir = get_next_version_dir(base_dir=str(base_dir), prefix=prefix)

    # Assert
    expected_dir = os.path.join(base_dir, f"{prefix}1")
    assert next_dir == expected_dir
    assert os.path.isdir(expected_dir)

def test_get_next_version_dir_existing_runs(tmp_path):
    """
    Testa se a próxima versão sequencial é criada quando já existem outras.
    """
    # Arrange
    base_dir = tmp_path
    prefix = "run"
    # Cria algumas pastas de exemplo fora de ordem
    os.makedirs(os.path.join(base_dir, f"{prefix}1"))
    os.makedirs(os.path.join(base_dir, f"{prefix}3"))

    # Act
    next_dir = get_next_version_dir(base_dir=str(base_dir), prefix=prefix)

    # Assert
    # O maior número é 3, então o próximo deve ser 4
    expected_dir = os.path.join(base_dir, f"{prefix}4")
    assert next_dir == expected_dir
    assert os.path.isdir(expected_dir)

def test_get_next_version_dir_ignores_other_prefixes(tmp_path):
    """
    Testa se a função ignora diretórios com outros prefixos.
    """
    # Arrange
    base_dir = tmp_path
    prefix = "run"
    # Cria pastas com prefixos diferentes
    os.makedirs(os.path.join(base_dir, "predict1"))
    os.makedirs(os.path.join(base_dir, "train5"))

    # Act
    next_dir = get_next_version_dir(base_dir=str(base_dir), prefix=prefix)

    # Assert
    # Como não há pastas com o prefixo 'run', deve criar 'run1'
    expected_dir = os.path.join(base_dir, f"{prefix}1")
    assert next_dir == expected_dir
    assert os.path.isdir(expected_dir)

def test_get_next_version_dir_base_dir_does_not_exist(tmp_path):
    """
    Testa se o diretório base é criado se ele não existir.
    """
    # Arrange
    # tmp_path cria a pasta pai, nós definimos uma subpasta que não existe
    base_dir = os.path.join(tmp_path, "non_existent_runs_dir")
    prefix = "run"

    # Act
    next_dir = get_next_version_dir(base_dir=base_dir, prefix=prefix)

    # Assert
    expected_dir = os.path.join(base_dir, f"{prefix}1")
    assert os.path.isdir(base_dir) # Garante que a pasta base foi criada
    assert next_dir == expected_dir
    assert os.path.isdir(expected_dir)
