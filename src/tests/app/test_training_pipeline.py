import pytest
import pandas as pd
import numpy as np
import yaml
import os

# Assumimos que seus scripts de pipeline podem ser chamados como funções.
# Se eles rodam de outra forma, podemos adaptar o teste.
from src.app import train_pipeline 

# Fixture: Nosso Ambiente de Teste Controlado
@pytest.fixture(scope="module")
def pipeline_setup(tmp_path_factory):
    """
    Cria um ambiente completo para o teste de pipeline em um diretório temporário.
    
    Isso inclui:
    - Um diretório de 'run' para os resultados.
    - Um diretório de 'data' com um dataset sintético.
    - Um arquivo de configuração ('config.yaml') que aponta para esses diretórios.
    
    A fixture retorna o caminho para o config, que é o que o pipeline precisa.
    """
    # Cria um diretório temporário para esta sessão de teste
    temp_dir = tmp_path_factory.mktemp("pipeline_test_env")
    
    # Cria a estrutura de pastas temporárias
    data_path = temp_dir / "data"
    raw_path = data_path / "raw"
    synthetic_processed_path = data_path / "processed_synthetic" # Onde nossos dados de teste REAIS vão ficar
    dummy_processed_path = data_path / "processed_dummy" # Onde a etapa de process_data vai escrever
    run_path = temp_dir / "runs"
    model_dir = run_path / "train1"

    raw_path.mkdir(parents=True, exist_ok=True)
    synthetic_processed_path.mkdir(parents=True, exist_ok=True)
    dummy_processed_path.mkdir(parents=True, exist_ok=True)
    run_path.mkdir(exist_ok=True)

    # Cria um arquivo 'raw' FALSO para a etapa process_data
    # Esta etapa precisa rodar, mas não queremos que ela sobrescreva nossos dados de teste.
    dummy_raw_df = pd.DataFrame({
        'id': range(10), 'Time': range(10), 'Amount': range(10), 'Class': [0]*10,
        **{f'V{i}': [0]*10 for i in range(1, 29)}
    })
    dummy_raw_path = raw_path / "dummy_data.csv"
    dummy_raw_df.to_csv(dummy_raw_path, index=False)
    
    # Cria o dataset sintético
    n_samples = 300
    n_features = 29 # V1-V28 + Amount
    n_obvious_fraud = 30

    X = pd.DataFrame(np.random.rand(n_samples, n_features), columns=[f'V{i}' for i in range(1, 29)] + ['Amount'])
    y = pd.DataFrame(np.zeros(n_samples), columns=['Class'])

    X.iloc[:n_obvious_fraud, 13] = -100 + np.random.randn(n_obvious_fraud) * 0.1
    y.iloc[:n_obvious_fraud] = 1
    X['Time'] = np.arange(n_samples)

    # Salva os dados de treino e teste sintéticos
    train_features_path = synthetic_processed_path / "train_processed.csv"
    train_target_path = synthetic_processed_path / "train_processed_target.csv"
    test_features_path = synthetic_processed_path / "test_processed.csv"
    test_target_path = synthetic_processed_path / "test_processed_target.csv"

    X.to_csv(train_features_path, index=False)
    y.to_csv(train_target_path, index=False)
    X.to_csv(test_features_path, index=False)
    y.to_csv(test_target_path, index=False)

    # Cria o arquivo de configuração temporário COMPLETO
    config_data = {
        'data': {
            # Aponta para o DUMMY, para que process_data não dê erro e escreva em um lugar isolado
            'raw_data_path': str(dummy_raw_path), 
            'processed_data_dir': str(dummy_processed_path),
            
            # Aponta para os dados SINTÉTICOS para treino e avaliação
            'train_features_path': str(train_features_path),
            'train_target_path': str(train_target_path),
            'test_features_path': str(test_features_path),
            'test_target_path': str(test_target_path),
        },
        'preprocessing': {
            'test_data_ratio': 0.2
        },
        'features': {
            'feature_selection': 'all',
            'top_n_features': 15 
        },
        'training': {
            'model_type': 'RandomForest',
            'params': {
                'n_estimators': 10,
                'max_depth': 5,
                'random_state': 42
            }
        },
        'train': { # Chave 'train' para compatibilidade com o teste original
            'run_path': str(run_path),
            'model_path': str(model_dir / 'model.pkl'),
            'metrics_path': str(model_dir / 'metrics.yaml')
        },
        'evaluate': { 
            'model_path': str(model_dir / 'model.pkl'),
            'metrics_path': str(model_dir / 'metrics.yaml')
        }
    }
    
    config_path = temp_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)
        
    return str(config_path)

# O Teste de Integração

def test_train_evaluate_flow(pipeline_setup):
    """
    Testa o fluxo end-to-end de treinamento e avaliação.
    
    Usa a fixture 'pipeline_setup' para obter um ambiente limpo e controlado.
    """
    # Arrange
    config_path = pipeline_setup
    
    # O pipeline ignora o config e escreve em './runs'. Vamos monitorar essa pasta.
    base_run_path = 'runs'
    os.makedirs(base_run_path, exist_ok=True)
    
    # Pega os dirs existentes ANTES de rodar o pipeline
    before_dirs = {d for d in os.listdir(base_run_path) if os.path.isdir(os.path.join(base_run_path, d))}

    # Act
    train_pipeline.run_pipeline(config_path=config_path)

    # Assert
    # Descobre o novo diretório criado pelo pipeline
    after_dirs = {d for d in os.listdir(base_run_path) if os.path.isdir(os.path.join(base_run_path, d))}
    new_dirs = after_dirs - before_dirs
    
    assert len(new_dirs) == 1, f"Deveria ter sido criado apenas um novo diretório de run, mas foram criados: {new_dirs}"
    latest_run_dir = os.path.join(base_run_path, new_dirs.pop())

    model_path = os.path.join(latest_run_dir, 'model.pkl')
    metrics_path = os.path.join(latest_run_dir, 'metrics.yaml')

    # Verificar se os artefatos foram criados
    assert os.path.exists(model_path), f"Arquivo do modelo não foi criado em: {model_path}"
    assert os.path.exists(metrics_path), f"Arquivo de métricas não foi criado em: {metrics_path}"

    # Verificar a métrica mínima de performance
    with open(metrics_path, 'r') as f:
        metrics = yaml.safe_load(f)
    
    recall = metrics.get('classification_report', {}).get('1.0', {}).get('recall')
    assert recall is not None, "Métrica 'recall' para a classe '1.0' não encontrada."
    
    min_recall = 0.7 
    assert recall >= min_recall, f"Recall {recall:.2f} é menor que o mínimo esperado de {min_recall}."
