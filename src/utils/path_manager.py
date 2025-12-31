import os
import re

def get_next_run_dir(base_dir: str = 'runs') -> str:
    """
    Encontra o próximo número de 'run', cria o diretório e o retorna.

    A função procura por diretórios no formato 'train<numero>' dentro
    do diretório base, determina o próximo número sequencial e cria
    um novo diretório com esse número.

    Args:
        base_dir (str): O diretório base onde os runs de treinamento são armazenados.

    Returns:
        str: O caminho para o diretório do novo run (ex: 'runs/train1').
    """
    os.makedirs(base_dir, exist_ok=True)
    
    existing_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    max_run_num = 0
    for dir_name in existing_dirs:
        match = re.match(r'train(\d+)', dir_name)
        if match:
            run_num = int(match.group(1))
            if run_num > max_run_num:
                max_run_num = run_num
                
    next_run_num = max_run_num + 1
    next_run_dir = os.path.join(base_dir, f'train{next_run_num}')
    
    os.makedirs(next_run_dir, exist_ok=True)
    
    return next_run_dir
