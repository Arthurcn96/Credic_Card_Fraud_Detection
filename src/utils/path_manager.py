import os
import re

import os
import re

def get_next_version_dir(base_dir: str = 'runs', prefix: str = 'run') -> str:
    """
    A função procura por diretórios no formato '<prefix><numero>' dentro
    do diretório base, determina o próximo número sequencial e cria
    um novo diretório com esse número.

    Args:
        base_dir (str): O diretório base onde os runs são armazenados.
        prefix (str): O prefixo para os diretórios de run (ex: 'train', 'predict').

    Returns:
        str: O caminho para o diretório do novo run (ex: 'runs/train1').
    """
    os.makedirs(base_dir, exist_ok=True)
    
    existing_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    max_num = 0
    for dir_name in existing_dirs:
        match = re.match(rf'{prefix}(\d+)', dir_name)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
                
    next_num = max_num + 1
    next_dir = os.path.join(base_dir, f'{prefix}{next_num}')
    
    os.makedirs(next_dir, exist_ok=True)
    
    return next_dir

