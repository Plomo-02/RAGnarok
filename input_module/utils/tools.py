from pathlib import Path


def find_project_root(marker_name="RAGnarok") -> Path:
    """
    Cerca verso l'alto dalla directory dello script corrente
    per trovare una directory con il nome specificato.
    """
    try:
        current_path = Path(__file__).resolve()
    except NameError:
        current_path = Path.cwd().resolve()
        print(f"Attenzione: __file__ non definito. Uso la directory corrente: {current_path}")

    project_root = None
    for parent in current_path.parents:
        if parent.name == marker_name:
            project_root = parent
            break

    if project_root is None:
            if current_path.name == marker_name:
                project_root = current_path
            else:
                raise FileNotFoundError(
                    f"Impossibile trovare la directory radice del progetto '{marker_name}' "
                    f"risalendo da {current_path}"
                )
    return project_root
