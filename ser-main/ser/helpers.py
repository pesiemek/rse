import datetime
from ser.constants import PROJECT_ROOT, TIMESTAMP_FORMAT


def set_paths(name:str, timestamp:None):
    if not timestamp:
        timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    
    results_path = PROJECT_ROOT / f"Results" / "{name}"
    model_path = results_path / f"model_{timestamp}.pt"
    params_path = results_path / f"params_{timestamp}.json"
    results_path.mkdir(parents=True, exist_ok=True)

    return model_path, params_path