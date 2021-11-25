import datetime
from ser.constants import PROJECT_ROOT, TIMESTAMP_FORMAT


def set_paths(name:str):
    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    results_path = PROJECT_ROOT / "Results" / "{name}".format(name=name)
    model_path = results_path / "model_{timestamp}.pt".format(timestamp=timestamp)
    params_path = results_path / "params_{timestamp}.json".format(timestamp=timestamp)
    results_path.mkdir(parents=True, exist_ok=True)

    return model_path, params_path