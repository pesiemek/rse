from datetime import datetime
from ser.constants import PROJECT_ROOT, TIMESTAMP_FORMAT
from ser.loaders import load_test
from ser.transforms import transform


def set_paths(name: str, timestamp=None):
    if not timestamp:
        timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    
    results_path = PROJECT_ROOT / "Results" / f"{name}"
    model_path = results_path / f"model_{timestamp}.pt"
    params_path = results_path / f"params_{timestamp}.json"
    results_path.mkdir(parents=True, exist_ok=True)

    return model_path, params_path



def select_test_image(label, transforms):
    test_data = load_test(1, transforms)
    
    image, label = next(iter(test_data))
    while label[0].item() != label:
        image, label = next(iter(test_data))
    
    return image