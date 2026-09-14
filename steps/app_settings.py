import json
import os

SETTINGS_PATH = os.path.join('DATA', 'input_main', 'app_settings.json')


def load_settings() -> dict:
    if not os.path.exists(SETTINGS_PATH):
        return {}
    try:
        with open(SETTINGS_PATH, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def save_settings(settings: dict) -> None:
    os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
    with open(SETTINGS_PATH, 'w') as f:
        json.dump(settings, f, indent=2)


def get_setting(key, default=None):
    return load_settings().get(key, default)


def set_settings(updates: dict) -> None:
    settings = load_settings()
    settings.update(updates)
    save_settings(settings)
