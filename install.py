import json
from pathlib import Path

ROOT_DIR = Path().absolute()  # Webui root path


def load_config():
    file_name = ROOT_DIR / "config.json"

    try:
        with open(file_name, "r") as file:
            config = json.load(file)
        return config, file_name  # config と file_name の両方を返す
    except Exception as e:
        raise RuntimeError(f"Failed to read configuration file: {e}")


# load config
config, file_name = load_config()  # config と file_name を取得

# set "restart_steps"
config["restart_steps"] = 6

# JSONファイルを書き込む
with open(file_name, "w") as json_file:
    json.dump(config, json_file, indent=4)
