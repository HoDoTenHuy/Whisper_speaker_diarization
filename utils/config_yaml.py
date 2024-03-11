from utils.get_common_config import parse_config


class ConfigManager:
    def __init__(self):
        self.yaml_path = 'config/default.yaml'
        self.config_data = None

    def get_config(self):
        if self.config_data is None:
            self.config_data = parse_config(path=self.yaml_path)
        return self.config_data
