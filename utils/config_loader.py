import json

class ConfigLoader:
    """
    Class to load and handle configuration settings for the experiment.
    """

    @staticmethod
    def load_config(json_path: str) -> dict:
        """
        Load the configuration for the experiment.

        :return: dict containing configuration settings
        """
        with open(json_path, "r") as file:
            config = json.load(file)

        return config