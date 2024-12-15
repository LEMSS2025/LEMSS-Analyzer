import logging
import os


class LoggingSetup:
    """
    Utility class for setting up logging in the project.
    """
    @staticmethod
    def configure_logging(log_file_path):
        """
        Configure logging to output both to console and a log file.
        :param log_file_path: Path to the log file.
        """
        # Create the directory for log files if it doesn't exist
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file_path),
                logging.StreamHandler()
            ]
        )