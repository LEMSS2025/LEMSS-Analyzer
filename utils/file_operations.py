import os
import shutil
import logging
from typing import Union


class FileOperations:
    """
    Utility class for common file operations like copying files.
    """

    @staticmethod
    def copy_files_to_experiment_folder(trectext_path: Union[str, os.PathLike], game_history_path: Union[str, os.PathLike],
                                        experiment_folder: Union[str, os.PathLike]) -> None:
        """
        Copy necessary files to the experiment folder.

        :param trectext_path: Path to the TREC text data file.
        :param game_history_path: Path to the game history file.
        :param experiment_folder: Path to the experiment folder.
        """
        # Ensure the experiment folder exists
        os.makedirs(experiment_folder, exist_ok=True)

        # Copy TREC text file
        trec_dest = os.path.join(experiment_folder, os.path.basename(trectext_path))
        shutil.copy(trectext_path, trec_dest)
        logging.debug(f"Copied {trectext_path} to {trec_dest}")

        # Copy game history file
        game_history_dest = os.path.join(experiment_folder, os.path.basename(game_history_path))
        shutil.copy(game_history_path, game_history_dest)
        logging.debug(f"Copied {game_history_path} to {game_history_dest}")

        logging.info(f"All files successfully copied to {experiment_folder}")
