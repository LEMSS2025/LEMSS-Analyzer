import os

import pandas as pd
import numpy as np

from constants.constants import (HISTORY_DF_RANK_COLUMN, HISTORY_DF_QUERY_ID_COLUMN, HISTORY_DF_PLAYER_COLUMN,
                                 REPORT_TABLE_BEST_AGENT_COLUMN, REPORT_TABLE_WINNING_HOMOGENEITY_COLUMN,
                                 COMPEITION_HISTORY_FILE_NAME, REPORT_TABLE_FILE_NAME)


class ReportTable:
    def __init__(self, experiment_path: str):
        """
        Initializes the ReportTable with the experiment path and loads competition history data.

        Parameters:
        - experiment_path (str): Path to the folder containing the competition history CSV file.
        """
        self.experiment_path = experiment_path
        self._competition_history = pd.read_csv(os.path.join(experiment_path, COMPEITION_HISTORY_FILE_NAME))

    def generate_report_table(self):
        """
        Generates a report table summarizing the competition history data.

        Returns:
        - pd.DataFrame: A DataFrame containing the report table.
        """
        report_table = {
            REPORT_TABLE_BEST_AGENT_COLUMN: [self.__best_agent()],
            REPORT_TABLE_WINNING_HOMOGENEITY_COLUMN: [self.__calculate_winning_homogeneity()]
        }

        pd.DataFrame(report_table).to_csv(os.path.join(self.experiment_path, REPORT_TABLE_FILE_NAME), index=False)

    def __get_player_wins(self, query_id: str) -> dict:
        """
        Creates a dictionary with the number of wins (rank 1) per player for a given query.

        Parameters:
        - query_id (str): The query identifier to filter competition history data.

        Returns:
        - dict: A dictionary where keys are player names and values are the count of rank 1 finishes.
        """
        df = self._competition_history[self._competition_history[HISTORY_DF_QUERY_ID_COLUMN] == query_id]
        player_wins = {player: len(df[(df[HISTORY_DF_PLAYER_COLUMN] == player) & (df[HISTORY_DF_RANK_COLUMN] == 1)])
                       for player in df[HISTORY_DF_PLAYER_COLUMN].unique()}
        return player_wins

    def __calculate_winning_homogeneity(self) -> float:
        """
        Calculates the winning homogeneity metric across all queries, which indicates
        how dominant the best-performing player is relative to the average.

        Returns:
        - float: The average winning homogeneity score across all queries.
        """
        wins_per_query = [
            max(player_wins.values()) / np.mean(list(player_wins.values()))
            for query_id in self._competition_history[HISTORY_DF_QUERY_ID_COLUMN].unique()
            if (player_wins := self.__get_player_wins(query_id))
        ]

        return np.mean(wins_per_query)

    def __best_agent(self) -> float:
        """
        Identifies the player with the highest aggregate wins across all queries.

        Returns:
        - str: The player with the most overall wins.
        """
        total_wins = {}
        for query_id in self._competition_history[HISTORY_DF_QUERY_ID_COLUMN].unique():
            player_wins = self.__get_player_wins(query_id)
            for player, wins in player_wins.items():
                total_wins[player] = total_wins.get(player, 0) + wins

        return max(total_wins, key=total_wins.get)

