import os.path

import pandas as pd
import numpy as np

from constants.constants import (COMPEITION_HISTORY_FILE_NAME, HISTORY_DF_ROUND_COLUMN, HISTORY_DF_RANK_COLUMN,
                                 HISTORY_DF_PLAYER_COLUMN, HISTORY_DF_QUERY_ID_COLUMN, HISTORY_DF_USER_QUERY_COLUMN,
                                 COMPETITION_HISTORY_PIVOIT_FILE_NAME)


class RankAnalysis:
    def __init__(self, experiment_path):
        """
        Initializes RankAnalysis with competition history data.

        Parameters:
        - competition_history_path (str): Path to the CSV file containing competition history.
        """
        self.experiment_path = experiment_path

        self._competition_history = pd.read_csv(os.path.join(experiment_path, COMPEITION_HISTORY_FILE_NAME))
        self._NUM_ROUNDS = int(self._competition_history[HISTORY_DF_ROUND_COLUMN].max())  # Total rounds in competition
        self._MAX_RANK = int(self._competition_history[HISTORY_DF_RANK_COLUMN].max())  # Maximum rank value

    def rank_analysis(self, experiment_name):
        """
        Public method to perform analysis on ranking changes and probabilities based on historical data.

        Parameters:
        - experiment_name (str): Name of the experiment or analysis.
        """
        df = self._preprocess_df(self._competition_history)
        self._changes_analysis(df, f'{experiment_name}:')
        self._prob_analysis(df)

    def _preprocess_df(self, df):
        """
        Private method to preprocess the data for rank analysis by creating a pivot table of ranks by round.

        Parameters:
        - df (DataFrame): Competition history dataframe.

        Returns:
        - DataFrame: Transformed dataframe with ranks as columns for each round.
        """
        df = df.copy()
        df = df[df[HISTORY_DF_ROUND_COLUMN] != 0]  # Exclude round 0 if it exists
        df[HISTORY_DF_USER_QUERY_COLUMN] = df[HISTORY_DF_PLAYER_COLUMN] + '_' + df[HISTORY_DF_QUERY_ID_COLUMN].astype(str)
        df = df[[HISTORY_DF_USER_QUERY_COLUMN, HISTORY_DF_ROUND_COLUMN, HISTORY_DF_RANK_COLUMN]]
        df = df.pivot(index=HISTORY_DF_USER_QUERY_COLUMN, columns=HISTORY_DF_ROUND_COLUMN, values=HISTORY_DF_RANK_COLUMN)
        df.reset_index(inplace=True)
        df.index.name = 'index'

        df.to_csv(os.path.join(self.experiment_path, COMPETITION_HISTORY_PIVOIT_FILE_NAME), index=True)

        return df

    def _changes_analysis(self, df, text=None):
        """
        Private method to analyze rank position changes between consecutive rounds.

        Parameters:
        - df (DataFrame): Processed dataframe with ranks per round.
        - text (str): Optional text to print before analysis output.
        """
        sum_changes = []
        num_changes = []

        # Analyze changes in rank position across rounds
        for i in range(1, min(30, self._NUM_ROUNDS)):
            num_changes.append(int((df[i] != df[i + 1]).sum()))
            sum_changes.append(int((df[i] - df[i + 1]).abs().sum()))

        if text:
            print(text)
        print('Number of changes:', num_changes)
        print('Sum of changes:', sum_changes)
        print('Total sum of changes:', sum(sum_changes))
        print('Total number of changes:', sum(num_changes))

    def _calc_prob(self, df, curr, diff=1):
        """
        Private method to calculate the transition probabilities from a given rank to all possible ranks.

        Parameters:
        - df (DataFrame): Processed dataframe with ranks per round.
        - curr (int): The current rank to calculate transitions from.
        - diff (int): Number of rounds between comparisons (default is 1 for consecutive rounds).

        Returns:
        - ndarray: Probability distribution of transitioning to each rank from the current rank.
        """
        prob = [0] * self._MAX_RANK

        # Compute transition probabilities
        for i in range(1, self._NUM_ROUNDS + 1 - diff):
            times = (df[i] == curr)
            for pos in range(1, self._MAX_RANK + 1):
                prob[pos - 1] += (times & (df[i + diff] == pos)).sum()

        prob = np.array(prob) / sum(prob)
        return prob

    def _prob_analysis(self, df):
        """
        Private method to perform probability analysis for each rank's transition probabilities, expected value, and standard deviation.

        Parameters:
        - df (DataFrame): Processed dataframe with ranks per round.
        - text (str): Optional text to print before analysis output.
        """

        # Calculate probability, expected value, and standard deviation for each rank
        for i in range(1, self._MAX_RANK + 1):
            prob = self._calc_prob(df, i)
            expected_value = (prob * np.arange(1, self._MAX_RANK + 1)).sum()
            variance = (prob * (np.arange(1, self._MAX_RANK + 1) ** 2)).sum() - expected_value ** 2
            sd = np.sqrt(variance)

            print(f'Position {i}: Probabilities: {prob.round(2)}, Expected Value: {expected_value:.2f}, SD: {sd:.2f}')