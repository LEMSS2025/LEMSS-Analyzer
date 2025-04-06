from constants.constants import NUM_TO_STR, NUM_TO_REPRESENTATION, EMBEDDING_FILENAMES, CONFIDENCE_INTERVAL

import pickle
import os
from itertools import combinations
from typing import Union, List, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class EmbeddingAnalyzer:
    def __init__(self, experiment_path: str) -> None:
        """
        Initialize the EmbeddingAnalyzer with the given experiment path.

        Parameters:
            experiment_path (str): Path to the experiment folder.
        """
        self.experiment_path = experiment_path
        os.makedirs(os.path.join(self.experiment_path, "embeddings_graphs"), exist_ok=True)
        self.__extract_embeddings()
        self.__extract_competition_history()

    def __extract_embeddings(self) -> None:
        """
        Extract embeddings from the experiment folder and store them in self.representations.
        """
        self.representations = []

        # Loop over the embedding filenames and load each embedding
        for filename in EMBEDDING_FILENAMES:
            file_path = os.path.join(self.experiment_path, filename)
            # Check if the file exists
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"Embedding file not found: {file_path}")
            # Load the embedding and append to self.representations
            with open(file_path, "rb") as f:
                self.representations.append(pickle.load(f))

    def __extract_competition_history(self) -> None:
        """
        Extract the competition history from the experiment folder.
        """
        competition_history_path = os.path.join(self.experiment_path, "competition_history.csv")
        df = pd.read_csv(competition_history_path)
        self.competition_history = self.__filter_competition_history(df)

    def __filter_competition_history(self, competition_history: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out documents that are the same as the initial document for each query.

        Parameters:
            competition_history (pd.DataFrame): Competition history dataframe.

        Returns:
            pd.DataFrame: Filtered competition history dataframe.
        """
        # Get initial documents for each query_id where round is 0
        init_docs = \
        competition_history[competition_history["round"] == 0].drop_duplicates(subset=['query_id']).set_index(
            'query_id')['document']

        # Merge initial documents back into the competition_history
        competition_history = competition_history.merge(init_docs.rename('init_doc'), on='query_id', how='left',
                                                        validate="many_to_one")

        # Filter out rows where 'document' equals 'init_doc'
        filtered_competition_history = competition_history[
            competition_history['document'] != competition_history['init_doc']
            ]

        # Select desired columns
        filtered_competition_history = filtered_competition_history[
            ["docno", "query_id", "round", "rank", "player", "document"]
        ]

        return filtered_competition_history

    def plot_graphs(self) -> None:
        """
        Generate all plots for each embedding representation.
        """
        # Loop over each representation and generate plots
        for representation_num in range(len(self.representations)):
            self.__plot_consecutive_winner_similarity_excluding_players(self.competition_history, representation_num,
                                                                        ["teacher", "writer"])
            self.__plot_winner_to_non_winner_similarity(self.competition_history, representation_num,
                                                        ["teacher", "writer"])
            self.__plot_non_winner_to_winner_similarity(self.competition_history, representation_num,
                                                        ["teacher", "writer"])
            self.__plot_average_similarity_of_player_documents_consecutive_rounds_excluding_players(
                self.competition_history, representation_num, ["teacher", "writer"])
            self.__plot_average_similarity_of_player_documents_consecutive_rounds_only_players(self.competition_history,
                                                                                               representation_num,
                                                                                               ["teacher", "writer"])
            self.__plot_diameter_and_average_over_time_only_players(self.competition_history, representation_num,
                                                                    ["teacher", "writer"])
            self.__plot_diameter_and_average_over_time_excluding_players(self.competition_history, representation_num,
                                                                         ["teacher", "writer"])
            self.__plot_diameter_and_average_over_time(self.competition_history, representation_num)
            self.__plot_average_similarity_of_player_documents_consecutive_rounds(self.competition_history,
                                                                                  representation_num)
            self.__save_rank_diameter_and_average_last_round(self.competition_history, representation_num)
            self.__plot_average_and_diameter_of_player_documents(self.competition_history, representation_num)
            self.__plot_consecutive_winner_similarity_over_time(self.competition_history, representation_num)
            self.__plot_first_second_similarity_over_time(self.competition_history, representation_num, [1, 2])
            self.__plot_rank_diameter_and_average_over_time(self.competition_history, representation_num, 1)

        # Generate the average unique documents plot
        self.__plot_average_unique_documents(self.competition_history)

    def __calculate_pairwise_similarity(self, group: pd.DataFrame,
                                        representations: Dict[int, Dict[str, Dict[str, float]]],
                                        rep_num: int, return_min: bool = False) -> Union[float, List[float]]:
        """
        Calculate pairwise similarities between documents in a group.

        Parameters:
            group (pd.DataFrame): DataFrame containing document identifiers under the 'docno' column.
            representations (Dict[int, Dict[str, Dict[str, float]]]): Nested dictionary of representations.
            rep_num (int): Key for selecting the specific representation.
            return_min (bool, optional): If True, returns both mean and minimum distances. Defaults to False.

        Returns:
            float or List[float]: Mean distance, or [mean_distance, min_distance] if return_min is True.
        """
        # Extract document numbers from the group DataFrame
        docnos = group["docno"].values
        # Access the specific representation based on rep_num
        representation = representations[rep_num]

        # Calculate distances between all pairs of documents
        distances = [
            representation[docno1][docno2]
            for docno1, docno2 in combinations(docnos, 2)
            if docno1 in representation and docno2 in representation[docno1]
        ]

        # If distances are available, compute mean and minimum distances
        if distances:
            mean_distance = np.mean(distances)
            min_distance = np.min(distances)
        else:
            # Default to zero if no distances are found
            mean_distance = min_distance = 0.0

        # Return both mean and minimum distances if return_min is True; otherwise, return mean_distance
        return [mean_distance, min_distance] if return_min else mean_distance

    def __plot_graph(self, X: np.array, y: np.array, title: str, xlabel: str, ylabel: str, save_path: str,
                     error_bars: np.array = None) -> None:
        """
        Plot a graph with the given X and y values, title, x-axis label, y-axis label, and save path.

        Parameters:
            X (List): List of x-axis values.
            y (List): List of y-axis values.
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            save_path (str): Path to save the plot.
            error_bars (List, optional): List of error bars for the y-axis values. Defaults to None.
        """
        plt.figure(figsize=(6.5, 6.5))
        plt.plot(X, y, color="blue")

        if error_bars is not None:
            plt.errorbar(X, y, yerr=error_bars, fmt='o', color="blue")

        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.title(title, fontsize=16)

        # Move graph a little bit to the right
        plt.subplots_adjust(left=0.18)
        plt.savefig(save_path)

    def __save_numpy(self, data: np.array, dir_name: str, file_path: str) -> None:
        """
        Save the given NumPy array to a file in the specified directory.

        Parameters:
            data (np.array): NumPy array to save.
            dir_name (str): Directory to save the file in.
            file_path (str): Path of the file to save.
        """
        os.makedirs(dir_name, exist_ok=True)
        np.save(file_path, data)

    def __plot_first_second_similarity_over_time(self, df: pd.DataFrame, representation_num: int,
                                                 compare_positions=[1, 2]) -> None:
        """
        Plot the average similarity between the first and second ranked documents over time.

        Parameters:
            df (pd.DataFrame): DataFrame containing the competition history.
            representation_num (int): Index of the representation to use.
            compare_positions (List[int]): List of positions to compare.
        """
        df = df[df["rank"].isin(compare_positions)]
        avg_first_second_similarity, sd_first_second_similarity, len_rounds = [], [], []
        round_similarity_to_save = []

        for round_num in range(1, df["round"].max() + 1):
            round_df = df[df["round"] == round_num]
            groups = round_df.groupby('query_id').filter(lambda x: len(x) > 1).groupby('query_id')
            round_similarity = groups.apply(self.__calculate_pairwise_similarity, self.representations,
                                            representation_num)

            round_similarity_to_save.append(round_similarity)
            avg_first_second_similarity.append(round_similarity.mean())
            sd_first_second_similarity.append(round_similarity.std())
            len_rounds.append(len(groups))

        # Save the results
        folder_to_save = os.path.join(self.experiment_path, "embeddings_graphs",
                                      "plot_first_second_similarity_over_time")
        file_to_to_save = os.path.join(folder_to_save,
                                       f"{compare_positions[0]}-{compare_positions[1]}-{NUM_TO_REPRESENTATION[representation_num]}-{os.path.basename(self.experiment_path)}.npy")
        self.__save_numpy(np.array(round_similarity_to_save, dtype=np.float32), folder_to_save, file_to_to_save)

        self.__plot_graph(range(1, df["round"].max() + 1), np.array(avg_first_second_similarity),
                          f"Average {NUM_TO_STR[compare_positions[0]]}-{NUM_TO_STR[compare_positions[1]]} ranked players \nsimilarity vs round",
                          'Round',
                          f"Average {NUM_TO_STR[compare_positions[0]]}-{NUM_TO_STR[compare_positions[1]]} ranked players similarity",
                          file_to_to_save.replace(".npy", ".png"),
                          error_bars=CONFIDENCE_INTERVAL * np.array(sd_first_second_similarity) / np.sqrt(
                              np.array(len_rounds)[0]))

    def __plot_rank_diameter_and_average_over_time(self, df: pd.DataFrame, representation_num: int, rank: int) -> None:
        """
        Plot the rank diameter and average similarity for the given rank over time.

        Parameters:
            df (pd.DataFrame): DataFrame containing the competition history.
            representation_num (int): Index of the representation to use.
            rank (int): Rank to plot the diameter and average similarity for.
        """
        df = df[df["rank"] == rank]
        avg_similarity, diameter_similarity = [], []
        round_rank_measurements_mean_to_save, round_rank_measurements_min_to_save = [], []

        for round_num in range(2, df["round"].max() + 1):
            round_df = df[df["round"] <= round_num]
            groups = round_df.groupby('query_id').filter(lambda x: len(x) > 1).groupby('query_id')
            round_rank_measurements = groups.apply(self.__calculate_pairwise_similarity,
                                                   self.representations, representation_num, return_min=True)

            round_rank_measurements_mean_to_save.append(round_rank_measurements.apply(lambda x: x[0]))
            round_rank_measurements_min_to_save.append(round_rank_measurements.apply(lambda x: x[1]))

            avg_similarity.append(round_rank_measurements.apply(lambda x: x[0]).mean())
            diameter_similarity.append(round_rank_measurements.apply(lambda x: x[1]).mean())

        # Save the results
        folder_to_save_mean = os.path.join(self.experiment_path, "embeddings_graphs",
                                           "plot_rank_diameter_and_average_over_time-mean")
        file_to_save_mean = os.path.join(folder_to_save_mean,
                                         f"{rank}-{NUM_TO_REPRESENTATION[representation_num]}-{os.path.basename(self.experiment_path)}.npy")
        self.__save_numpy(np.array(round_rank_measurements_mean_to_save, dtype=np.float32), folder_to_save_mean,
                          file_to_save_mean)

        folder_to_save_min = os.path.join(self.experiment_path, "embeddings_graphs",
                                          "plot_rank_diameter_and_average_over_time-min")
        file_to_save_min = os.path.join(folder_to_save_min,
                                        f"{rank}-{NUM_TO_REPRESENTATION[representation_num]}-{os.path.basename(self.experiment_path)}.npy")
        self.__save_numpy(np.array(round_rank_measurements_min_to_save, dtype=np.float32), folder_to_save_min,
                          file_to_save_min)

        self.__plot_graph(range(2, df["round"].max() + 1), np.array(avg_similarity, dtype=np.float32),
                          f"Average {NUM_TO_STR[rank]}-ranked players \nsimilarity vs round",
                          'Round', f'Average {NUM_TO_STR[rank]}-ranked players similarity',
                          file_to_save_mean.replace(".npy", ".png"))
        self.__plot_graph(range(2, df["round"].max() + 1), np.array(diameter_similarity, dtype=np.float32),
                          f"Diameter {NUM_TO_STR[rank]}-ranked players \nsimilarity vs round",
                          'Round', f'Diameter {NUM_TO_STR[rank]}-ranked players similarity',
                          file_to_save_min.replace(".npy", ".png"))

    def __plot_consecutive_winner_similarity_over_time(self, df: pd.DataFrame, representation_num: int) -> None:
        """
        Save the similarity matrix for consecutive winners and plot the average similarity over time.

        Parameters:
            df (pd.DataFrame): DataFrame containing the competition history.
            representation_num (int): Index of the representation to use.
        """
        similarity_matrix = []

        # Compute the raw similarity matrix with an indicator for same player comparisons
        for round_num in range(2, df["round"].max() + 1):
            rounds_df = df[((df["round"] == round_num) | (df["round"] == round_num - 1)) & (df["rank"] == 1)]
            groups = rounds_df.groupby('query_id')

            # Compute similarity for each group and apply 2 for same-player comparisons
            group_similarities = [
                self.__calculate_pairwise_similarity(group, self.representations, representation_num) + 2
                if len(group["player"].unique()) == 1 else self.__calculate_pairwise_similarity(
                    group, self.representations, representation_num)
                for query_id, group in groups
            ]

            similarity_matrix.append(group_similarities)

        # Calculate the average similarity while removing same-player comparisons
        average_similarity = [
            np.mean([similarity for similarity in group_similarities if similarity < 2])
            for group_similarities in similarity_matrix
        ]

        # Save the results
        folder_to_save = os.path.join(self.experiment_path, "embeddings_graphs", "winner_similarity_over_time")
        file_to_save = os.path.join(folder_to_save,
                                    f"{NUM_TO_REPRESENTATION[representation_num]}-{os.path.basename(self.experiment_path)}.npy")
        self.__save_numpy(np.array(similarity_matrix, dtype=np.float32), folder_to_save, file_to_save)

        self.__plot_graph(range(2, df["round"].max() + 1), np.array(average_similarity, dtype=np.float32),
                          "Average similarity between \nconsecutive winners vs round",
                          'Round', 'Average similarity between consecutive winners',
                          file_to_save.replace(".npy", ".png"))

    def __plot_consecutive_winner_similarity_excluding_players(self, df: pd.DataFrame, representation_num: int,
                                                               excluded_players: list) -> None:
        """
        Save the similarity matrix for consecutive winners and plot the average similarity over time,
        excluding specified players.

        Parameters:
            df (pd.DataFrame): DataFrame containing the competition history.
            representation_num (int): Index of the representation to use.
            excluded_players (list): List of player names to exclude from the analysis.
        """
        similarity_matrix = []
        sample_counts = []
        max_samples = 0

        # Compute the raw similarity matrix with an indicator for same player comparisons
        for round_num in range(2, df["round"].max() + 1):
            # Filter out the excluded players
            filtered_df = df[~df["player"].isin(excluded_players)]

            rounds_df = filtered_df[((filtered_df["round"] == round_num) | (filtered_df["round"] == round_num - 1)) & (
                    filtered_df["rank"] == 1)]
            groups = rounds_df.groupby('query_id')

            # Compute similarity for each group and apply 2 for same-player comparisons
            group_similarities = []
            for query_id, group in groups:
                if len(group) < 2:  # Skip if we don't have entries for both rounds
                    continue

                similarity = self.__calculate_pairwise_similarity(group, self.representations, representation_num)
                if len(group["player"].unique()) == 1:
                    similarity += 2  # Mark same-player comparisons

                group_similarities.append(similarity)

            if group_similarities:
                similarity_matrix.append(group_similarities)
                sample_counts.append(len(group_similarities))
                max_samples = max(max_samples, len(group_similarities))
            else:
                # If no similarities for this round, add an empty list
                similarity_matrix.append([])
                sample_counts.append(0)

        # Pad arrays with -1 to make them rectangular
        padded_matrix = []
        for similarities in similarity_matrix:
            if not similarities:
                padded_matrix.append([-1] * max_samples)
            else:
                padded = similarities + [-1] * (max_samples - len(similarities))
                padded_matrix.append(padded)

        # Calculate the average similarity while removing same-player comparisons
        average_similarity = []
        for i, similarities in enumerate(similarity_matrix):
            valid_similarities = [s for s in similarities if s < 2 and s >= 0]
            if valid_similarities:
                average_similarity.append(np.mean(valid_similarities))
            else:
                average_similarity.append(np.nan)  # Use NaN for rounds with no valid data

        # Save the results
        folder_to_save = os.path.join(self.experiment_path, "embeddings_graphs",
                                      "winner_similarity_over_time_excluded_players")
        file_to_save = os.path.join(folder_to_save,
                                    f"{NUM_TO_REPRESENTATION[representation_num]}-excluded-players({','.join(excluded_players)})-{os.path.basename(self.experiment_path)}.npy")
        np_result = np.array(padded_matrix, dtype=np.float32)
        self.__save_numpy(np_result, folder_to_save, file_to_save)

        # Save sample counts separately
        counts_file = os.path.join(folder_to_save,
                                   f"{NUM_TO_REPRESENTATION[representation_num]}-excluded-counts-players({','.join(excluded_players)})-{os.path.basename(self.experiment_path)}.npy")
        self.__save_numpy(np.array(sample_counts, dtype=np.int32), folder_to_save, counts_file)

        # Filter out NaN values for plotting
        valid_rounds = []
        valid_averages = []
        for i, avg in enumerate(average_similarity):
            if not np.isnan(avg):
                valid_rounds.append(i + 2)  # +2 because rounds start at 2
                valid_averages.append(avg)

        if valid_averages:
            self.__plot_graph(valid_rounds, np.array(valid_averages, dtype=np.float32),
                              f"Average similarity between \nconsecutive winners vs round (excluding players)",
                              'Round', 'Average similarity between consecutive winners',
                              file_to_save.replace(".npy", ".png"))
        else:
            print("No valid data points for plotting after excluding players.")

        # Plot samples vs rounds graph
        rounds_for_samples = range(2, df["round"].max() + 1)
        self.__plot_graph(rounds_for_samples, np.array(sample_counts, dtype=np.int32),
                          f"Number of samples per round (excluding players)",
                          'Round', 'Number of samples',
                          file_to_save.replace(".npy", "_samples.png"))

    def __plot_winner_to_non_winner_similarity(self, df: pd.DataFrame, representation_num: int,
                                               players_list: list) -> None:
        """
        Calculate similarity for cases where a player from the provided list was the winner in the previous round
        and a player NOT from the list is the winner in the current round.

        Parameters:
            df (pd.DataFrame): DataFrame containing the competition history.
            representation_num (int): Index of the representation to use.
            players_list (list): List of player names to consider for the winner-to-non-winner transition.
        """
        similarity_matrix = []
        sample_counts = []
        max_samples = 0

        # Compute the raw similarity matrix for transitions from players in list to players not in list
        for round_num in range(2, df["round"].max() + 1):
            # Get previous round winners who are in the players_list
            prev_round_winners = df[
                (df["round"] == round_num - 1) & (df["rank"] == 1) & (df["player"].isin(players_list))]
            prev_winner_queries = set(prev_round_winners["query_id"].unique())

            # Get current round winners who are NOT in the players_list
            curr_round_winners = df[(df["round"] == round_num) & (df["rank"] == 1) & (~df["player"].isin(players_list))]
            curr_winner_queries = set(curr_round_winners["query_id"].unique())

            # Find queries where there's a transition from list player to non-list player
            common_queries = prev_winner_queries.intersection(curr_winner_queries)

            group_similarities = []
            if common_queries:
                # For each relevant query, compute the similarity
                for query_id in common_queries:
                    group = df[((df["round"] == round_num) | (df["round"] == round_num - 1)) &
                               (df["rank"] == 1) &
                               (df["query_id"] == query_id)]

                    if len(group) == 2:  # Ensure we have both rounds
                        similarity = self.__calculate_pairwise_similarity(group, self.representations,
                                                                          representation_num)
                        group_similarities.append(similarity)

            if group_similarities:
                similarity_matrix.append(group_similarities)
                sample_counts.append(len(group_similarities))
                max_samples = max(max_samples, len(group_similarities))
            else:
                # If no similarities for this round, add an empty list
                similarity_matrix.append([])
                sample_counts.append(0)

        # Pad arrays with -1 to make them rectangular
        padded_matrix = []
        for similarities in similarity_matrix:
            if not similarities:
                padded_matrix.append([-1] * max_samples)
            else:
                padded = similarities + [-1] * (max_samples - len(similarities))
                padded_matrix.append(padded)

        # Calculate the average similarity for plotting
        average_similarity = []
        for similarities in similarity_matrix:
            valid_similarities = [s for s in similarities if s >= 0]
            if valid_similarities:
                average_similarity.append(np.mean(valid_similarities))
            else:
                average_similarity.append(np.nan)

        # Save the results
        folder_to_save = os.path.join(self.experiment_path, "embeddings_graphs", "list_to_nonlist_winner_similarity")
        file_to_save = os.path.join(folder_to_save,
                                    f"{NUM_TO_REPRESENTATION[representation_num]}-list-to-nonlist-players({','.join(players_list)})-{os.path.basename(self.experiment_path)}.npy")
        np_result = np.array(padded_matrix, dtype=np.float32)
        self.__save_numpy(np_result, folder_to_save, file_to_save)

        # Save sample counts separately
        counts_file = os.path.join(folder_to_save,
                                   f"{NUM_TO_REPRESENTATION[representation_num]}-list-to-nonlist-counts-players({','.join(players_list)})-{os.path.basename(self.experiment_path)}.npy")
        self.__save_numpy(np.array(sample_counts, dtype=np.int32), folder_to_save, counts_file)

        # Filter out NaN values for plotting
        valid_rounds = []
        valid_averages = []
        for i, avg in enumerate(average_similarity):
            if not np.isnan(avg):
                valid_rounds.append(i + 2)  # +2 because rounds start at 2
                valid_averages.append(avg)

        if valid_averages:
            self.__plot_graph(valid_rounds, np.array(valid_averages, dtype=np.float32),
                              "Average similarity when winner transitions\nfrom list player to non-list player",
                              'Round', 'Average similarity',
                              file_to_save.replace(".npy", ".png"))
        else:
            print("No valid data points for plotting list-to-nonlist transitions.")

        # Plot samples vs rounds graph
        rounds_for_samples = range(2, df["round"].max() + 1)
        self.__plot_graph(rounds_for_samples, np.array(sample_counts, dtype=np.int32),
                          f"Number of list-to-nonlist transitions per round",
                          'Round', 'Number of samples',
                          file_to_save.replace(".npy", "_samples.png"))

    def __plot_non_winner_to_winner_similarity(self, df: pd.DataFrame, representation_num: int,
                                               players_list: list) -> None:
        """
        Calculate similarity for cases where a player NOT from the provided list was the winner in the previous round
        and a player FROM the list is the winner in the current round.

        Parameters:
            df (pd.DataFrame): DataFrame containing the competition history.
            representation_num (int): Index of the representation to use.
            players_list (list): List of player names to consider for the non-winner-to-winner transition.
        """
        similarity_matrix = []
        sample_counts = []
        max_samples = 0

        # Compute the raw similarity matrix for transitions from non-list players to list players
        for round_num in range(2, df["round"].max() + 1):
            # Get previous round winners who are NOT in the players_list
            prev_round_winners = df[
                (df["round"] == round_num - 1) & (df["rank"] == 1) & (~df["player"].isin(players_list))]
            prev_winner_queries = set(prev_round_winners["query_id"].unique())

            # Get current round winners who ARE in the players_list
            curr_round_winners = df[(df["round"] == round_num) & (df["rank"] == 1) & (df["player"].isin(players_list))]
            curr_winner_queries = set(curr_round_winners["query_id"].unique())

            # Find queries where there's a transition from non-list player to list player
            common_queries = prev_winner_queries.intersection(curr_winner_queries)

            group_similarities = []
            if common_queries:
                # For each relevant query, compute the similarity
                for query_id in common_queries:
                    group = df[((df["round"] == round_num) | (df["round"] == round_num - 1)) &
                               (df["rank"] == 1) &
                               (df["query_id"] == query_id)]

                    if len(group) == 2:  # Ensure we have both rounds
                        similarity = self.__calculate_pairwise_similarity(group, self.representations,
                                                                          representation_num)
                        group_similarities.append(similarity)

            if group_similarities:
                similarity_matrix.append(group_similarities)
                sample_counts.append(len(group_similarities))
                max_samples = max(max_samples, len(group_similarities))
            else:
                # If no similarities for this round, add an empty list
                similarity_matrix.append([])
                sample_counts.append(0)

        # Pad arrays with -1 to make them rectangular
        padded_matrix = []
        for similarities in similarity_matrix:
            if not similarities:
                padded_matrix.append([-1] * max_samples)
            else:
                padded = similarities + [-1] * (max_samples - len(similarities))
                padded_matrix.append(padded)

        # Calculate the average similarity for plotting
        average_similarity = []
        for similarities in similarity_matrix:
            valid_similarities = [s for s in similarities if s >= 0]
            if valid_similarities:
                average_similarity.append(np.mean(valid_similarities))
            else:
                average_similarity.append(np.nan)

        # Save the results
        folder_to_save = os.path.join(self.experiment_path, "embeddings_graphs", "nonlist_to_list_winner_similarity")
        file_to_save = os.path.join(folder_to_save,
                                    f"{NUM_TO_REPRESENTATION[representation_num]}-nonlist-to-list-players({','.join(players_list)})-{os.path.basename(self.experiment_path)}.npy")
        np_result = np.array(padded_matrix, dtype=np.float32)
        self.__save_numpy(np_result, folder_to_save, file_to_save)

        # Save sample counts separately
        counts_file = os.path.join(folder_to_save,
                                   f"{NUM_TO_REPRESENTATION[representation_num]}-nonlist-to-list-counts-players({','.join(players_list)})-{os.path.basename(self.experiment_path)}.npy")
        self.__save_numpy(np.array(sample_counts, dtype=np.int32), folder_to_save, counts_file)

        # Filter out NaN values for plotting
        valid_rounds = []
        valid_averages = []
        for i, avg in enumerate(average_similarity):
            if not np.isnan(avg):
                valid_rounds.append(i + 2)  # +2 because rounds start at 2
                valid_averages.append(avg)

        if valid_averages:
            self.__plot_graph(valid_rounds, np.array(valid_averages, dtype=np.float32),
                              "Average similarity when winner transitions\nfrom non-list player to list player",
                              'Round', 'Average similarity',
                              file_to_save.replace(".npy", ".png"))
        else:
            print("No valid data points for plotting nonlist-to-list transitions.")

        # Plot samples vs rounds graph
        rounds_for_samples = range(2, df["round"].max() + 1)
        self.__plot_graph(rounds_for_samples, np.array(sample_counts, dtype=np.int32),
                          f"Number of nonlist-to-list transitions per round",
                          'Round', 'Number of samples',
                          file_to_save.replace(".npy", "_samples.png"))

    def __plot_average_unique_documents(self, df: pd.DataFrame) -> None:
        """
        Plot the average number of unique documents per query by round.

        Parameters:
            df (pd.DataFrame): DataFrame containing the competition history.
        """
        unique_documents_counts_per_round = [
            df[df["round"] == round_num].groupby('query_id')["document"].nunique().values
            for round_num in range(1, df["round"].max() + 1)
        ]

        average_unique_documents = [counts.mean() for counts in unique_documents_counts_per_round]

        # Save the results
        folder_to_save = os.path.join(self.experiment_path, "embeddings_graphs", "average_unique_documents_over_time")
        file_to_save = os.path.join(folder_to_save, f"{os.path.basename(self.experiment_path)}.npy")
        self.__save_numpy(np.array(unique_documents_counts_per_round, dtype=np.float32), folder_to_save, file_to_save)

        self.__plot_graph(range(1, df["round"].max() + 1), np.array(average_unique_documents, dtype=np.float32),
                          "Average number of unique documents vs round",
                          'Round', 'Average number of unique documents',
                          file_to_save.replace(".npy", ".png"))

    def __plot_average_and_diameter_of_player_documents(self, df: pd.DataFrame, representation_num: int) -> None:
        """
        Plot the average similarity and diameter of player documents over time

        Parameters:
            df (pd.DataFrame): DataFrame containing the competition history.
            representation_num (int): Index of the representation to use.
        """
        matrix_metrics = []

        for query in df["query_id"].unique():
            game_df = df[df["query_id"] == query]
            all_players_game_metrics = []

            for player in game_df["player"].unique():
                player_game_metrics = []
                for round_num in range(2, game_df["round"].max() + 1):
                    player_df = game_df[(game_df["round"] <= round_num) & (game_df["player"] == player)]
                    round_rank_measurements = self.__calculate_pairwise_similarity(
                        player_df, self.representations, representation_num, return_min=True)
                    player_game_metrics.append(round_rank_measurements)

                all_players_game_metrics.append(player_game_metrics)

            matrix_metrics.append(all_players_game_metrics)

        numpy_matrix = np.array(matrix_metrics, dtype=np.float32)
        mean_matrix = numpy_matrix[:, :, :, 0]
        min_matrix = numpy_matrix[:, :, :, 1]

        # Save the results
        folder_to_save_mean = os.path.join(self.experiment_path, "embeddings_graphs",
                                           "average_and_diameter_of_player_documents-mean")
        file_to_save_mean = os.path.join(folder_to_save_mean,
                                         f"{NUM_TO_REPRESENTATION[representation_num]}-{os.path.basename(self.experiment_path)}.npy")
        self.__save_numpy(mean_matrix, folder_to_save_mean, file_to_save_mean)

        folder_to_save_min = os.path.join(self.experiment_path, "embeddings_graphs",
                                          "average_and_diameter_of_player_documents-min")
        file_to_save_min = os.path.join(folder_to_save_min,
                                        f"{NUM_TO_REPRESENTATION[representation_num]}-{os.path.basename(self.experiment_path)}.npy")
        self.__save_numpy(min_matrix, folder_to_save_min, file_to_save_min)

        mean_matrix_for_graph = mean_matrix.swapaxes(1, 2)
        min_matrix_for_graph = min_matrix.swapaxes(1, 2)
        mean_matrix_for_graph = mean_matrix_for_graph.mean(axis=2).mean(axis=0)
        min_matrix_for_graph = min_matrix_for_graph.mean(axis=2).mean(axis=0)

        self.__plot_graph(range(2, df["round"].max() + 1), mean_matrix_for_graph,
                          "Average similarity of player documents vs round",
                          'Round', 'Average similarity of player documents',
                          file_to_save_mean.replace(".npy", ".png"))
        self.__plot_graph(range(2, df["round"].max() + 1), min_matrix_for_graph,
                          "Diameter similarity of player documents vs round",
                          'Round', 'Diameter similarity of player documents',
                          file_to_save_min.replace(".npy", ".png"))

        df = pd.DataFrame({"average_and_diameter_of_player_documents_mean": mean_matrix.mean(),
                           "average_and_diameter_of_player_documents_min": min_matrix.mean()},
                          index=[0])
        df.to_csv(
            os.path.join(self.experiment_path, "embeddings_graphs", "average_and_diameter_of_player_documents.csv"))

    def __save_rank_diameter_and_average_last_round(self, df: pd.DataFrame, representation_num: int) -> None:
        """
        Plot the rank diameter and average similarity for the last round.

        Parameters:
            df (pd.DataFrame): DataFrame containing the competition history.
            representation_num (int): Index of the representation to use.
        """
        round_df = df[df["round"] == df["round"].max()]
        groups = round_df.groupby('query_id').filter(lambda x: len(x) > 1).groupby('query_id')
        round_rank_measurements = groups.apply(self.__calculate_pairwise_similarity, self.representations,
                                               representation_num, return_min=True)

        round_avg_similarity = round_rank_measurements.apply(lambda x: x[0])
        round_diameter_similarity = round_rank_measurements.apply(lambda x: x[1])

        # Save the results
        numpy_mean_to_save = np.array(round_avg_similarity, dtype=np.float32)
        folder_to_save_mean = os.path.join(self.experiment_path, "embeddings_graphs",
                                           "rank_diameter_and_average_last_round-mean")
        file_to_save_mean = os.path.join(folder_to_save_mean,
                                         f"{NUM_TO_REPRESENTATION[representation_num]}-{os.path.basename(self.experiment_path)}.npy")
        self.__save_numpy(numpy_mean_to_save, folder_to_save_mean, file_to_save_mean)

        numpy_min_to_save = np.array(round_diameter_similarity, dtype=np.float32)
        folder_to_save_min = os.path.join(self.experiment_path, "embeddings_graphs",
                                          "rank_diameter_and_average_last_round-min")
        file_to_save_min = os.path.join(folder_to_save_min,
                                        f"{NUM_TO_REPRESENTATION[representation_num]}-{os.path.basename(self.experiment_path)}.npy")
        self.__save_numpy(numpy_min_to_save, folder_to_save_min, file_to_save_min)

        df = pd.DataFrame({"rank_diameter_and_average_last_round_mean": numpy_mean_to_save.mean(),
                           "rank_diameter_and_average_last_round_min": numpy_min_to_save.mean()},
                          index=[0])
        df.to_csv(os.path.join(self.experiment_path, "embeddings_graphs", "rank_diameter_and_average_last_round.csv"))

    def __plot_average_similarity_of_player_documents_consecutive_rounds(self, df: pd.DataFrame,
                                                                         representation_num: int) -> None:
        """
        Plot the average similarity of player documents between consecutive rounds.

        Parameters:
            df (pd.DataFrame): DataFrame containing the competition history.
            representation_num (int): Index of the representation to use.
        """
        matrix_metrics = []

        for query in df["query_id"].unique():
            game_df = df[df["query_id"] == query]
            all_players_game_metrics = [
                [
                    self.__calculate_pairwise_similarity(
                        game_df[(game_df["round"] == round_num) | (game_df["round"] == round_num - 1)]
                        [game_df["player"] == player],
                        self.representations, representation_num
                    )
                    for round_num in range(2, game_df["round"].max() + 1)
                ]
                for player in game_df["player"].unique()
            ]
            matrix_metrics.append(all_players_game_metrics)

        # Save the results
        numpy_matrix = np.array(matrix_metrics, dtype=np.float32)
        folder_to_save = os.path.join(self.experiment_path, "embeddings_graphs",
                                      "average_of_player_documents_consecutive_rounds")
        file_to_save = os.path.join(folder_to_save,
                                    f"{NUM_TO_REPRESENTATION[representation_num]}-{os.path.basename(self.experiment_path)}.npy")
        self.__save_numpy(numpy_matrix, folder_to_save, file_to_save)

        mean_matrix_for_graph = numpy_matrix.swapaxes(1, 2)
        mean_matrix_for_graph = mean_matrix_for_graph.mean(axis=2).mean(axis=0)

        self.__plot_graph(range(2, df["round"].max() + 1), mean_matrix_for_graph,
                          "Average similarity of \nplayer documents vs consecutive rounds",
                          'Round', 'Average similarity of \nplayer documents between consecutive rounds',
                          file_to_save.replace(".npy", ".png"))

        df = pd.DataFrame({"average_of_player_documents_consecutive_rounds_mean": numpy_matrix.mean()},
                          index=[0])
        df.to_csv(os.path.join(self.experiment_path, "embeddings_graphs",
                               "average_of_player_documents_consecutive_rounds.csv"))

    def __plot_average_similarity_of_player_documents_consecutive_rounds_excluding_players(self, df: pd.DataFrame,
                                                                                           representation_num: int,
                                                                                           excluded_players: list) -> None:
        """
        Plot the average similarity of player documents between consecutive rounds, excluding specified players.

        Parameters:
            df (pd.DataFrame): DataFrame containing the competition history.
            representation_num (int): Index of the representation to use.
            excluded_players (list): List of player names to exclude from the analysis.
        """
        matrix_metrics = []

        df = df[~df["player"].isin(excluded_players)]

        for query in df["query_id"].unique():
            game_df = df[df["query_id"] == query]
            all_players_game_metrics = [
                [
                    self.__calculate_pairwise_similarity(
                        game_df[(game_df["round"] == round_num) | (game_df["round"] == round_num - 1)]
                        [game_df["player"] == player],
                        self.representations, representation_num
                    )
                    for round_num in range(2, game_df["round"].max() + 1)
                ]
                for player in game_df["player"].unique()
            ]
            matrix_metrics.append(all_players_game_metrics)

        # Save the results
        numpy_matrix = np.array(matrix_metrics, dtype=np.float32)
        folder_to_save = os.path.join(self.experiment_path, "embeddings_graphs",
                                      "average_of_player_documents_consecutive_rounds_excluding_players")
        file_to_save = os.path.join(folder_to_save,
                                    f"{NUM_TO_REPRESENTATION[representation_num]}-excluded({excluded_players})-{os.path.basename(self.experiment_path)}.npy")

        self.__save_numpy(numpy_matrix, folder_to_save, file_to_save)

        mean_matrix_for_graph = numpy_matrix.swapaxes(1, 2)
        mean_matrix_for_graph = mean_matrix_for_graph.mean(axis=2).mean(axis=0)

        self.__plot_graph(range(2, df["round"].max() + 1), mean_matrix_for_graph,
                          "Average similarity of \nplayer documents vs consecutive rounds (excluding players)",
                          'Round',
                          'Average similarity of \nplayer documents between consecutive rounds (excluding players)',
                          file_to_save.replace(".npy", ".png"))

        df = pd.DataFrame(
            {"average_of_player_documents_consecutive_rounds_mean (excluding players)": numpy_matrix.mean()},
            index=[0])
        df.to_csv(os.path.join(self.experiment_path, "embeddings_graphs",
                               "average_of_player_documents_consecutive_rounds (excluding players).csv"))

    def __plot_average_similarity_of_player_documents_consecutive_rounds_only_players(self, df: pd.DataFrame,
                                                                                      representation_num: int,
                                                                                      included_players: list) -> None:
        """
        Plot the average similarity of player documents between consecutive rounds, including only specified players.

        Parameters:
            df (pd.DataFrame): DataFrame containing the competition history.
            representation_num (int): Index of the representation to use.
            included_players (list): List of player names to include in the analysis.
        """
        matrix_metrics = []

        df = df[df["player"].isin(included_players)]

        for query in df["query_id"].unique():
            game_df = df[df["query_id"] == query]
            all_players_game_metrics = [
                [
                    self.__calculate_pairwise_similarity(
                        game_df[(game_df["round"] == round_num) | (game_df["round"] == round_num - 1)]
                        [game_df["player"] == player],
                        self.representations, representation_num
                    )
                    for round_num in range(2, game_df["round"].max() + 1)
                ]
                for player in game_df["player"].unique()
            ]
            matrix_metrics.append(all_players_game_metrics)

        # Save the results
        numpy_matrix = np.array(matrix_metrics, dtype=np.float32)
        folder_to_save = os.path.join(self.experiment_path, "embeddings_graphs",
                                      "average_of_player_documents_consecutive_rounds_including_players")
        file_to_save = os.path.join(folder_to_save,
                                    f"{NUM_TO_REPRESENTATION[representation_num]}-included({included_players})-{os.path.basename(self.experiment_path)}.npy")

        self.__save_numpy(numpy_matrix, folder_to_save, file_to_save)

        mean_matrix_for_graph = numpy_matrix.swapaxes(1, 2)
        mean_matrix_for_graph = mean_matrix_for_graph.mean(axis=2).mean(axis=0)

        self.__plot_graph(range(2, df["round"].max() + 1), mean_matrix_for_graph,
                          "Average similarity of \nplayer documents vs consecutive rounds (including players)",
                          'Round',
                          'Average similarity of \nplayer documents between consecutive rounds (including players)',
                          file_to_save.replace(".npy", ".png"))

        df = pd.DataFrame(
            {"average_of_player_documents_consecutive_rounds_mean (including players)": numpy_matrix.mean()},
            index=[0])
        df.to_csv(os.path.join(self.experiment_path, "embeddings_graphs",
                               "average_of_player_documents_consecutive_rounds (including players).csv"))

    def __plot_diameter_and_average_over_time(self, df: pd.DataFrame, representation_num: int) -> None:
        """
        Plot the diameter and average similarity over time.

        Parameters:
            df (pd.DataFrame): DataFrame containing the competition history.
            representation_num (int): Index of the representation to use.
        """
        avg_similarity, diameter_similarity = [], []
        round_rank_measurements_mean_to_save, round_rank_measurements_min_to_save = [], []

        for round_num in range(1, df["round"].max() + 1):
            round_df = df[df["round"] == round_num]
            groups = round_df.groupby('query_id').filter(lambda x: len(x) > 1).groupby('query_id')
            round_rank_measurements = groups.apply(self.__calculate_pairwise_similarity, self.representations,
                                                   representation_num, return_min=True)

            round_rank_measurements_mean_to_save.append(round_rank_measurements.apply(lambda x: x[0]))
            round_rank_measurements_min_to_save.append(round_rank_measurements.apply(lambda x: x[1]))

            avg_similarity.append(round_rank_measurements.apply(lambda x: x[0]).mean())
            diameter_similarity.append(round_rank_measurements.apply(lambda x: x[1]).mean())

        # Save the results
        folder_to_save_mean = os.path.join(self.experiment_path, "embeddings_graphs",
                                           "plot_diameter_and_average_over_time-mean")
        file_to_save_mean = os.path.join(folder_to_save_mean,
                                         f"{NUM_TO_REPRESENTATION[representation_num]}-{os.path.basename(self.experiment_path)}.npy")
        self.__save_numpy(np.array(round_rank_measurements_mean_to_save, dtype=np.float32), folder_to_save_mean,
                          file_to_save_mean)

        folder_to_save_min = os.path.join(self.experiment_path, "embeddings_graphs",
                                          "plot_diameter_and_average_over_time-min")
        file_to_save_min = os.path.join(folder_to_save_min,
                                        f"{NUM_TO_REPRESENTATION[representation_num]}-{os.path.basename(self.experiment_path)}.npy")
        self.__save_numpy(np.array(round_rank_measurements_min_to_save, dtype=np.float32), folder_to_save_min,
                          file_to_save_min)

        self.__plot_graph(range(1, df["round"].max() + 1), np.array(avg_similarity, dtype=np.float32),
                          "Average group similarity vs round",
                          'Round', 'Average group similarity',
                          file_to_save_mean.replace(".npy", ".png"))
        self.__plot_graph(range(1, df["round"].max() + 1), np.array(diameter_similarity, dtype=np.float32),
                          "Diameter group similarity vs round",
                          'Round', 'Diameter group similarity',
                          file_to_save_min.replace(".npy", ".png"))

    def __plot_diameter_and_average_over_time_only_players(self, df: pd.DataFrame, representation_num: int,
                                                           included_players: list) -> None:
        """
        Plot the diameter and average similarity over time for a specific set of players.

        Parameters:
            df (pd.DataFrame): DataFrame containing the competition history.
            representation_num (int): Index of the representation to use.
            included_players (list): List of player names to include in the analysis.
        """

        # Filter the DataFrame to include only the specified players
        df = df[df["player"].isin(included_players)]

        avg_similarity, diameter_similarity = [], []
        round_rank_measurements_mean_to_save, round_rank_measurements_min_to_save = [], []

        for round_num in range(1, df["round"].max() + 1):
            round_df = df[df["round"] == round_num]
            groups = round_df.groupby('query_id').filter(lambda x: len(x) > 1).groupby('query_id')
            round_rank_measurements = groups.apply(self.__calculate_pairwise_similarity, self.representations,
                                                   representation_num, return_min=True)

            round_rank_measurements_mean_to_save.append(round_rank_measurements.apply(lambda x: x[0]))
            round_rank_measurements_min_to_save.append(round_rank_measurements.apply(lambda x: x[1]))

            avg_similarity.append(round_rank_measurements.apply(lambda x: x[0]).mean())
            diameter_similarity.append(round_rank_measurements.apply(lambda x: x[1]).mean())

        # Save the results
        folder_to_save_mean = os.path.join(self.experiment_path, "embeddings_graphs",
                                           "plot_diameter_and_average_over_time-mean")
        file_to_save_mean = os.path.join(folder_to_save_mean,
                                         f"{NUM_TO_REPRESENTATION[representation_num]}-included({included_players})-{os.path.basename(self.experiment_path)}.npy")
        self.__save_numpy(np.array(round_rank_measurements_mean_to_save, dtype=np.float32), folder_to_save_mean,
                          file_to_save_mean)

        folder_to_save_min = os.path.join(self.experiment_path, "embeddings_graphs",
                                          "plot_diameter_and_average_over_time-min")
        file_to_save_min = os.path.join(folder_to_save_min,
                                        f"{NUM_TO_REPRESENTATION[representation_num]}-included({included_players})-{os.path.basename(self.experiment_path)}.npy")
        self.__save_numpy(np.array(round_rank_measurements_min_to_save, dtype=np.float32), folder_to_save_min,
                          file_to_save_min)

        self.__plot_graph(range(1, df["round"].max() + 1), np.array(avg_similarity, dtype=np.float32),
                          "Average group similarity vs round",
                          'Round', 'Average group similarity',
                          file_to_save_mean.replace(".npy", ".png"))
        self.__plot_graph(range(1, df["round"].max() + 1), np.array(diameter_similarity, dtype=np.float32),
                          "Diameter group similarity vs round",
                          'Round', 'Diameter group similarity',
                          file_to_save_min.replace(".npy", ".png"))

    def __plot_diameter_and_average_over_time_excluding_players(self, df: pd.DataFrame, representation_num: int,
                                                                excluded_players: list) -> None:
        """
        Plot the diameter and average similarity over time, excluding specified players.

        Parameters:
            df (pd.DataFrame): DataFrame containing the competition history.
            representation_num (int): Index of the representation to use.
            excluded_players (list): List of player names to exclude from the analysis.
        """
        # Filter the DataFrame to exclude the specified players
        df = df[~df["player"].isin(excluded_players)]

        avg_similarity, diameter_similarity = [], []
        round_rank_measurements_mean_to_save, round_rank_measurements_min_to_save = [], []

        for round_num in range(1, df["round"].max() + 1):
            round_df = df[df["round"] == round_num]
            groups = round_df.groupby('query_id').filter(lambda x: len(x) > 1).groupby('query_id')
            round_rank_measurements = groups.apply(self.__calculate_pairwise_similarity, self.representations,
                                                   representation_num, return_min=True)

            round_rank_measurements_mean_to_save.append(round_rank_measurements.apply(lambda x: x[0]))
            round_rank_measurements_min_to_save.append(round_rank_measurements.apply(lambda x: x[1]))

            avg_similarity.append(round_rank_measurements.apply(lambda x: x[0]).mean())
            diameter_similarity.append(round_rank_measurements.apply(lambda x: x[1]).mean())

        # Save the results
        folder_to_save_mean = os.path.join(self.experiment_path, "embeddings_graphs",
                                           "plot_diameter_and_average_over_time-mean")
        file_to_save_mean = os.path.join(folder_to_save_mean,
                                         f"{NUM_TO_REPRESENTATION[representation_num]}-excluded({excluded_players})-{os.path.basename(self.experiment_path)}.npy")
        self.__save_numpy(np.array(round_rank_measurements_mean_to_save, dtype=np.float32), folder_to_save_mean,
                          file_to_save_mean)

        folder_to_save_min = os.path.join(self.experiment_path, "embeddings_graphs",
                                          "plot_diameter_and_average_over_time-min")
        file_to_save_min = os.path.join(folder_to_save_min,
                                        f"{NUM_TO_REPRESENTATION[representation_num]}-excluded({excluded_players})-{os.path.basename(self.experiment_path)}.npy")
        self.__save_numpy(np.array(round_rank_measurements_min_to_save, dtype=np.float32), folder_to_save_min,
                          file_to_save_min)

        self.__plot_graph(range(1, df["round"].max() + 1), np.array(avg_similarity, dtype=np.float32),
                          "Average group similarity vs round",
                          'Round', 'Average group similarity',
                          file_to_save_mean.replace(".npy", ".png"))
        self.__plot_graph(range(1, df["round"].max() + 1), np.array(diameter_similarity, dtype=np.float32),
                          "Diameter group similarity vs round",
                          'Round', 'Diameter group similarity',
                          file_to_save_min.replace(".npy", ".png"))