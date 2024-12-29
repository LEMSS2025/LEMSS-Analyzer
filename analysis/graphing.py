import os

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

from analysis.feature_history import FeatureHistoryRetriever


class Graphing:
    """
    Class responsible for generating various graphs and saving them.
    """

    def __init__(self, experiment_folder: str):
        """
        Initialize the Graphing class and configure the graph settings.

        :param experiment_folder: Path to the folder where graphs will be saved.
        """
        self.experiment_folder = experiment_folder
        self.__configure_graphs()

    def __configure_graphs(self) -> None:
        """
        Configure the aesthetic settings for the graphs using seaborn and matplotlib.
        """
        sns.set_style("white")
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['figure.facecolor'] = 'white'
        mpl.rcParams['font.size'] = 12
        mpl.rcParams['font.weight'] = 'bold'

    def plot_number_matches_won(self, data: pd.DataFrame, ax=None) -> None:
        """
        Plot a histogram showing the number of matches won by each player.

        :param data: DataFrame containing merged data of features and game history.
        :param ax: Matplotlib axis to plot on (optional).
        """
        if ax is None:
            _, ax = plt.subplots()

        df = data.groupby(['qid', 'player'])['is_winner'].sum().reset_index()
        sns.histplot(data=df, x='is_winner', multiple='dodge', ax=ax, shrink=0.75, discrete=True, kde=True)
        ax.set_xlabel('Number of Matches Won')
        ax.set_ylabel('Number of LLMs')
        ax.set_title('Number of Matches Won')
        ax.set_xticks(range(0, 11))

    def plot_consecutive_matches_won(self, data: pd.DataFrame, ax=None) -> None:
        """
        Plot a histogram showing the number of consecutive matches won by each player.

        :param data: DataFrame containing merged data of features and game history.
        :param ax: Matplotlib axis to plot on (optional).
        """
        rounds_won = data[data['is_winner']].groupby(['qid', 'player'])['round'].apply(list).reset_index()
        rounds_won['consecutive'] = rounds_won['round'].apply(self.num_consecutive)
        rounds_won = rounds_won.explode('consecutive')

        if ax is None:
            _, ax = plt.subplots()

        sns.histplot(data=rounds_won, x='consecutive', multiple='dodge', ax=ax, shrink=0.75, discrete=True, kde=True)
        ax.set_xlabel('Number of Consecutive Matches Won')
        ax.set_ylabel('Number of LLMs')
        ax.set_title('Number of Consecutive Matches Won')
        ax.set_xticks(range(1, 11))

    @staticmethod
    def num_consecutive(rounds_won: list) -> list:
        """
        Calculate the number of consecutive rounds won.

        :param rounds_won: List of rounds won by a player.
        :return: List of consecutive win counts.
        """
        consecutive = []
        i = 0
        while i < len(rounds_won):
            if i == len(rounds_won) - 1:
                consecutive.append(i + 1)
                break
            elif rounds_won[i] == rounds_won[i + 1] - 1:
                i += 1
            else:
                consecutive.append(i + 1)
                rounds_won = rounds_won[i + 1:]
                i = 0

        return consecutive

    def __save_legend_to_file(self, handles, labels, filename="legend.png", ncol=3, bbox_to_anchor=(0.5, 0)) -> None:
        """
        Save the legend of a plot to a separate file.

        :param handles: Handles from the plot's legend.
        :param labels: Labels from the plot's legend.
        :param filename: Filename to save the legend to.
        :param ncol: Number of columns in the legend.
        :param bbox_to_anchor: Positioning of the legend relative to the plot.
        """
        fig_legend = plt.figure(figsize=(3, 2))
        ax_legend = fig_legend.add_subplot(111)

        legend = ax_legend.legend(handles, labels, loc='upper center', ncol=ncol, frameon=False, bbox_to_anchor=bbox_to_anchor)
        ax_legend.axis('off')

        fig_legend.canvas.draw()
        bbox = legend.get_window_extent().transformed(fig_legend.dpi_scale_trans.inverted())

        fig_legend.savefig(filename, bbox_inches=bbox, transparent=True)
        plt.close(fig_legend)

    def plot_feature_improvers(
        self, df: pd.DataFrame, feature: str, num_rounds_back: int = 4, feature_title: str = None,
        save_path: str = None, with_legend: bool = False, user_other_queries: bool = False
    ) -> None:
        """
        Plot the feature improvers for a given feature over several rounds.

        :param df: DataFrame containing the data.
        :param feature: The feature to be plotted.
        :param num_rounds_back: Number of rounds back to consider.
        :param feature_title: Title of the feature being plotted.
        :param save_path: Path to save the plot.
        :param with_legend: Whether to include the legend in the plot.
        :param user_other_queries: Whether to consider other queries for the user.
        """
        first_win_by_query = df[df['round'] > num_rounds_back]
        first_win_by_query = first_win_by_query[(first_win_by_query['is_winner'])].groupby(
            ['player', 'qid', 'round']).apply(
            lambda x: True).reset_index()
        first_win_by_query['hist_round'] = first_win_by_query.apply(
            lambda x: FeatureHistoryRetriever.get_rounds_history(df.copy(), x['qid'], x['player'], x['round'],
                                                                 num_rounds_back), axis=1)
        first_win_by_query['len_hist_round'] = first_win_by_query['hist_round'].apply(lambda x: len(x))
        first_win_by_query = first_win_by_query[first_win_by_query['len_hist_round'] == 1]

        columns_map = {}

        # Retrieve user feature history
        first_win_by_query['user_history'] = first_win_by_query.apply(
            lambda x: FeatureHistoryRetriever.get_user_feature_history(df.copy(), feature, player=x['player'], qid=x['qid'],
                                               win_round=x['round'], num_rounds_back=num_rounds_back), axis=1)
        columns_map['user_history'] = {'user': 'loser', 'feature': 'query'}

        if user_other_queries:
            first_win_by_query['user_other_queries_history'] = first_win_by_query.apply(
                lambda x: FeatureHistoryRetriever.get_user_feature_history_other_queries(df.copy(), feature, player=x['player'], qid=x['qid'],
                                                                 win_round=x['round'], num_rounds_back=num_rounds_back),
                axis=1)
            columns_map['user_other_queries_history'] = {'user': 'loser', 'feature': 'other queries'}

        # Retrieve winner feature history
        first_win_by_query['winner_history'] = first_win_by_query.apply(
            lambda x: FeatureHistoryRetriever.get_winner_feature_history(df.copy(), feature, qid=x['qid'],
                                                 win_round=x['round'], num_rounds_back=num_rounds_back), axis=1)
        columns_map['winner_history'] = {'user': 'winner in query', 'feature': 'query'}

        # Label based on user and winner feature comparison
        first_win_by_query['label'] = first_win_by_query.apply(
            lambda x: FeatureHistoryRetriever.get_label(winner=x['winner_history'][0], user=x['user_history'][0]), axis=1)

        # Plotting setup
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        groups = ['all']

        label_to_color = {'L<=W': 'orange', 'L>W': 'blue', 'W': 'black'}
        losers_labels = {'L<=W': r'$L \leq W$', 'L>W': r'L > W'}

        for group, ax in zip(groups, [ax]):
            df_g = first_win_by_query

            # Plot for first time winners
            for label in sorted(df_g['label'].unique()):
                user_history = df_g[(df_g['label'] == label)]['user_history'].values
                user_history_mean = np.stack(user_history).mean(axis=0)
                ax.plot(np.arange(-num_rounds_back, 1), user_history_mean, label=losers_labels[label],
                        color=label_to_color[label], marker='s')

                if user_other_queries:
                    user_other_queries_history = df_g[(df_g['label'] == label)]['user_other_queries_history'].values
                    user_other_queries_history_mean = np.stack(user_other_queries_history).mean(axis=0)
                    ax.plot(np.arange(-num_rounds_back, 1), user_other_queries_history_mean, label=losers_labels[label],
                            color=label_to_color[label], marker='s', linestyle='--')

            # Plot for winners
            winner_history = df_g['winner_history'].values
            winner_history_mean = np.stack(winner_history).mean(axis=0)
            ax.plot(np.arange(-num_rounds_back, 1), winner_history_mean, label="W",
                    color='black', marker='|')

            ax.set_xticks(np.arange(-num_rounds_back, 1))

        # Handle legend
        handles, labels = ax.get_legend_handles_labels()
        if with_legend:
            fig.legend(handles, labels, loc='upper center', ncol=len(label_to_color) + 1, bbox_to_anchor=(0.5, 1.1))
        else:
            self.__save_legend_to_file(handles, labels, filename=os.path.join(self.experiment_folder, "features-legend.jpg"),
                                       ncol=len(label_to_color) + 1, bbox_to_anchor=(0.5, 1.1))

        if feature_title is not None:
            fig.suptitle(feature_title, fontsize=20)
        fig.tight_layout()

        # Save the plot if a save path is provided
        if save_path is not None:
            fig.savefig(save_path, bbox_inches='tight')
