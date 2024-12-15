import pandas as pd


class FeatureHistoryRetriever:
    """
    Class responsible for retrieving feature history for players and winners.
    """

    @staticmethod
    def get_rounds_history(df, qid, player, win_round, num_rounds_back=3):
        rounds_to_consider = [win_round - i for i in range(num_rounds_back + 1)]
        winner_df = df[(df['is_winner']) & (df['qid'] == qid) & (df['round'].isin(rounds_to_consider)) & (
                    df['player'] == player)].sort_values('round')
        return winner_df['round'].values

    @staticmethod
    def get_user_feature_history(
        df: pd.DataFrame, feature: str, player: int, qid: int, win_round: int, num_rounds_back: int = 4
    ) -> list:
        """
        Retrieve the history of a feature for a specific user in a specific query.

        :param df: DataFrame containing all data.
        :param feature: The feature to be analyzed (column name).
        :param player: The player (user) for whom the history is being retrieved.
        :param qid: The query ID associated with the feature.
        :param win_round: The round in which the user won.
        :param num_rounds_back: The number of rounds back to consider.
        :return: List of feature values for the specified rounds.
        """
        rounds_to_consider = [win_round - i for i in range(num_rounds_back + 1)]
        user_df = df[(df['player'] == player) & (df['qid'] == qid) & (df['round'].isin(rounds_to_consider))].sort_values('round')

        return user_df[feature].values

    @staticmethod
    def get_user_feature_history_other_queries(
        df: pd.DataFrame, feature: str, player: int, qid: int, win_round: int, num_rounds_back: int = 4
    ) -> list:
        """
        Retrieve the history of a feature for a user in other queries within the same topic.

        :param df: DataFrame containing all data.
        :param feature: The feature to be analyzed (column name).
        :param player: The player (user) for whom the history is being retrieved.
        :param qid: The query ID associated with the feature.
        :param win_round: The round in which the user won.
        :param num_rounds_back: The number of rounds back to consider.
        :return: List of average feature values across other queries.
        """
        rounds_to_consider = [win_round - i for i in range(num_rounds_back + 1)]
        user_df = df[(df['player'] == player) & (df['qid'] == qid) & (df['round'].isin(rounds_to_consider))].sort_values('round')

        return user_df.groupby('round')[feature].mean().reset_index().sort_values('round')[feature].values

    @staticmethod
    def get_winner_feature_history(
        df: pd.DataFrame, feature: str, qid: int, win_round: int, num_rounds_back: int = 4
    ) -> list:
        """
        Retrieve the history of a feature for the winner of a specific query.

        :param df: DataFrame containing all data.
        :param feature: The feature to be analyzed (column name).
        :param qid: The query ID associated with the feature.
        :param win_round: The round in which the query was won.
        :param num_rounds_back: The number of rounds back to consider.
        :return: List of feature values for the specified rounds.
        """
        rounds_to_consider = [win_round - i for i in range(num_rounds_back + 1)]
        winner_df = df[(df['is_winner']) & (df['qid'] == qid) & (df['round'].isin(rounds_to_consider))].sort_values('round')

        return winner_df[feature].values

    @staticmethod
    def get_winner_feature_history_other_queries(
        df: pd.DataFrame, feature: str, qid: int, win_round: int, num_rounds_back: int = 4
    ) -> list:
        """
        Retrieve the history of a feature for the winner in other queries in the same topic, with respect to the query in which they won.

        :param df: DataFrame containing all data.
        :param feature: The feature to be analyzed (column name).
        :param qid: The query ID associated with the feature.
        :param win_round: The round in which the query was won.
        :param num_rounds_back: The number of rounds back to consider.
        :return: List of average feature values for the winner across other queries.
        """
        rounds_to_consider = [win_round - i for i in range(num_rounds_back + 1)]
        winners_docnos = df[(df['is_winner']) & (df['qid'] == qid) & (df['round'].isin(rounds_to_consider))]['docno'].values

        return df[(df['docno'].isin(winners_docnos)) & (df['qid'] == qid)].groupby('round')[feature].mean().reset_index().sort_values('round')[feature].values

    @staticmethod
    def get_other_winner_feature_history_other_queries(
        df: pd.DataFrame, feature: str, qid: int, win_round: int, num_rounds_back: int = 4
    ) -> list:
        """
        Retrieve the history of a feature for the winner in other queries in the same topic, with respect to the query in which they won.

        :param df: DataFrame containing all data.
        :param feature: The feature to be analyzed (column name).
        :param qid: The query ID associated with the feature.
        :param win_round: The round in which the query was won.
        :param num_rounds_back: The number of rounds back to consider.
        :return: List of average feature values for the winner across other queries.
        """
        rounds_to_consider = [win_round - i for i in range(num_rounds_back + 1)]
        winner_df = df[(df['is_winner']) & (df['qid'] == qid) & (df['round'].isin(rounds_to_consider))].sort_values('round')

        return winner_df.groupby('round')[feature].mean().reset_index().sort_values('round')[feature].values

    @staticmethod
    def get_other_queries_winner_feature_history(
        df: pd.DataFrame, feature: str, qid: int, win_round: int, num_rounds_back: int = 4
    ) -> list:
        """
        Retrieve the history of a feature for the winner in other queries within the same topic, with respect to the current query.

        :param df: DataFrame containing all data.
        :param feature: The feature to be analyzed (column name).
        :param qid: The query ID associated with the feature.
        :param win_round: The round in which the query was won.
        :param num_rounds_back: The number of rounds back to consider.
        :return: List of average feature values for the winner across other queries.
        """
        rounds_to_consider = [win_round - i for i in range(num_rounds_back + 1)]
        f_history = []

        for r in rounds_to_consider:
            other_winners_docnos = df[(df['is_winner']) & (df['qid'] == qid) & (df['round'] == r)]['docno'].values
            other_winners_values_in_current_query = df[(df['docno'].isin(other_winners_docnos)) & (df['qid'] == qid)][feature].mean()
            f_history.append(other_winners_values_in_current_query)

        return f_history

    @staticmethod
    def get_topic_winners_feature_history(
        df: pd.DataFrame, feature: str, qid: int, win_round: int, num_rounds_back: int = 4, agg: str = 'mean'
    ) -> list:
        """
        Retrieve the history of a feature for the winners in the same topic.

        :param df: DataFrame containing all data.
        :param feature: The feature to be analyzed (column name).
        :param qid: The query ID associated with the feature.
        :param win_round: The round in which the query was won.
        :param num_rounds_back: The number of rounds back to consider.
        :param agg: The aggregation method to apply (default is 'mean').
        :return: List of aggregated feature values for the specified rounds.
        """
        rounds_to_consider = [win_round - i for i in range(num_rounds_back + 1)]
        winners_docnos = df[(df['is_winner']) & (df['qid'] == qid) & (df['round'].isin(rounds_to_consider))]['docno'].values

        return df[(df['docno'].isin(winners_docnos)) & (df['qid'] == qid)].groupby('round')[feature].agg(agg).reset_index().sort_values('round')[feature].values

    @staticmethod
    def get_label(winner: float, user: float) -> str:
        """
        Determine the label based on the comparison of user and winner feature values.

        :param winner: Feature value for the winner.
        :param user: Feature value for the user.
        :return: Label indicating whether the user's value is less than, equal to, or greater than the winner's.
        """
        return 'L<=W' if user <= winner else 'L>W'
