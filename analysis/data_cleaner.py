import pandas as pd

from constants.constants import INDEX_ORDER


class DataCleaner:
    """
    Class responsible for cleaning feature matrix and game history data.
    """

    @staticmethod
    def clean_features_df(features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the feature matrix DataFrame by renaming columns, splitting identifiers,
        and converting data types.

        :param features_df: DataFrame containing the feature matrix.
        :return: Cleaned DataFrame with adjusted columns.
        """
        # Rename 'query_id' to 'qid' for consistency
        features_df = features_df.rename(columns={'query_id': 'qid'})

        # Extract 'user', 'round', and 'query_id' from the 'document_id' column
        features_df['user'] = features_df['document_id'].apply(lambda x: x.split('-')[3])
        features_df['round'] = features_df['document_id'].apply(lambda x: x.split('-')[1])
        features_df['query_id'] = features_df['document_id'].apply(lambda x: x.split('-')[2])

        # Convert 'query_id', 'round', and 'user' to integers for consistency
        features_df['query_id'] = features_df['query_id'].astype(int)
        features_df['round'] = features_df['round'].astype(int)
        features_df['user'] = features_df['user'].astype(int)

        return features_df

    @staticmethod
    def clean_game_history_df(game_history_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the game history DataFrame by mapping and converting data types.

        :param game_history_df: DataFrame containing game history.
        :return: Cleaned DataFrame with adjusted columns.
        """
        # Map the 'player' column using INDEX_ORDER and convert to integer
        game_history_df['player'] = game_history_df['player'].map(INDEX_ORDER).astype(int)

        # Convert 'round' column to integer
        game_history_df['round'] = game_history_df['round'].astype(int)

        return game_history_df
