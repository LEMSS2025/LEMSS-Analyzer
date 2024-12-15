import logging

import pandas as pd

from constants.constants import (FEATURES_ORDER, COLUMNS_TO_FEATURE_NAMES,
                                 COLUMN_DOCUMENT, COLUMN_RENAMED_DOCNO, COLUMN_ROUND)


class DataProcessor:
    """
    Class responsible for processing the final DataFrame by combining extracted and additional features.
    """

    def process_final_data(self, final_data: pd.DataFrame) -> pd.DataFrame:
        """
        Combine extracted features with additional features, process and normalize the final DataFrame.

        :param final_data: DataFrame containing extracted features.
        :return: Processed and normalized final DataFrame.
        """
        logging.info("Combining all features into the final DataFrame...")

        # Remove unnecessary columns
        final_data.drop(columns=[COLUMN_DOCUMENT], inplace=True)

        # Extract the round number from 'docno' by splitting the string and converting to an integer
        final_data[COLUMN_ROUND] = final_data[COLUMN_RENAMED_DOCNO].str.split('-').str[1].astype(int)

        # Rearrange columns for easier readability and rename columns according to the specified mapping
        final_data = final_data[FEATURES_ORDER].rename(columns=COLUMNS_TO_FEATURE_NAMES).reset_index(drop=True)

        logging.info("Final DataFrame processed successfully.")

        return final_data
