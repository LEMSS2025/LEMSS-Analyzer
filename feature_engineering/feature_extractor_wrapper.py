import os
import logging

import pandas as pd
from pyserini.search.lucene.ltr import *
from tqdm import tqdm

from constants.constants import (NUM_THREADS, PYSERINI_BATCH_SIZE, PYSERINI_CONTENTS, PYSERINI_DOCIDS, ANALYZED,
                                 COLUMN_QID, COLUMN_RENAMED_DOCNO, COLUMN_COUNT)



class FeatureExtractorWrapper:
    """
    Wrapper class for the feature extraction process using Pyserini's LTR features.
    """

    def __init__(self, index_path: str):
        """
        Initialize the FeatureExtractorWrapper with the given index path.
        Set up the feature extractor and add the necessary features.

        :param index_path: Path to the Lucene index.
        """
        logging.info("Initializing Feature Extractor...")
        self.__feature_extractor = FeatureExtractor(index_path, max(os.cpu_count() // NUM_THREADS, 1))
        self.__add_features()
        logging.info("Feature Extractor initialized and features added.")

    def __add_features(self) -> None:
        """
        Add the required features to the feature extractor.
        """
        features_to_add = [
            (BM25Stat, SumPooler(), PYSERINI_CONTENTS),
            (LmDirStat, SumPooler(), PYSERINI_CONTENTS),
            (TfStat, SumPooler(), PYSERINI_CONTENTS),
            (NormalizedTfStat, SumPooler(), PYSERINI_CONTENTS),
            (DocSize, None, PYSERINI_CONTENTS)
        ]

        for feature_class, pooler, field in features_to_add:
            if pooler:
                # Add the feature with pooling (e.g., BM25, LMDir, TF, NormalizedTF)
                self.__feature_extractor.add(feature_class(pooler, field=field, qfield=ANALYZED))
            else:
                # Add features without pooling (e.g., DocSize)
                self.__feature_extractor.add(feature_class(field=field))

            logging.info(f"Added {feature_class.__name__} feature.")

    def batch_extract(self, df: pd.DataFrame, queries: dict) -> tuple:
        """
        Perform batch feature extraction on a DataFrame of documents and their corresponding queries.

        :param df: DataFrame containing documents with 'qid' and 'docno' columns.
        :param queries: Dictionary mapping query IDs to query texts.
        :return: Tuple containing DataFrames for info, features, and groups.
        """
        tasks = []
        task_infos = []
        group_lst = []

        info_dfs = []
        feature_dfs = []
        group_dfs = []

        logging.info("Starting batch feature extraction...")
        for qid, group in tqdm(df.groupby(COLUMN_QID), desc="Extracting Features"):
            task = {
                COLUMN_QID: str(qid),
                PYSERINI_DOCIDS: [str(t.docno) for t in group.itertuples()],
                "query_dict": queries[str(qid)]
            }
            tasks.append(task)
            task_infos.extend([(qid, t.docno) for t in group.itertuples()])
            group_lst.append((qid, len(task[PYSERINI_DOCIDS])))

            # Process tasks in batches
            if len(tasks) >= PYSERINI_BATCH_SIZE:
                self.__process_batch(tasks, task_infos, group_lst, info_dfs, feature_dfs, group_dfs)
                tasks, task_infos, group_lst = [], [], []

        # Process any remaining tasks
        if tasks:
            self.__process_batch(tasks, task_infos, group_lst, info_dfs, feature_dfs, group_dfs)

        info_df = pd.concat(info_dfs, axis=0, ignore_index=True)
        feature_df = pd.concat(feature_dfs, axis=0, ignore_index=True)
        group_df = pd.concat(group_dfs, axis=0, ignore_index=True)

        logging.info("Batch feature extraction completed.")

        return info_df, feature_df, group_df

    def __process_batch(self, tasks: list, task_infos: list, group_lst: list,
                       info_dfs: list, feature_dfs: list, group_dfs: list) -> None:
        """
        Process a batch of tasks and append the results to the respective lists.

        :param tasks: List of tasks to process.
        :param task_infos: List of task info tuples.
        :param group_lst: List of group tuples.
        :param info_dfs: List to store DataFrames of task info.
        :param feature_dfs: List to store DataFrames of extracted features.
        :param group_dfs: List to store DataFrames of groups.
        """
        # Extract features for the current batch of tasks
        features = self.__feature_extractor.batch_extract(tasks)
        info_df = pd.DataFrame(task_infos, columns=[COLUMN_QID, COLUMN_RENAMED_DOCNO])
        group_df = pd.DataFrame(group_lst, columns=[COLUMN_QID, COLUMN_COUNT])

        # Append the results to the respective lists
        info_dfs.append(info_df)
        feature_dfs.append(features)
        group_dfs.append(group_df)

        logging.info(f"Processed batch of size {len(tasks)}.")
