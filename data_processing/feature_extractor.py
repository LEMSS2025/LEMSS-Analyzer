import logging
import os

import pandas as pd

from parsers.query_parser import QueryParser
from parsers.trec_parser import TrecParser
from feature_engineering.feature_extractor_wrapper import FeatureExtractorWrapper
from feature_engineering.custom_features import CustomFeatures
from feature_engineering.bert_scorer import BertScorer
from feature_engineering.e5 import E5
from feature_engineering.sbert import SBERT
from feature_engineering.tf_idf import TFIDF
from data_processing.data_processor import DataProcessor
from faiss_index.faiss_index import FaissIndex
from constants.constants import (SBERT_DIMENSION, E5_DIMENSION, SBERT_INDEX_FILENAME, E5_INDEX_FILENAME, PROJECT_DIR,
                                 FAISS_FOLDER, COLUMN_DOCUMENT, COLUMN_RENAMED_DOCNO, COLUMN_QID)


class FeatureExtractor:
    """
    Class responsible for extracting features from documents and queries.
    """

    def __init__(self, trectext_path: str, queries_folder: str, output_path: str, index_path: str, experiment_folder: str,
                 embedding_mode=True):
        """
        Initialize the FeatureExtractor.

        :param trectext_path: Path to the TREC text data.
        :param queries_folder: Path to the folder containing query XML files.
        :param output_path: Path to save the extracted features.
        :param index_path: Path to the document index.
        :param experiment_folder: Path to the experiment folder.
        :param embedding: Whether to use embeddings for feature extraction.
        """
        self.__trectext_path = trectext_path
        self.__queries_folder = queries_folder
        self.__output_path = output_path
        self.__index_path = index_path
        self.__experiment_folder = experiment_folder
        self.__embedding_mode = embedding_mode

        if embedding_mode:
            self.__faiss_index_bert, self.__bert_index_path = self.__init_faiss_indices_embeddings(SBERT_INDEX_FILENAME, SBERT_DIMENSION)
            self.__faiss_index_e5, self.__e5_index_path = self.__init_faiss_indices_embeddings(E5_INDEX_FILENAME, E5_DIMENSION)

    def extract_features(self) -> None:
        """
        Extract features from the documents and queries, including custom and BERT-based features.
        """
        logging.info("Starting feature extraction process...")

        # Parse documents
        docs_df = self.__parse_documents()

        # Load and preprocess queries
        queries = self.__load_queries()

        # Calculate custom features
        docs_df = self.__calculate_custom_features(docs_df)

        # Extract features using FeatureExtractorWrapper
        docs_df = self.__extract_features_with_wrapper(docs_df, queries)

        # Calculate BERT scores and embeddings
        feature_matrix = self.__calculate_bert_scores_and_embeddings(docs_df, queries)

        # Save the updated index
        if self.__embedding_mode:
            self.__faiss_index_bert.save_index(self.__bert_index_path)
            self.__faiss_index_e5.save_index(self.__e5_index_path)

        # Process and save the final data
        self.__process_and_save_final_data(feature_matrix)

        logging.info(f"Feature extraction completed and feature matrix saved to {self.__output_path}")

    def __init_faiss_indices_embeddings(self, index_name, dimension) -> None:
        # Check if folder exists
        if not os.path.exists(os.path.join(PROJECT_DIR, self.__experiment_folder, FAISS_FOLDER)):
            os.makedirs(os.path.join(PROJECT_DIR, self.__experiment_folder, FAISS_FOLDER))

        # Initialize a FAISS index
        index_path = os.path.join(PROJECT_DIR, self.__experiment_folder, FAISS_FOLDER, index_name)
        if os.path.exists(index_path):
            faiss_index = FaissIndex(dimension=dimension, index_file=index_path)
        else:
            faiss_index = FaissIndex(dimension=dimension)

        return faiss_index, index_path

    def __parse_documents(self) -> pd.DataFrame:
        """
        Parse the TREC text data into a DataFrame.

        :return: DataFrame containing the parsed documents.
        """
        logging.info("Parsing TREC data...")
        trec_parser = TrecParser(self.__trectext_path)
        docs_df = trec_parser.parse_trec()
        logging.info("TREC data parsed successfully.")

        return docs_df

    def __load_queries(self) -> dict:
        """
        Load and preprocess queries from the XML files.

        :return: Dictionary mapping query IDs to query text.
        """
        query_files = [os.path.join(self.__queries_folder, file) for file in os.listdir(self.__queries_folder) if file.endswith('.xml')]
        query_parser = QueryParser(query_files)
        queries = query_parser.query_loader()
        logging.info("Queries loaded and preprocessed successfully.")

        return queries

    def __calculate_custom_features(self, docs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate custom features (stopword fraction, stopword coverage, entropy) for the documents.

        :param docs_df: DataFrame containing the documents.
        :return: DataFrame with custom features added.
        """
        custom_features = CustomFeatures()
        docs_df['frac_stop'] = custom_features.frac_stop(docs_df[COLUMN_DOCUMENT].str.lower())
        docs_df['stop_cover'] = custom_features.stop_cover(docs_df[COLUMN_DOCUMENT].str.lower())
        docs_df['entropy'] = custom_features.entropy(docs_df[COLUMN_DOCUMENT].str.lower())

        if self.__embedding_mode:
            tf_idf = TFIDF(self.__experiment_folder, self.__index_path, docs_df[COLUMN_RENAMED_DOCNO].tolist())
            tf_idf.extract_tf_idf_similarities()
        logging.info("Custom features calculated successfully.")

        return docs_df

    def __extract_features_with_wrapper(self, docs_df: pd.DataFrame, queries: dict) -> pd.DataFrame:
        """
        Extract additional features using the FeatureExtractorWrapper.

        :param docs_df: DataFrame containing the documents.
        :param queries: Dictionary mapping query IDs to query text.
        :return: DataFrame with extracted features.
        """
        fe_wrapper = FeatureExtractorWrapper(self.__index_path)
        info, data, _ = fe_wrapper.batch_extract(docs_df, queries)
        data[COLUMN_RENAMED_DOCNO] = info[COLUMN_RENAMED_DOCNO]
        logging.info("Feature extraction completed successfully.")

        # Align docs_df to the order of 'docno' in info and merge with extracted features
        docs_df = docs_df.set_index(COLUMN_RENAMED_DOCNO).loc[info[COLUMN_RENAMED_DOCNO]].reset_index()
        final_data = pd.merge(data, docs_df, on=COLUMN_RENAMED_DOCNO)

        return final_data

    def __calculate_bert_scores_and_embeddings(self, docs_df: pd.DataFrame, queries: dict) -> pd.DataFrame:
        """
        Calculate BERT similarity scores for the document-query pairs.

        :param docs_df: DataFrame containing the documents.
        :param queries: Dictionary mapping query IDs to query text.
        :return: DataFrame with BERT scores added.
        """
        queries_to_docnos = docs_df.groupby(COLUMN_QID)[COLUMN_RENAMED_DOCNO].apply(list).to_dict()
        docs_dict = docs_df.set_index(COLUMN_RENAMED_DOCNO)[COLUMN_DOCUMENT].to_dict()
        queries_dict = {qid: queries[str(qid)] for qid in docs_df[COLUMN_QID].unique()}

        bert_scorer = BertScorer()
        e5 = E5()
        sbert = SBERT()

        if self.__embedding_mode:
            e5.create_e5_embeddings(queries_to_docnos, docs_dict, self.__faiss_index_e5, self.__experiment_folder)
            sbert.create_sbert_embeddings(queries_to_docnos, docs_dict, self.__faiss_index_bert, self.__experiment_folder)
        bert_scores = bert_scorer.get_bert_scores(queries_to_docnos, docs_dict, queries_dict)
        docs_df['bert_score'] = docs_df[COLUMN_RENAMED_DOCNO].map(bert_scores)
        logging.info("BERT scores calculated successfully.")

        return docs_df

    def __process_and_save_final_data(self, feature_matrix: pd.DataFrame) -> None:
        """
        Process the extracted features and save the final DataFrame to a CSV file.

        :param feature_matrix: DataFrame containing the documents with features.
        """
        data_processor = DataProcessor()
        final_data = data_processor.process_final_data(feature_matrix)
        final_data.to_csv(self.__output_path, index=False)
        logging.info("Final data processed and saved successfully.")