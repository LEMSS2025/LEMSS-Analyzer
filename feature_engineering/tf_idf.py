import os
import logging
import pickle

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pyserini.vectorizer import TfidfVectorizer

from constants.constants import (TF_IDF_FILENAME, TF_IDF_SIMILARITY_FILENAME, TF_IDF_JACCARD_FILENAME,
                                 TF_IDF_JACCARD_SIMILARITY_FILENAME)


class TFIDF:
    def __init__(self, experiment_folder, index_path: str, docnos: list):
        """
        Initialize the TFIDF class with the experiment folder for saving results.

        :param experiment_folder: Path to the folder where results will be saved.
        :param index_path: Path to the Pyserini index.
        :param docnos: List of document IDs.
        """
        self.__experiment_folder = experiment_folder
        self.__docnos = docnos
        self.__tf_idf = self.__build_tf_idf(index_path, docnos)

    def __build_tf_idf(self, index_path, docnos: list) -> np.ndarray:
        """
        Calculate the TF-IDF vectors for the documents in the index.

        :param index_path: Path to the Pyserini index.
        :param docnos: List of DOCNOs.

        :return: Sparse matrix containing the TF-IDF vectors for the documents.
        """
        logging.info("Calculating TF-IDF vectors...")
        vectorizer = TfidfVectorizer(index_path)
        tf_idf = vectorizer.get_vectors(docnos)
        logging.info("Finished calculating TF-IDF vectors.")
        return tf_idf

    def extract_tf_idf_similarities(self) -> None:
        """
        Extract TF-IDF and Jaccard similarities between documents and save the results to pickle files.

        :param docnos: List of document IDs.
        """
        # Calculate and save TF-IDF dictionary and cosine similarities
        tf_idf_dict = self.__compute_tf_idf_dict(self.__tf_idf, self.__docnos)
        tf_idf_similarities = self.__compute_cosine_similarities(self.__tf_idf, self.__docnos)

        # Generate a binary TF-IDF matrix for Jaccard calculation
        binary_tf_idf = self.__tf_idf.copy()
        binary_tf_idf.data = np.where(binary_tf_idf.data != 0, 1, 0) # Check without data, check the speed and if moves to memory

        # Calculate and save Jaccard similarities
        jaccard_dict = self.__compute_jaccard_similarity(binary_tf_idf, self.__docnos)

        # Step 5: Save dictionaries to pickle files
        self.__save_to_pickle(TF_IDF_FILENAME, tf_idf_dict)
        self.__save_to_pickle(TF_IDF_SIMILARITY_FILENAME, tf_idf_similarities)
        self.__save_to_pickle(TF_IDF_JACCARD_FILENAME, binary_tf_idf)
        self.__save_to_pickle(TF_IDF_JACCARD_SIMILARITY_FILENAME, jaccard_dict)

        logging.info("TF-IDF and Jaccard similarities extracted and saved successfully.")

    def __compute_tf_idf_dict(self, tf_idf, docnos):
        """Create a dictionary mapping document IDs to their respective TF-IDF vectors."""
        return {docno: tf_idf[i] for i, docno in enumerate(docnos)}

    def __compute_cosine_similarities(self, matrix, docnos):
        """
        Compute cosine similarities between documents and store non-zero similarities in a dictionary.

        :param matrix: Input matrix (e.g., TF-IDF matrix or Jaccard similarity matrix).
        :param docnos: List of document IDs.
        :return: Dictionary of cosine similarities.
        """
        cosine_sim_matrix = cosine_similarity(matrix)
        return {
            docno: {
                docno2: cosine_sim_matrix[i, j]
                for j, docno2 in enumerate(docnos) if i != j
            }
            for i, docno in enumerate(docnos)
        }

    def __compute_jaccard_similarity(self, binary_tf_idf, docnos):
        """
        Compute the Jaccard similarity matrix based on binary TF-IDF vectors.

        :param binary_tf_idf: Binary version of the TF-IDF matrix.
        :param docnos: List of document IDs.
        :return: Dictionary of Jaccard similarities between documents.
        """
        intersection_matrix = binary_tf_idf @ binary_tf_idf.T
        non_zero_counts = intersection_matrix.diagonal()
        union_matrix = non_zero_counts[:, None] + non_zero_counts[None, :] - intersection_matrix.toarray()

        jaccard_similarity_matrix = np.divide(intersection_matrix.toarray(), union_matrix, where=union_matrix != 0)
        return {
            docnos[i]: {
                docnos[j]: jaccard_similarity_matrix[i, j]
                for j in range(len(docnos)) if i != j
            }
            for i in range(len(docnos))
        }

    def __save_to_pickle(self, filename, data):
        """Save data to a pickle file in the experiment folder."""
        file_path = os.path.join(self.__experiment_folder, filename)
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
