import logging

import faiss


class FaissIndex:
    def __init__(self, dimension, index_file=None):
        """
        Initialize a FAISS index with ID mapping for embedding extraction by ID.

        :param dimension: Dimension of the embeddings.
        :param index_file: Path to the saved index file (if loading an existing index).
        """
        self.__dimension = dimension
        self.__index_file = index_file

        if index_file:
            # Load existing index from file
            self.index = faiss.read_index(index_file)
            logging.info(f"Index loaded from {index_file}")
        else:
            # Initialize a flat index with L2 distance and ID map for ID-based retrieval
            flat_index = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIDMap2(flat_index)
            logging.info("Initialized a new FAISS index with ID mapping")

    def add_embeddings(self, embeddings, ids):
        """
        Add embeddings with specific IDs to the index.

        :param embeddings: A numpy array of embeddings to add.
        :param ids: A numpy array of unique IDs corresponding to each embedding.
        """
        self.index.add_with_ids(embeddings, ids)
        logging.info(f"Added {len(ids)} embeddings to the index")

    def extract_by_id(self, id):
        """
        Extract an embedding by its ID.

        :param id: The ID of the embedding to extract.
        :return: The embedding vector if it exists, else None.
        """
        try:
            vector = self.index.reconstruct(id)
            logging.info(f"Vector for ID {id} retrieved")
            return vector
        except RuntimeError:
            logging.error(f"ID {id} not found in the index.")
            return None

    def save_index(self, index_file):
        """
        Save the current index to a file.

        :param index_file: Path to save the index file.
        """
        faiss.write_index(self.index, index_file)
        logging.info(f"Index saved to {index_file}")