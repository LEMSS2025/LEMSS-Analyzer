import os
import pickle
import logging

import numpy as np
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from utils.utils import hash_id
from constants.constants import SBERT_MODEL_LINK, SBERT_SIMILARITY_FILENAME


class SBERT:
    """
    Class responsible for creating SBERT embeddings.
    """

    def __init__(self, model_name: str = SBERT_MODEL_LINK):
        """
        Initialize the SBERT with the specified tokenizer and model.

        :param model_name: Name of the SBERT model to use for embeddings.
        """
        logging.info("Loading model...")
        self.__sbert_model_embedding = SentenceTransformer(model_name,
                                                           device=torch.device("cuda" if torch.cuda.is_available()
                                                                               else "cpu"))
        logging.info("Model loaded successfully.")

    def create_sbert_embeddings(self, queries_to_docnos: dict, docs_dict: dict, faiss_index=None,
                                experiment_folder: str = None) -> None:
        """
        Create SBERT embeddings for a given set of queries and documents.

        :param queries_to_docnos: A dictionary mapping query IDs to lists of document IDs.
        :param docs_dict: A dictionary mapping document IDs to document texts.
        :param faiss_index: A FaissIndex object to store document embeddings.
        :param experiment_folder: Path to the experiment folder.
        """
        logging.info("Starting SBERT embedding creation...")
        cosine_similarities = {}
        for qid in tqdm(queries_to_docnos, desc="SBERT", total=len(queries_to_docnos)):
            docs_qid = queries_to_docnos[qid]
            input_texts = [docs_dict[docno] for docno in docs_qid]

            embeddings = self.__sbert_model_embedding.encode(input_texts, normalize_embeddings=True)
            docnos = np.array([hash_id(docno) for docno in docs_qid])

            # Add to faiss index
            faiss_index.add_embeddings(embeddings, docnos)

            embeddings_dict = {docno: emb for docno, emb in zip(docs_qid, embeddings)}
            for docno in docs_qid:
                cosine_similarities[docno] = {docno2: cos_sim(embeddings_dict[docno],
                                                              embeddings_dict[docno2]).cpu().item() for docno2 in
                                              queries_to_docnos[qid] if docno != docno2}

        # Save cosine similarities
        with open(os.path.join(experiment_folder, SBERT_SIMILARITY_FILENAME), "wb") as f:
            pickle.dump(cosine_similarities, f)
