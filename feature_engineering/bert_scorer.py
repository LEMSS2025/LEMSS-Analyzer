import logging

from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

from constants.constants import (BERT_MODEL_LINK,
                                 BERT_MAX_LEN, SBERT_SIMILARITY_FILENAME)

class BertScorer:
    """
    Class responsible for calculating BERT-based similarity scores and extracting document embeddings.
    """

    def __init__(self, model_name: str = BERT_MODEL_LINK):
        """
        Initialize the BertScorer with the specified tokenizer and model.

        :param model_name: Name of the BERT model to use for scoring.
        """
        logging.info("Loading tokenizer and model...")
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.__bert_model_embedding = AutoModel.from_pretrained(model_name).to(self.__device).eval()
        self.__bert_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.__device).eval()
        logging.info("Tokenizer and model loaded successfully.")

    def get_bert_scores(self, queries_to_docnos: dict, docs_dict: dict, queries_dict: dict) -> dict:
        """
        Calculate BERT scores for a given set of queries and documents.

        :param queries_to_docnos: A dictionary mapping query IDs to lists of document IDs.
        :param docs_dict: A dictionary mapping document IDs to document texts.
        :param queries_dict: A dictionary mapping query IDs to query texts.

        :return: A dictionary mapping document IDs to BERT scores or embeddings.
        """
        results_dict = {}

        logging.info("Starting BERT score calculation...")
        for qid in tqdm(queries_to_docnos, desc="BERT", total=len(queries_to_docnos)):
            for docno in queries_to_docnos[qid]:
                doc_text = docs_dict[docno]
                # Process query-document pair for BERT score
                query_text = queries_dict[qid]['original_query']
                tokenized_input = self.__tokenizer.encode_plus(
                    query_text,
                    doc_text,
                    max_length=BERT_MAX_LEN,
                    truncation=True,
                    return_token_type_ids=True,
                    return_tensors='pt'
                )

                with torch.no_grad():
                    input_ids = tokenized_input['input_ids'].to(self.__device)
                    token_type_ids = tokenized_input['token_type_ids'].to(self.__device)
                    output = self.__bert_model(input_ids, token_type_ids=token_type_ids, return_dict=False)[0]

                    # Single-label or multi-label classification
                    if output.size(1) > 1:
                        score = torch.nn.functional.softmax(output, dim=1)[0, -1].item()
                    else:
                        score = output.item()

                results_dict[docno] = score

        logging.info("BERT score calculation completed.")

        return results_dict