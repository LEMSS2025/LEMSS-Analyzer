import re
import logging
from typing import List, Tuple

import pandas as pd

from constants.constants import (COLUMN_DOCNO, COLUMN_TEXT, COLUMN_QID,
                                 COLUMN_RENAMED_DOCNO, COLUMN_DOCUMENT,
                                 TAG_DOC, TAG_DOCNO, TAG_TEXT)


class TrecParser:
    """
    Class responsible for parsing TREC text files and generating query IDs.
    """

    def __init__(self, trectext_path: str):
        """
        Initialize the TrecParser with the path to the TREC text file.

        :param trectext_path: Path to the TREC text file.
        """
        self.__trectext_path = trectext_path
    def __parse_trectext(self) -> pd.DataFrame:
        """
        Parse the TREC text file to extract document numbers and texts.

        :return: DataFrame containing document numbers and texts.
        """
        logging.info(f"Parsing TREC text file: {self.__trectext_path}")

        try:
            with open(self.__trectext_path, 'r', encoding='utf-8') as file:
                content = file.read()
        except IOError as e:
            logging.error(f"Error reading TREC text file: {e}")
            raise

        docs = self.__extract_docs(content)
        if not docs:
            logging.error("No documents found in TREC text file.")
            raise ValueError("No documents found in TREC text file.")

        logging.info(f"Parsed {len(docs)} documents from TREC text file.")
        return pd.DataFrame(docs, columns=[COLUMN_DOCNO, COLUMN_TEXT])

    def __extract_docs(self, content: str) -> List[Tuple[str, str]]:
        """
        Extract document numbers and texts from the TREC text content.

        :param content: The full text content of the TREC file.
        :return: A list of tuples containing document numbers and texts.
        """
        docs = re.findall(rf'<{TAG_DOC}>(.*?)</{TAG_DOC}>', content, re.DOTALL)
        parsed_docs = []

        for doc in docs:
            docno = self.__extract_tag_content(doc, TAG_DOCNO)
            text = self.__extract_tag_content(doc, TAG_TEXT)
            if docno and text:
                parsed_docs.append((docno, text))

        return parsed_docs

    def __extract_tag_content(self, doc: str, tag: str) -> str:
        """
        Extract the content of a specified XML tag from a document string.

        :param doc: The document string containing the XML tags.
        :param tag: The XML tag to extract content from.
        :return: The content of the specified XML tag.
        """
        match = re.search(rf'<{tag}>(.*?)</{tag}>', doc, re.DOTALL)
        if not match:
            logging.warning(f"Tag <{tag}> not found in document.")
        return match.group(1).strip() if match else ''

    def __generate_qid(self, docno: str) -> str:
        """
        Generate a query ID from the document number.

        :param docno: Document number.
        :return: Query ID extracted from the document number.
        """
        parts = docno.split('-')
        if len(parts) > 2:
            return parts[2]
        logging.error(f"Unexpected DOCNO format: {docno}")
        return ''

    def __merge_data(self) -> pd.DataFrame:
        """
        Merge the parsed TREC text with generated query IDs.

        :return: Merged DataFrame containing query IDs, document IDs, and texts.
        """
        logging.info("Merging TREC text data...")
        documents_df = self.__parse_trectext()

        # Generate query IDs and rename the columns
        documents_df[COLUMN_QID] = documents_df[COLUMN_DOCNO].apply(self.__generate_qid)
        final_df = documents_df.rename(
            columns={COLUMN_DOCNO: COLUMN_RENAMED_DOCNO, COLUMN_TEXT: COLUMN_DOCUMENT}
        )
        logging.info("Data merged successfully.")

        return final_df

    def parse_trec(self) -> pd.DataFrame:
        """
        Run the data parsing, merging, and saving process.

        :return: Final merged DataFrame.
        """
        logging.info("Starting data preprocessing...")
        final_df = self.__merge_data()
        logging.info("Data preprocessed successfully.")

        return final_df
