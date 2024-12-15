import pandas as pd
import logging

import nltk
from nltk.corpus import stopwords
from scipy.stats import entropy as scipy_entropy
from pyserini.analysis import Analyzer, get_lucene_analyzer

from constants.constants import PYSEIRNI_KROVETZ_STEMMER

class CustomFeatures:
    """
    Class responsible for calculating custom features such as stopword fraction, stopword coverage,
    and entropy of term distributions in document.
    """

    def __init__(self):
        """
        Initialize the CustomFeatures class by downloading NLTK stopwords
        and setting up the stopwords set.
        """
        logging.info("Downloading NLTK stopwords...")
        nltk.download('stopwords')
        self.__stop_words = set(stopwords.words('english'))
        logging.info("Stopwords downloaded and set up successfully.")

        logging.info("Initializing Krovetz stemmer and analyzer...")
        self.__analyzer_with_stopwrods = Analyzer(get_lucene_analyzer(stemmer=PYSEIRNI_KROVETZ_STEMMER, stopwords=False))
        self.__analyzer_without_stopwrods = Analyzer(get_lucene_analyzer(stemmer=PYSEIRNI_KROVETZ_STEMMER, stopwords=True))
        logging.info("Analyzer initialized successfully.")

    def frac_stop(self, text_series: pd.Series) -> pd.Series:
        """
        Calculate the fraction of stopwords in each document of the text series.

        :param text_series: Series of document texts.
        :return: Series of stopword fractions for each document.
        """
        logging.info("Calculating fraction of stopwords...")

        def calculate_frac_stop(doc_text: str) -> float:
            """
            Calculate the fraction of stopwords in a single document text.

            :param doc_text: String representing the document text.
            :return: Fraction of stopwords in the document.
            """
            terms = self.__analyzer_with_stopwrods.analyze(doc_text)
            if not terms:
                return 0.0
            stopword_count = sum(1 for term in terms if term in self.__stop_words)

            return stopword_count / len(terms)

        result = text_series.apply(calculate_frac_stop)
        logging.info("Fraction of stopwords calculation completed.")

        return result

    def stop_cover(self, text_series: pd.Series) -> pd.Series:
        """
        Calculate the stopword coverage for each document in the text series.

        :param text_series: Series of document texts.
        :return: Series of stopword coverage values for each document.
        """
        logging.info("Calculating stopword coverage...")

        def calculate_stop_cover(doc_text: str) -> float:
            """
            Calculate the stopword coverage in a single document text.

            :param doc_text: String representing the document text.
            :return: Coverage of stopwords in the document.
            """
            terms_set = set(self.__analyzer_with_stopwrods.analyze(doc_text))
            if not terms_set:
                return 0.0
            stopword_count = sum(1 for stopword in self.__stop_words if stopword in terms_set)

            return stopword_count / len(self.__stop_words)

        result = text_series.apply(calculate_stop_cover)
        logging.info("Stopword coverage calculation completed.")

        return result

    def entropy(self, text_series: pd.Series) -> pd.Series:
        """
        Calculate the entropy of the term distribution for each document in the text series.

        :param text_series: Series of document texts.
        :return: Series of entropy values for each document.
        """
        logging.info("Calculating entropy of term distributions...")

        def calculate_entropy(doc_text: str) -> float:
            """
            Calculate the entropy of the term distribution in a single document text.

            :param doc_text: String representing the document text.
            :return: Entropy of the term distribution in the document.
            """
            terms = self.__analyzer_without_stopwrods.analyze(doc_text)
            if not terms:
                return 0.0
            term_counts = pd.Series(terms).value_counts(normalize=True)
            return scipy_entropy(term_counts)

        result = text_series.apply(calculate_entropy)
        logging.info("Entropy calculation completed.")

        return result
