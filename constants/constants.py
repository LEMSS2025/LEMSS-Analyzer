import os

FEATURES_ORDER = ['docno', 'qid', 'round', 'contents_analyzed_BM25Stat_sum_k1_0.90_b_0.40',
                             'contents_analyzed_LmDirStat_sum_mu_1000', 'contents_analyzed_TfStat_sum',
                             'contents_analyzed_NormalizedTfStat_sum', 'contents_DocSize',
                             'frac_stop', 'stop_cover', 'entropy', 'bert_score']
COLUMNS_TO_FEATURE_NAMES = {
        'docno': 'document_id',
        'qid': 'query_id',
        'contents_analyzed_BM25Stat_sum_k1_0.90_b_0.40': 'Okapi',
        'contents_analyzed_LmDirStat_sum_mu_1000': 'LM',
        'contents_analyzed_TfStat_sum': 'TF',
        'contents_analyzed_NormalizedTfStat_sum': 'NormTF',
        'contents_DocSize': 'LEN',
        'frac_stop': 'FracStop',
        'stop_cover': 'StopCover',
        'entropy': 'ENT',
        'bert_score': 'BERT'
    }

FEATURES_TO_NORMALIZE = ['Okapi', 'LM', 'TF', 'NormTF', 'LEN', 'FracStop', 'StopCover', 'ENT', 'BERT']
INDEX_ORDER = {'student': 0, 'writer': 1, 'editor': 2, 'teacher': 3, 'professor': 4}

NUM_TO_STR = {1: "first", 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth', 6: 'sixth', 7: 'seventh', 8: 'eighth', 9: 'ninth', 10: 'tenth', 11: 'eleventh', 12: 'twelfth', 13: 'thirteenth', 14: 'fourteenth', 15: 'fifteenth', 16: 'sixteenth', 17: 'seventeenth', 18: 'eighteenth', 19: 'nineteenth', 20: 'twentieth', 21: 'twenty-first', 22: 'twenty-second', 23: 'twenty-third', 24: 'twenty-fourth', 25: 'twenty-fifth', 26: 'twenty-sixth', 27: 'twenty-seventh', 28: 'twenty-eighth', 29: 'twenty-ninth', 30: 'thirtieth', 31: 'thirty-first', 32: 'thirty-second', 33: 'thirty-third', 34: 'thirty-fourth', 35: 'thirty-fifth', 36: 'thirty-sixth', 37: 'thirty-seventh', 38: 'thirty-eighth', 39: 'thirty-ninth', 40: 'fortieth', 41: 'forty-first', 42: 'forty-second', 43: 'forty-third', 44: 'forty-fourth', 45: 'forty-fifth', 46: 'forty-sixth', 47: 'forty-seventh', 48: 'forty-eighth', 49: 'forty-ninth', 50: 'fiftieth', 51: 'fifty-first', 52: 'fifty-second', 53: 'fifty-third', 54: 'fifty-fourth', 55: 'fifty-fifth', 56: 'fifty-sixth', 57: 'fifty-seventh', 58: 'fifty-eighth', 59: 'fifty-ninth', 60: 'sixtieth', 61: 'sixty-first', 62: 'sixty-second', 63: 'sixty-third', 64: 'sixty-fourth', 65: 'sixty-fifth', 66: 'sixty-sixth', 67: 'sixty-seventh', 68: 'sixty-eighth', 69: 'sixty-ninth', 70: 'seventieth', 71: 'seventy-first', 72: 'seventy-second', 73: 'seventy-third', 74: 'seventy-fourth', 75: 'seventy-fifth', 76: 'seventy-sixth', 77: 'seventy-seventh', 78: 'seventy-eighth', 79: 'seventy-ninth', 80: 'eightieth', 81: 'eighty-first', 82: 'eighty-second', 83: 'eighty-third', 84: 'eighty-fourth', 85: 'eighty-fifth', 86: 'eighty-sixth', 87: 'eighty-seventh', 88: 'eighty-eighth', 89: 'eighty-ninth', 90: 'ninetieth', 91: 'ninety-first', 92: 'ninety-second', 93: 'ninety-third', 94: 'ninety-fourth', 95: 'ninety-fifth', 96: 'ninety-sixth', 97: 'ninety-seventh', 98: 'ninety-eighth', 99: 'ninety-ninth', 100: 'one hundredth'}

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

E5_SIMILARITY_FILENAME = "e5_cosine_similarities_dict.pkl"
SBERT_SIMILARITY_FILENAME = "bert_similarities_dict.pkl"

TF_IDF_FILENAME = "tfidf-krovetz_dict.pkl"
TF_IDF_SIMILARITY_FILENAME = "tfidf-krovetz_similarity.pkl"
TF_IDF_JACCARD_FILENAME = "tfidf-krovetz-jaccard_dict.pkl"
TF_IDF_JACCARD_SIMILARITY_FILENAME = "tfidf-krovetz-jaccard_similarity.pkl"

SBERT_INDEX_FILENAME = "sbert_index.index"
E5_INDEX_FILENAME = "e5_index.index"

FAISS_FOLDER = "faiss"

EMBEDDING_FILENAMES = ["tfidf-krovetz-jaccard_similarity.pkl", "tfidf-krovetz_similarity.pkl", "e5_cosine_similarities_dict.pkl", "bert_similarities_dict.pkl"]
NUM_TO_REPRESENTATION ={0: "tf_idf_jaccard_similarity", 1: "tf_idf_similarity", 2: "e5_embeddings_similarity", 3: "bert_embeddings_similarity"}

SBERT_DIMENSION = 384
E5_DIMENSION = 1024
BERT_MAX_LEN = 512

E5_MODEL_LINK = "intfloat/e5-large-unsupervised"
BERT_MODEL_LINK = "amberoad/bert-multilingual-passage-reranking-msmarco"
SBERT_MODEL_LINK = "all-MiniLM-L6-v2"

COMPEITION_HISTORY_FILE_NAME = "competition_history.csv"
CONFIG_FILE_NAME = "config.json"
OUTPUT_TRECTEXT_FILE_NAME = "output.trectext"
WEB_TRACK_FOLDER = "web_track"
DATA_FOLDER = "data"
EXPERIMENTS_FOLDER = "experiments"

FEATURE_MATRIX_FILE_NAME = "feature_matrix.csv"

CONFIDENCE_INTERVAL = 1.96
HEX_BASE = 16
HASH_TRUNCATION_LIMIT = 10 ** 7
NUM_THREADS = 2
PYSERINI_BATCH_SIZE = 10000

TAG_DOC = "DOC"
TAG_DOCNO = "DOCNO"
TAG_TEXT = "TEXT"

HISTORY_DF_RANK_COLUMN = "rank"
HISTORY_DF_QUERY_ID_COLUMN = "query_id"
HISTORY_DF_PLAYER_COLUMN = "player"
HISTORY_DF_ROUND_COLUMN = "round"
HISTORY_DF_USER_QUERY_COLUMN = "user_query"

COLUMN_DOCNO = "DOCNO"
COLUMN_TEXT = "TEXT"
COLUMN_QID = "qid"
COLUMN_COUNT = "count"
COLUMN_RENAMED_DOCNO = "docno"
COLUMN_DOCUMENT = "document"
COLUMN_ROUND = "round"

XML_TOPIC_HEADER = "topic"
XML_NUMBER_HEADER = "number"
XML_QUERY_HEADER = "query"

ANALYZED_QUERY = "analyzed_query"
ORIGINAL_QUERY = "original_query"
ANALYZED = "analyzed"

PYSERINI_CONTENTS = "contents"
PYSERINI_DOCIDS = "docIds"
PYSEIRNI_KROVETZ_STEMMER = "krovetz"

REPORT_TABLE_BEST_AGENT_COLUMN = "Best Agent"
REPORT_TABLE_WINNING_HOMOGENEITY_COLUMN = "Winning Homogeneity"
REPORT_TABLE_FILE_NAME = "report_table.csv"

COMPETITION_HISTORY_PIVOIT_FILE_NAME = "competition_history_pivot.csv"

EXPERIMENT_LOG_FILE_NAME = "experiment.log"