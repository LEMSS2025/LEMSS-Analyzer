import os
import hashlib
import argparse

from utils.config_loader import ConfigLoader
from data_processing.index_manager import IndexManager
from data_processing.feature_extractor import FeatureExtractor
from analysis.analyzer import Analyzer
from analysis.embedding_analyzer import EmbeddingAnalyzer
from analysis.rank_analysis import RankAnalysis
from analysis.report_table import ReportTable
from utils.file_operations import FileOperations
from utils.logging_setup import LoggingSetup
from constants.constants import (COMPEITION_HISTORY_FILE_NAME, CONFIG_FILE_NAME, OUTPUT_TRECTEXT_FILE_NAME,
                                 FEATURE_MATRIX_FILE_NAME, WEB_TRACK_FOLDER, DATA_FOLDER, EXPERIMENT_LOG_FILE_NAME,
                                 EXPERIMENTS_FOLDER)


def main(input_folder, index_path):
    # Load configuration from the input folder
    config_loader = ConfigLoader()
    config_path = os.path.join(input_folder, CONFIG_FILE_NAME)
    config = config_loader.load_config(config_path)

    # Generate unique folder name based on config hash
    folder_name = hashlib.md5(str(config).encode()).hexdigest()
    experiment_folder = os.path.join(EXPERIMENTS_FOLDER, folder_name)
    os.makedirs(experiment_folder, exist_ok=True)

    # Set up logging
    log_file_path = os.path.join(experiment_folder, EXPERIMENT_LOG_FILE_NAME)
    LoggingSetup.configure_logging(log_file_path)

    # Save the configuration to the experiment folder
    config_file_path = os.path.join(experiment_folder, CONFIG_FILE_NAME)
    os.system(f"cp {config_path} {config_file_path}")

    # Define paths and parameters based on the input folder
    trectext_path = os.path.join(input_folder, OUTPUT_TRECTEXT_FILE_NAME)
    queries_folder = os.path.join(DATA_FOLDER, WEB_TRACK_FOLDER)
    feature_matrix_path = os.path.join(experiment_folder, FEATURE_MATRIX_FILE_NAME)
    competition_history_path = os.path.join(input_folder, COMPEITION_HISTORY_FILE_NAME)
    num_rounds_back = 4

    # Add documents to the index
    index_manager = IndexManager(trectext_path, index_path)
    index_manager.add_documents_to_index()

    # Extract features
    feature_extractor = FeatureExtractor(trectext_path, queries_folder, feature_matrix_path, index_path,
                                         experiment_folder)
    feature_extractor.extract_features()

    # Copy necessary files to the experiment folder
    FileOperations.copy_files_to_experiment_folder(trectext_path, competition_history_path, experiment_folder)

    # Analyze the features
    feature_analyzer = Analyzer(feature_matrix_path, competition_history_path, experiment_folder)
    feature_analyzer.analyze(num_rounds_back)

    # Analyze the embeddings
    embedding_analyzer = EmbeddingAnalyzer(experiment_folder)
    embedding_analyzer.plot_graphs()

    # Analyze the rank
    rank_analysis = RankAnalysis(experiment_folder)
    rank_analysis.rank_analysis(folder_name)

    # Generate Report Table
    report_table = ReportTable(experiment_folder)
    report_table.generate_report_table()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Analysis")
    parser.add_argument('--input_folder', type=str, required=True, help="Path to the input folder")
    parser.add_argument('--index_path', type=str, required=True, help="Path to the index path")

    args = parser.parse_args()

    main(args.input_folder, args.index_path)
