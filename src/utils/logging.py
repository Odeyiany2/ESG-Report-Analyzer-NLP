import logging
import pathlib 
import os
from pathlib import Path

#get current working directory
cwd = Path.cwd()

def setup_logger(
        logger_name:str,
        log_file: str,
        log_level: int = logging.INFO) -> logging.Logger:
    """
    This function allows the system to create and write log data of the entire system's operations.
    Args:
        logger_name (str): the name of the log file to create
        log_file (str): file path to the log file
        log_level (int): the value of the log type (warn, info, debug)
    Returns:
        logging.Logger: the logger object
    """
    #create the logger object
    logger = logging.getLogger(logger_name)

    # set the log level
    logger.setLevel(log_level)

    #create the file handler to write the log file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)

    #create the log format
    format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(format)
    logger.addHandler(file_handler)

    return logger

def create_folder_and_log_file(
        folder_name:str, 
        file_name:str
) -> pathlib.Path:
    """
    This function creates a folder and a log file inside that folder if they do not already exist.
    Args:
        folder_name (str): the name of the folder to create
        file_name (str): the name of the log file to create
    Returns:
        pathlib.Path: the path to the log file
    """

    new_path = cwd.joinpath(folder_name)

    # create folder_path only once if not exist
    new_path.mkdir(exist_ok=True)
    log_file_path = new_path.joinpath(file_name)

    # create file if not exist
    log_file_path.touch()

    return log_file_path

# Define the folder name for logs
folder_name = "ESG_logs"

# List of log files to create
log_files_to_create = [
    "doc_handler.log",
    "embedding.log",
    "fastapi_logic.log",
    "streamlit_app.log",
    "retriever.log"]

# Create the folder and log files
for log_file in log_files_to_create:
    create_folder_and_log_file(folder_name, log_file)


# setup loggers for different modules
doc_handler_logger = setup_logger(
    "doc_handler_logger",
    os.path.join(cwd, folder_name, "doc_handler.log")
)
embedding_logger = setup_logger(
    "embedding_logger",
    os.path.join(cwd, folder_name, "embedding.log")
)
fastapi_logic_logger = setup_logger(
    "fastapi_logic_logger",
    os.path.join(cwd, folder_name, "fastapi_logic.log")
)
streamlit_app_logger = setup_logger(
    "streamlit_app_logger",
    os.path.join(cwd, folder_name, "streamlit_app.log")
)
retriever_logger = setup_logger(
    "retriever_logger",
    os.path.join(cwd, folder_name, "retriever.log")
)