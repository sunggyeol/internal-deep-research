import logging
import datetime
import os

logger = None

def initial_logger(logging_path: str = "log", enable_stdout: bool = False) -> None:
    global logger
    now = datetime.datetime.now()
    log_file = os.path.join(logging_path, f"deep_research_py_{now.strftime('%Y%m%d_%H%M%S')}.log")
    if not os.path.exists(logging_path):
        os.makedirs(logging_path)
    handlers = [logging.FileHandler(log_file)]
    if enable_stdout:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=handlers
    )
    logger = logging.getLogger("deep_research_py")
    logger.setLevel(logging.INFO)

def log_event(event_desc: str) -> None:
    if logger is not None:
        logger.info(f"Event: {event_desc}")

def log_error(error_desc: str) -> None:
    if logger is not None:
        logger.error(f"Error: {error_desc}")

def log_warning(warning_desc: str) -> None:
    if logger is not None:
        logger.warning(f"Warning: {warning_desc}")
