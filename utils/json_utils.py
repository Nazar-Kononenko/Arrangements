import json
import os
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class JsonHandler:
    @staticmethod
    def save(data: dict, file_path: str):
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"JSON data saved to: {file_path}")

    @staticmethod
    def load(file_path: str):
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r") as f:
            data = json.load(f)

        logger.info(f"JSON data loaded from: {file_path}")
        return data
