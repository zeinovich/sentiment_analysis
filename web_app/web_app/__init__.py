import logging
from datetime import datetime
import os

if not os.path.exists("../logs"):
    os.mkdir("../logs")
    
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"../logs/web_app_{datetime.date(datetime.now())}.log"),
        logging.StreamHandler(),
    ],
)

LOGGER = logging.getLogger(__name__)
LOGGER.info("Logging initialized")
