# Import core libraries
import os
import logging
import warnings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Disable warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Check for required environment variables
def check_env_variables():
    required_vars = ["GROQ_API_KEY", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    if missing_vars:
        logger.warning(f"Missing environment variables: {', '.join(missing_vars)}")
        return False
    return True

# Define configuration settings
GROQ_MODEL = "llama3-8b-8192"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
VECTORSTORE_TOP_K = 5

# Directory configurations
PDF_DIR = os.environ.get("PDF_DIR", "./data")
EXTRACTED_IMG_DIR = os.environ.get("EXTRACTED_IMG_DIR", "./extracted_images")
PERSIST_DIR = os.environ.get("PERSIST_DIR", "./chroma_db")
TEMP_UPLOAD_DIR = "./tmp_uploads"

# Create required directories
def create_directories():
    for directory in [PDF_DIR, EXTRACTED_IMG_DIR, PERSIST_DIR, TEMP_UPLOAD_DIR]:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory created/verified: {directory}")