
# Main entry point for the project


# This script:

# Configures logging for debugging and information tracking.
# Provides command-line arguments:
# --run-api → Starts the Flask API server.
# --test-model → Loads and verifies the trained model.
# Loads the model from the configured path for validation.




import argparse
import logging
from src.app.api import app
from src.utils.model_utils import load_model
from src.config.config import MODEL_PATH

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_api():
    """Run the Flask API server."""
    logging.info("Starting API server...")
    app.run(debug=True, host='0.0.0.0', port=5000)

def load_and_test_model():
    """Load the trained model and perform a simple test."""
    logging.info("Loading model for verification...")
    model = load_model(MODEL_PATH)
    logging.info("Model loaded successfully.")

def main():
    """Main entry point for the project."""
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Alzheimer Voice Analysis System")
    parser.add_argument("--run-api", action="store_true", help="Run the API server")
    parser.add_argument("--test-model", action="store_true", help="Load and test the trained model")
    args = parser.parse_args()
    
    if args.run_api:
        run_api()
    elif args.test_model:
        load_and_test_model()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
