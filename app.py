import logging
from modules.ui import render_ui
from config import check_env_variables, create_directories

def main():
    """Main application entry point"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create required directories
    create_directories()
    
    # Check environment variables
    check_env_variables()
    
    # Render the UI
    render_ui()

if __name__ == "__main__":
    main()