# src/utils/logger_config.py

import logging
import os
from datetime import datetime

def setup_logger(log_dir='logs'):
    """Set up logging configuration"""
    
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create timestamp for log filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(log_dir, f'real_estate_analysis_{timestamp}.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # File handler
            logging.FileHandler(log_filename),
            # Console handler
            logging.StreamHandler()
        ]
    )
    
    # Get the logger
    logger = logging.getLogger('RealEstateAnalysis')
    
    logger.info(f'Logger initialized. Log file: {log_filename}')
    return logger