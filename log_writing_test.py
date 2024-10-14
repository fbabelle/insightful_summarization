import os
import sys
import logging
from datetime import datetime

# Set up log directory and file
root_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(root_dir, 'log')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'log_test.log')

print(f"Log directory: {log_dir}")
print(f"Log file: {log_file}")

# Function to write directly to the log file
def write_to_log(message):
    try:
        with open(log_file, 'a') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} - {message}\n")
        print(f"Written to log: {message}")
    except Exception as e:
        print(f"Error writing to log: {e}")

# Test direct writing to the log file
write_to_log("Direct writing test")

# Set up logging
def setup_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create file handler
    try:
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        print(f"File handler added for {log_file}")
    except Exception as e:
        print(f"Error setting up file handler: {e}")

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger

# Set up logging
logger = setup_logging()

# Test logging
logger.info("Logging initialized with explicit setup")
logger.info("This is a test log message")

# Ensure logs are written immediately
for handler in logger.handlers:
    handler.flush()

write_to_log("After logging test")

def main():
    logger.info("Main function started")
    # Simulating some work
    for i in range(5):
        logger.info(f"Processing item {i}")
        write_to_log(f"Directly writing about item {i}")
    logger.info("Main function completed")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
    finally:
        logger.info("Script execution completed")
        write_to_log("Final direct write")
        # Final flush of all handlers
        for handler in logger.handlers:
            handler.flush()
            handler.close()
        logging.shutdown()

print("Script finished. Please check the log file.")