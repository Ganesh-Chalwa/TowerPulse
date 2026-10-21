import os

# Get the folder where this config.py file is located.
# This helps us create reliable paths for the Flask project files.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path of the trained Random Forest model.
MODEL_FILE_PATH = os.path.join(
    BASE_DIR,
    "project_app",
    "Random_Forest.pkl"
)

# Path of the project data JSON file.
# This file contains the feature information required by the model.
JSON_FILE_PATH = os.path.join(
    BASE_DIR,
    "project_app",
    "project_data.json"
)