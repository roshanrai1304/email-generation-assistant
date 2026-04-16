#!/bin/bash

# Suppress all warnings and verbose output
export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3
export TRANSFORMERS_VERBOSITY=error
export STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
export STREAMLIT_LOGGER_LEVEL=error

# Launch Streamlit with clean output
streamlit run streamlit_app.py 2>&1 | grep -v "torchvision" | grep -v "ModuleNotFoundError" | grep -v "Accessing"
