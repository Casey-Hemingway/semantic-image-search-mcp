#!/bin/bash
# MCP Server startup script with correct Python path and environment

# Change to project directory so config.yml can be found
cd "/Users/casey-hemingway/Documents/Projects/HT/photo-library"

# Set environment variables
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH="/Users/casey-hemingway/Documents/Projects/HT/photo-library"

# Use the Python that has all packages installed
exec /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
    "/Users/casey-hemingway/Documents/Projects/HT/photo-library/run_server.py"
