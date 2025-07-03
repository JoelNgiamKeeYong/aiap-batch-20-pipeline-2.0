#!/bin/bash

# Reset the project by deleting contents of data, models, and output folders
echo "🔄 Resetting the project..."

# Define directories to clear
DIRECTORIES=("data" "output")

# Loop through each directory and delete contents if it exists
for DIR in "${DIRECTORIES[@]}"; do
    if [ -d "$DIR" ]; then
        echo "   └── Deleting contents of '$DIR' folder..."
        rm -rf "$DIR"/*
    else
        echo "   └── '$DIR' folder does not exist. Skipping deletion."
    fi
done

# For the models folder, only delete contents of 'trained_models'
if [ -d "models" ]; then
    echo "   └── Deleting contents of 'models'..."
    rm -rf models/*
else
    echo "   └── 'models' folder does not exist. Skipping deletion."
fi

# Check if the /archives/training_logs.txt file exists and ask for confirmation
LOG_FILE="archives/training_logs.txt"
if [ -f "$LOG_FILE" ]; then
    echo
    read -p "❓  Do you want to delete the '$LOG_FILE' file? (y/n): " CONFIRMATION
    if [[ "$CONFIRMATION" == "y" || "$CONFIRMATION" == "Y" ]]; then
        echo "   └── Deleting '$LOG_FILE' file..."
        rm -f "$LOG_FILE"
    else
        echo "   └── Skipping deletion of '$LOG_FILE'."
    fi
fi

# Final confirmation
echo
echo "✅ Project reset completed successfully!"