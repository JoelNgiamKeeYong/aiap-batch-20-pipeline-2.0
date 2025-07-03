#!/bin/bash

# Function to display usage instructions
usage() {
    echo "Usage: bash run.sh [--lite] [--model lr] [--model rf] [--model xgb] [--model lgbm]"
    echo "       --lite  : Run the pipeline in lite mode (for quick debugging)."
    echo "       --model : Specify which model(s) to run (lr, rf, xgb, lgbm). If no models are specified, all models will be run."
    exit 1
}

# Parse arguments
LITE_MODE=false
MODELS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lite)
            LITE_MODE=true
            shift
            ;;
        --model)
            if [[ -z "$2" ]]; then
                echo "❌ Error: Missing model name after '--model'."
                usage
            fi
            case "$2" in
                lr|rf|xgb|lgbm)
                    MODELS+=("$2")
                    shift 2
                    ;;
                *)
                    echo "❌ Error: Invalid model name '$2'. Valid options are 'lr', 'rf', 'xgb', 'lgbm'."
                    usage
                    ;;
            esac
            ;;
        *)
            echo "❌ Error: Unknown argument '$1'."
            usage
            ;;
    esac
done

# Default behavior if no models are specified
if [[ ${#MODELS[@]} -eq 0 && $LITE_MODE == false ]]; then
    MODELS=("lr" "rf" "xgb" "lgbm")
fi

# Path to the configuration file
CONFIG_FILE="config.yaml"

# Check if the configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Configuration file '$CONFIG_FILE' not found."
    exit 1
fi

# Extract from the YAML file
DATA_URL=$(grep '^data_url:' "$CONFIG_FILE" | awk '{print $2}' | sed 's/"//g')
SAVE_PATH=$(grep '^db_path:' "$CONFIG_FILE" | awk '{print $2}' | sed 's/"//g')

# Ensure the extracted paths are valid
if [ -z "$DATA_URL" ] || [ -z "$SAVE_PATH" ]; then
    echo "❌ Error: Failed to extract DATA_URL or SAVE_PATH from '$CONFIG_FILE'."
    exit 1
fi

# Check if the dataset exists
if [ ! -f "$SAVE_PATH" ]; then
    echo "⚠️  Dataset '$SAVE_PATH' not found in the 'data' folder."
    
    # Prompt the user to download the dataset automatically
    read -p "   Do you want to download the dataset automatically? (y/n): " choice
    
    # Convert the input to lowercase for case-insensitive comparison
    choice=$(echo "$choice" | tr '[:upper:]' '[:lower:]')
    
    if [ "$choice" == "y" ] || [ "$choice" == "yes" ]; then
        echo "   └── Creating directory for $SAVE_PATH if it doesn't exist..."
        mkdir -p "$(dirname "$SAVE_PATH")"

        echo "   └── Downloading from $DATA_URL..."
        if curl -o "$SAVE_PATH" "$DATA_URL" --fail --silent --show-error; then
            echo "   └── Saving to $SAVE_PATH..."
            echo
            echo "✅ Dataset downloaded successfully!"
            echo
        else
            echo "❌ Error: Download failed."
            exit 1
        fi
    elif [ "$choice" == "n" ] || [ "$choice" == "no" ]; then
        echo "   Exiting pipeline. Please manually place the dataset in the 'data' folder and try again."
        exit 1
    else
        echo "❌ Invalid input. Please run the command again and enter 'y' or 'n'."
        echo "   Exiting pipeline."
        exit 1
    fi
fi

# Convert MODELS array to a space-separated string
MODEL_ARGS=$(IFS=" "; echo "${MODELS[*]}")

# Run the pipeline
if $LITE_MODE; then
    echo "🚀🚀💡 Running pipeline in Lite Mode... (<1 min)"
    echo
    python src/pipeline.py --lite
else
    echo "🚀🚀🚀 Running the machine learning pipeline... (~5 min)"
    echo
    python src/pipeline.py --model $MODEL_ARGS
fi

# Check if the pipeline executed successfully
if [ $? -eq 0 ]; then
    echo "🍻 Pipeline executed successfully!"
else
    echo
    echo "❌ Error: Pipeline execution failed."
    exit 1
fi