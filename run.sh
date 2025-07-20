#!/bin/bash

# Function to display usage instructions
usage() {
    echo "ℹ️  Usage: bash run.sh [--lite] [--debug classification|regression] [--model lr|rf|xgb|lgbm]"
    echo "          --lite                          : Run the pipeline in lite mode (quick testing and debugging)."
    echo "          --debug [task]                  : Run in debug mode with task type. Options: ['classification', 'regression')."
    echo "          --model [model name]            : Specify single model to run. Options: ['lr', 'rf', 'xgb', 'lgbm']"
    exit 1
}

# Parse arguments
DEBUG_MODE=false
TASK=""
LITE_MODE=false
MODELS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --lite)
            LITE_MODE=true
            shift
            ;;
        --debug)
            DEBUG_MODE=true
            if [[ -n "$2" && "$2" != --* ]]; then
                if [[ "$2" != "classification" && "$2" != "regression" ]]; then
                    echo "❌ Error: Invalid debug task '$2'. Choose 'classification' or 'regression'."
                    usage
                fi
                DEBUG_TASK="$2"
                shift 2
            else
                echo "❌ Error: Missing task after '--debug'."
                usage
            fi
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

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Configuration file '$CONFIG_FILE' not found."
    exit 1
fi

# Extract data paths from YAML
DATA_URL=$(grep '^data_url:' "$CONFIG_FILE" | awk '{print $2}' | sed 's/"//g')
SAVE_PATH=$(grep '^db_path:' "$CONFIG_FILE" | awk '{print $2}' | sed 's/"//g')

if [ -z "$DATA_URL" ] || [ -z "$SAVE_PATH" ]; then
    echo "❌ Error: Failed to extract DATA_URL or SAVE_PATH from '$CONFIG_FILE'."
    exit 1
fi

# Download dataset if missing
if [ ! -f "$SAVE_PATH" ]; then
    echo "⚠️  Dataset '$SAVE_PATH' not found."
    read -p "   Do you want to download the dataset automatically? (y/n): " choice
    choice=$(echo "$choice" | tr '[:upper:]' '[:lower:]')
    if [[ "$choice" == "y" || "$choice" == "yes" ]]; then
        echo "   └── Creating directory..."
        mkdir -p "$(dirname "$SAVE_PATH")"
        echo "   └── Downloading from $DATA_URL..."
        if curl -o "$SAVE_PATH" "$DATA_URL" --fail --silent --show-error; then
            echo "   ✅  Dataset downloaded successfully!"
            echo
        else
            echo "❌ Error: Download failed."
            exit 1
        fi
    else
        echo "   Exiting. Please place dataset manually in 'data' folder."
        exit 1
    fi
fi

# Determine main Python script to run
PYTHON_SCRIPT="src/pipeline.py"

# Determine mode of execution
if $DEBUG_MODE; then
    echo "🧪 Debug mode enabled via CLI. Running debug pipeline..."
    python "$PYTHON_SCRIPT" --debug "$DEBUG_TASK"
elif $LITE_MODE; then
    echo "⚡ LITE mode enabled via CLI. Running LITE pipeline..."
    python "$PYTHON_SCRIPT" --lite
else
    echo "🚀 Running the pipeline..."
    MODEL_ARGS=$(IFS=" "; echo "${MODELS[*]}")  # Convert MODELS array to space-separated string
    python "$PYTHON_SCRIPT" ${MODEL_ARGS:+--model $MODEL_ARGS}
fi

# Final check
if [ $? -eq 0 ]; then
    echo
    echo "🍻 Pipeline executed successfully!"
else
    echo
    echo "❌ Error: Pipeline execution failed."
    exit 1
fi