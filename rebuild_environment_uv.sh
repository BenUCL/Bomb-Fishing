#!/bin/bash

# Script to rebuild the bomb audio environment using uv
# This script will create a new virtual environment with compatible versions

echo "=== Bomb Audio Environment Rebuild Script (UV) ==="
echo ""

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed or not in PATH"
    echo "Install uv with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Ask user which environment to create
echo "Choose which environment to create:"
echo "1. Old stack (more stable, TensorFlow 2.17 + AutoKeras 1.1.4)"
echo "2. New stack (future-proof, TensorFlow 2.19 + AutoKeras 2.2.0)"
echo ""
read -p "Enter your choice (1 or 2): " choice

case $choice in
    1)
        REQUIREMENTS_FILE="requirements_old_stack.txt"
        ENV_NAME="bomb-audio-old-stack"
        echo "Creating old stack environment..."
        ;;
    2)
        REQUIREMENTS_FILE="requirements_new_stack.txt"
        ENV_NAME="bomb-audio-new-stack"
        echo "Creating new stack environment..."
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "Step 1: Creating new virtual environment..."
uv venv .venv

echo ""
echo "Step 2: Installing dependencies from $REQUIREMENTS_FILE..."
uv pip install -r "$REQUIREMENTS_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Environment created successfully!"
    echo ""
    echo "Step 3: Activating the environment..."
    echo "To activate the environment, run:"
    echo "source .venv/bin/activate"
    echo ""
    echo "Step 4: Testing the installation..."
    echo "After activation, you can test with:"
    echo "python test_environment.py"
    echo ""
    echo "Step 5: Clean up AutoKeras cache (if needed):"
    echo "rm -rf ./autokeras_model_friday"
    echo ""
    echo "Step 6: Install Jupyter kernel (optional):"
    echo "python -m ipykernel install --user --name=$ENV_NAME --display-name='Bomb Audio ($ENV_NAME)'"
    echo ""
    echo "🎉 Environment rebuild complete!"
    echo ""
    echo "Quick start:"
    echo "source .venv/bin/activate"
    echo "python test_environment.py"
else
    echo ""
    echo "❌ Failed to create environment. Check the error messages above."
    exit 1
fi 