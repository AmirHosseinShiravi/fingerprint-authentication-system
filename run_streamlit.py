import os
import sys
import streamlit.web.cli as stcli

# Get path to the app.py file
def main():
    dirname = os.path.dirname(__file__)
    app_path = os.path.join(dirname, "streamlit_app", "app.py")
    
    if not os.path.exists(app_path):
        print(f"Error: Could not find application at {app_path}")
        sys.exit(1)
    
    # Create required directories
    os.makedirs(os.path.join(dirname, "streamlit_app"), exist_ok=True)
    os.makedirs(os.path.join(dirname, "uploaded_fingerprints"), exist_ok=True)
    
    # Ensure the model directory exists
    checkpoints_dir = os.path.join(dirname, "training_model", "triplet_checkpoints")
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir, exist_ok=True)
        print("Warning: Model checkpoint directory created, but model file may be missing.")
        print(f"Please ensure the model file exists in {checkpoints_dir}")
    
    # Launch Streamlit
    sys.argv = ["streamlit", "run", app_path, "--server.port=8501", "--server.address=0.0.0.0"]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main() 