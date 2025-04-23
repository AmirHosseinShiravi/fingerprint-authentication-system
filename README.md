# Fingerprint Authentication System

## Overview

This project implements a fingerprint identification system (1:N matching) using a deep learning model and a Streamlit-based user interface. The system allows users to register, store multiple fingerprint images, and authenticate by comparing an input fingerprint against the database. It leverages a custom Convolutional Neural Network (CNN) trained with Triplet Loss on the challenging SOCOFing dataset to achieve high accuracy even with altered or poor-quality fingerprints. For efficient similarity search, it utilizes the `sqlite-vec` extension, providing a balance between performance and deployment simplicity for this scale.

## Features

*   **Fingerprint Enrollment:** Register new users with basic information (username, name, email).
*   **Multi-Fingerprint Storage:** Each user can store up to 5 fingerprint images.
*   **Fingerprint Authentication (1:N):** Upload a fingerprint image on the home page. The system extracts features using the trained model and searches the database for the closest match above a similarity threshold (0.75).
*   **User Management:** View all registered users.
*   **Edit User Information:** Modify user details (username, name, email).
*   **Manage Fingerprints:** Add new fingerprints or delete existing ones for a user via the edit page.
*   **Robust Matching:** Thanks to the model trained on the SOCOFing dataset, the system can handle variations and alterations (easy, medium, hard) in fingerprint images.

## Dataset: SOCOFing

The model was trained on the **SOCOFing (Sokoto Coventry Fingerprint Dataset)**. This dataset is particularly useful for training robust models because it includes:
*   **Real Fingerprints:** A baseline set of clean fingerprint images.
*   **Altered Fingerprints:** Corresponding fingerprints with various augmentations applied, categorized into:
    *   Altered-Easy
    *   Altered-Medium
    *   Altered-Hard

This challenging dataset ensures the trained model learns features resilient to noise, rotation, occlusion, and other real-world imperfections often found in fingerprint images. The dataset contains images from 600 subjects, totaling ~55,000 images across real and altered categories.

## Model: Custom CNN with Triplet Loss

A custom Convolutional Neural Network (CNN) was designed and trained to generate discriminative fingerprint embeddings. Initial experiments with pre-trained ResNet18 and ResNet50 backbones did not yield satisfactory performance on this specific task. The custom architecture achieved better results.

*   **Architecture:** The model consists of several convolutional layers with ReLU activations and Max Pooling, followed by fully connected layers. An embedding dimension of 128 was chosen after experimentation; this relatively low dimension helps optimize inference speed and reduces the computational cost of the vector similarity search in the database, which is crucial for production-like performance. (See `training_model/model.py` and the training notebook for exact architectural details).

    ```
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1           [-1, 16, 94, 94]             160
                  ReLU-2           [-1, 16, 94, 94]               0
             MaxPool2d-3           [-1, 16, 31, 31]               0
                Conv2d-4           [-1, 32, 29, 29]           4,640
                  ReLU-5           [-1, 32, 29, 29]               0
             MaxPool2d-6             [-1, 32, 9, 9]               0
               Flatten-7                 [-1, 2592]               0
                Linear-8                  [-1, 256]         663,808
                  ReLU-9                  [-1, 256]               0
               Linear-10                  [-1, 384]          98,688
                 ReLU-11                  [-1, 384]               0
               Linear-12                  [-1, 128]          49,280
    ================================================================
    Total params: 816,576
    Trainable params: 816,576
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.04
    Forward/backward pass size (MB): 2.74
    Params size (MB): 3.11
    Estimated Total Size (MB): 5.89
    ----------------------------------------------------------------
    ```
*   **Training:**
    *   **Loss Function:** Triplet Margin Loss (margin=0.25) was used to learn embeddings where fingerprints from the same person are pulled closer in the embedding space, while fingerprints from different people are pushed farther apart.
    *   **Epochs:** 40
    *   **Optimizer:** AdamW
    *   **Scheduler:** ReduceLROnPlateau
    *   **Framework:** PyTorch
    *   **Dataset Split:** SOCOFing was split into 70% training, 15% validation, and 15% test sets based on unique finger instances.
*   **Performance:** The trained model achieves excellent performance on the test set:
    *   **Overall Rank-1 Accuracy (All Test Data):** ~97.47%
    ![](/training_model/result_images/retrieval_performance_over_test_dataset_all.png)
    
    *   **Rank-1 Accuracy (Real Images Test Set):** 100.00%
    ![](/training_model/result_images/retrieval_performance_over_real_test_subset.png)
    
    *   **Rank-1 Accuracy (Altered-Easy Test Set):** ~99.00%
    ![](/training_model/result_images/retrieval_performance_over_easy_test_subset.png)
    
    *   **Rank-1 Accuracy (Altered-Medium Test Set):** ~97.65%
    ![](/training_model/result_images/retrieval_performance_over_medium_test_subset.png)

    *   **Rank-1 Accuracy (Altered-Hard Test Set):** ~94.25%
    ![](/training_model/result_images/retrieval_performance_over_hard_test_subset.png)

    <!-- *   **TAR@FAR=0.1%:** > 98% (True Accept Rate at 0.1% False Accept Rate) -->
    <!-- *   (Refer to `training_model/advanced_cv_project_1_training_model_code_final.ipynb` for detailed evaluation plots and metrics like ROC curves and score distributions). -->

The best performing model checkpoint is saved at `training_model/triplet_checkpoints/best_model.pth`.

## Technology Stack

*   **Backend/ML:** Python, PyTorch, NumPy, OpenCV-Python
*   **Frontend:** Streamlit
*   **Database:** SQLite
*   **Vector Search:** `sqlite-vec` extension (chosen for simplicity and speed in this context over alternatives like Milvus which require separate server setups).
*   **Notebook Environment:** Jupyter/Google Colab (for training)

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd fingerprint-authentication-system
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    # Activate the environment
    # Windows (Command Prompt/PowerShell)
    .\.venv\Scripts\activate
    # Linux/macOS
    # source .venv/bin/activate
    ```
3.  **Install dependencies for the Streamlit application:**
    ```bash
    pip install -r streamlit_requirements.txt
    ```
    <!-- *Note: This includes `sqlite-vec`. If you encounter issues with `sqlite-vec` installation, refer to its documentation or run the provided troubleshooting script: `python fix_sqlite_vec.py`* -->

4.  **(Optional) Install dependencies for model training/evaluation:**
    If you want to retrain the model or run the evaluation notebook, install dependencies from the training folder:
    ```bash
    pip install -r training_model/requirements.txt
    ```

## Usage

<!-- 1.  **Ensure `sqlite-vec` is working:** The application relies on the `sqlite-vec` extension. If it's the first run or if you encounter database errors, you might need to run the helper script:
    ```bash
    python fix_sqlite_vec.py
    ``` -->
1.  **Run the Streamlit application:**
    ```bash
    streamlit run streamlit_app/app.py
    ```
    Alternatively, use the provided helper script:
    ```bash
    python run_streamlit.py
    ```
2.  **Navigate the UI:**
    *   **Home:** Upload a fingerprint for authentication. Results (matched user or failure) will be displayed.
    *   **Add User:** Fill in user details and optionally upload up to 5 fingerprint images.
    *   **All Users:** View registered users. Click "Edit" to modify details or manage fingerprints. Click "Delete" to remove a user.


## Screen Record

[screen_record1.webm](https://github.com/user-attachments/assets/3edbef87-2c37-4868-ab56-9d8739df8ea3)

<!-- ## Potential Improvements / Future Work

*   Explore more advanced model architectures (e.g., Vision Transformers).
*   Implement fingerprint enhancement preprocessing steps.
*   Integrate with a more scalable vector database solution (e.g., Milvus, Qdrant) if the user base grows significantly.
*   Add more robust error handling and user feedback in the UI.
*   Implement liveness detection to prevent spoofing attacks. -->

## Credits

This project was developed as part of an Advanced Computer Vision course.

## Prerequisites

<!-- - Docker -->
<!-- - Docker Compose -->
- NVIDIA GPU with CUDA support (for optimal performance)

## License

MIT License
