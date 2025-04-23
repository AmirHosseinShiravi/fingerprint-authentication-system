import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import tempfile
import time
import sys
import glob
import logging # Add logging
import shutil # <--- Add import for shutil

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the parent directory to sys.path to import from project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from streamlit_app.fingerprint_model import FingerprintModel
from streamlit_app.database import UserDatabase

# Initialize session state for storing temporary data
if 'temp_fingerprints' not in st.session_state:
    st.session_state.temp_fingerprints = []
if 'edit_user' not in st.session_state:
    st.session_state.edit_user = None
if 'vector_search_available' not in st.session_state:
    st.session_state.vector_search_available = False
# Add state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Constants
UPLOAD_FOLDER = "uploaded_fingerprints"
FINGERPRINT_LIMIT = 5
SIMILARITY_THRESHOLD = 0.75
MODEL_PATH = "training_model/triplet_checkpoints/best_model.pth"  # Update with actual path

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the fingerprint model
@st.cache_resource
def load_model():
    try:
        model = FingerprintModel(model_path=MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Failed to load fingerprint model: {str(e)}")
        return None

# Initialize database connection
@st.cache_resource # Cache the DB connection resource
def get_database():
    """Get or create a database connection."""
    try:
        db = UserDatabase()
        # Update session state with vector search availability
        if 'vector_search_available' not in st.session_state: # Initialize if not present
             st.session_state.vector_search_available = (db.vector_service is not None and 
                                                       db.vector_service.is_sqlite_vec_available)
        return db
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        return None

# Main function
def main():
    # Set page config
    st.set_page_config(
        page_title="Fingerprint Authentication System",
        page_icon="👆",
        layout="wide"
    )

    # Header
    st.title("Fingerprint Authentication System")
    
    # Initialize model and database
    model = load_model()
    db = get_database()
    
    if db is None:
        st.error("❌ Failed to initialize database. Please check the logs for details.")
        st.info("Try restarting the application or running the troubleshooting script: `python fix_sqlite_vec.py`")
        return
    
    # Update vector search availability status on each run
    st.session_state.vector_search_available = (db.vector_service is not None and 
                                                   db.vector_service.is_sqlite_vec_available)
    
    # Show warning if vector search is not available
    if not st.session_state.vector_search_available:
        st.warning("⚠️ Vector similarity search is not available. Authentication functionality will be limited.")
    
    # Store previous state for cleanup checks
    prev_page = st.session_state.page
    prev_edit_user_state = st.session_state.get('edit_user') # Capture before state modification

    # --- Sidebar Navigation --- 
    with st.sidebar:
        st.title("Navigation")
        nav_options = ["Home", "All Users", "Add User"]
        page_changed_by_sidebar = False

        # Create buttons for navigation
        if st.button("🏠 Home", key="nav_home", use_container_width=True):
            if st.session_state.page != "Home":
                st.session_state.page = "Home"
                page_changed_by_sidebar = True
        
        if st.button("👥 All Users", key="nav_all_users", use_container_width=True):
            if st.session_state.page != "All Users":
                 st.session_state.page = "All Users"
                 page_changed_by_sidebar = True

        if st.button("➕ Add User", key="nav_add_user", use_container_width=True):
             if st.session_state.page != "Add User":
                 st.session_state.page = "Add User"
                 page_changed_by_sidebar = True
        
        # Optionally highlight the active page (can be done with more complex styling or components)
        # For simplicity, we just rely on the button click to change state.

    # --- Cleanup Logic --- 
    # Check if the main page navigation was changed by a sidebar button click
    if page_changed_by_sidebar:
        # If the main page changed WHILE we were previously in the edit state, clear the edit state
        if prev_edit_user_state is not None:
            logging.info("Navigated away from Edit User page via sidebar, clearing edit state.")
            st.session_state.edit_user = None 
        
        # If navigating *away* from Add User page (page changed), clear temp fingerprints
        if prev_page == "Add User":
             if st.session_state.temp_fingerprints:
                 logging.info("Navigating away from Add User page, clearing temporary fingerprints.")
                 # Clean up the actual temp files (code omitted for brevity)
                 for temp_path in st.session_state.temp_fingerprints:
                     if os.path.exists(temp_path):
                         try:
                             os.remove(temp_path)
                         except OSError as e:
                             logging.warning(f"Could not remove temp file {temp_path} on navigation: {e}")
                 st.session_state.temp_fingerprints = []
        
        # Rerun the script to apply the navigation change
        st.rerun()

    # --- Page Routing --- 
    # (Routing logic remains the same, prioritizing edit_user state)
    if st.session_state.get('edit_user') is not None:
        edit_user_page(model, db)
    elif st.session_state.page == "Home":
        home_page(model, db)
    elif st.session_state.page == "All Users":
        users_page(model, db)
    elif st.session_state.page == "Add User":
        add_user_page(model, db)
    else:
        # Fallback if state is somehow invalid
        st.warning("Invalid page state, redirecting to Home.")
        st.session_state.page = "Home"
        st.rerun()

def home_page(model, db):
    """Home page with fingerprint authentication."""
    st.header("Fingerprint Authentication")
    
    if not model:
        st.error("Fingerprint model could not be loaded. Authentication is not available.")
        return
        
    if not st.session_state.vector_search_available:
        st.info("Vector similarity search is not available. You can still add users and fingerprints, but fingerprint matching functionality is limited.")
    
    st.write("Upload a fingerprint image to authenticate a user.")
    
    # --- Layout for uploader and button --- 
    col_uploader, col_button = st.columns([2, 1]) # Adjusted proportions

    uploaded_file = None # Initialize
    with col_uploader:
        # File uploader
        uploaded_file = st.file_uploader("Upload fingerprint image", type=["jpg", "jpeg", "png", "bmp"], label_visibility="collapsed")
    
    # --- Check if uploaded file changed and clear previous results --- 
    # Use file ID for comparison; reset if file is removed (uploaded_file is None)
    current_file_id = uploaded_file.file_id if uploaded_file else None
    if 'processed_file_id' not in st.session_state:
        st.session_state.processed_file_id = None

    if current_file_id != st.session_state.processed_file_id:
        # New file uploaded or existing file removed, clear previous results
        logging.info(f"File uploader changed (new ID: {current_file_id}, old ID: {st.session_state.processed_file_id}). Clearing auth results.")
        st.session_state.auth_results = None
        st.session_state.auth_error = None
        st.session_state.auth_duration = None
        # Update the ID of the file currently shown by the uploader
        st.session_state.processed_file_id = current_file_id 

    with col_button:
         # Button to perform authentication - disabled until file is uploaded
         authenticate_disabled = uploaded_file is None
         if st.button("Authenticate", disabled=authenticate_disabled, use_container_width=True):
             # Clear previous results (redundant due to above check, but safe) 
             # and start timer
             st.session_state.auth_results = None 
             st.session_state.auth_error = None
             st.session_state.auth_duration = None 
             start_time = time.perf_counter() 
             st.session_state.processed_file_id = uploaded_file.file_id # Mark this file ID as processed

             if not st.session_state.vector_search_available:
                 st.error("Authentication is not available without vector search functionality.")
                 st.info("Please check the errors in the terminal and ensure sqlite-vec is properly installed.")
                 st.session_state.auth_error = "Vector search not available."
                 # Still need to record duration even if we exit early
                 end_time = time.perf_counter()
                 st.session_state.auth_duration = end_time - start_time
             else:
                 with st.spinner("Processing fingerprint..."):
                     temp_path = None
                     try:
                         # --- Feature Extraction and Search --- 
                         with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                             temp_path = temp_file.name
                             with open(temp_path, "wb") as f:
                                 f.write(uploaded_file.getvalue())
                         
                         features = model.extract_features(temp_path)
                         results = db.search_similar_fingerprint(features, SIMILARITY_THRESHOLD)
                         # --- End of core processing --- 
                         
                         if results:
                             st.session_state.auth_results = results # Store results
                             st.success(f"✅ Authentication Successful! Match found.") # Show immediate feedback
                         else:
                             st.error("❌ Authentication Failed: No matching fingerprint found")
                             st.session_state.auth_error = "No match found."
                     
                     except Exception as e:
                         st.error(f"Error processing fingerprint: {str(e)}")
                         logging.error(f"Fingerprint processing error: {e}", exc_info=True)
                         st.session_state.auth_error = f"Processing error: {e}"
                     finally:
                         # --- Stop timer and cleanup --- 
                         end_time = time.perf_counter() # Stop timer
                         st.session_state.auth_duration = end_time - start_time # Store duration
                         
                         if temp_path and os.path.exists(temp_path):
                             try:
                                 os.remove(temp_path)
                             except OSError as e_rem:
                                  logging.warning(f"Could not remove temp file {temp_path}: {e_rem}")

    st.divider()

    # --- Display Area: Uploaded Image and Authentication Results --- 
    if uploaded_file is not None:
         col_img, col_details = st.columns(2)
         
         with col_img:
             st.image(uploaded_file, caption="Uploaded Fingerprint", use_column_width=False)
         
         with col_details:
             # Display results or error stored in session state
             auth_results = st.session_state.get('auth_results')
             auth_error = st.session_state.get('auth_error')
             auth_duration = st.session_state.get('auth_duration')

             if auth_results:
                 top_match = auth_results[0]
                 user = top_match["user"]
                 similarity = top_match["similarity"]
                 with st.container(border=True):
                     st.write(f"#### Match Details")
                     st.write(f"**Name:** {user['first_name']} {user['last_name']}")
                     st.write(f"**Username:** {user['username']}")
                     st.write(f"**Email:** {user['email']}")
                     st.write(f"**Similarity Score:** {similarity:.2f}")
             elif auth_error:
                 st.error(f"Authentication failed: {auth_error}")
             else:
                 # If file is uploaded but button not pressed or processing underway
                 st.info("Click 'Authenticate' to process the fingerprint.")

             # Display duration if available
             if auth_duration is not None:
                 st.caption(f"Processing time: {auth_duration:.3f} seconds")

def users_page(model, db):
    """Page to display and manage users."""
    st.header("All Users")
    
    # Fetch all users
    try:
        users = db.get_all_users()
    except Exception as e:
        st.error(f"Failed to fetch users: {e}")
        logging.error("Failed to fetch users", exc_info=True)
        users = []
    
    if not users:
        st.info("No users found. Add users from the 'Add User' page.")
    else:
        # Display users in a grid of cards
        num_columns = 3
        cols = st.columns(num_columns) # LEVEL 1 nesting (main grid)
        for i, user in enumerate(users):
            with cols[i % num_columns]: # Inside one of the main grid columns
                # Use an expander for each user card for better organization
                with st.expander(f"{user['first_name']} {user['last_name']} ({user['username']})", expanded=False):
                    st.markdown(f"**Email:** {user['email']}")
                    st.caption(f"User ID: {user['id']}")
                    
                    # --- Buttons for edit and delete --- (LEVEL 2 nesting)
                    edit_btn_col, delete_btn_col = st.columns(2)
                    with edit_btn_col:
                        # Edit User Button - JUST sets edit_user state
                        if st.button("✏️ Edit", key=f"edit_{user['id']}"):
                            st.session_state.edit_user = user # Store user info for editing
                            # NO page state change here
                            # NO rerun here - let the main routing handle it
                    with delete_btn_col:
                        # Delete User Button - Use a unique key for the button
                        if st.button("🗑️ Delete", key=f"delete_btn_{user['id']}"):
                            # Set state to show confirmation section below
                            st.session_state[f'confirm_delete_{user["id"]}'] = True 
                            st.rerun() # Re-run to show confirmation

                    # --- Confirmation Section --- 
                    # Placed directly under expander, NOT inside the delete_btn_col
                    if st.session_state.get(f'confirm_delete_{user["id"]}'):
                        st.warning(f"⚠️ Are you sure you want to delete {user['username']}?")
                        
                        # Confirmation Buttons (LEVEL 2 nesting - allowed here)
                        confirm_yes_col, confirm_no_col = st.columns(2)
                        with confirm_yes_col:
                            if st.button("✅ Yes, Delete", key=f"confirm_yes_{user['id']}"):
                                try:
                                    db.delete_user(user['id'])
                                    st.success(f"User {user['username']} deleted successfully!")
                                    # Clean up confirmation state
                                    del st.session_state[f'confirm_delete_{user["id"]}'] 
                                    time.sleep(1.5) # Give time to read success message
                                    st.rerun() # Refresh user list
                                except Exception as e:
                                    st.error(f"Failed to delete user {user['username']}: {e}")
                                    logging.error(f"Failed to delete user {user['id']}", exc_info=True)
                                    # Clean up confirmation state even on error
                                    if f'confirm_delete_{user["id"]}' in st.session_state:
                                        del st.session_state[f'confirm_delete_{user["id"]}'] 
                                    st.rerun()
                        with confirm_no_col:
                            if st.button("❌ No, Cancel", key=f"confirm_no_{user['id']}"):
                                # Just clean up confirmation state and rerun
                                if f'confirm_delete_{user["id"]}' in st.session_state:
                                    del st.session_state[f'confirm_delete_{user["id"]}'] 
                                st.rerun()

def add_user_page(model, db):
    """Page to add a new user."""
    st.header("➕ Add New User")
    
    if not model and not st.session_state.vector_search_available:
        st.warning("⚠️ Vector search is not available and model couldn't be loaded. You can add users but fingerprint features won't be stored properly.")

    # User information form
    with st.form("add_user_form", clear_on_submit=True): # Clear form on successful submission
        # --- Top row within form for Save button --- 
        _, col_save = st.columns([4, 1]) # Adjust ratio as needed
        
        with col_save:
             # Apply type="primary" for emphasis (color depends on theme)
             submitted = st.form_submit_button("💾 Save User", use_container_width=True, type="primary")

        username = st.text_input("Username *", value="", key="add_username")
        first_name = st.text_input("First Name", value="", key="add_firstname")
        last_name = st.text_input("Last Name", value="", key="add_lastname")
        email = st.text_input("Email", value="", key="add_email")
        
        # submitted = st.form_submit_button("💾 Save User")
        
        if submitted:
            if not username:
                st.error("Username is required.")
            else:
                try:
                    # Add new user
                    user_id = db.add_user(username, first_name, last_name, email)
                    st.success(f"User '{username}' (ID: {user_id}) added successfully! Proceeding to add fingerprints.")
                    
                    # Process any temporary fingerprints if model is available
                    if model and st.session_state.temp_fingerprints:
                        st.write("Adding associated fingerprints...")
                        added_count = 0
                        processed_temp_paths = list(st.session_state.temp_fingerprints) # Copy list for iteration
                        st.session_state.temp_fingerprints = [] # Clear state immediately

                        for temp_path in processed_temp_paths:
                            if not os.path.exists(temp_path):
                                logging.warning(f"Temp file {temp_path} missing during save, skipping.")
                                continue
                            try:
                                # Create a more permanent-like path structure
                                perm_dir = os.path.join(UPLOAD_FOLDER, str(user_id))
                                os.makedirs(perm_dir, exist_ok=True)
                                # Use timestamp for unique filenames
                                base, ext = os.path.splitext(os.path.basename(temp_path))
                                ts = int(time.time() * 1000) # Milliseconds for more uniqueness
                                perm_filename = f"{username}_{base}_{ts}{ext}"
                                perm_path = os.path.join(perm_dir, perm_filename)
                                
                                # Move the file using shutil.move
                                logging.info(f"Moving temp file {temp_path} to {perm_path}")
                                shutil.move(temp_path, perm_path) # <--- USE SHUTIL.MOVE
                                
                                # Extract features and add to database
                                features = model.extract_features(perm_path)
                                db.add_fingerprint(user_id, perm_path, features)
                                added_count += 1
                                st.write(f"  - Added: {os.path.basename(perm_path)}")
                            except Exception as e_fp:
                                st.error(f"Error processing fingerprint {os.path.basename(temp_path)}: {str(e_fp)}")
                                logging.error(f"Error processing fingerprint {temp_path} for user {user_id}", exc_info=True)
                                # Attempt to clean up temp file if move failed and it still exists
                                if os.path.exists(temp_path):
                                     try:
                                         os.remove(temp_path)
                                         logging.info(f"Cleaned up temp file {temp_path} after move error.")
                                     except OSError as e_rem:
                                          logging.warning(f"Could not clean up temp file {temp_path} after error: {e_rem}")
                        
                        if added_count > 0:
                             st.success(f"Added {added_count} fingerprints for user '{username}'.")
                        
                        # Clear temporary fingerprints list (already done above)
                        # st.session_state.temp_fingerprints = [] 
                        
                        # Navigate to user list after successful add + fingerprint processing
                        time.sleep(2) # Give time to read messages
                        st.session_state.page = "All Users"
                        st.rerun()
                    
                    elif not st.session_state.temp_fingerprints:
                         st.info("No fingerprints were added. You can add them later by editing the user.")
                         # Navigate to user list after successful add 
                         time.sleep(2)
                         st.session_state.page = "All Users"
                         st.rerun()

                except ValueError as ve:
                     st.error(f"Error saving user: {ve}") # Handle username exists error
                except Exception as e:
                    st.error(f"Error saving user: {str(e)}")
                    logging.error(f"Failed to add user {username}", exc_info=True)

    # --- Fingerprint upload section (common for add/edit, but handled within the form) ---
    # This section is now outside the form, allows adding fingerprints *before* saving user
    if model:
        st.subheader("Add Fingerprints (Optional)")
        st.write(f"You can pre-upload up to {FINGERPRINT_LIMIT} fingerprint images before saving the user.")
        
        # Method selection
        upload_method = st.radio("Upload Method", ["Upload Individual Images", "Upload Multiple Images"], key="add_upload_method")
        
        if upload_method == "Upload Individual Images":
            # Individual upload - provide a unique key that changes to help reset
            # Using a simple counter in session state might work
            if 'add_uploader_counter' not in st.session_state:
                 st.session_state.add_uploader_counter = 0
                 
            uploaded_file = st.file_uploader("Upload fingerprint image", 
                                             type=["jpg", "jpeg", "png", "bmp"], 
                                             key=f"add_uploader_single_{st.session_state.add_uploader_counter}")
            
            if uploaded_file is not None:
                # Check fingerprint limit
                if len(st.session_state.temp_fingerprints) >= FINGERPRINT_LIMIT:
                    st.warning(f"Maximum of {FINGERPRINT_LIMIT} fingerprints allowed. Delete some to add more.")
                else:
                    try:
                        # Save the uploaded image to a temporary file with unique name
                        with tempfile.NamedTemporaryFile(prefix=f"fp_{username if username else 'new'}_", suffix=".jpg", delete=False) as temp_file:
                            temp_path = temp_file.name
                            # Save the uploaded file to disk
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getvalue())
                            
                            # Add to temporary fingerprints
                            st.session_state.temp_fingerprints.append(temp_path)
                            st.success(f"Fingerprint ready to be added! ({len(st.session_state.temp_fingerprints)}/{FINGERPRINT_LIMIT})")
                            # Increment counter to change key for next time, resetting uploader
                            st.session_state.add_uploader_counter += 1 
                            st.rerun() # Rerun AFTER incrementing key and processing
                    except Exception as e_upload:
                         st.error(f"Error processing uploaded file: {e_upload}")
                         logging.error("Error during single file upload processing", exc_info=True)

        else: # Bulk upload
            # Similar key strategy for multi-uploader if needed, but often less problematic
            if 'multi_uploader_counter' not in st.session_state:
                 st.session_state.multi_uploader_counter = 0
                 
            uploaded_files = st.file_uploader("Upload multiple fingerprint images", 
                                            type=["jpg", "jpeg", "png", "bmp"], 
                                            accept_multiple_files=True, 
                                            key=f"add_uploader_multi_{st.session_state.multi_uploader_counter}")
            
            if uploaded_files:
                 # Process immediately upon selection
                 num_uploaded = len(uploaded_files)
                 available_slots = FINGERPRINT_LIMIT - len(st.session_state.temp_fingerprints)
                 files_to_process = uploaded_files[:available_slots]
                 num_to_process = len(files_to_process)
                 
                 if num_uploaded > available_slots:
                     st.warning(f"You selected {num_uploaded}, but only {available_slots} slots are free. Adding the first {available_slots} images.")
                 
                 added_count = 0
                 if num_to_process > 0:
                     with st.spinner(f"Processing {num_to_process} images..."):
                         for u_file in files_to_process:
                             try:
                                 # Save the uploaded image to a temporary file
                                 with tempfile.NamedTemporaryFile(prefix=f"fp_multi_{username if username else 'new'}_", suffix=".jpg", delete=False) as temp_file:
                                     temp_path = temp_file.name
                                     # Save the uploaded file to disk
                                     with open(temp_path, "wb") as f:
                                         f.write(u_file.getvalue())
                                     
                                     # Add to temporary fingerprints
                                     st.session_state.temp_fingerprints.append(temp_path)
                                     added_count += 1
                             except Exception as e_multi:
                                 st.error(f"Error processing file {u_file.name}: {e_multi}")
                                 logging.error("Error during multi file upload processing", exc_info=True)
                     
                     if added_count > 0:
                         st.success(f"{added_count} fingerprints ready to be added! ({len(st.session_state.temp_fingerprints)}/{FINGERPRINT_LIMIT})")
                         # Increment counter to change key for next time, resetting uploader
                         st.session_state.multi_uploader_counter += 1 
                         st.rerun() # Rerun AFTER incrementing key and processing
        
        # Display current temporary fingerprints
        if st.session_state.temp_fingerprints:
            st.subheader("Fingerprints Pending Save")
            
            # Create a grid of images
            num_cols_display = 5
            cols_display = st.columns(min(num_cols_display, len(st.session_state.temp_fingerprints)))
            
            indices_to_remove = []
            for i, temp_path in enumerate(st.session_state.temp_fingerprints):
                 # Check if path still exists before displaying
                 if not os.path.exists(temp_path):
                     logging.warning(f"Temp file {temp_path} at index {i} missing, marking for removal.")
                     indices_to_remove.append(i)
                     continue # Skip display if file is gone
                     
                 with cols_display[i % num_cols_display]:
                     try:
                         # Display image
                         img = Image.open(temp_path)
                         st.image(img, width=100, caption=f"FP {i+1}")
                         
                         # Delete button for this temp fingerprint
                         if st.button(f"❌ Remove", key=f"delete_temp_fp_{i}"):
                             # Mark for removal
                             indices_to_remove.append(i)
                     except FileNotFoundError:
                         # Should be caught by the check above, but handle defensively
                         st.error("Temp file missing!")
                         indices_to_remove.append(i) # Mark for removal
                     except Exception as e_disp:
                          st.error(f"Error displaying: {e_disp}")
                          # Don't remove just because display failed, but log it
                          logging.error(f"Error displaying temp file {temp_path}: {e_disp}", exc_info=True)
            
            # Remove marked fingerprints outside the loop
            if indices_to_remove:
                # Sort indices in reverse to avoid issues when removing
                indices_to_remove.sort(reverse=True)
                updated_temp_fingerprints = []
                original_paths = list(st.session_state.temp_fingerprints) # Get a copy
                
                # Build the new list excluding the removed indices
                for i, path in enumerate(original_paths):
                    if i not in indices_to_remove:
                        updated_temp_fingerprints.append(path)
                    else:
                         # If marked for removal, try deleting the file
                         if os.path.exists(path):
                             try:
                                 os.remove(path)
                                 logging.info(f"Removed temporary file: {path}")
                             except OSError as e_rem_fp:
                                 logging.warning(f"Error removing temp fingerprint file {path}: {e_rem_fp}")
                                 
                st.session_state.temp_fingerprints = updated_temp_fingerprints
                st.rerun() # Rerun after removal

# New function for editing users
def edit_user_page(model, db):
    """Page to edit an existing user."""
    st.header("✏️ Edit User")

    # --- Back Button --- 
    if st.button("⬅️ Back to All Users", key="edit_back_button"):
        st.session_state.edit_user = None 
        st.session_state.page = "All Users"
        st.rerun()
    
    st.divider()

    if st.session_state.edit_user is None:
        st.error("No user selected for editing.")
        return

    user = st.session_state.edit_user

    # --- User information form for editing --- 
    with st.form("edit_user_form"):
        # --- Top row within form for Save button --- 
        col_info, col_save = st.columns([4, 1]) # Adjust ratio as needed
        with col_info:
             st.caption(f"Editing User ID: {user['id']}") # Display ID on the left
        with col_save:
             # Apply type="primary" for emphasis (color depends on theme)
             submitted = st.form_submit_button("💾 Save Changes", use_container_width=True, type="primary")

        # --- Form fields below the top row ---
        username = st.text_input("Username *", value=user['username'], key="edit_username")
        first_name = st.text_input("First Name", value=user['first_name'], key="edit_firstname")
        last_name = st.text_input("Last Name", value=user['last_name'], key="edit_lastname")
        email = st.text_input("Email", value=user['email'], key="edit_email")
        
        # Process submission (logic remains the same)
        if submitted:
            if not username:
                st.error("Username is required.")
            else:
                try:
                    db.update_user(
                        user_id=user['id'],
                        username=username,
                        first_name=first_name,
                        last_name=last_name,
                        email=email
                    )
                    st.success(f"User '{username}' updated successfully!")
                    # Clear edit state and navigate back to user list
                    st.session_state.edit_user = None 
                    st.session_state.page = "All Users"
                    time.sleep(1.5)
                    st.rerun()
                except ValueError as ve:
                     st.error(f"Error updating user: {ve}") # Handle username exists error
                except Exception as e:
                    st.error(f"Error updating user: {str(e)}")
                    logging.error(f"Failed to update user {user['id']}", exc_info=True)

    # --- Fingerprint Management Section --- 
    st.subheader("Manage Fingerprints")
    try:
        existing_fingerprints = db.get_user_fingerprints(user['id'])
        st.write(f"Current fingerprints: {len(existing_fingerprints)} / {FINGERPRINT_LIMIT}")
    except Exception as e_get_fp:
         st.error(f"Error loading fingerprints: {e_get_fp}")
         existing_fingerprints = []

    # Display existing fingerprints with delete buttons
    if existing_fingerprints:
        num_cols_disp = 5
        cols_disp = st.columns(min(num_cols_disp, len(existing_fingerprints)))
        
        fingerprints_to_delete_later = [] # Store IDs to delete outside the display loop if needed

        for i, fp in enumerate(existing_fingerprints):
            with cols_disp[i % num_cols_disp]:
                image_path_to_display = fp['image_path']
                try:
                    # Read image bytes to avoid holding file lock
                    if os.path.exists(image_path_to_display):
                        with open(image_path_to_display, 'rb') as f:
                            image_bytes = f.read()
                        st.image(image_bytes, width=100, caption=f"FP {i+1}")
                    else:
                        st.error("Image file missing!")
                        fingerprints_to_delete_later.append(fp['id']) # Mark DB record for deletion
                        continue # Skip button if image missing

                    # Delete Button for this Fingerprint
                    if st.button("🗑️ Delete FP", key=f"delete_fp_{fp['id']}"):
                        try:
                            image_path_to_delete = fp['image_path'] # Get path again for deletion
                            # 1. Delete from Database (handles vector deletion too)
                            db.delete_fingerprint(fp['id'])
                            logging.info(f"Deleted fingerprint record {fp['id']} from DB.")
                            
                            # 2. Delete the image file
                            if os.path.exists(image_path_to_delete):
                                os.remove(image_path_to_delete)
                                logging.info(f"Deleted image file {image_path_to_delete}.")
                            else:
                                 logging.warning(f"Image file {image_path_to_delete} not found during delete attempt, but DB record removed.")
                                 
                            st.success(f"Fingerprint {i+1} deleted.")
                            time.sleep(0.5) # Shorter delay
                            st.rerun()
                        except Exception as e_del_fp:
                             st.error(f"Error deleting fingerprint {fp['id']}: {e_del_fp}")
                             logging.error(f"Failed to delete fingerprint {fp['id']}", exc_info=True)
                             
                except FileNotFoundError:
                    # This case handles if the file disappears between the check and open
                    st.error(f"Image not found at {image_path_to_display}. Marking record for deletion...")
                    fingerprints_to_delete_later.append(fp['id'])
                except Exception as e_disp_fp:
                     st.error(f"Error displaying fingerprint {i+1}: {e_disp_fp}")
                     logging.error(f"Error displaying fingerprint {fp['id']}", exc_info=True)
        
        # Process deletions for missing files outside the loop
        if fingerprints_to_delete_later:
             logging.warning(f"Attempting to delete DB records for missing fingerprint files: {fingerprints_to_delete_later}")
             deleted_count = 0
             for fp_id in fingerprints_to_delete_later:
                 try:
                     db.delete_fingerprint(fp_id)
                     deleted_count += 1
                 except Exception as e_del_missing:
                      st.error(f"Error deleting DB record for missing FP ID {fp_id}: {e_del_missing}")
             if deleted_count > 0:
                 st.rerun() # Rerun if records were deleted

    # Add new fingerprints section (only if limit not reached and model available)
    if model and len(existing_fingerprints) < FINGERPRINT_LIMIT:
        st.subheader("Add New Fingerprint")
        uploaded_file = st.file_uploader("Upload new fingerprint image", 
                                         type=["jpg", "jpeg", "png", "bmp"], 
                                         key="edit_uploader_single")
        
        if uploaded_file is not None:
            if st.button("Add This Fingerprint", key="edit_add_fp_btn"):
                with st.spinner("Adding fingerprint..."):
                    try:
                        # Save to permanent location
                        perm_dir = os.path.join(UPLOAD_FOLDER, str(user['id']))
                        os.makedirs(perm_dir, exist_ok=True)
                        # Avoid filename clashes with timestamp
                        base, ext = os.path.splitext(uploaded_file.name)
                        ts = int(time.time())
                        perm_filename = f"{user['username']}_{base}_{ts}{ext}"
                        perm_path = os.path.join(perm_dir, perm_filename)
                        
                        with open(perm_path, "wb") as f:
                             f.write(uploaded_file.getvalue())
                        
                        # Extract features and add to database
                        features = model.extract_features(perm_path)
                        db.add_fingerprint(user['id'], perm_path, features)
                        st.success(f"Fingerprint added successfully!")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e_add_fp:
                         st.error(f"Error adding fingerprint: {e_add_fp}")
                         logging.error(f"Failed to add fingerprint for user {user['id']}", exc_info=True)
    elif len(existing_fingerprints) >= FINGERPRINT_LIMIT:
        st.info("Maximum number of fingerprints reached.")

if __name__ == "__main__":
    main() 