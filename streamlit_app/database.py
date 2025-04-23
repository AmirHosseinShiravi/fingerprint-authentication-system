import sqlite3
import json
import os
import numpy as np
from streamlit_app.sqlite_vec_service import SqliteVecService
import logging

class UserDatabase:
    # def __init__(self, db_path="streamlit_app/user_db.sqlite", vector_db_path="streamlit_app/vector_store.db"):
    def __init__(self, db_path="streamlit_app/user_db.sqlite", vector_db_path="streamlit_app/vector_store.db"):
        """Initialize the user database."""
        self.db_path = db_path
        self.conn = None
        self.vector_service = None
        
        # Initialize database connection
        try:
            self._connect_and_initialize()
        except Exception as e:
            logging.error(f"Failed to initialize database: {e}")
            print(f"Error: Database initialization failed: {e}")
        
        # Try to initialize vector service
        try:
            self.vector_service = SqliteVecService(db_path=vector_db_path, table_name="fingerprints", dimension=128)
        except Exception as e:
            logging.error(f"Failed to initialize vector service: {e}")
            print(f"Warning: Vector similarity search will not be available. Error: {e}")

    def _connect_and_initialize(self):
        """Connect to the SQLite database and initialize tables if needed."""
        # Ensure directory exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            
        # Connect to database
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        # Enable foreign key constraints
        self.conn.execute("PRAGMA foreign_keys = ON")
        cursor = self.conn.cursor()
        
        # Create users table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            first_name TEXT,
            last_name TEXT,
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create fingerprints table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS fingerprints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            image_path TEXT NOT NULL,
            vector_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
        """)
        
        self.conn.commit()

    def _ensure_connection(self):
        """Ensure database connection is established."""
        if self.conn is None:
            try:
                self._connect_and_initialize()
            except Exception as e:
                raise ConnectionError(f"Could not establish database connection: {e}")
        return self.conn

    def add_user(self, username, first_name, last_name, email):
        """Add a new user to the database."""
        conn = self._ensure_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO users (username, first_name, last_name, email) VALUES (?, ?, ?, ?)",
                (username, first_name, last_name, email)
            )
            conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            raise ValueError(f"User with username '{username}' already exists")

    def update_user(self, user_id, username=None, first_name=None, last_name=None, email=None):
        """Update user information."""
        conn = self._ensure_connection()
        cursor = conn.cursor()
        
        # Get existing user data
        cursor.execute("SELECT username, first_name, last_name, email FROM users WHERE id = ?", (user_id,))
        existing = cursor.fetchone()
        
        if not existing:
            raise ValueError(f"User with ID {user_id} not found")
            
        # Use existing values if new values are not provided
        username = username if username is not None else existing[0]
        first_name = first_name if first_name is not None else existing[1]
        last_name = last_name if last_name is not None else existing[2]
        email = email if email is not None else existing[3]
        
        cursor.execute(
            "UPDATE users SET username = ?, first_name = ?, last_name = ?, email = ? WHERE id = ?",
            (username, first_name, last_name, email, user_id)
        )
        conn.commit()
        return user_id

    def delete_user(self, user_id):
        """Delete a user and their fingerprints."""
        conn = self._ensure_connection()
        cursor = conn.cursor()
        
        # Get vector IDs to delete from vector database
        cursor.execute("SELECT vector_id FROM fingerprints WHERE user_id = ?", (user_id,))
        vector_ids = [row[0] for row in cursor.fetchall() if row[0]]
        
        # Delete user
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        
        # fingerprints will be deleted automatically due to ON DELETE CASCADE
        
        # Remove vectors from vector database
        if self.vector_service:
            for vector_id in vector_ids:
                try:
                    self.vector_service.delete_vector(vector_id)
                except Exception as e:
                    print(f"Error deleting vector {vector_id}: {e}")
        
        conn.commit()

    def get_user(self, user_id):
        """Get user information by ID."""
        conn = self._ensure_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, username, first_name, last_name, email, created_at FROM users WHERE id = ?",
            (user_id,)
        )
        user = cursor.fetchone()
        
        if not user:
            return None
            
        return {
            "id": user[0],
            "username": user[1],
            "first_name": user[2],
            "last_name": user[3],
            "email": user[4],
            "created_at": user[5]
        }

    def get_all_users(self):
        """Get all users."""
        try:
            conn = self._ensure_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, username, first_name, last_name, email, created_at FROM users"
            )
            users = []
            for user in cursor.fetchall():
                users.append({
                    "id": user[0],
                    "username": user[1],
                    "first_name": user[2],
                    "last_name": user[3],
                    "email": user[4],
                    "created_at": user[5]
                })
            return users
        except Exception as e:
            print(f"Error fetching users: {e}")
            return []

    def add_fingerprint(self, user_id, image_path, feature_vector):
        """Add a fingerprint for a user."""
        # Store vector in the vector database
        vector_id = f"user_{user_id}_{os.path.basename(image_path)}"
        
        # Add to vector database if available
        if self.vector_service:
            try:
                metadata = {
                    "id": vector_id,
                    "user_id": user_id
                }
                
                self.vector_service.add_vector(
                    vector=feature_vector.flatten().tolist(),
                    metadata=metadata
                )
            except Exception as e:
                print(f"Warning: Could not add vector to vector database: {e}")
        
        # Store reference in SQL database
        conn = self._ensure_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO fingerprints (user_id, image_path, vector_id) VALUES (?, ?, ?)",
            (user_id, image_path, vector_id)
        )
        conn.commit()
        
        return cursor.lastrowid

    def get_user_fingerprints(self, user_id):
        """Get all fingerprints for a user."""
        conn = self._ensure_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, image_path, vector_id, created_at FROM fingerprints WHERE user_id = ?",
            (user_id,)
        )
        fingerprints = []
        for fp in cursor.fetchall():
            fingerprints.append({
                "id": fp[0],
                "image_path": fp[1],
                "vector_id": fp[2],
                "created_at": fp[3]
            })
        return fingerprints

    def delete_fingerprint(self, fingerprint_id):
        """Delete a fingerprint."""
        conn = self._ensure_connection()
        cursor = conn.cursor()
        
        # Get vector ID
        cursor.execute("SELECT vector_id FROM fingerprints WHERE id = ?", (fingerprint_id,))
        result = cursor.fetchone()
        
        if not result:
            raise ValueError(f"Fingerprint with ID {fingerprint_id} not found")
            
        vector_id = result[0]
        
        # Delete from SQL database
        cursor.execute("DELETE FROM fingerprints WHERE id = ?", (fingerprint_id,))
        
        # Delete from vector database
        if self.vector_service and vector_id:
            try:
                self.vector_service.delete_vector(vector_id)
            except Exception as e:
                print(f"Error deleting vector {vector_id}: {e}")
        
        conn.commit()

    def search_similar_fingerprint(self, feature_vector, threshold=0.75):
        """Search for similar fingerprints."""
        if not self.vector_service:
            print("Vector service not available. Cannot perform similarity search.")
            return []
            
        try:
            # Flatten feature vector and search
            flat_vector = feature_vector.flatten().tolist()
            results = self.vector_service.search_vector(flat_vector, k=5)
            
            # Filter results by similarity threshold
            filtered_results = []
            for result in results:
                if result["score"] >= threshold:
                    # Get user details
                    user_id = result["metadata"]["user_id"]
                    user = self.get_user(user_id)
                    
                    if user:
                        filtered_results.append({
                            "user": user,
                            "similarity": result["score"]
                        })
                        
            return filtered_results
        except Exception as e:
            print(f"Error searching for similar fingerprints: {e}")
            return []
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            try:
                self.conn.close()
                self.conn = None
            except Exception as e:
                print(f"Error closing database connection: {e}")
        
        if self.vector_service:
            try:
                self.vector_service.close()
            except Exception as e:
                print(f"Error closing vector service: {e}")
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def __del__(self):
        self.close() 