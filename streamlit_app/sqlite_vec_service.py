import sqlite3
import json
import os
import logging
import importlib.util
import sqlite_vec

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SqliteVecService:
    """
    A service class to manage vector embeddings using SQLite with the sqlite-vec extension.

    Handles database connection, table creation, vector insertion, KNN search,
    deletion by metadata ID, and vector counting.
    """
    def __init__(self, db_path="vectors.db", table_name="embeddings", dimension=128):
        """
        Initialize the SQLite Vector Service.
        
        Args:
            conn: SQLite connection object. If None, a new connection will be created.
            table_name: Name of the table to store vectors in.
            dimension: The dimension of the vectors to store.
        """
        self.conn = None
        self.db_path = db_path
        self.table_name = table_name
        self.dimension = dimension
        self.is_sqlite_vec_available = False

        
        # If no connection provided, create one
        if self.conn is None:
            self.conn = self._connect_and_initialize()
        
        # Check if sqlite_vec extension is available
        if self.conn:
            try:
                cursor = self.conn.cursor()
                cursor.execute("SELECT vec_version()")
                version = cursor.fetchone()
                logging.info(f"SQLite-vec extension version: {version}")
                self.is_sqlite_vec_available = True
                # Create the vectors table if it doesn't exist
                self._create_vectors_table()
            except sqlite3.OperationalError as e:
                if "no such function: sqlite_vec_version" in str(e).lower():
                    logging.error("SQLite-vec extension not available. Vector search functionality will be disabled.")
                    logging.error("To enable vector search, install sqlite-vec: https://github.com/asg017/sqlite-vec")
                else:
                    logging.error(f"Error initializing SQLiteVecService: {e}")
                self.is_sqlite_vec_available = False
            except Exception as e:
                logging.error(f"Unexpected error initializing SQLiteVecService: {e}")
                self.is_sqlite_vec_available = False

    def _find_extension_path(self):
        """Tries to locate the sqlite-vec extension file."""
        try:
            # Try finding through the package path first
            spec = importlib.util.find_spec("sqlite_vec")
            if spec and spec.origin:
                package_dir = os.path.dirname(spec.origin)
                # Common patterns for extension files within the package
                possible_filenames = ['vec0.dll', 'vec0.so', 'vec0.dylib']
                for name_part in ['build', 'lib', '']: # Check common build/lib subdirs
                    for filename in possible_filenames:
                        potential_path = os.path.join(package_dir, name_part, filename)
                        logging.info(f"Checking potential extension path: {potential_path}")
                        if os.path.exists(potential_path):
                            logging.info(f"Found potential extension at: {potential_path}")
                            return potential_path
        except Exception as e:
            logging.warning(f"Could not find sqlite_vec package location via importlib: {e}")

        # Fallback: check common names in current/parent dirs (less reliable)
        # Determine the directory of the current script
        try:
            current_script_dir = os.path.dirname(__file__)
        except NameError:
            current_script_dir = os.getcwd() # Fallback if __file__ is not defined (e.g., interactive)

        check_dirs = ['.', current_script_dir, os.path.dirname(self.db_path)]
        possible_filenames = ['vec0.dll', 'vec0.so', 'vec0.dylib']
        for check_dir in check_dirs:
             # Ensure check_dir is a valid directory before joining
             if os.path.isdir(check_dir):
                 for filename in possible_filenames:
                      potential_path = os.path.join(check_dir, filename)
                      if os.path.exists(potential_path):
                           logging.info(f"Found potential extension via fallback search: {potential_path}")
                           return potential_path

        logging.warning("Could not automatically locate the sqlite-vec extension file.")
        return None


    def _connect_and_initialize(self):
        """Connects to the database and initializes the sqlite-vec extension and table."""
        try:
            # Ensure the directory for the db exists
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                 os.makedirs(db_dir, exist_ok=True)
                 logging.info(f"Created database directory: {db_dir}")

            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.enable_load_extension(True)

            # --- Load sqlite-vec extension ---
            loaded = False
            try:
                # Preferred method: use the sqlite_vec package loader
                import sqlite_vec
                sqlite_vec.load(self.conn)
                logging.info("sqlite-vec extension loaded successfully using sqlite_vec.load().")
                loaded = True
            except ImportError:
                logging.warning("sqlite_vec python package not found or failed to load. Trying manual load.")
            except Exception as e:
                 logging.warning(f"sqlite_vec.load() failed: {e}. Trying manual load.")

            if not loaded:
                # Attempt manual load if package loading fails
                extension_path = self._find_extension_path()
                if extension_path:
                    try:
                        self.conn.load_extension(extension_path)
                        logging.info(f"sqlite-vec extension loaded manually from {extension_path}.")
                        loaded = True
                    except sqlite3.OperationalError as e:
                        logging.error(f"Failed to load sqlite-vec extension from {extension_path}: {e}")
                else:
                     logging.error("Could not find sqlite-vec extension file to load manually.")

            # Disable extension loading for security, regardless of whether we succeeded
            self.conn.enable_load_extension(False)
            
            # The actual version check will be done in the constructor
            # This function only establishes the connection and attempts to load the extension
            return self.conn

        except sqlite3.Error as e:
            logging.error(f"Database connection or initialization error: {e}")
            if self.conn:
                self.conn.close()
                self.conn = None # Ensure connection is None on failure
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred during initialization: {e}")
            if self.conn:
                self.conn.close()
                self.conn = None
            raise

    def _create_vectors_table(self):
        """Create the vectors table if it doesn't exist."""
        if not self.conn or not self.is_sqlite_vec_available:
            return
            
        try:
            cursor = self.conn.cursor()
            # Using TEXT for metadata, assumed to be JSON.
            create_table_sql = f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {self.table_name} USING vec0(
                vec FLOAT[{self.dimension}],
                metadata TEXT
            );
            """
            cursor.execute(create_table_sql)
            self.conn.commit()
            logging.info(f"Vector table '{self.table_name}' ({self.dimension}d) ensured")
        except sqlite3.Error as e:
            logging.error(f"Error creating vector table: {e}")
            raise

    def add_vector(self, vector: list[float], metadata: dict):
        """
        Adds a vector with associated metadata to the table.

        Args:
            vector (list[float]): The vector embedding. Must match the dimension specified during initialization.
            metadata (dict): Metadata associated with the vector (e.g., {'id': 'some_id', 'source': 'doc1'}).
                             Must be JSON serializable. 'id' key is recommended for deletion/retrieval.

        Returns:
            int: The rowid of the inserted row.

        Raises:
            ConnectionError: If the database connection is not established.
            ValueError: If vector dimension mismatch or metadata is not JSON serializable.
            sqlite3.Error: For database-related errors during insertion.
        """
        if not self.conn or not self.is_sqlite_vec_available:
            raise ConnectionError("Database connection is not established or vector extension not available.")
            
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension mismatch. Expected {self.dimension}, got {len(vector)}.")

        try:
            vector_str = json.dumps(vector)
            metadata_str = json.dumps(metadata) # Ensure metadata is serializable
        except TypeError as e:
            logging.error(f"Metadata is not JSON serializable: {metadata}")
            raise ValueError("Metadata must be JSON serializable.") from e

        try:
            cursor = self.conn.cursor()
            # Using INSERT. Caller should handle duplicate logic if needed
            insert_sql = f"INSERT INTO {self.table_name} (vec, metadata) VALUES (?, ?)"
            cursor.execute(insert_sql, (vector_str, metadata_str))
            self.conn.commit()
            last_row_id = cursor.lastrowid
            logging.debug(f"Vector added (rowid={last_row_id}) with metadata: {metadata}")
            return last_row_id
        except sqlite3.Error as e:
            logging.error(f"Error adding vector: {e}")
            self.conn.rollback()
            raise

    def search_vector(self, query_vector, k=5, metadata_filter=None):
        """
        Search for the k nearest vectors to the query vector.

        Args:
            query_vector (list): The query vector
            k (int): Number of nearest neighbors to return
            metadata_filter (dict, optional): Filter results by metadata values

        Returns:
            list: List of dictionaries containing the search results with metadata and scores
        """
        if not self.conn or not self.is_sqlite_vec_available:
            logging.error("SQLite-vec not available or connection is not established")
            return []

        # Ensure the query vector has the correct dimension
        if len(query_vector) != self.dimension:
            raise ValueError(f"Query vector dimension ({len(query_vector)}) does not match expected dimension ({self.dimension})")

        results = []
        params = []
        
        # Convert the query vector to a string representation (required by sqlite-vec)
        query_vec_str = "[" + ",".join(map(str, query_vector)) + "]"
        params.append(query_vec_str)
        
        # Base query structure
        # Select rowid (often the primary key link), distance, and metadata
        base_query = f"""
            SELECT rowid, distance, metadata
            FROM {self.table_name}
        """
        
        # Build WHERE clause
        where_conditions = []
        
        # KNN condition using MATCH and k (preferred method)
        where_conditions.append(f"vec MATCH ?") # Use column name 'vec'
        where_conditions.append(f"k = ?")
        params.append(k)

        # Add metadata filters if provided
        if metadata_filter:
            for key, value in metadata_filter.items():
                # Use json_extract for filtering JSON metadata
                where_conditions.append(f"json_extract(metadata, '$.{key}') = ?")
                params.append(value)
                
        # Construct the full query
        full_query = f"""
            {base_query}
            WHERE { " AND ".join(where_conditions) }
        """
        
        try:
            cursor = self.conn.cursor()
            
            # Execute the query using k parameter (standard method)
            logging.debug(f"Executing KNN query: {full_query} with params {params}")
            cursor.execute(full_query, tuple(params))
            results = self._process_vector_results(cursor)
            logging.info(f"Vector search successful using k parameter, found {len(results)} results")
            return results
            
        except sqlite3.OperationalError as e:
            # Check if the error is related to the 'k =' syntax (might indicate older SQLite < 3.41)
            # Note: Documentation says LIMIT is only for SQLite 3.41+, k= is preferred.
            # If the primary method fails, we log the error but don't attempt LIMIT fallback here
            # as the k= syntax is the documented standard for vec0 KNN.
            logging.error(f"KNN query failed: {e}. Query: {full_query}, Params: {params}")
            logging.error("Ensure SQLite version is compatible and sqlite-vec extension is correctly loaded.")
            return [] # Return empty list on failure
        except Exception as e:
            logging.error(f"Unexpected error during vector search: {e}")
            return []

    def _process_vector_results(self, cursor):
        """
        Process cursor results into a standardized format
        
        Args:
            cursor: Database cursor with query results
            
        Returns:
            list: Processed results with metadata and normalized scores
        """
        results = []
        rows = cursor.fetchall()
        
        for i, (rowid, distance, metadata_json) in enumerate(rows):
            try:
                metadata = json.loads(metadata_json) if metadata_json else {}
                # Calculate a similarity score (1.0 is perfect match, decreasing as distance increases)
                # For Euclidean distance, we need to convert to a similarity score
                # A common approach is to use 1/(1+distance)
                similarity = 1.0 / (1.0 + float(distance))
                
                result = {
                    "id": rowid,
                    "metadata": metadata,
                    "score": similarity,
                    "distance": distance,
                    "position": i + 1  # 1-based position in results
                }
                results.append(result)
            except Exception as e:
                logging.error(f"Error processing search result: {e}")
                
        return results

    def delete_vector(self, vector_id: str):
        """
        Deletes vectors by their metadata id field.

        Args:
            vector_id (str): The ID value in the metadata to identify the vector(s) to delete.
                              This looks for an 'id' field in the metadata.

        Returns:
            int: Number of vectors deleted.

        Raises:
            ConnectionError: If the database connection is not established.
            sqlite3.Error: For database-related errors during deletion.
        """
        if not self.conn or not self.is_sqlite_vec_available:
            raise ConnectionError("Database connection is not established or vector extension not available.")

        try:
            cursor = self.conn.cursor()
            
            # First, count how many will be deleted for the return value
            count_sql = f"""
            SELECT COUNT(*) FROM {self.table_name}
            WHERE json_extract(metadata, '$.id') = ?
            """
            cursor.execute(count_sql, (vector_id,))
            count_result = cursor.fetchone()
            count = count_result[0] if count_result else 0
            
            if count == 0:
                logging.warning(f"No vectors found with id '{vector_id}' to delete.")
                return 0
            
            # Perform the deletion
            delete_sql = f"""
            DELETE FROM {self.table_name}
            WHERE json_extract(metadata, '$.id') = ?
            """
            cursor.execute(delete_sql, (vector_id,))
            self.conn.commit()
            
            logging.info(f"Deleted {count} vector(s) with id '{vector_id}'")
            return count
        except sqlite3.Error as e:
            logging.error(f"Error deleting vector(s) with id '{vector_id}': {e}")
            self.conn.rollback()
            raise

    def get_vector_by_id(self, vector_id: str):
        """
        Retrieves a vector by its metadata id field.

        Args:
            vector_id (str): The ID value in the metadata to identify the vector.

        Returns:
            dict or None: Dictionary with 'vector', 'metadata', and 'rowid' if found, 
                         None if no matching vector is found.

        Raises:
            ConnectionError: If the database connection is not established.
            sqlite3.Error: For database-related errors during retrieval.
        """
        if not self.conn or not self.is_sqlite_vec_available:
            raise ConnectionError("Database connection is not established or vector extension not available.")

        try:
            cursor = self.conn.cursor()
            
            select_sql = f"""
            SELECT rowid, vec, metadata
            FROM {self.table_name}
            WHERE json_extract(metadata, '$.id') = ?
            LIMIT 1
            """
            cursor.execute(select_sql, (vector_id,))
            result = cursor.fetchone()
            
            if not result:
                return None
                
            rowid, vector_str, metadata_str = result
            
            try:
                vector = json.loads(vector_str)
            except json.JSONDecodeError:
                logging.error(f"Error parsing vector JSON for id '{vector_id}'")
                return None
                
            try:
                metadata = json.loads(metadata_str) if metadata_str else {}
            except json.JSONDecodeError:
                logging.warning(f"Error parsing metadata JSON for id '{vector_id}'")
                metadata = {"_raw": metadata_str}
            
            return {
                "rowid": rowid,
                "vector": vector,
                "metadata": metadata
            }
        except sqlite3.Error as e:
            logging.error(f"Error retrieving vector with id '{vector_id}': {e}")
            raise

    def count_vectors(self):
        """
        Counts the total number of vectors in the table.

        Returns:
            int: The count of vectors.

        Raises:
            ConnectionError: If the database connection is not established.
            sqlite3.Error: For database-related errors during counting.
        """
        if not self.conn:
            raise ConnectionError("Database connection is not established.")
            
        # If the extension isn't available, return 0 as there can't be any vectors
        if not self.is_sqlite_vec_available:
            return 0

        try:
            cursor = self.conn.cursor()
            count_sql = f"SELECT COUNT(*) FROM {self.table_name}"
            cursor.execute(count_sql)
            result = cursor.fetchone()
            return result[0] if result else 0
        except sqlite3.Error as e:
            logging.error(f"Error counting vectors: {e}")
            raise

    def close(self):
        """Closes the database connection."""
        if self.conn:
            try:
                self.conn.close()
                logging.info(f"Closed database connection to {self.db_path}")
            except Exception as e:
                logging.error(f"Error closing database connection: {e}")
            finally:
                self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close() 