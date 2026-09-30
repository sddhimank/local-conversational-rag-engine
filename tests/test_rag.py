import os
import pytest
from unittest.mock import MagicMock, patch
from app.ingest import get_relative_source_path

def test_get_relative_source_path_root():
    """Validates that root prefix folders are stripped accurately to support nested configurations."""
    assert get_relative_source_path("./data/strategy.txt") == "strategy.txt"
    assert get_relative_source_path("/workspace/data/sub/test_log.txt") == "sub/test_log.txt"
    assert get_relative_source_path("upload/20260930/elevoc.txt") == "20260930/elevoc.txt"

def test_get_relative_source_path_no_match():
    """Validates fallback path handling when standard storage root prefixes are missing."""
    assert get_relative_source_path("raw_filename.txt") == "raw_filename.txt"

@patch('app.ingest.create_engine')
def test_delete_vectors_for_file_execution(mock_create_engine):
    """Validates that the database deletion engine processes and executes SQL queries cleanly."""
    from app.ingest import delete_vectors_for_file
    
    # Set up mocks for the SQLAlchemy engine connection lifecycle
    mock_engine = MagicMock()
    mock_conn = MagicMock()
    mock_create_engine.return_value = mock_engine
    mock_engine.begin.return_value.__enter__.return_value = mock_conn
    
    # Trigger the function under test
    delete_vectors_for_file("./data/sub/test_log.txt")
    
    # Verify the database execution commands ran precisely
    mock_engine.begin.assert_called_once()
    mock_conn.execute.assert_called_once()
    
    # Validate the parameter binding logic to ensure no SQL syntax errors occur
    args, kwargs = mock_conn.execute.call_args

    # Extract structural components safely from position-based argument blocks
    sql_query_string = str(args[0])
    bind_parameters_dict = args[1] if len(args) > 1 else kwargs

    assert "DELETE FROM langchain_pg_embedding" in str(args[0])
    # ✅ FIX: Evaluates the dictionary parameters securely out of the positional routing array
    assert bind_parameters_dict["source_suffix"] == "%test_log.txt"
