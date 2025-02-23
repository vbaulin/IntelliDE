#debug.py Loads and tests the vector database. Outputs the structure.
import pickle
import os
from typing import Dict, Any, List, Union, Optional

# Import vector classes directly from vector_fns.py
from vector_fns import (
    Vector, TextVector, ScoreVector, ParameterVector, TableVector,
    BinaryVector, RelationshipVector, TemplateInstanceVector, SectionVector,
    SubsectionVector
)

VECTOR_CACHE_FILE = os.path.join(".embedding_cache", "vectors_cache.pkl")

def print_vector_structure(data: Union[Dict[str, 'TemplateInstanceVector'], List[Vector], Vector], indent: int = 0):
    """Recursively prints the structure of the loaded vector data."""
    prefix = "  " * indent

    if isinstance(data, dict):
        print(f"{prefix}Dictionary with {len(data)} items:")
        for key, value in data.items():
            print(f"{prefix}  Key: {key}")
            print_vector_structure(value, indent + 2)

    elif isinstance(data, list):
        print(f"{prefix}List with {len(data)} items:")
        for i, item in enumerate(data):
            print(f"{prefix}  Item {i+1}:")
            print_vector_structure(item, indent + 2)

    elif isinstance(data, TemplateInstanceVector):
        print(f"{prefix}TemplateInstanceVector (ID: {data.vector_id}, Type: {data.vector_type})")
        print(f"{prefix}  Metadata: {data.metadata}")
        if hasattr(data, 'embedding') and data.embedding is not None:
            print(f"{prefix}  Embedding (first 5 elements): {data.embedding[:5]}...")
        print_vector_structure(data.sections, indent + 2)

    elif isinstance(data, SectionVector):
        print(f"{prefix}SectionVector (ID: {data.vector_id}, Type: {data.vector_type})")
        print(f"{prefix}  Metadata: {data.metadata}")
        if hasattr(data, 'embedding') and data.embedding is not None:
             print(f"{prefix}  Embedding (first 5 elements): {data.embedding[:5]}...")
        print_vector_structure(data.subsections, indent + 2)

    elif isinstance(data, SubsectionVector):
        print(f"{prefix}SubsectionVector (ID: {data.vector_id}, Type: {data.vector_type})")
        print(f"{prefix}  Metadata: {data.metadata}")
        if hasattr(data, 'embedding') and data.embedding is not None:
            print(f"{prefix}  Embedding (first 5 elements): {data.embedding[:5]}...")
        print_vector_structure(data.content, indent + 2)

    elif isinstance(data, TextVector):
        print(f"{prefix}TextVector (ID: {data.vector_id}, Type: {data.vector_type})")
        print(f"{prefix}  Content: {data.content[:100]}...")  # Truncate content
        print(f"{prefix}  Metadata: {data.metadata}")
        if hasattr(data, 'embedding') and data.embedding is not None:
            print(f"{prefix}  Embedding (first 5 elements): {data.embedding[:5]}...")

    elif isinstance(data, ScoreVector):
        print(f"{prefix}ScoreVector (ID: {data.vector_id}, Type: {data.vector_type})")
        print(f"{prefix}  Score: {data.score}")
        print(f"{prefix}  Metadata: {data.metadata}")

    elif isinstance(data, ParameterVector):
        print(f"{prefix}ParameterVector (ID: {data.vector_id}, Type: {data.vector_type})")
        print(f"{prefix}  Parameter Name: {data.parameter_name}")
        print(f"{prefix}  Value: {data.value}")
        print(f"{prefix}  Units: {data.units}")
        print(f"{prefix}  Metadata: {data.metadata}")

    elif isinstance(data, TableVector):
        print(f"{prefix}TableVector (ID: {data.vector_id}, Type: {data.vector_type})")
        print(f"{prefix}  Headers: {data.headers}")
        print(f"{prefix}  Rows ({len(data.rows)}):")
        if data.rows:
            print(f"{prefix}    First Row: {data.rows[0]}")
        print(f"{prefix}  Metadata: {data.metadata}")

    elif isinstance(data, BinaryVector):
        print(f"{prefix}BinaryVector (ID: {data.vector_id}, Type: {data.vector_type})")
        print(f"{prefix}  Value: {data.value}")
        print(f"{prefix}  Metadata: {data.metadata}")

    elif isinstance(data, RelationshipVector):
        print(f"{prefix}RelationshipVector (ID: {data.vector_id}, Type: {data.vector_type})")
        print(f"{prefix}  Relationship Type: {data.relationship_type}")
        print(f"{prefix}  Source Vector ID: {data.source_vector_id}")
        print(f"{prefix}  Target Vector ID: {data.target_vector_id}")
        print(f"{prefix}  Metadata: {data.metadata}")

    else:
        print(f"{prefix}Unknown Vector Type: {type(data)}")

def main():
    """Loads the vector cache and prints its structure."""

    if not os.path.exists(VECTOR_CACHE_FILE):
        print(f"Cache file not found: {VECTOR_CACHE_FILE}")
        return

    try:
        with open(VECTOR_CACHE_FILE, "rb") as f:
            loaded_data: Dict[str, TemplateInstanceVector] = pickle.load(f)
    except Exception as e:
        print(f"Error loading cache: {e}")
        return

    print("----- Cache Structure -----")
    print_vector_structure(loaded_data)

if __name__ == "__main__":
    main()
