# vector_fns.py
import os
import re
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Union, Tuple, Optional
import datetime
import uuid

# --- Vector Classes ---
class Vector:
    def __init__(self, vector_id: str, vector_type: str, metadata: Dict):
        self.vector_id = vector_id
        self.vector_type = vector_type
        self.metadata = metadata
        self.embedding = None

class TextVector(Vector):
    def __init__(self, vector_id: str, content: str, metadata: Dict):
        super().__init__(vector_id, "Text", metadata)
        self.content = content

class ScoreVector(Vector):
    def __init__(self, vector_id: str, score: Union[int, float, str], metadata: Dict):
        super().__init__(vector_id, "Score", metadata)
        self.score = score

class ParameterVector(Vector):
    def __init__(self, vector_id: str, parameter_name: str, value: Union[str, float, int, None], units: Optional[str], metadata: Dict):
        super().__init__(vector_id, "Parameter", metadata)
        self.parameter_name = parameter_name
        self.value = value
        self.units = units

class TableVector(Vector):
    def __init__(self, vector_id: str, headers: List[str], rows: List[Dict], metadata: Dict):
        super().__init__(vector_id, "Table", metadata)
        self.headers = headers
        self.rows = rows

class BinaryVector(Vector):
     def __init__(self, vector_id: str, value: str, metadata: Dict):
        super().__init__(vector_id, "Binary", metadata)
        self.value = value

class RelationshipVector(Vector):
    def __init__(self, vector_id: str, relationship_type: str, source_vector_id: str, target_vector_id: str, metadata: Dict):
        super().__init__(vector_id, "Relationship", metadata)
        self.relationship_type = relationship_type
        self.source_vector_id = source_vector_id
        self.target_vector_id = target_vector_id

class TemplateInstanceVector(Vector):
    def __init__(self, vector_id: str, sections: List[Vector], metadata: Dict):
        super().__init__(vector_id, "TemplateInstance", metadata)
        self.sections = sections

class SectionVector(Vector):
    def __init__(self, vector_id: str, subsections: List[Vector], metadata: Dict): #ADDED subsections
        super().__init__(vector_id, "Section", metadata)
        self.subsections = subsections

class SubsectionVector(Vector):
    def __init__(self, vector_id: str, content: List[Vector], metadata: Dict):
        super().__init__(vector_id, "Subsection", metadata)
        self.content = content


# --- Helper Functions ---

def generate_unique_id():  return str(uuid.uuid4())


def create_metadata(bibtex_data: Dict, paper_type: str, bibtex_key: str, section_id: str = None, subsection_id: str =None, gin_node_type=None, gin_edge_type=None):
    metadata = {
        "id": generate_unique_id(),  # Use a UUID for the ID *within* metadata
        "date": bibtex_data.get('year', str(datetime.date.today().year)),
        "title": bibtex_data.get('title', ''),
        "paper_type": paper_type,
        "section_id": section_id,
        "subsection_id": subsection_id,
        'gin_node_type': gin_node_type,
        'gin_edge_type': gin_edge_type,
        "bibtex_key": bibtex_key,
    }
    return metadata


def create_text_vector(content: str, bibtex_data, paper_type, section_id, subsection_id, bibtex_key, gin_node_type=None, gin_edge_type=None):
    vector_id = f"{bibtex_key}_text_{section_id}_{subsection_id}_{gin_node_type}_{uuid.uuid4()}"
    return TextVector(vector_id, content, create_metadata(bibtex_data, paper_type, bibtex_key, section_id, subsection_id, gin_node_type, gin_edge_type))


def create_score_vector(score: Union[int, float, str], bibtex_data, paper_type, section_id, subsection_id, bibtex_key, gin_node_type=None, gin_edge_type=None):
    vector_id = f"{bibtex_key}_score_{section_id}_{subsection_id}_{gin_node_type}_{uuid.uuid4()}"
    return ScoreVector(vector_id, score, create_metadata(bibtex_data, paper_type, bibtex_key, section_id, subsection_id, gin_node_type, gin_edge_type))


def create_parameter_vector(parameter_name: str, value: Union[str, float, int, None], units: Optional[str], bibtex_data, paper_type, section_id, subsection_id, bibtex_key, gin_node_type=None, gin_edge_type=None):
    vector_id = f"{bibtex_key}_param_{section_id}_{subsection_id}_{gin_node_type}_{uuid.uuid4()}"
    return ParameterVector(vector_id, parameter_name, value, units, create_metadata(bibtex_data, paper_type, bibtex_key, section_id, subsection_id, gin_node_type, gin_edge_type))


def create_table_vector(headers: List[str], rows: List[Dict], bibtex_data, paper_type, section_id, subsection_id, bibtex_key, gin_node_type=None, gin_edge_type=None):
    vector_id = f"{bibtex_key}_table_{section_id}_{subsection_id}_{gin_node_type}_{uuid.uuid4()}"
    return TableVector(vector_id, headers, rows, create_metadata(bibtex_data, paper_type, bibtex_key, section_id, subsection_id, gin_node_type, gin_edge_type))


def create_binary_vector(value: str, bibtex_data, paper_type, section_id, subsection_id, bibtex_key, gin_node_type=None, gin_edge_type=None):
    vector_id = f"{bibtex_key}_binary_{section_id}_{subsection_id}_{gin_node_type}_{uuid.uuid4()}"
    return BinaryVector(vector_id, value, create_metadata(bibtex_data, paper_type, bibtex_key, section_id, subsection_id, gin_node_type, gin_edge_type))


def create_relationship_vector(relationship_type: str, source_vector_id: str, target_vector_id: str, bibtex_data, paper_type, bibtex_key, section_id=None, subsection_id=None, gin_edge_type=None):
    # Determine gin_edge_type based on relationship_type if not provided
    if gin_edge_type is None:
        gin_edge_type = relationship_type  # Default to relationship_type

    vector_id = f"{bibtex_key}_rel_{relationship_type}_{source_vector_id}_{target_vector_id}_{uuid.uuid4()}"
    return RelationshipVector(vector_id, relationship_type, source_vector_id, target_vector_id, create_metadata(bibtex_data, paper_type, bibtex_key, section_id, subsection_id, gin_edge_type=gin_edge_type))


def create_template_instance_vector(sections: List[SectionVector], bibtex_data, paper_type, bibtex_key, gin_node_type=None):
    vector_id = f"{bibtex_key}_templateInstance_{uuid.uuid4()}"
    return TemplateInstanceVector(vector_id, sections, create_metadata(bibtex_data, paper_type, bibtex_key, gin_node_type=gin_node_type))


def create_section_vector(subsections: List[SubsectionVector], bibtex_data, paper_type, section_id, bibtex_key, gin_node_type=None):
    vector_id = f"{bibtex_key}_section_{section_id}_{uuid.uuid4()}"
    return SectionVector(vector_id, subsections, create_metadata(bibtex_data, paper_type, bibtex_key, section_id, gin_node_type=gin_node_type))


def create_subsection_vector(content: List[Vector], bibtex_data, paper_type, section_id, subsection_id, bibtex_key, gin_node_type=None, gin_edge_type=None):
    vector_id = f"{bibtex_key}_subsection_{section_id}_{subsection_id}_{gin_node_type}_{uuid.uuid4()}"
    return SubsectionVector(vector_id, content, create_metadata(bibtex_data, paper_type, bibtex_key, section_id, subsection_id, gin_node_type, gin_edge_type))

def save_to_pickle(data: List[TemplateInstanceVector], filename: str):
    filepath = f".embedding_cache/{filename}.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(data, f)

def load_from_pickle(filename: str) -> List[TemplateInstanceVector]:
    filepath = f".embedding_cache/{filename}.pkl"
    with open(filepath, "rb") as f:
        return pickle.load(f)


# --- Embedding Functions ---

def get_embedding_model(model_name: str = "allenai/scibert_scivocab_uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

def embed_text(text: str, tokenizer, model, max_length: int = 512) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embedding
