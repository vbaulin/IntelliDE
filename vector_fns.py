# vector_fns.py
import os
import re
import pickle
import torch
import logging
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Union, Tuple, Optional, Pattern  # Added Pattern
import datetime
import uuid
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # For stop words
import nltk
from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')

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
        self.publication_vector = None  # ADDED THIS LINE - Essential fix!

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

def build_vector_hierarchy(publication_data: Dict, parsed_markdown: Dict, scibert_model, scibert_tokenizer, pub_key: str) -> Optional[TemplateInstanceVector]:
    """Builds the hierarchical structure, adding embeddings, metadata, and filtering for M1-M10 sections."""
    logger = logging.getLogger(__name__)
    logger.info(f"Building vector hierarchy for: {pub_key}")

    paper_type = publication_data.get("paper_type", "Unknown")
    bibtex_data = publication_data.get("bibtex_data", {})

    # Create TemplateInstanceVector
    template_instance = create_template_instance_vector([], bibtex_data, paper_type, pub_key, gin_node_type="template_instance")
    template_instance.metadata = create_metadata(bibtex_data, paper_type, bibtex_key=pub_key)
    template_instance.metadata['bibtex_key'] = pub_key

    # --- Publication-level TextVector ---
    logger.debug(f"  Creating publication_vector for: {pub_key}")
    pub_content = publication_data.get("content")
    if not pub_content:
        logger.error(f"  ❌ No content found for publication: {pub_key} in publication_data.  Cannot create publication_vector.")
        return None

    pub_embedding = embed_text(pub_content, scibert_tokenizer, scibert_model)
    pub_text_vector = create_text_vector(pub_content, bibtex_data, paper_type, None, None, pub_key, gin_node_type="publication")
    pub_text_vector.embedding = pub_embedding.tolist() if pub_embedding is not None else None
    template_instance.publication_vector = pub_text_vector
    template_instance.metadata['text_vector_id'] = pub_text_vector.vector_id #the id of the vector
    logger.debug(f"  ✅ publication_vector created and assigned for: {pub_key}")

    # --- Section, Subsection, and Content Vectors ---
    sections_list: List[SectionVector] = []
    for section_key, section_data in parsed_markdown.items():
        logger.debug(f"  Processing section: {section_key}")

        # --- M1-M10 Filtering ---
        match = re.match(r"^M(\d+)(:|$)", section_key)
        if not match:
            logger.debug(f"    Skipping section (not M1-M10): {section_key}")
            continue  # Skip to the next section
        section_number = int(match.group(1))
        if not (1 <= section_number <= 10):
            logger.debug(f"    Skipping section (section number out of range): {section_key}")
            continue
        # --- CONSTRUCT SECTION TEXT (Concatenate Subsection Texts) --- #GET TEXT FROM SUBSECTIONS
        section_text = ""
        for subsection_key, subsection_data in section_data.get("subsections", {}).items():
            section_text += subsection_data.get("text", "")

        if not section_text.strip():
            logger.warning(f"    ⚠️ Section {section_key} has empty text content.")

        section_vector = create_section_vector([], bibtex_data, paper_type, section_key, pub_key, gin_node_type="section")
        section_vector.embedding = embed_text(section_text, scibert_tokenizer, scibert_model).tolist()
        section_vector.metadata['original_text'] = section_text  # Store the *combined* section text
        # Add section-level metadata
        section_vector.metadata.update(section_data.get("metadata", {})) #update section metadata


        subsections_list: List[SubsectionVector] = []
        for subsection_key, subsection_data in section_data.get("subsections", {}).items():
            logger.debug(f"    Processing subsection: {subsection_key}")
            subsection_text = subsection_data.get("text", "") #get text from subsection
            if not subsection_text.strip():
                logger.warning(f"      ⚠️ Subsection {subsection_key} has empty text content.")

            subsection_vector = create_subsection_vector([], bibtex_data, paper_type, section_key, subsection_key, pub_key, gin_node_type="subsection")
            subsection_vector.embedding = embed_text(subsection_text, scibert_tokenizer, scibert_model).tolist()

            subsection_vector.metadata['original_text'] = subsection_text #store subsection text
            # Add subsection-level metadata, it is crucial to include ALL metadata.
            subsection_vector.metadata.update(subsection_data.get("metadata", {}))


            subsections_list.append(subsection_vector)
            section_vector.subsections.append(create_relationship_vector("has_subsection", section_vector.vector_id, subsection_vector.vector_id, bibtex_data, paper_type, pub_key, gin_edge_type="has_subsection"))
        section_vector.subsections = subsections_list #assign subsections
        sections_list.append(section_vector)  # add section
        template_instance.sections.append(create_relationship_vector("has_section", template_instance.vector_id, section_vector.vector_id,  bibtex_data, paper_type, pub_key, gin_edge_type="has_section"))

    template_instance.sections = sections_list #add all sections
    relationship_pub_template = create_relationship_vector("has_template_instance", pub_text_vector.vector_id, template_instance.vector_id, bibtex_data, paper_type, pub_key, gin_edge_type="has_template_instance") #create relationship vector
    template_instance.metadata['relationship_pub_template_id'] = relationship_pub_template.vector_id
    return template_instance

def extract_data_for_bertopic(
    vector_cache: Dict[str, TemplateInstanceVector],
    section_regex: str = r"^M(\d+)(:|$)",  # Matches M1, M1:1, M2, M10, etc.
    include_subsections: bool = True,
    stop_words: Optional[List[str]] = None,
    exclude_pubs: Optional[List[str]] = None,
    return_type: str = "both",
) -> Union[Tuple[List[str], List[List[float]]], List[str], List[List[float]]]:

    logger = logging.getLogger(__name__)
    logger.debug("Starting extract_data_for_bertopic")

    texts = []
    embeddings = []
    compiled_regex = re.compile(section_regex)
    #Add nltk stop words
    stop_words = stop_words or []  # Ensure stop_words is a list
    stop_words = list(set(stop_words + nltk.corpus.stopwords.words('english')))
    #Add lemmatizer
    lemmatizer = WordNetLemmatizer()


    for pub_key, template_instance in vector_cache.items():
        if exclude_pubs and pub_key in exclude_pubs:
            continue

        for section_vector in template_instance.sections:
            if not isinstance(section_vector, SectionVector):
                continue

            section_key = section_vector.metadata.get('section_id')
            if not section_key:
                continue

            match = compiled_regex.match(section_key)
            if match:
                if include_subsections or ":" not in section_key:
                    # --- Iterate through Subsections ---
                    for subsection_vector in section_vector.subsections:
                         if isinstance(subsection_vector, SubsectionVector):
                            # --- Get text and embedding DIRECTLY from SubsectionVector ---
                            subsection_text = subsection_vector.metadata.get('original_text', '') #HERE IS THE TEXT
                            subsection_embedding = subsection_vector.embedding

                            if subsection_embedding is not None:
                                embeddings.append(subsection_embedding) #add embedding FIRST

                                if return_type in ("both", "text"):
                                    if subsection_text:
                                        # --- Text Cleaning and Lemmatization ---
                                        # 1. Lowercase
                                        subsection_text = subsection_text.lower()
                                        # 2. Remove punctuation (you might have this already)
                                        subsection_text = re.sub(r'[^\w\s]', '', subsection_text)
                                        # 3. Tokenize
                                        tokens = nltk.word_tokenize(subsection_text)
                                        # 4. Remove stop words
                                        filtered_tokens = [word for word in tokens if word not in stop_words]
                                        # 5. Lemmatize
                                        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
                                        # 6. Join back into a string
                                        subsection_text = " ".join(lemmatized_tokens)

                                    if subsection_text.strip():  # Add text only if it's not empty
                                        texts.append(subsection_text)
                                    else: #remove embedding if text is empty
                                        embeddings.pop()

                    # If there are NO subsections, get data directly from SectionVector
                    if not section_vector.subsections:
                        section_text = section_vector.metadata.get('original_text', '')
                        section_embedding = section_vector.embedding
                        if section_embedding is not None:
                            embeddings.append(section_embedding)  #add embedding FIRST
                            if return_type in ("both", "text"):
                                    if section_text:
                                        # --- Text Cleaning and Lemmatization ---
                                        section_text = section_text.lower()
                                        section_text = re.sub(r'[^\w\s]', '', section_text)
                                        tokens = nltk.word_tokenize(section_text)
                                        filtered_tokens = [word for word in tokens if word not in stop_words]
                                        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
                                        section_text = " ".join(lemmatized_tokens)
                                    if section_text.strip():
                                        texts.append(section_text)
                                    else: #remove embedding if text is empty
                                        embeddings.pop()



    if return_type == "text":
        return texts
    elif return_type == "embeddings":
        return embeddings
    elif return_type == "both":
        return texts, embeddings
    else:
        raise ValueError("Invalid return_type")