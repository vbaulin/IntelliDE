import os
import re
import json
import pickle
import unicodedata
import hashlib
import numpy as np
import logging
import time
import glob
from typing import Dict, Union
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt

EMBEDDING_CACHE_DIR = ".embedding_cache"
EMBEDDING_CACHE_FILE = os.path.join(EMBEDDING_CACHE_DIR, "embeddings_cache.pkl")

# --- Helper Functions ---
def setup_logging(name=__name__):
    """Set up logging configuration."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(name)
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)
    return logger

def clean_text(text):
    """
    Cleans text by removing special characters and extra whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFKD', text)
    text = "".join(c for c in text if unicodedata.category(c) != 'Mn')
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove non-alphanumeric
    text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace
    return text

def load_data_from_json(json_file_path):
    """
    Loads data from JSON files, handling the new folder structure.
    """
    all_data = []
    for folder_name in os.listdir(json_file_path):
        folder_path = os.path.join(json_file_path, folder_name)
        if os.path.isdir(folder_path):
            json_files = glob.glob(os.path.join(folder_path, "*.json"))
            for file in json_files:
                try:
                    with open(file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        all_data.append(data)  # Ensure each file's content is a single item in the list
                except UnicodeDecodeError:
                    try:
                        with open(file, "r", encoding="latin-1") as f:
                            data = json.load(f)
                            all_data.append(data)
                    except Exception as e:
                        print(f"❌ Error reading file {file}: {e}")
                except Exception as e:
                    print(f"❌ Error reading file {file}: {e}")
    return all_data

def save_topic_tree(tree, filename="topic_tree.pickle"):
    """
    Saves the topic tree data from a BERTopic model to a file using pickle.

    Args:
        topic_model: The fitted BERTopic model.
        filename: The name of the file to save the topic tree to.
    """
    try:


        with open(filename, "w") as f:
            pickle.dump(tree, f)

        print(f"✅ Topic tree saved to {filename}")

    except Exception as e:
        print(f"❌ Error saving topic tree: {e}")

def read_markdown_file(filepath):
    """
    Reads the content of a markdown file.

    Args:
        filepath: Path to the markdown file.

    Returns:
        The content of the file as a string, or None if an error occurred.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"❌ Error: Markdown file not found at {filepath}")
        return None
    except Exception as e:
        print(f"❌ Error reading markdown file: {e}")
        return None


def load_prompt(file_path):
    """
    Loads a prompt from a text file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"❌ Error: Prompt file not found at {file_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading prompt from {file_path}: {e}")
        return None

def load_template(template_file):
    """
    Loads the template structure from the specified Markdown file,
    generating unique keys for each section and subsection.
    """
    template_sections = {}
    section_counter = 0
    subsection_counter = 0

    with open(template_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("## "):
                current_section = line[3:].strip()
                section_counter += 1
                section_key = f"section_{section_counter}"
                template_sections[section_key] = {
                    "name": current_section,
                    "subsections": {},
                }
                current_subsection = None
                subsection_counter = 0  # Reset subsection counter for each section
                print(f"Loaded section: {section_key} - {current_section}")
            elif line.startswith("### "):
                current_subsection = line[4:].strip()
                current_subsection = current_subsection.replace(" [Text]", "")

                subsection_counter += 1  # Increment subsection counter
                subsection_key = f"{section_key}_subsection_{subsection_counter}"

                if current_section:
                    template_sections[section_key]["subsections"][
                        subsection_key
                    ] = current_subsection
                    print(f"  Loaded subsection: {subsection_key} - {current_subsection}")

    print("✅ Template loaded successfully!")
    return template_sections

def get_embedding(text, embedding_model, pub_key, section_key, subsection_key):
    """
    Generates an embedding for a given text using the specified model.
    Caches the embeddings to a single Pickle file to avoid redundant computations.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    # Ensure the cache directory exists
    os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)

    # Get a shortened model name using hashing
    try:
        model_name = embedding_model._first_module().config._name_or_path
    except (AttributeError, StopIteration):
        model_name = "default_model_name"
    model_name_hash = hashlib.sha256(model_name.encode()).hexdigest()[:16]

    # Create a unique key for the text based on its hash and the model name hash
    text_hash = hashlib.sha256(text.encode()).hexdigest()
    cache_key = f"{text_hash}_{model_name_hash}"

    # Load the cache if it exists
    cache = {}
    if os.path.exists(EMBEDDING_CACHE_FILE):
        try:
            with open(EMBEDDING_CACHE_FILE, "rb") as f:
                cache = pickle.load(f)
        except Exception as e:
            print(f"❌ Error loading cache from {EMBEDDING_CACHE_FILE}: {e}")
            cache = {}

    # Check if the embedding is present in the cache and if the pub_key, section_key, and subsection_key exist
    if pub_key in cache and section_key in cache[pub_key] and subsection_key in cache[pub_key][section_key] and cache_key in cache[pub_key][section_key][subsection_key]:
        try:
            embedding = cache[pub_key][section_key][subsection_key][cache_key]
            #print(f"✅ Loaded embedding from cache for text: '{text[:50]}...'")
            return embedding
        except Exception as e:
            print(f"❌ Error loading embedding from cache: {e}")

    # Generate embedding if not found in cache
    try:
        embedding = embedding_model.encode(text, convert_to_numpy=True)
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return None

    # Store the embedding in the cache
    if pub_key not in cache:
        cache[pub_key] = {}
    if section_key not in cache[pub_key]:
        cache[pub_key][section_key] = {}
    if subsection_key not in cache[pub_key][section_key]:
        cache[pub_key][section_key][subsection_key] = {}
    cache[pub_key][section_key][subsection_key][cache_key] = embedding

    # Save the updated cache to the file
    try:
        with open(EMBEDDING_CACHE_FILE, "wb") as f:
            pickle.dump(cache, f)
        #print(f"✅ Embedding generated and cached for text: '{text[:50]}...'")
    except (pickle.PicklingError, OSError) as e:
        print(f"❌ Error saving embedding to cache: {e}")

    return embedding

def combine_embeddings(
    embeddings: Dict[str, Union[np.ndarray, None]]
) -> np.ndarray:
    """
    Combines embeddings, handling None values.

    Parameters
    ----------
    embeddings : Dict[str, Union[np.ndarray, None]]
        Dictionary mapping keys to embeddings (or None).

    Returns
    -------
    np.ndarray
        Combined embedding vector.
    """
    valid_embeddings = [e for e in embeddings.values() if e is not None]

    if not valid_embeddings:
        return np.array([])  # Return empty array if all embeddings are None

    if len(valid_embeddings) == 1:
        return valid_embeddings[0]  # Return the single embedding if only one is valid

    # Concatenate valid embeddings
    combined_embedding = np.concatenate(valid_embeddings)

    return combined_embedding

def extract_answers(publication, template_sections):
    """
    Extracts answers based on the order of sections and subsections in the template,
    ignoring any numbering discrepancies in the JSON content.
    Handles subsection content ending with newline characters.
    Prints only warnings and errors.
    """
    answers = {}
    content = publication.get("content", "").replace('\\n', '\n')
    pub_key = publication.get('metadata', {}).get('key')

    section_index = 0
    for section_key, section_data in template_sections.items():
        section_index += 1
        section_name = section_data["name"]
        answers[section_key] = {}
        #print(f"  - Looking for section: {section_index}. {section_name}")

        subsection_index = 0
        header_start_index = 0  # Start searching from the beginning of the content

        for subsection_key, subsection_name in section_data["subsections"].items():
            subsection_index += 1
            #print(f"    -- Looking for subsection: {section_index}.{subsection_index}. {subsection_name}")

            # Find the start of the next subsection header (###)
            header_start_index = content.find("###", header_start_index)

            if header_start_index != -1:
                #print(f"      -- Found subsection header start at index: {header_start_index}")

                # Find the end of the subsection header (newline)
                header_end_index = content.find("\n", header_start_index + 3)
                if header_end_index == -1:
                    print(f"      -- ⚠️ Warning: Could not find end of header for subsection {subsection_key} in publication {pub_key}")
                    header_start_index += 3
                    continue

                # Find the end of the subsection (next header or end of content)
                end_index = len(content)
                for next_header_tag in ["###", "##", "#"]:
                    next_header_index = content.find(next_header_tag, header_end_index + 1)
                    if next_header_index != -1 and next_header_index < end_index:
                        end_index = next_header_index
                #print(f"      -- Found subsection end at index: {end_index}")

                # Extract the answer text, handling trailing newline
                answer_text = content[header_end_index + 1:end_index].strip() # Skip the header title
                if answer_text.endswith('\n'):
                    answer_text = answer_text[:-1].strip()
                #print(f"      -- Extracted raw answer text:\n'{answer_text}'")

                # Check for numbering mismatch
                extracted_numbering = re.search(r"(\d+\.\d+)\.\s", answer_text)
                if extracted_numbering:
                    extracted_numbering = extracted_numbering.group(1)
                    expected_numbering = f"{section_index}.{subsection_index}"
                    if extracted_numbering != expected_numbering:
                        print(f"      -- ⚠️ Warning: Numbering mismatch. Expected {expected_numbering}, found {extracted_numbering} in publication {pub_key}")

                # Clean and store the answer
                cleaned_answer = clean_text(answer_text)
                if cleaned_answer:
                    answers[section_key][subsection_key] = cleaned_answer
                    #print(f"      -- ✅ Successfully extracted and stored subsection text.")
                else:
                    print(f"      -- ⚠️ Warning: Extracted text is empty after cleaning for subsection {subsection_key} in publication {pub_key}")

                header_start_index = end_index  # Move past the current subsection

            else:
                answers[section_key][subsection_key] = ""
                print(f"      -- ❌ Warning: Could not find subsection {section_index}.{subsection_index} in publication {pub_key}")

    # print(f"✅ Successfully extracted content from publication: {pub_key}.")
    return answers

def plot_within_cluster_distances(label, llm_cluster_title, keys, distances, template_sections):
    """
    Plots the distribution of distances within a single cluster.
    Simplified to accept pre-calculated distances.
    """
    if not distances:
        print(f"No distances data available for cluster {label}.")
        return

    # Convert distances dictionary to a list of float values
    distances_list = [float(value) for value in distances.values()]

    plt.figure()
    plt.hist(distances_list, bins=20, color="skyblue", edgecolor="black")
    plt.title(f"Distribution of Distances within Cluster {label}: {llm_cluster_title}")
    plt.xlabel("Cosine Distance")
    plt.ylabel("Number of Publications")
    plt.grid(True)
    plt.show()
