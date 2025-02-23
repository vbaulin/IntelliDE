# main.py
import os
import re
import json
import pickle
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModel  # For type hints
import torch  # For type hints
from typing import Dict, Any, List, Literal, Union, Optional
import time
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm  # Import tqdm


# Import your utility functions
from utils import (
    setup_logging,
    load_template,
    extract_pdf_text,
    parse_bib_file,
    generate_paper_prompt,
    extract_paper_type,
    parse_markdown_output
)

# Import LLM interaction functions
from OpenRouter_Methods import get_llm_response

# Import BERTopic related functions
from analysis import perform_bertopic_analysis, analyze_clusters

# Import vector-related functions and classes
from vector_fns import (
    get_embedding_model,
    embed_text,
    create_text_vector,
    create_score_vector,
    create_parameter_vector,
    create_table_vector,
    create_relationship_vector,
    create_template_instance_vector,
    create_section_vector,
    create_subsection_vector,
    create_metadata,  # Import create_metadata
    TemplateInstanceVector,
    SectionVector,
    SubsectionVector,
    TextVector,
    ScoreVector,
    ParameterVector,
    TableVector,
    RelationshipVector,
    Vector
)

# Constants
TEMPLATE_FILE = "../Catechism/paper6GIN_10.md"
PROMPT_FILE = "prompts/analyze_paper.txt"
OUTPUT_DIR = "../synthetic/publications6GIN10"
INPUT_DIR = "../AI-4/"                                                        #location of PDFs
LLM_PROVIDER: Literal["google", "openrouter"] = "openrouter"
EMBEDDING_CACHE_DIR = ".embedding_cache"  # Keep directory
VECTOR_CACHE_FILE = os.path.join(EMBEDDING_CACHE_DIR, "vectors_cache.pkl")
MAX_LLM_RETRIES = 5
CLUSTER_ANALYSIS_OUTPUT_DIR = "cluster_analysis_results"
PERSPECTIVE_FILE = "../Catechism/Perspective2.md"

def build_vector_hierarchy(publication_data: Dict, parsed_markdown: Dict, scibert_model, scibert_tokenizer, pub_key: str) -> TemplateInstanceVector:
    """Builds the hierarchical structure using vector classes, adding embeddings,
    and incorporating metadata from the BibTeX data.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Building vector hierarchy for: {pub_key}")

    paper_type = publication_data.get("paper_type", "Unknown")
    bibtex_data = publication_data.get("bibtex_data", {})  # Get BibTeX data

    # Create the top-level TemplateInstanceVector
    # ADDED gin_node_type
    template_instance = create_template_instance_vector([], publication_data, paper_type, pub_key, gin_node_type="template_instance")
    template_instance.metadata = create_metadata(bibtex_data, paper_type, bibtex_key=pub_key) # Use create_metadata, ADDED bibtex_key
    template_instance.metadata['bibtex_key'] = pub_key  # Store BibTeX key

    # Publication-level TextVector
    pub_embedding = embed_text(publication_data["content"], scibert_tokenizer, scibert_model)
    # ADDED bibtex_key, gin_node_type
    pub_text_vector = create_text_vector(publication_data["content"], bibtex_data, paper_type, None, None, pub_key, gin_node_type="publication") # Use bibtex_data
    pub_text_vector.embedding = pub_embedding.tolist() if pub_embedding is not None else None
    template_instance.metadata['text_vector_id'] = pub_text_vector.vector_id
    # --- Section Vectors ---
    sections_list: List[SectionVector] = []
    for section_key, section_data in parsed_markdown.items():
        #logger.debug(f"  Processing section: {section_key}") # REMOVED debug logging

        # Create SectionVector (BEFORE processing subsections)
        section_embedding = embed_text(section_data.get("text", ""), scibert_tokenizer, scibert_model)
        # ADDED bibtex_key, gin_node_type  # FIXED: Pass pub_key, not bibtex_key
        section_vector = create_section_vector([], bibtex_data, paper_type, section_key, pub_key, gin_node_type="section") # Use bibtex_data, pass section_key
        section_vector.embedding = section_embedding.tolist() if section_embedding is not None else None

        # --- Subsection Vectors ---
        subsections_list: List[SubsectionVector] = []
        if "subsections" in section_data:
            for subsection_key, subsection_data in section_data["subsections"].items():
                #logger.debug(f"    Processing subsection: {subsection_key} in section: {section_key}") # REMOVED debug logging
                content_vectors: List[Union[TextVector, ScoreVector, RelationshipVector, ParameterVector, TableVector]] = []

                # Create SubsectionVector (BEFORE processing probes)
                subsection_embedding = embed_text(subsection_data.get("text", ""), scibert_tokenizer, scibert_model)
                # ADDED bibtex_key
                subsection_vector = create_subsection_vector([], bibtex_data, paper_type, section_key, subsection_key, pub_key, gin_node_type="subsection") # Use bibtex_data
                subsection_vector.embedding = subsection_embedding.tolist() if subsection_embedding is not None else None

                # --- Probe Vectors ---
                if "probes" in subsection_data:
                    for probe_key, probe_data in subsection_data["probes"].items():
                        probe_text = probe_data["text"]
                        probe_score = probe_data.get("score")  # Handle potential missing score
                        probe_justification = probe_data.get("justification") # Handle potential missing justification

                        # Probe TextVector
                        probe_embedding = embed_text(probe_text, scibert_tokenizer, scibert_model)
                        # ADDED bibtex_key
                        probe_text_vector = create_text_vector(probe_text, bibtex_data, paper_type, section_key, subsection_key, pub_key, gin_node_type="probe") # Use bibtex_data
                        probe_text_vector.embedding = probe_embedding.tolist() if probe_embedding is not None else None
                        content_vectors.append(probe_text_vector)

                        # Probe ScoreVector (if score exists)
                        if probe_score is not None:
                            # ADDED bibtex_key
                            score_vector = create_score_vector(probe_score, bibtex_data, paper_type, section_key, subsection_key, pub_key, gin_node_type="score") # Use bibtex_data
                            content_vectors.append(score_vector)

                        # Probe Justification TextVector (if justification exists)
                        if probe_justification:
                            # ADDED bibtex_key
                            justification_vector = create_text_vector(probe_justification, bibtex_data, paper_type, section_key, subsection_key, pub_key, gin_node_type="justification")# Use bibtex_data
                            content_vectors.append(justification_vector)

                         # Relationship: Subsection HAS_PROBE Probe
                        # ADDED bibtex_key, gin_edge_type
                        relationship_vector = create_relationship_vector("has_probe", subsection_vector.vector_id, probe_text_vector.vector_id, bibtex_data, paper_type, pub_key, gin_edge_type="has_probe") # Use bibtex_data
                        content_vectors.append(relationship_vector)

                # --- Handling Tables and Parameters ---
                # Example: Assuming you have logic to extract tables/parameters
                if "table" in subsection_data: # Check if table exists
                    table_data = subsection_data["table"]
                    # Create TableVector
                    if isinstance(table_data, dict) and "headers" in table_data and "rows" in table_data:
                        # ADDED bibtex_key
                        table_vector = create_table_vector(table_data["headers"], table_data["rows"], bibtex_data, paper_type, section_key, subsection_key, pub_key)
                        content_vectors.append(table_vector)
                        # ADDED bibtex_key, gin_edge_type
                        relationship_subsection_table = create_relationship_vector("has_table", subsection_vector.vector_id, table_vector.vector_id,  bibtex_data, paper_type, pub_key, gin_edge_type="has_table")
                        content_vectors.append(relationship_subsection_table)


                if "parameters" in subsection_data:  # Check if parameters exist
                    for param_name, param_data in subsection_data["parameters"].items():
                         # Create ParameterVector
                         if isinstance(param_data, dict) and "value" in param_data:
                            param_value = param_data["value"]
                            param_units = param_data.get("units")  # Optional units
                            # ADDED bibtex_key
                            parameter_vector = create_parameter_vector(param_name, param_value, param_units, bibtex_data, paper_type, section_key, subsection_key, pub_key) # Use bibtex_data
                            content_vectors.append(parameter_vector)
                            # ADDED bibtex_key, gin_edge_type
                            relationship_subsection_param = create_relationship_vector("has_parameter", subsection_vector.vector_id, parameter_vector.vector_id,  bibtex_data, paper_type, pub_key, gin_edge_type="has_parameter")
                            content_vectors.append(relationship_subsection_param)

                # Add all content vectors to the subsection
                subsection_vector.content = content_vectors
                subsections_list.append(subsection_vector)

                # Relationship: Section HAS_SUBSECTION Subsection
                # ADDED bibtex_key, gin_edge_type
                relationship_section_subsection = create_relationship_vector("has_subsection", section_vector.vector_id, subsection_vector.vector_id, bibtex_data, paper_type, pub_key, gin_edge_type="has_subsection")
                section_vector.subsections.append(relationship_section_subsection) #CHANGED: add to section vector

        section_vector.subsections = subsections_list  # Assign subsections to section
        sections_list.append(section_vector)
        #Relationship vector Template - Section
        # ADDED bibtex_key, gin_edge_type
        relationship_template_section = create_relationship_vector("has_section", template_instance.vector_id, section_vector.vector_id,  bibtex_data, paper_type, pub_key, gin_edge_type="has_section") #use bibtex
        template_instance.sections.append(relationship_template_section)

    # Assign sections to the TemplateInstanceVector
    template_instance.sections = sections_list
    #Relationship vector between Publication vector and Template Instance # ADDED
    relationship_pub_template = create_relationship_vector("has_template_instance", pub_text_vector.vector_id, template_instance.vector_id, bibtex_data, paper_type, pub_key, gin_edge_type="has_template_instance") #use bibtex
    template_instance.metadata['relationship_pub_template_id'] = relationship_pub_template.vector_id #to use further

    logger.info(f"Vector hierarchy built SUCCESSFULLY for: {pub_key}")
    return template_instance

def get_vector_by_address(template_instance: TemplateInstanceVector, address: str) -> Optional[Vector]:
    """Retrieves a vector from the hierarchy using a dot-separated address string."""
    parts = address.split(".")
    current_element: Union[TemplateInstanceVector, SectionVector, SubsectionVector, List[Vector]] = template_instance

    try:
        for i, part in enumerate(parts):
            if i == 0:  # Top-level (TemplateInstanceVector)
                if part == "metadata":
                    return template_instance.metadata  # Return metadata dict
                elif part == "sections":
                    continue  # Go to the next part (section index)
                else:
                    return None # Invalid address
            elif isinstance(current_element, TemplateInstanceVector): #FIXED
                 if part.isdigit() and int(part) < len(current_element.sections):
                     current_element = current_element.sections[int(part)]
                 else:
                     return None

            elif isinstance(current_element, SectionVector): #FIXED
                if part == "subsections":
                    continue
                elif part.isdigit() and int(part) < len(current_element.subsections):
                    current_element = current_element.subsections[int(part)]
                else:
                    return None #invalid

            elif isinstance(current_element, SubsectionVector): #FIXED
                if part.isdigit() and int(part) < len(current_element.content):
                    current_element = current_element.content[int(part)]
                else:
                    return None #invalid

            elif isinstance(current_element, list): #if it is list of Vectors #FIXED
                if part.isdigit() and int(part) < len(current_element):
                     current_element = current_element[int(part)] #get vector
                else:
                    return None #invalid

            else: #other cases
                return None

        return current_element  # Return the found element (Vector or list)

    except (KeyError, IndexError, TypeError) as e:
        # Handle cases where keys are missing, indices are out of bounds, or types are unexpected
        print(f"Error accessing vector: {e}")
        return None

def main():
    """Main function to process publications."""
    print("Starting main function")
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # --- Load Template and Count Expected Sections ---
    template_text = load_template(TEMPLATE_FILE)
    if not template_text:
        logger.error("❌ Template loading failed. Exiting.")
        return
    expected_sections = len(re.findall(r"^##\s+", template_text, re.MULTILINE))
    logger.info(f"Expected number of sections (from template): {expected_sections}")

    # --- Load SciBERT Model and Tokenizer ---
    logger.info("Loading SciBERT tokenizer and model...")
    scibert_tokenizer, scibert_model = get_embedding_model()
    scibert_model.eval()
    logger.info("✅ SciBERT tokenizer and model loaded.")

    # --- Create Output/Cache Directory ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)
    logger.info(f"Output directories created: {OUTPUT_DIR}, {EMBEDDING_CACHE_DIR}")

    # --- Get Publication Data ---
    bib_file_path = os.path.join(INPUT_DIR, 'AI-4.bib')
    presenters = parse_bib_file(INPUT_DIR, bib_file_path)
    if not presenters:
        logger.error("❌ No publication data found after BibTeX parsing. Exiting.")
        return
    logger.info(f"🔬 Found {len(presenters)} entries from BibTeX.")

    # --- LOAD VECTOR CACHE ---
    vector_cache: Dict[str, TemplateInstanceVector] = {}
    if os.path.exists(VECTOR_CACHE_FILE):
        try:
            with open(VECTOR_CACHE_FILE, "rb") as f:
                vector_cache = pickle.load(f)
            logger.info(f"✅ Vector cache loaded from: {VECTOR_CACHE_FILE}")
        except Exception as e:
            logger.warning(f"⚠️ Error loading vector cache: {e}. Starting with an empty cache.")
            vector_cache = {}  # Initialize as empty dict
    else:
        logger.info("ℹ️ No vector cache file found. Starting with an empty cache.")

    # --- Process Each Publication ---
    logger.info("--- Starting publication processing loop ---")
    # Use tqdm to create a progress bar
    for key, data in tqdm(presenters.items(), desc="Processing Publications"):
        markdown_file = os.path.join(OUTPUT_DIR, f"{key}.md")
        #logger.info(f"--- Processing publication: {key} ---")

        # --- 1. Check Vector Cache FIRST ---
        if key in vector_cache:
            logger.info(f"ℹ️ Skipping {key}: Already processed. Using cached vector data.")
            continue

        # --- 2. Process from Markdown or PDF ---
        if os.path.exists(markdown_file):
            logger.info(f"📄 Markdown file found for {key}. Processing from Markdown.")
            with open(markdown_file, "r", encoding="utf-8") as f:
                analysis_text = f.read()
            publication_data = {
                "key": key,
                "title": data.get("title", key),
                "content": analysis_text,
                "pdf_path": data.get('pdf_path'),
                "year": data.get("year", "N/A"),
                "doi": data.get("doi", "N/A"),
                "authors": data.get("authors", "N/A"),
                "paper_type": extract_paper_type(analysis_text) if analysis_text else "Unknown",  # ADDED: Extract paper type
                "bibtex_data": data,  # Store the BibTeX data
            }
            parsed_markdown = parse_markdown_output(analysis_text)
            logger.info(f"Markdown parsed for {key}, sections found: {len(parsed_markdown)}")


        # ADDED: Handle missing PDF case
        elif data.get('pdf_path') == 'N/A':
            logger.warning(f"⚠️ Skipping {key}: No PDF found.")
            publication_data = {
                "key": key,
                "title": data.get("title", key),
                "content": "No content available",  # Minimal content
                "pdf_path": 'N/A',
                "year": data.get("year", "N/A"),
                "doi": data.get("doi", "N/A"),
                "authors": data.get("authors", "N/A"),
                "paper_type": "Unknown",  # Default paper type
                "bibtex_data": data,
            }
            parsed_markdown = {} #empty, no parsing
            #You can skip creating vector if no content
            continue

        else:  # Process from PDF (or title if no PDF)
            logger.info(f"No Markdown file found for {key}. Processing from PDF.")
            pdf_path = data.get('pdf_path')
            text_content = extract_pdf_text(pdf_path) if pdf_path and os.path.exists(pdf_path) else data.get("title", key)
            paper_type = extract_paper_type(text_content) if text_content != data.get("title", key) else "Unknown"

            publication_data = {
                "key": key,
                "title": data.get("title", key),
                "content": text_content,
                "pdf_path": pdf_path,
                "year": data.get("year", "N/A"),
                "doi": data.get("doi", "N/A"),
                "authors": data.get("authors", "N/A"),
                "paper_type": paper_type, # ADDED
                "bibtex_data": data,  # Store the BibTeX data
            }

            # --- LLM Processing (if no Markdown) ---
            prompt = generate_paper_prompt(PROMPT_FILE, TEMPLATE_FILE, PERSPECTIVE_FILE, publication_data)
            if not prompt:
                logger.warning(f"⚠️ Skipping {key}: Prompt generation failed.")
                continue

            analysis_text = None
            retries = 0
            while retries < MAX_LLM_RETRIES:
                try:
                    logger.info(f"Attempting LLM response for {key}, retry: {retries + 1}/{MAX_LLM_RETRIES}")
                    analysis_text = get_llm_response(prompt, LLM_PROVIDER)
                    if analysis_text:
                        parsed_markdown = parse_markdown_output(analysis_text)
                        if len(parsed_markdown) == expected_sections:
                            logger.info(f"✅ LLM response received and validated for {key}.")
                            break  # Exit retry loop
                        else:
                            logger.warning(f"⚠️ Truncated response. Retrying... ({retries + 1}/{MAX_LLM_RETRIES})")
                            retries += 1
                            time.sleep(2)
                    else:
                        logger.warning(f"⚠️ No LLM response. Retrying... ({retries + 1}/{MAX_LLM_RETRIES})")
                        retries += 1
                        time.sleep(2)
                except Exception as e:
                    logger.error(f"❌ LLM response error for {key}: {e}")
                    break

            if analysis_text is None:
                logger.error(f"❌ Skipping {key}: Failed to get LLM response.")
                continue

            # Save Markdown (if successful)
            with open(markdown_file, "w", encoding="utf-8") as f:
                f.write(analysis_text)
            logger.info(f"✅ Markdown output SAVED to: {markdown_file}")
            parsed_markdown = parse_markdown_output(analysis_text)

        # --- Build Vector Hierarchy (for both Markdown and PDF paths) ---
        publication_vector_hierarchy = build_vector_hierarchy(publication_data, parsed_markdown, scibert_model, scibert_tokenizer, key)
        vector_cache[key] = publication_vector_hierarchy

        # --- Save to Vector Cache ---
        try:
            with open(VECTOR_CACHE_FILE, "wb") as f:
                pickle.dump(vector_cache, f)
            logger.info(f"💾 Vector hierarchy SAVED to cache for {key}")
        except Exception as e:
            logger.error(f"❌ Error saving vector cache for {key}: {e}")

     # --- Example of how to access vectors using the address string ---
    logger.info("--- Vector Access Examples ---")
    for pub_key, template_instance in vector_cache.items():
        # Access the publication's title from metadata
        title = template_instance.metadata.get('title')
        logger.info(f"Publication Title ({pub_key}): {title}")

        # Access a specific vector (e.g., first probe in the first subsection of the first section)
        address = "sections.0.subsections.0.content.0"  # Address string -দরা
        vector = get_vector_by_address(template_instance, address)
        if vector and isinstance(vector, TextVector):
            logger.info(f"  Vector at {address}: Text: {vector.content[:50]}...")  # First 50 chars
            if vector.embedding:
                logger.info(f"  Vector at {address}: Embedding (first 5 elements): {vector.embedding[:5]}...")
        elif vector:
            logger.info(f"Vector at {address}: {vector}") #if not text vector
        else:
            logger.warning(f"  Could not retrieve vector at address: {address}")

        #Access relationship vectors
        address = "sections.0"  # Address string
        vector = get_vector_by_address(template_instance, address)
        if vector and isinstance(vector, RelationshipVector):
             logger.info(f"  Relationship Vector at {address}: Source ID {vector.source_vector_id}, Target ID: {vector.target_vector_id}...")  # First 50 chars
        elif vector and isinstance(vector, SectionVector):
            for rel_vector in vector.subsections: #check if it is relationship
                if isinstance(rel_vector, RelationshipVector):
                    logger.info(f" Relationship in Section Vector {address}: Source ID {rel_vector.source_vector_id}, Target ID: {rel_vector.target_vector_id}...")  # First 50 chars
        else:
            logger.warning(f"  Could not retrieve vector at address: {address}")

    # --- BERTopic Analysis (remains largely the same) ---
    logger.info("--- Publication processing loop COMPLETE. ---")
    logger.info("Loading vector cache for BERTopic analysis...")
    try:
        with open(VECTOR_CACHE_FILE, "rb") as f:
            vector_cache = pickle.load(f)
        logger.info(f"✅ Vector cache LOADED for BERTopic from: {VECTOR_CACHE_FILE}")
    except Exception as e:
        logger.error(f"❌ Error loading vector cache for BERTopic: {e}. Cannot perform analysis.")
        return
    #Prepare texts and embeddings
    combined_texts = []
    publication_keys = []
    all_combined_embeddings = []

    logger.info("Preparing texts and embeddings for BERTopic...")
    for pub_key, template_instance in vector_cache.items():
        publication_keys.append(pub_key)
        pub_text_vector_id = template_instance.metadata.get('text_vector_id')
        if pub_text_vector_id:
            pub_text_vector = None
            for section in template_instance.sections:
                for subsection in section.subsections:
                    for content_vector in subsection.content:
                        if content_vector.vector_id == pub_text_vector_id and isinstance(content_vector, TextVector):
                            pub_text_vector = content_vector
                            break  # Found
                    if pub_text_vector:
                        break
                if pub_text_vector:
                    break

            if pub_text_vector:
                pub_text = pub_text_vector.content
                combined_texts.append(pub_text)

                if pub_text_vector.embedding:
                    all_combined_embeddings.append(pub_text_vector.embedding)
                else:
                    logger.warning(f"⚠️ No embedding for pub text vector: {pub_key}")
                    all_combined_embeddings.append(None)
            else:
                logger.warning(f"⚠️ Could not find TextVector for pub: {pub_key}")
                combined_texts.append("")
                all_combined_embeddings.append(None)
        else:
            logger.warning(f"⚠️ No text_vector_id in metadata for pub: {pub_key}")
            combined_texts.append("")
            all_combined_embeddings.append(None)

    # Filter
    filtered_texts = []
    filtered_embeddings = []
    filtered_keys = []

    for text, embedding, key in zip(combined_texts, all_combined_embeddings, publication_keys):
        if text.strip() and embedding is not None:
            filtered_texts.append(text)
            filtered_embeddings.append(embedding)
            filtered_keys.append(key)

    # Run
    if filtered_embeddings:
        logger.info("\n--- Running Combined BERTopic across all publications ---")
        vectorizer_model = CountVectorizer(stop_words="english", min_df=2)
        all_combined_embeddings_np = np.array(filtered_embeddings)

        combined_model, combined_labels, combined_probs = perform_bertopic_analysis(
            filtered_texts,
            embeddings=all_combined_embeddings_np,
            embedding_model=scibert_model,  # For consistency
            vectorizer_model=vectorizer_model
        )

        if combined_model:
            logger.info("\n--- Analyzing Clusters with LLM ---")
            os.makedirs(CLUSTER_ANALYSIS_OUTPUT_DIR, exist_ok=True)
            analyze_clusters(
                combined_labels,
                filtered_texts,
                filtered_keys,
                template_text,
                vector_cache,  # Pass vector cache
                combined_model,
                combined_probs,
                PERSPECTIVE_FILE
            )
            logger.info("✅ Cluster analysis COMPLETE and results saved.")
        else:
            logger.warning("⚠️ BERTopic model could not be trained.")
    else:
        logger.warning("⚠️ No valid combined embeddings. Skipping BERTopic.")

    logger.info("\n--- Analysis Complete ---")
    print("\n--- Analysis Complete --- Check log for details ---")

if __name__ == "__main__":
    main()
