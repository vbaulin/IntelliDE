# main.py
import os
import re
import json
import pickle
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
from typing import Dict, Any, List, Literal, Union, Optional
import time
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import matplotlib.pyplot as plt
from bertopic import BERTopic
import plotly.io as pio

# Import your utility functions
from utils import (
    setup_logging,
    load_template,
    extract_pdf_text,
    parse_bib_file,
    generate_paper_prompt,
    extract_paper_type,
    read_markdown_file,
    parse_markdown_output
)

# Import LLM interaction functions
from OpenRouter_Methods import get_llm_response

# Import BERTopic related functions
from analysis import perform_bertopic_analysis, analyze_clusters, save_topic_tree_to_markdown, calculate_average_similarity, visualize_intertopic_distance_map, visualize_topic_hierarchy, visualize_topic_similarity

# Import vector-related functions and classes
from vector_fns import (
    get_embedding_model,
    build_vector_hierarchy,
    embed_text,
    create_text_vector,
    create_score_vector,
    create_parameter_vector,
    create_table_vector,
    create_relationship_vector,
    create_template_instance_vector,
    create_section_vector,
    create_subsection_vector,
    create_metadata,
    TemplateInstanceVector,  # Import the corrected class
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
INPUT_DIR = "../AI-4/"
LLM_PROVIDER: Literal["google", "openrouter"] = "openrouter"
EMBEDDING_CACHE_DIR = ".embedding_cache"
VECTOR_CACHE_FILE = os.path.join(EMBEDDING_CACHE_DIR, "vectors_cache.pkl")
MAX_LLM_RETRIES = 5
CLUSTER_ANALYSIS_OUTPUT_DIR = "cluster_analysis_results"
PERSPECTIVE_FILE = "../Catechism/Perspective2.md"
VISUALIZATION_DIR = "visualization_results"



def save_visualization(fig, filename, formats=["html", "png"]):
    """Save visualizations in multiple formats"""
    for fmt in formats:
        if fmt == "html":
            pio.write_html(fig, f"{VISUALIZATION_DIR}/{filename}.html")
        elif fmt == "png":
            pio.write_image(fig, f"{VISUALIZATION_DIR}/{filename}.png", width=1200, height=800)

def create_topic_visualizations(model, texts, topics, probs, topic_labels=None):
    """Create and save standard BERTopic visualizations"""
    logger = logging.getLogger(__name__)

        # 1. Intertopic Distance Map
    logger.info("Generating Intertopic Distance Map...")
    try:
        fig_distance_map = model.visualize_topics()
        save_visualization(fig_distance_map, "intertopic_distance_map")
        logger.info("✅ Intertopic Distance Map saved")

    except Exception as e:
        logger.error(f"Error generating intertopic distance map: {e}")


    # 2. Topic Hierarchy (Hierarchical Clustering)
    logger.info("Generating Topic Hierarchy...")
    try:
        fig_hierarchy = model.visualize_hierarchy(custom_labels = topic_labels)
        save_visualization(fig_hierarchy, "topic_hierarchy")
        logger.info("✅ Topic Hierarchy visualization saved")
    except Exception as e:
        logger.error(f"Error generating topic hierarchy: {e}")

    # 3. Topic Similarity Matrix
    logger.info("Generating Topic Similarity Matrix...")
    try:
        fig_similarity = model.visualize_heatmap(custom_labels=topic_labels)
        save_visualization(fig_similarity, "topic_similarity_matrix")
        logger.info("✅ Topic Similarity Matrix saved")
    except Exception as e:
        logger.error(f"Error generating topic similarity matrix: {e}")

def main():
    """Main function to process publications."""
    print("Starting main function")
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # --- Create Output/Cache Directories ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    logger.info(f"Directories created: {OUTPUT_DIR}, {EMBEDDING_CACHE_DIR}, {VISUALIZATION_DIR}")

    # --- Load Template and Count Sections ---
    template_text = load_template(TEMPLATE_FILE)
    if not template_text: return
    expected_sections = len(re.findall(r"^##\s+", template_text, re.MULTILINE))
    logger.info(f"Expected sections: {expected_sections}")

    # --- Load SciBERT ---
    logger.info("Loading SciBERT...")
    scibert_tokenizer, scibert_model = get_embedding_model()
    scibert_model.eval()
    logger.info("✅ SciBERT loaded.")

    # --- Get Publication Data ---
    bib_file_path = os.path.join(INPUT_DIR, 'AI-4.bib')
    presenters = parse_bib_file(INPUT_DIR, bib_file_path)
    if not presenters: return
    logger.info(f"🔬 Found {len(presenters)} entries from BibTeX.")

    # --- Load/Create Vector Cache ---
    vector_cache: Dict[str, TemplateInstanceVector] = {}
    if os.path.exists(VECTOR_CACHE_FILE):
        try:
            with open(VECTOR_CACHE_FILE, "rb") as f:
                vector_cache = pickle.load(f)
            logger.info(f"✅ Vector cache loaded from: {VECTOR_CACHE_FILE}")
        except Exception as e:
            logger.warning(f"⚠️ Error loading cache: {e}. Using empty cache.")
    else:
        logger.info("ℹ️ No vector cache found. Starting with an empty cache.")

    # --- OPTION TO CLEAR CACHE (for debugging) ---
    clear_cache = False  # Set to True if you want to force a cache rebuild
    if clear_cache:
        logger.warning("⚠️ Clearing vector cache...")
        vector_cache = {}  # Reset the cache
        if os.path.exists(VECTOR_CACHE_FILE):
            os.remove(VECTOR_CACHE_FILE)  # Delete the cache file
            logger.warning("⚠️ Vector cache file deleted.")

    # --- Process Each Publication ---
    logger.info("--- Starting publication processing ---")
    for key, data in tqdm(presenters.items(), desc="Processing Publications"):
        markdown_file = os.path.join(OUTPUT_DIR, f"{key}.md")

        if key in vector_cache:
            logger.info(f"ℹ️ Skipping {key}: Already processed.")
            continue

        if os.path.exists(markdown_file):
            logger.info(f"📄 Markdown found for {key}. Processing from Markdown.")
            with open(markdown_file, "r", encoding="utf-8") as f:
                analysis_text = f.read()
            publication_data = {
                "key": key, "title": data.get("title", key), "content": analysis_text,
                "pdf_path": data.get('pdf_path'), "year": data.get("year"),
                "doi": data.get("doi"), "authors": data.get("authors"),
                "paper_type": extract_paper_type(analysis_text) if analysis_text else "Unknown",
                "bibtex_data": data,
            }
            parsed_markdown = parse_markdown_output(analysis_text)

        elif data.get('pdf_path') == 'N/A':
            logger.warning(f"⚠️ Skipping {key}: No PDF.")
            continue

        else:  # Process from PDF
            logger.info(f"No Markdown for {key}. Processing from PDF.")
            pdf_path = data.get('pdf_path')
            text_content = extract_pdf_text(pdf_path) if pdf_path and os.path.exists(pdf_path) else data.get("title", key)
            paper_type = extract_paper_type(text_content) if text_content != data.get("title", key) else "Unknown"
            publication_data = {"key": key, "title": data.get("title"), "content": text_content,
                "pdf_path": pdf_path, "year": data.get("year"), "doi": data.get("doi"),
                "authors": data.get("authors"), "paper_type": paper_type, "bibtex_data": data,}
            logger.debug(f"  PDF content extracted for {key}: {text_content[:100]}...")  # Log part of the content


            prompt = generate_paper_prompt(PROMPT_FILE, TEMPLATE_FILE, PERSPECTIVE_FILE, publication_data)
            if not prompt:
                logger.warning(f"⚠️ Skipping {key}: Prompt generation failed.")
                continue

            analysis_text = None
            retries = 0
            while retries < MAX_LLM_RETRIES:
                try:
                    logger.info(f"LLM request for {key}, retry: {retries + 1}/{MAX_LLM_RETRIES}")
                    analysis_text = get_llm_response(prompt, LLM_PROVIDER)
                    if analysis_text:
                        parsed_markdown = parse_markdown_output(analysis_text)
                        if len(parsed_markdown) == expected_sections:
                            logger.info(f"✅ LLM response received for {key}.")
                            break
                        else:
                            logger.warning(f"⚠️ Truncated response. Retrying...")
                            retries += 1; time.sleep(2)
                    else:
                        logger.warning(f"⚠️ No LLM response. Retrying...")
                        retries += 1; time.sleep(2)
                except Exception as e:
                    logger.error(f"❌ LLM error for {key}: {e}")
                    break

            if analysis_text is None:
                logger.error(f"❌ Skipping {key}: Failed to get LLM response.")
                continue

            with open(markdown_file, "w", encoding="utf-8") as f:
                f.write(analysis_text)
            logger.info(f"✅ Markdown saved: {markdown_file}")
            parsed_markdown = parse_markdown_output(analysis_text)

        # --- Build Vector Hierarchy ---
        logger.debug(f"  Building vector hierarchy for: {key}")
        publication_vector_hierarchy = build_vector_hierarchy(publication_data, parsed_markdown, scibert_model, scibert_tokenizer, key)

        # Check if hierarchy was built successfully
        if publication_vector_hierarchy is None:
            logger.warning(f"⚠️ Skipping {key}: Vector hierarchy creation failed.")
            continue

        vector_cache[key] = publication_vector_hierarchy

        # Save to cache
        try:
            with open(VECTOR_CACHE_FILE, "wb") as f:
                pickle.dump(vector_cache, f)
            logger.info(f"💾 Vector hierarchy saved for {key}")
        except Exception as e:
            logger.error(f"❌ Error saving cache for {key}: {e}")


    logger.info("--- Analysis Complete ---")

if __name__ == "__main__":
    main()

