# utils.py
import os
import re
import json
import pickle
import unicodedata
import hashlib
import numpy as np
import logging
from typing import Dict, Union, Any, Optional, List, Tuple
from pathlib import Path
import bibtexparser
from bibtexparser.bparser import BibTexParser
import pdfplumber


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

def load_template(template_file):
    """
    Loads the template structure from the specified Markdown file.
    """
    try:
        with open(template_file, 'r', encoding='utf-8') as f:
            return f.read() #return as a string
        print("✅ Template loaded successfully!")
    except FileNotFoundError:
        print(f"❌ Error: Template file not found at {template_file}")
        return None
    except Exception as e:
        print(f"❌ Error loading template: {e}")
        return None

def extract_pdf_text(pdf_file: str) -> str:
    try:
        text_content = []
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_content.append(text)
        return "\n".join(text_content)
    except Exception as e:
        print(f"❌ An error occurred while reading the PDF: {e}")
        return ""

def parse_bib_file(INPUT_DIR, bib_file_path: str) -> Dict:
    """
    Parse the BibTeX file and extract relevant information.
    """
    presenters = {}
    logger = logging.getLogger(__name__)
    logger.info(f"Parsing BibTeX file: {bib_file_path}")

    try:
        with open(bib_file_path, 'r', encoding='utf-8') as bib_file:
            parser = BibTexParser()
            bib_database = bibtexparser.loads(bib_file.read(), parser=parser)
        logger.info(f"BibTeX file parsed, {len(bib_database.entries)} entries found.")

        for entry in bib_database.entries:
            key = entry.get('ID', 'unknown_key')
            #logger.info(f"--- Processing BibTeX entry: {key} ---")  # DEBUG
            pdf_path = entry.get('file', '')
            #logger.info(f"  Raw 'file' field: {pdf_path}")  # DEBUG
            if 'file' in entry and entry['file']:
                 try:
                     pdf_path_raw = entry.get('file', '')
                     pdf_path_split = pdf_path_raw.split(':')
                     if len(pdf_path_split) > 1:
                         pdf_path = pdf_path_split[1].strip()
                     else:
                         pdf_path = pdf_path_split[0].strip()
                     pdf_path = os.path.join(INPUT_DIR, pdf_path)
                     #logger.info(f"  Extracted pdf_path: {pdf_path}")  # DEBUG
                 except Exception as e:
                     logger.error(f" ❌ Error processing 'file' field: {e}")
                     pdf_path = 'N/A'
            else:
                pdf_path = 'N/A'
                #logger.warning(f"  No 'file' field found or empty for {key}, setting pdf_path to N/A")


            presenters[key] = {
                'name': entry.get('author', 'Unknown Author'),
                'title': entry.get('title', 'Unknown Title').replace('{', '').replace('}', '').replace('/', ' '),
                'year': entry.get('year', 'Unknown Year'),
                'doi': entry.get('doi', 'N/A'),
                'pdf_path': pdf_path,
                'key': key,
                "authors": entry.get('author', 'Unknown Author'),
            }
    except Exception as e:
        logger.error(f"Error reading BibTeX file: {e}")
        return {} # Return empty dict if error

    if not presenters:
        logger.warning("❌ No data extracted from the BibTeX file. Please check its structure.")

    return presenters

def save_markdown_report(content: str, filepath: Path, title: str):
    """Save report in markdown format"""
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# {title}\n\n")
            f.write(content)

    except Exception as e:
        logging.error(f"❌ Error saving markdown report: {e}")
        raise

def save_json_report(content: str, filepath: Path, metadata: dict):
    """Save report in JSON format"""
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": metadata,
            "content": content
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    except Exception as e:
        logging.error(f"❌ Error saving JSON report: {e}")
        raise

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

def read_description(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            description = file.read().strip()
        return description
    except FileNotFoundError:
        print(f"❌ Error: File {file_path} not found")
        return None
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None

def extract_paper_type(publication_content: str) -> str:
    """Extracts the paper type from the publication content."""
    content_lines = publication_content.splitlines()
    for i, line in enumerate(content_lines):
        if "__Paper Type:__" in line:
            # Extract everything after the marker on the same line.
            after_marker = line.split("__Paper Type:__")[-1].strip()
            if after_marker:
                # Check if it's enclosed in square brackets.
                match = re.search(r"\[(.*?)\]", after_marker)
                if match:
                    return match.group(1).lower()
                else:
                    return after_marker.lower()
            # If nothing is found after the marker on the same line,
            # then check the following lines.
            for next_line in content_lines[i + 1:]:
                next_line = next_line.strip()
                if not next_line or next_line.startswith("---") or next_line.startswith("*"):
                    continue
                match = re.search(r"\[(.*?)\]", next_line)
                if match:
                    return match.group(1).lower()
                else:
                    return next_line.lower()
    return "Unknown"  # Default if not found

def generate_paper_prompt(prompt, template, field_context, publication: Dict) -> str:
    """Generate research prompt using the external prompt file."""

    prompt_template = load_prompt(prompt)  # Load from file
    if not prompt_template:
        return ""  # Return empty string if loading fails

    # --- Check for content FIRST ---
    if not publication.get('content') or publication['content'] == "No content available":
        print("❌ No content available, cannot generate prompt")
        return ""  # Return empty string if no content

    # --- If we get here, we HAVE content ---
    paper_content = publication['content']
    field = read_description(field_context)  # Load field context
    catechism_template = read_description(template)  # Load template

    if not field or not catechism_template:
        print("❌ Could not load required files (field/catechism) for prompt generation.")
        return ""

    # Use .format() for clarity and safety
    # Changed 'catechism_template' to 'template' to match prompt file
    return prompt_template.format(
        publication_title=publication['title'],
        publication_content=paper_content,
        field=field,
        template=catechism_template  # <-- Changed key to 'template'
    )

def get_vector_by_key(cache: Dict[str, Any], full_key: str) -> Optional[List[float]]:
    """Retrieves a specific vector from the cache using its full key (pub_key_section_...)."""
    try:
        parts = full_key.split("_")
        pub_key = parts[0]
        if pub_key not in cache:
            return None

        if len(parts) == 1:  # Just publication key.  Corrected this.
            return cache[pub_key]["embeddings"].get(pub_key) # <--- Correct retrieval
        elif len(parts) == 3:  # Publication and Section
            section_key = parts[2]
            return cache[pub_key]["embeddings"].get(f"section_{section_key}")
        elif len(parts) == 4: # Publication, Section, Subsection
            section_key = parts[2]
            subsection_key = parts[3]
            return cache[pub_key]["embeddings"].get(f"section_{section_key}_{subsection_key}")
        elif len(parts) == 5: # Publication, Section, Subsection, Probe
            section_key = parts[2]
            subsection_key = parts[3]
            probe_key = parts[4]
            return cache[pub_key]["embeddings"].get(f"section_{section_key}_{subsection_key}_{probe_key}")
        else:
            return None  # Invalid key format

    except KeyError:
        return None  # Handle cases where keys are missing
    except Exception as e: #general exception
        print(f"An unexpected error occurred: {e}")
        return None


def get_text_by_key(cache: Dict[str, Any], full_key: str) -> Optional[str]:
    """Retrieves the text associated with a specific key."""
    parts = full_key.split("_")
    pub_key = parts[0]

    if pub_key not in cache:
        return None
    try:
        if len(parts) == 1:
            return cache[pub_key]['metadata']['title'] # Return the title of publication
        elif len(parts) == 3:  # Publication and Section
            section_key = parts[2]
            return cache[pub_key]["sections"][section_key]["text"]
        elif len(parts) == 4: # Publication, Section, Subsection
            section_key = parts[2]
            subsection_key = parts[3]
            return cache[pub_key]["sections"][section_key]["subsections"][subsection_key]["text"]
        elif len(parts) == 5: # Publication, Section, Subsection, Probe
            section_key = parts[2]
            subsection_key = parts[3]
            probe_key = parts[4]
            return cache[pub_key]["sections"][section_key]["subsections"][subsection_key]["probes"][probe_key]["text"]
        else:
            return None
    except KeyError:
        return None
    except Exception as e: #general exception
        print(f"An unexpected error occurred: {e}")
        return None

def filter_vectors(
    cache: Dict[str, Any],
    paper_type: Optional[str] = None,
    min_score: Optional[Dict[str, float]] = None,
    max_results: Optional[int] = None,
) -> List[str]:
    """
    Filters vectors in the cache based on metadata and score criteria.

    Args:
        cache: The loaded embedding cache.
        paper_type:  Optional paper type to filter by (e.g., "Experimental").
        min_score: Optional dictionary of {probe_key: min_score}.
        max_results: Optional maximum number of results (for future use, if needed).
    Returns:
        A list of full_keys (str) of the filtered vectors.
    """

    filtered_keys = []

    for pub_key, pub_data in cache.items():
        # Filter by paper_type (if provided)
        if paper_type and pub_data["metadata"].get("paper_type") != paper_type:
            continue

        # Iterate through all possible keys (sections, subsections, probes)
        for section_key, section_data in pub_data["sections"].items():
            for subsection_key, subsection_data in section_data["subsections"].items():
                for probe_key, probe_data in subsection_data["probes"].items():
                    full_key = f"{pub_key}_{section_key}_{subsection_key}_{probe_key}"

                    # Filter by score (if provided)
                    if min_score:
                        for score_key, score_threshold in min_score.items():
                            # Check if this probe matches the score_key
                            if score_key in probe_key: #Use "in" to find in a full name of probe
                                probe_score = probe_data.get("score")
                                if probe_score is None or probe_score < score_threshold:
                                    break  # Skip this vector
                        else:
                            # All score criteria met (or no score criteria)
                            filtered_keys.append(full_key)
                    else:
                        # No score criteria, so include the key
                        filtered_keys.append(full_key)

        if max_results and len(filtered_keys) >= max_results: #Added to be compatible
            break
    # Apply max_results (if provided)
    if max_results:
        return filtered_keys[:max_results]
    else:
        return filtered_keys

def parse_markdown_output(markdown_text: str) -> Dict[str, Any]:
    """Parses Markdown, extracting sections, subsections, probes, scores, and justifications."""
    logger = logging.getLogger(__name__)
    logger.debug("--- Starting parse_markdown_output (REVISED LOGIC) ---")
    sections = {}
    current_section = None
    current_subsection = None
    current_probe = None
    subsection_regex = r"^###\s+\**(\d*\.*\d*\s*[\w\s\d\.\:\-]+?)\**(?:\s|$)"

    lines = markdown_text.splitlines()

    for line_number, line in enumerate(lines, 1):
        stripped_line = line.strip()
        original_line = line  # Keep original line for text accumulation
        logger.debug(f"Line {line_number}: '{stripped_line}'")

        # Match Section (##)
        if section_match := re.match(r"^##\s+(.+)$", stripped_line):
            current_section = section_match.group(1).strip()
            sections[current_section] = {
                "subsections": {},
                "text": ""  # Initialize section text
            }
            current_subsection = None
            current_probe = None
            logger.debug(f"✅ Section MATCH: '{current_section}'")
            continue

        # Match Subsection (###)
        if current_section and (subsection_match := re.match(subsection_regex, stripped_line)):
            current_subsection = subsection_match.group(1).strip()
            sections[current_section]["subsections"][current_subsection] = {
                "probes": {},
                "text": ""  # Initialize subsection text
            }
            current_probe = None
            logger.debug(f"  ✅ Subsection MATCH: '{current_subsection}'")
            continue

        # Match Probe (####)
        if current_section and current_subsection and (probe_match := re.match(r"^####\s*(.+?)\s*$", stripped_line)):
            current_probe = probe_match.group(1).strip()
            sections[current_section]["subsections"][current_subsection]["probes"][current_probe] = {
                "text": "",
                "score": None,
                "justification": ""
            }
            logger.debug(f"    ✅ Probe MATCH: '{current_probe}'")
            continue

        # Text accumulation logic
        if current_probe:
            # Add to probe text
            sections[current_section]["subsections"][current_subsection]["probes"][current_probe]["text"] += original_line + "\n"
            logger.debug(f"      📝 Probe text: {current_probe}")

            # Extract score and justification ONLY if we are inside a probe
            if score_match := re.search(r"Score:\s*(\d+)", original_line):
                sections[current_section]["subsections"][current_subsection]["probes"][current_probe]["score"] = int(score_match.group(1))
            if justification_match := re.search(r"Justification:\s*(.*)", original_line, re.IGNORECASE):
                sections[current_section]["subsections"][current_subsection]["probes"][current_probe]["justification"] = justification_match.group(1).strip()

        elif current_subsection:
            # Add to subsection text
            sections[current_section]["subsections"][current_subsection]["text"] += original_line + "\n"
            logger.debug(f"    📝 Subsection text: {current_subsection}")

        elif current_section:
            # Add to section text
            sections[current_section]["text"] += original_line + "\n"
            logger.debug(f"  📝 Section text: {current_section}")

    logger.debug(f"Parsing complete. Found {len(sections)} sections")
    return sections
