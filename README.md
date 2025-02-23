# IntelliDE: Intelligence Discovery Engine

**Towards a Formal Definition of Cognition in Non-Biological Systems**

## Overview

The **Intelligence Discovery Engine (IntelliDE)** is a computational framework designed to facilitate the exploration and definition of intelligence in non-biological systems, with a particular focus on material intelligence. This project leverages advanced natural language processing (NLP) techniques, including Large Language Models (LLMs) and topic modeling, to analyze scientific literature and extract key insights related to the emerging field of intelligent materials.

The engine aims to:

1.  **Identify and characterize** different types or categories of material intelligence from a corpus of scientific publications.
2.  **Uncover the underlying principles and mechanisms** that enable intelligence in non-biological matter.
3.  **Discover hierarchical relationships and overlaps** between different approaches to achieving material intelligence.
4.  **Define criteria** for evaluating the "intelligence" or cognitive capabilities of these systems.
5.  **Generate insights** that can guide the design and development of novel intelligent materials.
6.  **Automate** the process of scientific discovery.

## Core Features

*   **Automated Text Processing Pipeline:** The engine processes a collection of research papers, extracts relevant information, and prepares it for analysis.
*   **LLM-Powered Analysis:** Employs Large Language Models (LLMs) through OpenRouter API to perform in-depth analysis of individual publications based on a structured framework, mimicking the critical evaluation of a Nature/Science referee.
*   **Topic Modeling with BERTopic:** Uses BERTopic, a state-of-the-art topic modeling technique, to cluster publications based on semantic similarity and identify key topics or categories of material intelligence [https://maartengr.github.io/BERTopic/](https://maartengr.github.io/BERTopic/).
*   **Hierarchical Topic Analysis:** Explores hierarchical relationships between topics, revealing a structured understanding of the field.
*   **Cluster Analysis and Evaluation:** Analyzes the characteristics of each cluster, generates descriptive titles and summaries using LLMs, and evaluates individual publications against cluster-specific criteria.
*   **Interactive Visualizations:** Generates various visualizations to help explore the data and interpret the results, including:
    *   Topic frequency plots (with and without outliers)
    *   Intertopic distance maps
    *   Topic hierarchy visualizations
    *   t-SNE plots of publications in 2D, colored by cluster, with interactive hover information
    *   Radar charts representing the average evaluation scores for each cluster
    *   Plot publications in 2D space with their distances to cluster centroids.
*   **Markdown and JSON Output:** Saves analysis results in both human-readable Markdown format and structured JSON format for further processing.

## Table of Contents

1.  [Introduction](#introduction)
2.  [Purpose and Goals](#purpose-and-goals)
3.  [The CT-GIN Framework](#the-ct-gin-framework)
4.  [System Architecture and Strategy](#system-architecture-and-strategy)
    *   [Everything is a Vector](#everything-is-a-vector)
    *   [Hierarchical Data Structure](#hierarchical-data-structure)
    *   [Caching and Idempotency](#caching-and-idempotency)
    *   [LLM Interaction](#llm-interaction)
    *   [Filtering and Contextualization](#filtering-and-contextualization)
    *   [Extensibility](#extensibility)
5.  [Advantages of this Approach](#advantages-of-this-approach)
6.  [File Structure and Descriptions](#file-structure-and-descriptions)
7.  [Installation and Setup](#installation-and-setup)
8.  [Usage](#usage)
    *   [Initial Processing (`main.py`)](#initial-processing-mainpy)
    *   [Re-Analysis from Cache (`analysis_from_cache.py`)](#re-analysis-from-cache-analysis_from_cachepy)
    *   [Filtering Examples](#filtering-examples)
9. [Dependencies](#dependencies)
10. [Contributing](#contributing)
11. [License](#license)

## Introduction

This project implements a novel framework, based on Category Theory (CT) and Graph Isomorphism Networks (GIN), for analyzing scientific literature.  The goal is to move beyond simple keyword or RAG searches and towards a **structured, interconnected and navegable knowledge base** from structured self-evolving templates, where existing knowledge is re-evaluated in the context of new information. Dynamic graph structure captures not just individual facts, but also the *relationships* between them. It can be used to:

*   Identify emergent patterns and relationships *between* different realizations of material intelligence.
*   Characterize the capabilities and limitations of existing systems.
*   Guide future research and development in the field.
* Perform deep re-analysis of the publication using existing knowledge from the database.

The system uses Large Language Models (LLMs) to extract information from scientific publications (PDFs) and represent it as a network of interconnected vectors.  This allows for sophisticated querying, filtering, and analysis, going beyond traditional literature review methods. As a result it creates a standardized, granular, and richly interconnected "knowledge tree," rather than just a collection of isolated documents.

## Purpose and Goals

The primary purposes of this system are:

1.  **Structured Knowledge Extraction:**  To automatically extract key information from scientific papers based on a predefined template (the CT-GIN framework, described below).  This includes not just keywords, but also:
    *   Quantitative parameters and their units.
    *   Scores assigned by the LLM based on specific criteria.
    *   Justifications for those scores.
    *   Relationships between different parts of the analysis (e.g., which section justifies a particular score).
    *   Metadata from BibTeX files.

2.  **Creation of a Vector-Based Knowledge Base:**  To represent *all* extracted information as vectors, allowing for:
    *   **Concept mapping:**  Finding concepts in publications, sections, or even individual probes that are semantically similar.
    *   **Filtering:** Selecting information based on metadata (e.g., publication type, year) and scores.
    *   **Aggregation and Synthesis:** Combining information from multiple sources to generate new insights.

3.  **Iterative and Idempotent Processing:** To ensure that:
    *   Publications are processed one at a time.
    *   Processing can be interrupted and resumed without re-doing work.
    *   The system avoids redundant LLM calls by caching results.

4.  **Deep Analysis:** To enable analysis of a given publication, *in the context of the entire knowledge base*.  This allows the LLM to compare and contrast the publication with others, identifying potential gaps, inconsistencies, or novel contributions.

5. **Facilitate Discovery:** To make the discovery of a new type of material intelligence based on similar principles or mechanisms easier.

## The CT-GIN Framework

The core of the system is the CT-GIN framework, which is defined in a Markdown template (`paper6GIN_10.md`).  This template specifies a series of modules (sections) and probes (subsections) that cover key aspects of material intelligence, including:

*   **System Overview & Implementation (M1):**  Basic description, implementation clarity, key parameters.
*   **Energy Flow (M2):**  Energy input, transduction, efficiency, and dissipation.
*   **Memory (M3):**  Presence, type, retention time, capacity, and other memory-related properties.
*   **Self-Organization and Emergent Order (M4):**  Local interaction rules, global order, and predictability.
*   **Computation (M5):** Embodied computation capabilities.
*   **Temporal Dynamics (M6):**  Relevant timescales and active inference.
*   **Adaptation (M7):**  Adaptive plasticity and learning mechanisms.
*   **Emergent Behaviors (M8):**  Description and robustness of behaviors.
*   **Cognitive Proximity (M9):**  Mapping to cognitive processes and overall cognitive-like behavior assessment.
*   **Criticality Assessment (M10):**  Whether the system operates near a critical point.
*   **Review/Theoretical Paper Specifics (M11, M12):** Conditional modules for specific paper types.
*   **Overall Assessment & Scoring (M13):**  CT-GIN readiness score and qualitative assessment.
* **CT-GIN Knowledge Graph (M14):** Schematic representation.
* **CT-GIN Template Self-Improvement Insights (M15):** Feedback on template

Each probe within the template has a unique identifier (e.g., "M1.1", "M3.2").  The LLM is instructed to answer each probe based on the content of the publication being analyzed, and provide scores and justifications where appropriate.

## System Architecture and Strategy

### Everything is a Vector

The fundamental principle is that *every* piece of extracted information is represented as a vector.  This includes:

*   The entire publication's content.
*   Each section of the CT-GIN template.
*   Each subsection (probe) within each section.
*   The text of each answer.
*   Metadata (title, authors, year, etc.).

These vectors are generated using the SciBERT model (`allenai/scibert_scivocab_uncased`), which is specifically trained on scientific text.

Okay, here's the corrected and improved "Enhanced Vectorization and Storage" section for your README.md, accurately reflecting the current code implementation:


*   **Vector Classes:** The system uses a set of Python classes to represent different types of information as vectors. This object-oriented approach provides structure and allows for easy access to attributes. The key vector classes are:

    *   `Vector`: A base class for all vectors, containing common attributes like `vector_id`, `vector_type`, and `metadata`.
    *   `TextVector`: Represents textual content, storing the `content` string and its SciBERT embedding.
    *   `ScoreVector`: Represents a numerical score, storing the `score` value.
    *   `ParameterVector`: Represents a parameter, storing the `parameter_name`, `value`, and optional `units`.
    *   `TableVector`: Represents a table, storing `headers` and `rows`.
    *   `BinaryVector`: Represents a binary value (e.g., "Yes" or "No").
    *   `RelationshipVector`: Represents a relationship *between* two other vectors, storing the `relationship_type`, `source_vector_id`, and `target_vector_id`.  This is *crucial* for encoding the hierarchical structure and connections within the data.
    *   `TemplateInstanceVector`: Represents the *entire analysis* of a single publication.  It contains a list of `SectionVector` objects (and potentially other top-level vectors in the future).
    *   `SectionVector`: Represents a section within the CT-GIN template (e.g., "M1: System Overview"). It contains a list of `SubsectionVector` objects, as well as `RelationshipVector` objects that define the relationships between sections and subsections.
    *   `SubsectionVector`: Represents a subsection (probe) within a section.  It contains a `content` list, which holds a collection of other vectors (`TextVector`, `ScoreVector`, `ParameterVector`, `TableVector`, and `RelationshipVector` instances) that represent the extracted information for that probe.

*   **Hierarchical Relationships:** The hierarchical relationships between these vectors are encoded through a combination of nesting and relationship vectors:

    *   **Nesting:** `TemplateInstanceVector` objects directly contain `SectionVector` objects. `SectionVector` objects, in turn, directly contain `SubsectionVector` objects (and relationship vectors to other sections.) This nesting mirrors the structure of the CT-GIN template itself.

    *   **Relationship Vectors:** `RelationshipVector` objects are used to explicitly link vectors. Critically, these are used to represent the relationship:
        *   Between a `SectionVector` and its `SubsectionVector` objects (`relationship_type = "has_subsection"`).
        *	Between publication text vector and the template instance (`relationship_type = "has_template_instance`).
        *	Between section and template instance (`relationship_type = "has_section"`).
        *   Between a `SubsectionVector` and the various vectors within its `content` (e.g., a `TextVector` representing a probe answer, a `ScoreVector` for the probe's score, etc.). For these, relationship types like `"has_probe"`, `"has_parameter"`, `"has_table"` are used.

*   **Metadata:** Each vector includes rich metadata, stored in a `metadata` dictionary.  This metadata includes:

    *   `id`: A universally unique identifier (UUID) for the vector *within its metadata*.  This ensures uniqueness even if other parts of the ID are the same.
    *   `bibtex_key`: The BibTeX key of the publication (e.g., "friston_free_2023"). This is prepended to the `vector_id` of all vectors derived from that publication, providing a clear link back to the source.
    *   `date`: The publication year (extracted from BibTeX).
    *   `title`: The publication title (extracted from BibTeX).
    *   `paper_type`: The type of paper (e.g., "Experimental", "Theoretical", "Review", "Perspective"), extracted from the publication content *early* in the processing pipeline.
    *   `section_id`: The ID of the section (e.g., "M1").
    *   `subsection_id`: The ID of the subsection (e.g., "1.1").  If a section doesn't have explicit subsections, a "default" subsection is used.
    *   `gin_node_type`: A string indicating the type of node this vector could represent in a Graph Isomorphism Networks (e.g., "publication", "section", "subsection", "probe", "score", "parameter," "table", "template_instance").  This facilitates potential future graph-based analysis.
    *   `gin_edge_type`: For `RelationshipVector` objects, this indicates the type of relationship (e.g., "has_probe", "has_parameter", "has_subsection").

*   **Storage:** The entire hierarchical structure, including all vectors and their metadata, is stored in a Python dictionary. This dictionary is then serialized to a Pickle file (`.embedding_cache/vectors_cache.pkl`) for persistent storage. While a dedicated vector database (e.g., ChromaDB, Faiss) could provide performance benefits for extremely large datasets, the current implementation uses Pickle for simplicity and ease of development. The in-memory dictionary structure directly mirrors the hierarchical relationships defined by the vector classes and their containment/relationship properties. This is *not* a flattened representation; it's a true nested structure.

**Conceptual Data Structure:**

```
TemplateInstanceVector (for a single publication)
    - vector_id:  "publication_key_templateInstance_..."
    - metadata: {bibtex_key: "publication_key", ...}
    - sections: [
        SectionVector
            - vector_id: "publication_key_section_M1_..."
            - metadata: {section_id: "M1", ...}
            - subsections: [
                SubsectionVector
                    - vector_id: "publication_key_subsection_M1_1.1_..."
                    - metadata: {subsection_id: "1.1", ...}
                    - content: [
                        TextVector (for probe text)
                            - vector_id: "publication_key_text_M1_1.1_probe_..."
                            - content: "..."
                            - embedding: [...]
                        ScoreVector (if applicable)
                            - vector_id: "publication_key_score_M1_1.1_probe_..."
                            - score: 8
                        RelationshipVector (linking subsection to probe text)
                            - vector_id: "publication_key_rel_has_probe_..."
                            - relationship_type: "has_probe"
                            - source_vector_id: (ID of SubsectionVector)
                            - target_vector_id: (ID of TextVector)
                        ... (other vectors for parameters, tables, etc.)
                    ]
                    - embedding: [...]
                 RelationshipVector (linking section to subsection)
                   -vector_id: "publication_key_rel_has_subsection..."
                   -relationship_type: "has_subsection"
                   -source_vector_id: ... (id Section Vector)
                   -target_vector_id: ... (id SubsectionVector)
                ... (other subsections)
            ]
            - embedding: [...]
        ... (other sections)
    ]
```

### Caching and Idempotency

The system uses a Pickle file (`.embedding_cache/vectors_cache.pkl`) to cache the entire knowledge base, *not* `embeddings_cache.pkl`. This is crucial for ensuring that processing is both efficient and idempotent.

*   **Idempotency:** Before processing a publication, the system checks if the cache (the `vectors_cache.pkl` file) already contains an entry for that publication's BibTeX key.  If the key exists, the publication is skipped, preventing redundant processing.  There is no separate `processing_complete` flag; the presence of the key in the cache *itself* indicates that the publication has been processed.

*   **Caching:**  The cache stores the *entire* hierarchical structure of `TemplateInstanceVector`, `SectionVector`, `SubsectionVector`, and all contained content vectors (`TextVector`, `ScoreVector`, etc.), including their embeddings and metadata.  It does *not* use a separate hierarchical key for individual embeddings.  Instead, each vector object (which includes its embedding) is stored within the nested structure.  This simplifies access and maintains the relationships between vectors.  The cache is invalidated and rebuilt if:
    *   The input PDF changes.
    *   The CT-GIN template (`paper6GIN_10.md`) changes.
    *   The prompt file (`analyze_paper.txt`) changes.
    *   The code logic within `main.py` that handles parsing or vector creation changes (this would usually be detected by a change in the version control system).
    *   The LLM provider or model changes.

*   **Incremental Updates:** The cache (`vectors_cache.pkl`) is loaded at the beginning of the `main.py` script.  It is then saved *after processing each publication*. This ensures that even if the script is interrupted (e.g., due to a power outage or error), the work done up to that point is preserved.  The next time the script is run, it will resume from where it left off, skipping already-processed publications.

### LLM Interaction

The system uses the `OpenRouter_Methods.py` module to interact with LLMs.  This module supports both the OpenRouter API and the Google Gemini API, with automatic retries and error handling.  The `get_llm_response` function provides a unified interface, allowing you to switch between providers using the `LLM_PROVIDER` constant.

### Analysis of user PDFs based on created Knowledge Graph

The `analysis_from_cache.py` script provides functionality for re-analyzing a publication *in the context of the entire knowledge base*.  It does this by:

1.  **Filtering:**  The `filter_vectors` function allows you to select vectors based on metadata (e.g., `paper_type`) and scores (e.g., `min_score` for a specific probe).
2.  **Context Generation:** The `generate_analysis_prompt` function constructs a prompt for the LLM that includes:
    *   Information from the *query* publication.
    *   Information from *other* publications that match the filter criteria.
3.  **LLM Re-Analysis:** The LLM is then given this enriched prompt, allowing it to perform a more informed and contextualized analysis.

### Extensibility
The system includes possibility to build GIN (Graph Isomorphism Networks) structure to provide additional knowledge representation as a graph, and possibility to extend it in a future.


## File Structure and Descriptions

```
project_root/
├── main.py             # Main processing script (builds the knowledge base).
├── analysis.py          # Script for performing BERTopic and creating graphs.
├── utils.py            # Utility functions for files operations (PDF extraction, BibTeX parsing, etc.).
├── vector_fns.py       # Utility functions for vector creation.
├── OpenRouter_Methods.py # LLM API interaction.
├── prompts/
│   ├── analyze_paper.txt   # Prompt template for the *initial* analysis in main.py.
│   └── perform_clustering_prompt.txt  # Prompt template for the description of each *cluster*.
├── Catechism/              # Contains the analysis template and background
│   ├── paper6GIN_10.md     # The CT-GIN analysis template in machine readable format (Markdown format).
│   ├── paper6.md           # The analysis template in human readable format (Markdown format).
│   └── Perspective2.md     # Background information on the research field.
├── ../AI-4/              # Directory containing PDF files to be processed by main.py.
├── ../synthetic/publications6GIN10/  # Output directory for Markdown reports.
└── .embedding_cache/
    └── vectors_cache.pkl  # The central vector database file (Pickle format).
```

*   **`main.py`:** The core script for processing individual publications.  It extracts text from PDFs, interacts with the LLM, parses the Markdown output, generates embeddings, and updates the cache.
*   **`utils.py`:** Contains utility functions used by both `main.py` and `analysis_from_cache.py`, including:
    *   `setup_logging`: Sets up logging.
    *   `load_template`: Loads the CT-GIN template from a Markdown file.
    *   `extract_pdf_text`: Extracts text from a PDF file using `pdfplumber`.
    *   `parse_bib_file`: Parses a BibTeX file to extract publication metadata.
    *   `save_markdown_report`: Saves text to a Markdown file.
    *   `load_prompt`: Loads text from a file (used for prompts).
    *   `read_description`: Reads the content of a text file.
    *   `generate_paper_prompt`:  Generates the prompt for the initial LLM analysis.
    *   `extract_paper_type`: extract paper type from the text.
    *   `parse_markdown_output`: parse the markdown output of LLM.

*   **`OpenRouter_Methods.py`:** Handles interaction with LLM APIs (OpenRouter and Google Gemini), including retries and error handling.
*   **`analysis.py`:** Contains different functions for BERTopic analysis and visualization.
*   **`vector_fns.py`:** Creates embeddings and defines vector classes.
*   **`prompts/analyze_paper.txt`:**  The prompt template used by the LLM for the initial analysis (in `main.py`).
*    **`prompts/perform_clustering_prompt.txt`:** prompt for LLM for cluster analysis.
*   **`../Catechism/paper6GIN_10.md`:** The Markdown template defining the CT-GIN framework.  This file *defines* the structure of the analysis.
*   **`../Catechism/Perspective2.md`:** A file containing the background information of the target research field (used as context in analysis).
*   **`../AI-4/`:**  A directory containing the PDF files to be analyzed.  `main.py` processes all PDFs in this directory and subdirectories.
*   **`../synthetic/publications6GIN10/`:** The output directory. `main.py` creates Markdown files (`.md`) here, one for each processed publication.
*   **`.embedding_cache/`:**  A directory containing the cache.
    *   **`vectors_cache.pkl`:**  The Pickle file containing the *entire* knowledge base (embeddings, metadata, hierarchical structure, and processing status).

## Installation and Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/vbaulin/IntelliDE/
    cd IntelliDE/
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    .venv\Scripts\activate  # On Windows
    ```

4.  **Set Up API Keys:**
    *   Create a `.env` file in the root directory of the project.
    *   Obtain API keys for your chosen LLM provider(s):
        *   **OpenRouter:** Obtain an API key from the OpenRouter website.
        *   **Google Gemini:** Obtain an API key from Google AI Studio.
    *   Add the API keys to your `.env` file:

        ```
        OPENROUTER_API_KEY=your_openrouter_api_key
        GEMINI_API_KEY=your_gemini_api_key
        ```
    *   **Choose LLM:** In `main.py`, set the `LLM_PROVIDER` constant to either `"openrouter"` or `"google"`.

5.  **Place PDF Files:** Place the PDF files you want to analyze in the `../AI-4/` directory with corresponding bib file created by Zotero. For this export collection from Zotero library with PDFs included.

## Usage

### Initial Processing (`main.py`)

Run `main.py` to perform the initial processing of the publications:

```bash
python main.py
```

This script will:

1.  Load the CT-GIN template (`paper6GIN_10.md`).
2.  Load LLM prompt template (`prompts/analyze_paper.txt`)
3.  Load the existing vector database (if any).
4.  Load bib file and iterate through all PDF files in the `../AI-4/` directory (and its subdirectories with fies).
5.  For each PDF:
    *   Extract the text.
    *   Extract paper type
    *   Generate a prompt for the LLM based on templates.
    *   Call the LLM to analyze the publication according to the template.
    *   Parse the LLM's Markdown output.
    *   Generate SciBERT embeddings for the publication, sections, subsections, and probes.
    *   Store the extracted data (text, scores, embeddings, metadata) in the vector database.
    *   Save the raw Markdown output to a file in the `../synthetic/publications6GIN10/` directory.
6.  Load vector cache for BERTopic analysis.
7.  Run BERTopic analysis.
8. Save the updated cache.

If the script is interrupted, you can simply re-run it.  It will automatically skip any publications that have already been fully processed (thanks to the cache and the `processing_complete` flag).

## Results

The Intelligence Discovery Engine (IdeliDE) analyzes scientific literature on material intelligence to produce a range of visualizations that illuminate the structure and relationships within this emerging field.

A key output is an interactive t-SNE plot that maps publications as points in a 2D space, colored by their assigned cluster, allowing users to explore the semantic similarity between documents and identify clusters of related research; hovering over a point reveals the publication's key, and circles are drawn around clusters to highlight potential overlaps.

![Cluster](Results/cluster.png)

The 2D visualization of publications generated by the Intelligence Discovery Engine is a scatter plot that represents each publication as a point in a two-dimensional space, primarily using t-SNE (t-distributed Stochastic Neighbor Embedding) for dimensionality reduction, with an option to use UMAP as well. This plot is designed to visually demonstrate the relationships and clustering of research papers based on their semantic similarity as determined by their high-dimensional embeddings.

![Overlap](Results/Overlap.png)

The program also generates a topic similarity matrix heatmap that uses LLM-generated titles to label axes, offering an intuitive visualization of how similar or dissimilar the identified topics are to each other.

![Similarity](Results/Similarity.png)

It also generates topic hierarchy plot, enhanced with LLM titles, presents a tree-like structure of the topics, revealing the hierarchical relationships between them.

![Hierarchy](Results/Hierarchy.png)

A topic frequency plot displays the prevalence of each identified topic using a bar chart, with a clear distinction between the frequency of topics in the entire dataset and in the dataset excluding outliers (topic -1), providing insights into the core themes and the presence of noise.  Lastly, radar charts are used to evaluate publications against predefined criteria relevant to material intelligence (such as the presence of specific sections or keywords extracted from the documents), with each cluster represented in a separate subplot showing the average scores across multiple evaluation criteria, effectively summarizing each cluster's alignment with aspects of intelligent behavior in soft matter.


## Dependencies

The project requires the following Python libraries:

*   `numpy`
*   `torch`
*   `transformers`
*    `scikit-learn`
*    `pathlib`
*    `bibtexparser`
*   `pdfplumber`
*   `openai`
*   `backoff`
*   `python-dotenv`
*    `google-generativeai`
*   `tqdm`
*   `sentence_transformers`
*    `bertopic`
*    `umap-learn`
*   `matplotlib`
*    `networkx`
*    `seaborn`
*   `plotly`
*   `pandas`
*   `scipy`
*   `textblob`

These can be installed using `pip install -r requirements.txt`.

## Contributing

Contributions to this project are welcome! Please follow these guidelines:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with clear, descriptive messages.
4.  Submit a pull request.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
