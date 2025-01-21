import os
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import Counter
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
import plotly.express as px
import logging
import time
import backoff
import json
from openai import OpenAI
from tqdm import tqdm
import plotly.graph_objects as go

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from utils import (
    load_data_from_json,
    load_template,
    extract_answers,
    get_embedding,
    combine_embeddings,
    plot_within_cluster_distances,
    save_topic_tree
)
from analysis import (
    analyze_clusters,
    visualize_intertopic_distance_map,
    visualize_clusters,
    perform_bertopic_analysis,
    calculate_topic_probabilities,
    visualize_topic_hierarchy,
    visualize_topic_similarity,
    analyze_topic_evolution,
    analyze_topic_distribution_by_section,
    visualize_embeddings_with_tsne,
    plot_topic_distances,
    visualize_representative_docs,
    plot_topic_frequencies_stacked,
    plot_topic_frequencies_grouped,
    plot_topic_frequencies_lollipop,
    plot_topic_frequencies_dot
)
from OpenRouter_Methods import get_openrouter_client, get_openrouter_response, load_prompt

# --- Configuration ---
DATA_SOURCE = "json_file"
JSON_FILE_PATH = "../synthetic/publications"
TEMPLATE_FILE = "../Catechism/Paper_Template.md"
PERSPECTIVE_FILE = "../Catechism/Perspective.md"
CLUSTER_ANALYSIS_OUTPUT_DIR = "cluster_analysis_results"

def analyze_material_intelligence(publications, template_sections, embedding_model):
    """
    Analyzes material intelligence, creating a hierarchical vector representation
    for each publication, including section and subsection embeddings.
    """
    publication_data = {}

    for publication in tqdm(publications, desc="Analyzing publications"):
        key = publication.get("metadata", {}).get("key")
        if not key:
            print("Warning: Publication is missing 'key' in metadata. Skipping.")
            continue

        answers = extract_answers(publication, template_sections)

        publication_data[key] = {
            "answers": answers,
            "section_embeddings": {},
            "subsection_embeddings": {},
            "publication_embedding": None,
        }

        publication_vector = []
        for section_key, section_data in template_sections.items():
            section_vector = []
            publication_data[key]["section_embeddings"][section_key] = {}
            for subsection_key, subsection_answers in answers[section_key].items():
                if subsection_answers:
                    subsection_embedding = get_embedding(
                        subsection_answers,
                        embedding_model,
                        key,
                        section_key,
                        subsection_key,
                    )
                    if subsection_embedding is not None:
                        publication_data[key]["subsection_embeddings"][
                            subsection_key
                        ] = subsection_embedding
                        section_vector.append(subsection_embedding)
                    else:
                        print(
                            f"      -- ⚠️ Warning: Could not generate embedding for subsection {subsection_key} in publication {key}."
                        )
                else:
                    print(
                        f"      -- ⚠️ Warning: No answer found for subsection {subsection_key} in publication {key}."
                    )

            if section_vector:
                section_embedding = np.mean(section_vector, axis=0)
                publication_data[key]["section_embeddings"][section_key] = section_embedding
                publication_vector.append(section_embedding)
            else:
                print(
                    f"      -- ⚠️ Warning: No valid subsection embeddings found for section {section_key} in publication {key}."
                )

        if publication_vector:
            publication_embedding = np.mean(publication_vector, axis=0)
            publication_data[key]["publication_embedding"] = publication_embedding
        else:
            print(
                f"    -- ❌ Warning: No valid section embeddings found for publication {key}."
            )

    return publication_data


def evaluate_publications_against_criteria(publication_data, cluster_analysis_dir, template_sections):
    cluster_evaluations = {}

    for filename in os.listdir(cluster_analysis_dir):
        if filename.startswith("cluster_") and filename.endswith(".json"):
            filepath = os.path.join(cluster_analysis_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                cluster_data = json.load(f)

            cluster_label = cluster_data["label"]
            cluster_criteria = cluster_data["description"]

            # Convert section_probabilities to a dictionary
            section_probabilities_dict = {
                section_key: dict(counts)
                for section_key, counts in cluster_data["section_probabilities"].items()
            }

            # Extract subsection names from template_sections
            subsection_names = []
            for section_key, section_data in template_sections.items():
                for subsection_key, subsection_name in section_data["subsections"].items():
                    subsection_names.append(subsection_name)

            cluster_evaluations[cluster_label] = []

            for pub_key in cluster_data["keys"]:
                if pub_key in publication_data:
                    publication = publication_data[pub_key]

                    evaluation_results = {}

                    # Example: Calculate a score based on the presence and probability of each subsection
                    for subsection_name in subsection_names:
                        subsection_score = 0
                        for section_key, section_data in template_sections.items():
                            if subsection_name in section_data["subsections"].values():
                                subsection_key = list(section_data["subsections"].keys())[list(section_data["subsections"].values()).index(subsection_name)]
                                if section_key in publication["answers"] and subsection_key in publication["answers"][section_key]:
                                    if publication["answers"][section_key][subsection_key]:
                                        # Check if probabilities_per_document is available and use it
                                        if cluster_data["probabilities_per_document"]:
                                            # Find the index of the publication in the cluster keys
                                            pub_index = cluster_data["keys"].index(pub_key)
                                            # Ensure the index is within the bounds of probabilities_per_document
                                            if pub_index < len(cluster_data["probabilities_per_document"]):
                                                topic_probs = cluster_data["probabilities_per_document"][pub_index]
                                                # Find the probability for the current cluster label
                                                topic_prob = topic_probs[cluster_label] if cluster_label < len(topic_probs) else 0
                                                subsection_score = topic_prob  # Use probability as the score
                                            else:
                                                print(f"Warning: Index {pub_index} is out of bounds for probabilities_per_document in cluster {cluster_label}.")
                                        else:
                                            subsection_score = 1  # Full score if present and probabilities are not used
                                        break  # Found the subsection, no need to continue searching
                        evaluation_results[subsection_name] = subsection_score

                    # Add evaluation results to cluster_evaluations
                    cluster_evaluations[cluster_label].append(evaluation_results)

    # Create radar charts for all clusters in one figure with subplots
    n_clusters = len(cluster_evaluations)
    if n_clusters > 0:
        # Define the grid layout based on the number of clusters
        if n_clusters <= 3:
            rows, cols = 1, n_clusters
        elif n_clusters == 4:
            rows, cols = 2, 2
        elif n_clusters <= 6:
            rows, cols = 2, 3
        elif n_clusters <= 9:
            rows, cols = 3, 3
        else:
            rows, cols = (n_clusters + 3) // 4, 4  # Ensure at least 4 columns

        fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), subplot_kw=dict(polar=True))
        axs = axs.flatten()

        # Ensure axs is iterable even if there's only one subplot
        if n_clusters == 1:
            axs = [axs]

        for i, (cluster_label, evaluations) in enumerate(cluster_evaluations.items()):
            if not evaluations:
                print(f"No evaluations found for cluster {cluster_label}. Skipping radar chart.")
                continue
            ax = axs[i]
            # Calculate average scores for each criterion
            avg_evaluation = {}
            for key in evaluations[0].keys():
                avg_evaluation[key] = np.mean([e[key] for e in evaluations if key in e])

            # Create radar chart data
            categories = list(avg_evaluation.keys())
            values = list(avg_evaluation.values())

            # Close the loop for the radar chart
            values += values[:1]
            categories += categories[:1]

            # Create radar chart
            ax.plot(np.linspace(0, 2 * np.pi, len(categories)), values, marker='o', linestyle='-', color='skyblue', linewidth=2, markersize=8)
            ax.fill(np.linspace(0, 2 * np.pi, len(categories)), values, color='skyblue', alpha=0.25)

            # Set title and labels
            ax.set_title(f"Cluster {cluster_label}", size=12, color='black', y=1.1)
            ax.set_xticks(np.linspace(0, 2 * np.pi, len(categories)))
            ax.set_xticklabels(categories, color='black', size=10)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color='black', size=8)
            ax.set_ylim(0, 1)
            ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.show()
    else:
        print("No clusters found for evaluation.")


def main():
    """
    Main function to run the material intelligence analysis.
    """
    # --- Load Data ---
    publications = load_data_from_json(JSON_FILE_PATH)
    print(f"✅ Loaded {len(publications)} publications from {JSON_FILE_PATH}.")

    # --- Load Template ---
    template_sections = load_template(TEMPLATE_FILE)

    # --- Initialize Sentence Transformer ---
    model_name = "all-mpnet-base-v2"
    embedding_model = SentenceTransformer(model_name)
    print(f"✅ Loaded Sentence Transformer model: {model_name}")

    # --- Analyze Material Intelligence and Get Embeddings ---
    publication_data = analyze_material_intelligence(
        publications, template_sections, embedding_model
    )
    print("✅ Material intelligence analysis complete.")

    publication_keys = list(publication_data.keys())
    all_combined_embeddings = [
        data["publication_embedding"]
        for data in publication_data.values()
        if data["publication_embedding"] is not None
    ]

    # --- Prepare Texts for BERTopic ---
    combined_texts = []
    for pub_key, data in publication_data.items():
        pub_text = ""
        for section_key, section_data in template_sections.items():
            for subsection_key in section_data["subsections"].keys():
                answer = data["answers"][section_key][subsection_key]
                pub_text += answer + " "
        combined_texts.append(pub_text.strip())

    # Create dummy texts for publications without valid combined embeddings
    dummy_text = "No content available"
    aligned_combined_texts = [
        combined_texts[i] if publication_data[key]["publication_embedding"] is not None else dummy_text
        for i, key in enumerate(publication_keys)
    ]

    # --- Run Combined BERTopic Analysis ---
    if all_combined_embeddings:
        print("\n--- Running Combined BERTopic across all publications ---")
        combined_model, combined_labels, combined_probs = perform_bertopic_analysis(
            aligned_combined_texts, embedding_model=embedding_model
        )

        if combined_model:
            # --- Analyze Clusters with LLM (and save results) ---
            print("\n--- Analyzing Clusters with LLM ---")
            # Ensure combined_model is not None before passing it to analyze_clusters
            if combined_model is not None:
                analyze_clusters(
                    combined_labels,
                    aligned_combined_texts,
                    publication_keys,
                    template_sections,
                    publication_data,
                    combined_model,
                    combined_probs,
                    PERSPECTIVE_FILE
                )
            else:
                print("⚠️ Combined model is None. Skipping cluster analysis.")

            # --- Visualization (separate step) ---
            print("\n--- Visualizing Clusters ---")
            all_cluster_data = visualize_clusters(CLUSTER_ANALYSIS_OUTPUT_DIR, template_sections, publication_data)
            if all_cluster_data is not None:
                # --- Visualize Topic Hierarchy ---
                print("\n--- Visualizing Topic Hierarchy ---")
                topic_labels = {int(label): cluster_data["title"] for label, cluster_data in all_cluster_data.items()}
                visualize_topic_hierarchy(combined_model, combined_texts, topic_labels)

                # --- Visualize Intertopic Distance Map ---
                print("\n--- Visualizing Intertopic Distance Map ---")
                visualize_intertopic_distance_map(combined_model, topic_labels)

                # --- Visualize Topic Similarity ---
                visualize_topic_similarity(combined_model, topic_labels)
            else:
                 print("all_cluster_data is None")

            # --- Plot Topic Frequencies ---
            print("\n--- Plotting Topic Frequencies ---")

            # --- Save Topic Tree (Markdown) ---
            print("\n--- Saving Topic Tree ---")
            tree = combined_model.get_topic_tree(topic_labels)
            save_topic_tree(tree)

            # Get topic frequencies, including outliers
            topic_info_all = combined_model.get_topic_info()

            # Get topic frequencies, excluding outliers
            topic_info_filtered = topic_info_all[topic_info_all.Topic != -1]

            # Convert to pandas DataFrames for easier plotting
            df_all = pd.DataFrame({"Topic": topic_info_all.Name, "Frequency": topic_info_all.Count})
            df_filtered = pd.DataFrame({"Topic": topic_info_filtered.Name, "Frequency": topic_info_filtered.Count})

            # Ensure both DataFrames have the same topics for proper alignment in plotting
            all_topics = set(df_all['Topic']).union(set(df_filtered['Topic']))

            for topic in all_topics:
                if topic not in df_all['Topic'].values:
                    df_all = pd.concat([df_all, pd.DataFrame([{'Topic': topic, 'Frequency': 0}])], ignore_index=True)
                if topic not in df_filtered['Topic'].values:
                    df_filtered = pd.concat([df_filtered, pd.DataFrame([{'Topic': topic, 'Frequency': 0}])], ignore_index=True)

            # Sort DataFrames by Topic for consistent plotting
            df_all = df_all.sort_values(by='Topic')
            df_filtered = df_filtered.sort_values(by='Topic')

            # Extract the 'Frequency' column as NumPy arrays
            frequencies_all = df_all['Frequency'].to_numpy()
            frequencies_filtered = df_filtered['Frequency'].to_numpy()

            # Extract numeric part of topic ID for sorting
            def extract_topic_id(topic_name):
                parts = str(topic_name).split('_', 1)
                return int(parts[0]) if parts[0].isdigit() else -1

            # Sort topic_labels based on numeric topic ID
            sorted_topic_labels = dict(sorted(topic_labels.items(), key=lambda item: item[0]))

            # Create a list of topic labels, sorted by topic ID
            topic_labels_list = [
                sorted_topic_labels.get(extract_topic_id(topic_id), f"Topic {topic_id}")
                for topic_id in df_filtered['Topic']
            ]

            # Choose one of the plotting functions:
            plot_topic_frequencies_stacked(frequencies_all, frequencies_filtered, topic_labels_list)
            #plot_topic_frequencies_grouped(frequencies_all, frequencies_filtered, topic_labels_list)
            #plot_topic_frequencies_lollipop(frequencies_all, frequencies_filtered, topic_labels_list)
            #plot_topic_frequencies_dot(frequencies_all, frequencies_filtered, topic_labels_list)


            # --- Evaluate Publications Against Cluster Criteria ---
            print("\n--- Evaluating Publications ---")
            evaluate_publications_against_criteria(publication_data, CLUSTER_ANALYSIS_OUTPUT_DIR, template_sections)

        else:
            print("⚠️ BERTopic model could not be trained.")

    else:
        print("⚠️ No valid combined embeddings found. Skipping combined BERTopic analysis.")

    print("\n--- Analysis Complete ---")

if __name__ == "__main__":
    main()
