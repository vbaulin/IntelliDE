import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap import UMAP
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from collections import Counter
from typing import Optional, Union, List, Tuple, Dict
from utils import load_prompt, read_markdown_file
import logging
import os
import json
from scipy.spatial import ConvexHull
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from OpenRouter_Methods import get_openrouter_client, get_openrouter_response, setup_logging, load_prompt

# --- File Paths ---
CLUSTER_ANALYSIS_OUTPUT_FILE = "cluster_analysis_material_intelligence.md"
PERFORM_CLUSTERING_TITLE_PROMPT_FILE = "perform_clustering_prompt_short.txt"
PERFORM_CLUSTERING_PROMPT_FILE = "perform_clustering_prompt.txt"
CLUSTER_ANALYSIS_OUTPUT_DIR = "cluster_analysis_results"
CLUSTER_ANALYSIS_OUTPUT_FILE = "cluster_analysis_material_intelligence.md"

def analyze_clusters(
    labels,
    texts,
    keys,
    template_sections,
    publication_data,
    model,
    probs,
    background_file
):
    unique_labels = sorted(set(labels))
    client = get_openrouter_client()
    background_info = read_markdown_file(background_file)

    os.makedirs(CLUSTER_ANALYSIS_OUTPUT_DIR, exist_ok=True)

    with open(CLUSTER_ANALYSIS_OUTPUT_FILE, "w", encoding="utf-8") as md_file:
        md_file.write("# Cluster Analysis Results\n\n")

        for label in unique_labels:
            if label == -1:
                continue

            cluster_indices = [i for i, l in enumerate(labels) if l == label]
            cluster_keys = [keys[i] for i in cluster_indices]
            cluster_texts = [texts[i] for i in cluster_indices]

            # 1. Calculate Cluster Centroid and Distances
            cluster_embeddings = [
                publication_data[key]["publication_embedding"]
                for key in cluster_keys
                if publication_data[key]["publication_embedding"] is not None
            ]
            centroid = np.mean(cluster_embeddings, axis=0) if cluster_embeddings else None

            distances = {}
            if centroid is not None:
                for key in cluster_keys:
                    if publication_data[key]["publication_embedding"] is not None:
                        distances[key] = cosine_distances(
                            [publication_data[key]["publication_embedding"]], [centroid]
                        )[0][0]

            # 2. Calculate Topic Probabilities (per section)
            section_probabilities = {
                section_key: Counter() for section_key in template_sections
            }
            for doc_index, (pub_key, topic_id) in enumerate(zip(keys, labels)):
                if pub_key in cluster_keys:
                    for section_key in template_sections:
                        section_probabilities[section_key][topic_id] += 1

            # 3. Get Representative Documents
            try:
                representative_docs = model.get_representative_docs(label)
            except Exception as e:
                print(f"Error getting representative documents for cluster {label}: {e}")
                representative_docs = []

            # 4. Get Cluster Representation from BERTopic
            topic_info = model.get_topic_info(label)
            # Use BERTopic's Name as an initial title
            cluster_title = topic_info['Name'].iloc[0] # This will be updated later by LLM
            cluster_representation = model.get_topic(label)

            # 5. Get Cluster Description from LLM
            description_prompt = load_prompt(PERFORM_CLUSTERING_PROMPT_FILE)
            if description_prompt:
                description_prompt = description_prompt.format(
                    background=background_info,
                    label=label,
                    title=cluster_title,
                    summaries=str(cluster_texts),
                    keys=str(cluster_keys),
                    distances=str(distances),
                    section_probabilities=str(section_probabilities),
                    representative_docs=str(representative_docs),
                    probabilities_per_document=str(probs[cluster_indices].tolist() if probs is not None else None),
                    cluster_representation=str(cluster_representation)
                )
                try:
                    topic_description = get_openrouter_response(client, description_prompt)
                except Exception as e:
                    print(f"❌ Error getting description from OpenRouter for cluster {label}: {e}")
                    topic_description = "No description available."
            else:
                topic_description = "No description available."

            # 6. Get LLM-Generated Title
            title_prompt = load_prompt(PERFORM_CLUSTERING_TITLE_PROMPT_FILE)
            if title_prompt:
                title_prompt = title_prompt.format(
                    description=topic_description
                )
                try:
                    llm_cluster_title = get_openrouter_response(client, title_prompt)
                except Exception as e:
                    print(f"❌ Error getting title from OpenRouter for cluster {label}: {e}")
                    llm_cluster_title = f"Cluster {label}"  # Fallback to a default title
            else:
                llm_cluster_title = f"Cluster {label}"  # Fallback if no prompt is found

            # 7. Save Results to File
            cluster_data = {
                "label": label,
                "title": llm_cluster_title,  # Use LLM-generated title
                "description": topic_description,
                "keys": cluster_keys,
                "centroid": centroid.tolist() if centroid is not None else None,
                "distances": {k: float(v) for k, v in distances.items()},
                "section_probabilities": {k: dict(v) for k, v in section_probabilities.items()},
                "texts": cluster_texts,
                "probabilities_per_document": probs[cluster_indices].tolist() if probs is not None else None,
                "representative_docs": representative_docs,
                "representation": cluster_representation
            }

            output_file = os.path.join(CLUSTER_ANALYSIS_OUTPUT_DIR, f"cluster_{label}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(cluster_data, f, indent=4)

            # 8. Write Cluster Information to Markdown File (Now with LLM-generated title)
            md_file.write(f"## {llm_cluster_title} (Cluster {label})\n\n")
            md_file.write(f"{topic_description}\n\n")

            print(f"✅ Cluster analysis results for cluster {label} saved to {output_file}")
            print(f"✅ Cluster description and title for cluster {label} added to {CLUSTER_ANALYSIS_OUTPUT_FILE}")

def perform_bertopic_analysis(
    text_data: List[str],
    embedding_model: Union[str, SentenceTransformer] = "all-mpnet-base-v2",
    min_topic_size: int = 5,
    n_gram_range: Tuple[int, int] = (1, 3),
    seed: Optional[int] = 42,
) -> Tuple[any, List[int], Optional[np.ndarray]]:
    from bertopic import BERTopic
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    vectorizer_model = CountVectorizer(
        ngram_range=n_gram_range, stop_words="english"
    )
    umap_model = UMAP(
        n_neighbors=15,
        n_components=20,
        min_dist=0.4,
        metric="cosine",
        random_state=seed,
    )
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=min_topic_size,
        nr_topics="auto",
        calculate_probabilities=True,
        verbose=True,
    )
    try:
        if any(not isinstance(doc, str) or not doc.strip() for doc in text_data):
            raise ValueError("All documents must be non-empty strings.")

        topics, probs = topic_model.fit_transform(text_data)
        logger.info("✅ BERTopic analysis completed successfully.")
        return topic_model, topics, probs
    except Exception as e:
        logger.error(f"❌ Error during BERTopic fitting or transformation: {e}")
        return None, [-1] * len(text_data), None

def calculate_topic_probabilities(combined_probs, publication_keys):
    if combined_probs is not None:
        print("\n--- Topic Probabilities for Each Document ---")
        for doc_index, pub_key in enumerate(publication_keys):
            print(f"\nDocument: {pub_key}")
            topic_probs = combined_probs[doc_index]
            sorted_probs = sorted(enumerate(topic_probs), key=lambda x: x[1], reverse=True)
            for topic_id, prob in sorted_probs:
                print(f"  Topic {topic_id}: {prob:.4f}")

def visualize_topic_hierarchy(combined_model, combined_texts, topic_labels):
    try:
        # Generate the hierarchical topics data
        hierarchical_topics = combined_model.hierarchical_topics(docs=combined_texts)

        # Create the hierarchy visualization using BERTopic's built-in function
        fig = combined_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)

        # Update node labels with LLM-generated titles from topic_labels
        for trace in fig.data:
            if trace.mode == 'markers+text':  # Check if it's a node trace
                updated_labels = []
                for label in trace.text:
                    try:
                        # Check if the label can be converted to an integer (topic ID)
                        topic_id = int(label.split('_')[0])  # Assuming label format is "0_topic"
                        # Get the corresponding title from topic_labels
                        updated_label = topic_labels.get(topic_id, label)
                        updated_labels.append(updated_label)
                    except ValueError:
                        # If the label is not an integer, keep the original label
                        updated_labels.append(label)
                trace.text = updated_labels
                trace.textposition = 'middle center'  # Ensure text is centered

        # Customize layout
        fig.update_layout(
            title="Topic Hierarchy",
            title_x=0.5,
            showlegend=True,
            margin=dict(l=40, r=40, b=85, t=100),
            plot_bgcolor='white'
        )

        fig.show()

    except Exception as e:
        print(f"Error generating hierarchical topics visualization: {e}")

def visualize_topic_similarity(combined_model, topic_labels):
    try:
        # Get the similarity matrix
        similarity_matrix = combined_model.visualize_heatmap()

        # Extract topic IDs and map them to LLM titles
        topic_ids = [int(label) for label in combined_model.get_topic_info().Topic if label != -1]
        llm_titles = [topic_labels.get(topic_id, f"Cluster {topic_id}") for topic_id in topic_ids]

        # Update the figure to use LLM titles instead of topic numbers
        fig = go.Figure(data=go.Heatmap(z=similarity_matrix.data[0].z,  # Access the z-data correctly
                                       x=llm_titles,
                                       y=llm_titles,
                                       colorscale='Viridis'))

        fig.update_layout(title='Topic Similarity Matrix with LLM Titles',
                          xaxis_title="Topic",
                          yaxis_title="Topic",
                          yaxis=dict(scaleanchor="x", scaleratio=1))  # Make y-axis scale match x-axis

        fig.show()

    except ValueError as e:
        print(f"Error visualizing topics heatmap: {e}")
        print("This may occur if topic frequencies are too low or not computed.")

def analyze_topic_evolution(combined_model, combined_texts, combined_labels, timestamps):
    if combined_model and all(timestamps):
        try:
            topics_over_time = combined_model.topics_over_time(
                docs=combined_texts,
                topics=combined_labels,
                timestamps=timestamps
            )
            fig = combined_model.visualize_topics_over_time(topics_over_time)
            fig.show()
        except Exception as e:
            print(f"Error visualizing topics over time: {e}")

def analyze_topic_distribution_by_section(template_sections, publication_keys, combined_labels, publication_data):
    section_topic_distribution = {}
    for section_key, section_data in template_sections.items():
        section_topic_distribution[section_key] = Counter()

    for doc_index, (pub_key, topic_id) in enumerate(zip(publication_keys, combined_labels)):
        if pub_key in publication_data:
            for section_key, section_data in template_sections.items():
                section_name = section_data["name"]
                if "answers" in publication_data[pub_key]:
                    for subsection_key, subsection_answers in publication_data[pub_key]["answers"][section_key].items():
                        if topic_id != -1:
                            section_topic_distribution[section_key][topic_id] += 1

    for section_key, section_data in template_sections.items():
        section_name = section_data["name"]
        print(f"\n--- Topic Distribution in Section: {section_name} ---")
        total_topics = sum(section_topic_distribution[section_key].values())
        for topic_id, count in section_topic_distribution[section_key].most_common():
            percentage = (count / total_topics) * 100 if total_topics > 0 else 0
            print(f"  Topic {topic_id}: {count} ({percentage:.2f}%)")

def visualize_embeddings_with_tsne(all_combined_embeddings, combined_labels, combined_topic_labels, combined_model):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_embeddings = tsne.fit_transform(np.array(all_combined_embeddings))
    plt.figure(figsize=(10, 8))
    unique_labels = sorted(set(combined_labels))
    n_colors = len(unique_labels)
    colormap = plt.cm.get_cmap("jet", n_colors)

    for i, label in enumerate(unique_labels):
        if label == -1:
            continue  # Skip outlier topic
        indices = [j for j, l in enumerate(combined_labels) if l == label]
        color = colormap(i)

        # Scatter plot of points
        plt.scatter(
            tsne_embeddings[indices, 0],
            tsne_embeddings[indices, 1],
            color=color,
            label=combined_topic_labels.get(label, f"Cluster {label}"),
        )

        # Calculate the centroid of each cluster
        centroid = np.mean(tsne_embeddings[indices], axis=0)

        # Add a cross marker at the centroid
        plt.scatter(centroid[0], centroid[1], marker='x', color=color, s=100)

        # Add cluster title near the centroid
        cluster_title = combined_topic_labels.get(label, "")  # Get title from combined_topic_labels
        if cluster_title:  # Only add a title if it exists
            plt.text(
                centroid[0],
                centroid[1],
                cluster_title,
                fontsize=9,
                fontweight='bold',
                color=color,  # Use the same color for the title as for the points
                ha='left',  # Horizontal alignment
                va='bottom',  # Vertical alignment
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')  # Optional: Add a white background for readability
            )

    plt.title("t-SNE Visualization of Combined Embeddings")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend().set_visible(False)
    plt.show()


import matplotlib.pyplot as plt
import numpy as np

def plot_topic_frequencies_grouped(frequencies_all, frequencies_filtered, topic_labels):
    """
    Plots the topic frequencies with and without outliers using grouped bars.
    """
    x = np.arange(len(topic_labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot bars for all frequencies
    rects1 = ax.bar(x - width/2, frequencies_all, width, label='All (including outliers)', color='skyblue')

    # Plot bars for filtered frequencies
    rects2 = ax.bar(x + width/2, frequencies_filtered, width, label='Filtered (excluding outliers)', color='dodgerblue')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Frequency')
    ax.set_title('Topic Frequencies')
    ax.set_xticks(x)
    ax.set_xticklabels(topic_labels, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.show()

def plot_topic_frequencies_lollipop(frequencies_all, frequencies_filtered, topic_labels):
    """
    Plots the topic frequencies with and without outliers using a lollipop chart.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create the lollipop stems
    for i in range(len(topic_labels)):
        ax.plot([frequencies_filtered[i], frequencies_all[i]], [i, i], color="gray", linestyle="-", linewidth=1)

    # Plot circles for all frequencies
    ax.scatter(frequencies_all, range(len(topic_labels)), label="All (including outliers)", color="skyblue", s=50)

    # Plot circles for filtered frequencies
    ax.scatter(frequencies_filtered, range(len(topic_labels)), label="Filtered (excluding outliers)", color="dodgerblue", s=50)

    # Customize the plot
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Topic")
    ax.set_title("Topic Frequencies (Lollipop Chart)")
    ax.set_yticks(range(len(topic_labels)))
    ax.set_yticklabels(topic_labels)
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_topic_frequencies_dot(frequencies_all, frequencies_filtered, topic_labels):
    """
    Plots the topic frequencies with and without outliers using a dot plot with lines.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot dots for all frequencies
    ax.plot(frequencies_all, range(len(topic_labels)), "o", color="skyblue", label="All (including outliers)")

    # Plot dots for filtered frequencies
    ax.plot(frequencies_filtered, range(len(topic_labels)), "o", color="dodgerblue", label="Filtered (excluding outliers)")

    # Connect the dots with lines
    for i in range(len(topic_labels)):
        ax.plot([frequencies_filtered[i], frequencies_all[i]], [i, i], color="gray", linestyle="-", linewidth=0.5)

    # Customize the plot
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Topic")
    ax.set_title("Topic Frequencies (Dot Plot)")
    ax.set_yticks(range(len(topic_labels)))
    ax.set_yticklabels(topic_labels)
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_topic_distances(publication_data, combined_labels, publication_keys, template_sections, all_cluster_data):
    unique_labels = sorted(set(combined_labels))
    n_clusters = len(unique_labels)

    # Calculate centroids for each cluster
    centroids = {}
    for label in unique_labels:
        if label == -1:
            continue  # Skip outlier cluster
        cluster_keys = [key for i, key in enumerate(publication_keys) if combined_labels[i] == label]
        cluster_embeddings = [publication_data[key]["publication_embedding"] for key in cluster_keys if
                              key in publication_data and publication_data[key]["publication_embedding"] is not None]
        if cluster_embeddings:
            centroids[label] = np.mean(cluster_embeddings, axis=0)

    # Combine all embeddings and centroids for dimensionality reduction
    all_embeddings = np.array([publication_data[key]["publication_embedding"] for key in publication_keys if
                               key in publication_data and publication_data[key]["publication_embedding"] is not None])
    centroid_embeddings = list(centroids.values())

    # Check if there are any valid embeddings before combining
    if len(all_embeddings) > 0 and len(all_embeddings[0]) >= 2:
        all_embeddings_combined = np.vstack([all_embeddings, centroid_embeddings])

        # Reduce dimensionality to 2D for visualization using t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings_combined) - 1))
        reduced_embeddings_combined = tsne.fit_transform(all_embeddings_combined)

        # Separate reduced embeddings for publications and centroids
        reduced_embeddings = reduced_embeddings_combined[:len(all_embeddings)]
        reduced_centroids = reduced_embeddings_combined[len(all_embeddings):]

        # Map reduced embeddings back to publication keys
        reduced_embeddings_dict = {key: reduced_embeddings[i] for i, key in enumerate(publication_keys) if
                                   key in publication_data and publication_data[key]["publication_embedding"] is not None}

        # Calculate 2D centroids based on reduced embeddings
        centroids_2d = {}
        for label in unique_labels:
            if label == -1:
                continue
            cluster_points = np.array([reduced_embeddings_dict[key] for i, key in enumerate(publication_keys) if
                                       combined_labels[i] == label and key in reduced_embeddings_dict])
            if len(cluster_points) > 0:
                centroids_2d[label] = np.mean(cluster_points, axis=0)

        # Create a 2D scatter plot
        plt.figure(figsize=(12, 10))
        for i, key in enumerate(publication_keys):
            if key in reduced_embeddings_dict:
                embedding = reduced_embeddings_dict[key]
                label = combined_labels[i]

                # Plot scatter points
                if label != -1 and label in centroids_2d:
                    plt.scatter(embedding[0], embedding[1], color=plt.cm.jet(label / float(n_clusters)))

        # Plot centroids and add LLM titles
        centroid_labels = list(centroids_2d.keys())
        for i, (label, centroid) in enumerate(centroids_2d.items()):
            plt.scatter(centroid[0], centroid[1], s=100, marker='X', color='black')

            # Get LLM title from all_cluster_data
            llm_title = all_cluster_data.get(label, {}).get("title", f"Cluster {label}")

            # Add LLM title near the centroid
            plt.text(centroid[0] + 0.2, centroid[1] + 0.2, llm_title, fontsize=9, ha='center', va='center',
                     color='black', fontweight='bold')

        # Create and plot convex hulls for each cluster
        for label in unique_labels:
            if label == -1 or label not in centroids_2d:
                continue
            cluster_points = np.array([reduced_embeddings_dict[key] for i, key in enumerate(publication_keys) if
                                       combined_labels[i] == label and key in reduced_embeddings_dict])
            if len(cluster_points) >= 3:  # Need at least 3 points to form a convex hull
                try:
                    hull = ConvexHull(cluster_points)
                    for simplex in hull.simplices:
                        plt.plot(cluster_points[simplex, 0], cluster_points[simplex, 1],
                                 color=plt.cm.jet(label / float(n_clusters)), alpha=0.3)
                    plt.fill(cluster_points[hull.vertices, 0], cluster_points[hull.vertices, 1],
                             color=plt.cm.jet(label / float(n_clusters)), alpha=0.1)
                except Exception as e:
                    print(f"Error creating convex hull for cluster {label}: {e}")

        plt.title('2D Visualization of Publications with Distances to Cluster Centroids')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
    else:
        print("Cannot reduce embeddings to 2D or no valid embeddings. Skipping plot.")

def visualize_intertopic_distance_map(topic_model, topic_labels):
    """
    Visualizes the intertopic distance map using the fitted BERTopic model.
    """
    try:
        # Visualize the intertopic distance map
        fig = topic_model.visualize_topics()

        # Update node labels with LLM-generated titles from topic_labels
        for trace in fig.data:
            if isinstance(trace, go.Scatter) and trace.mode == 'markers+text':
                updated_labels = []
                for label in trace.text:
                    try:
                        # Extract topic ID and get the corresponding title
                        topic_id = int(label.split(')')[0])
                        updated_label = topic_labels.get(topic_id, label)
                        updated_labels.append(updated_label)
                    except (ValueError, IndexError):
                        # Keep the original label if it cannot be parsed
                        updated_labels.append(label)
                trace.text = updated_labels

        # Update layout if needed
        fig.update_layout(
            title="Intertopic Distance Map with LLM Titles",
            title_x=0.5  # Center the title
        )

        fig.show()

    except Exception as e:
        print(f"Error generating intertopic distance map: {e}")

def visualize_clusters(cluster_data_dir, template_sections, publication_data):
    """
    Loads cluster data from files and performs visualizations.
    """
    combined_labels = []
    combined_texts = []
    publication_keys = []
    all_combined_embeddings = []
    all_cluster_data = {}  # Store data for each cluster for later use

    for filename in os.listdir(cluster_data_dir):
        if filename.startswith("cluster_") and filename.endswith(".json"):
            filepath = os.path.join(cluster_data_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                cluster_data = json.load(f)

            # Store cluster data for later use
            all_cluster_data[cluster_data["label"]] = cluster_data

            # Extract data from cluster_data
            label = cluster_data["label"]
            keys = cluster_data["keys"]
            publication_keys.extend(keys)
            centroid = np.array(cluster_data["centroid"]) if cluster_data["centroid"] is not None else None
            distances = cluster_data["distances"]
            section_probabilities = cluster_data["section_probabilities"]
            texts = cluster_data["texts"]
            combined_texts.extend(texts)
            probabilities_per_document = np.array(cluster_data["probabilities_per_document"]) if cluster_data["probabilities_per_document"] is not None else None
            combined_labels.extend([label] * len(keys))  # Assign the cluster label to each key

            # Add embeddings to the combined list
            for key in keys:
                if key in publication_data and publication_data[key]["publication_embedding"] is not None:
                    all_combined_embeddings.append(publication_data[key]["publication_embedding"])

            # Perform visualizations that can be done for each cluster individually
        #    print(f"Visualizing data for cluster {label}...")

        #    if probabilities_per_document is not None:
        #        plot_section_topic_distribution(section_probabilities, template_sections)

        #    plot_within_cluster_distances(label, cluster_data["title"], keys, distances, template_sections)

        #    print(f"Visualizations for cluster {label} complete.")

    # Perform visualizations that require combined data
    # Check if there are any valid embeddings before proceeding
    if all_combined_embeddings:
        all_combined_embeddings = np.array(all_combined_embeddings)

        # UMAP for combined embeddings
        reducer_combined = UMAP(n_components=15, random_state=42, n_neighbors=5)
        umap_embeddings_combined = reducer_combined.fit_transform(all_combined_embeddings)

        # Create combined_topic_labels using LLM-generated titles
        llm_combined_topic_labels = {label: cluster_data["title"] for label, cluster_data in all_cluster_data.items()}

        # Visualize combined clusters using t-SNE
        visualize_embeddings_with_tsne(all_combined_embeddings, combined_labels, llm_combined_topic_labels, None)

        # Plot distances
        plot_topic_distances(publication_data, combined_labels, publication_keys, template_sections, all_cluster_data)

        # Plot topic distribution
        section_topic_distribution = {section_key: Counter() for section_key in template_sections}
        for doc_index, (pub_key, topic_id) in enumerate(zip(publication_keys, combined_labels)):
            if pub_key in publication_data and topic_id != -1:
                for section_key in template_sections:
                    if "answers" in publication_data[pub_key] and section_key in publication_data[pub_key]["answers"]:
                        section_topic_distribution[section_key][topic_id] += 1
      #  plot_section_topic_distribution(section_topic_distribution, template_sections)

        # Visualize combined clusters using t-SNE with Plotly for interactivity
        if len(all_combined_embeddings[0]) >= 2:
            tsne = TSNE(n_components=2, random_state=42)
            tsne_embeddings = tsne.fit_transform(all_combined_embeddings)

            # Create a DataFrame for Plotly
            df = pd.DataFrame({
                'x': tsne_embeddings[:, 0],
                'y': tsne_embeddings[:, 1],
                'cluster': [str(label) for label in combined_labels],
                'key': publication_keys
            })

            fig = px.scatter(df, x='x', y='y', color='cluster', hover_data=['key'],
                             title="t-SNE Visualization of Combined Embeddings with LLM Titles")

            # Customize the hover template to show the key and cluster name
            fig.update_traces(
                hovertemplate='<b>Key</b>: %{hovertext}<br><b>Cluster</b>: %{marker.color}<extra></extra>')

            # Update marker size and opacity for better visualization
            fig.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=0.5, color='DarkSlateGrey')))

            # Customize layout with square aspect ratio
            fig.update_layout(
                legend_title_text='Cluster',
                xaxis_title="t-SNE Component 1",
                yaxis_title="t-SNE Component 2",
                font=dict(family="Courier New, monospace", size=12, color="RebeccaPurple"),
                legend=dict(font=dict(size=10)),
                width=800,  # Set the width of the plot
                height=800,  # Set the height of the plot to make it square
                xaxis=dict(scaleanchor="y", scaleratio=1),  # Ensure x and y axes have the same scale
                yaxis=dict(scaleanchor="x", scaleratio=1)  # Ensure x and y axes have the same scale
            )

            fig.show()

            # Estimate and plot cluster circles
            for label in sorted(df['cluster'].unique()):
                cluster_points = df[df['cluster'] == label][['x', 'y']].values
                centroid = cluster_points.mean(axis=0)
                radius = np.max(np.sqrt(np.sum((cluster_points - centroid)**2, axis=1)))
                cluster_title = all_cluster_data[int(label)]["title"] if int(label) in all_cluster_data else f"Cluster {label}"

                # Add a circle for the cluster
                fig.add_shape(
                    type="circle",
                    xref="x", yref="y",
                    x0=centroid[0] - radius, y0=centroid[1] - radius,
                    x1=centroid[0] + radius, y1=centroid[1] + radius,
                    line=dict(color="grey", dash="dot"),
                    fillcolor=px.colors.qualitative.Plotly[int(label) % len(px.colors.qualitative.Plotly)],
                    opacity=0.2,
                    label=dict(text=cluster_title, textposition="top center", font=dict(size=12, color="black"))
                )

                # Add a marker for the centroid
                fig.add_trace(go.Scatter(
                    x=[centroid[0]],
                    y=[centroid[1]],
                    mode='markers+text',
                    marker=dict(size=10, color='black'),
                    text=[cluster_title],
                    textposition='top center',
                    textfont=dict(size=12),
                    showlegend=False,
                    hoverinfo='none'  # Disable hover info for centroid
                ))

            fig.show()

        else:
            print("Insufficient dimensions for t-SNE visualization (at least 2 are required).")
    else:
        print("No valid embeddings found for visualization.")

    # Return all_cluster_data
    return all_cluster_data

def plot_topic_frequencies_stacked(frequencies_all, frequencies_filtered, topic_labels):
    """
    Plots the topic frequencies with and without outliers using stacked horizontal bars.
    """

    # Calculate the difference (outliers)
    outlier_counts = frequencies_all - frequencies_filtered

    # Create the stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 8))

    y_pos = np.arange(len(topic_labels))

    # Plot non-outlier frequencies
    ax.barh(y_pos, frequencies_filtered, align='center', color='skyblue', label='In-cluster')

    # Plot outlier frequencies on top
    ax.barh(y_pos, outlier_counts, left=frequencies_filtered, align='center', color='lightcoral', label='Outliers')

    # Customize the plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(topic_labels)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Frequency')
    ax.set_title('Topic Frequencies (Stacked)')
    ax.legend()
    plt.tight_layout()
    plt.show()

def visualize_representative_docs(
    topic_model,
    docs,
    topic_labels,
    publication_keys,
    top_n_docs=5,
):
    if not hasattr(topic_model, "get_representative_docs"):
        print("❌ The provided topic model does not support finding representative documents.")
        return

    unique_topics = set(topic_labels)
    for topic_id in unique_topics:
        if topic_id == -1:
            continue

        try:
            representative_docs = topic_model.get_representative_docs(topic_id)
            if not representative_docs:
                print(f"⚠️ No representative documents found for topic {topic_id}.")
                continue

            print(f"\n--- Representative Documents for Topic {topic_id} ---")

            for i, doc in enumerate(representative_docs):
                if i >= top_n_docs:
                    break

                doc_key = None
                for key, text in zip(publication_keys, docs):
                    if doc in text:
                        doc_key = key
                        break

                if doc_key is None:
                    doc_key = f"Key not found"

                print(f"  - **{doc_key}**: {doc[:200] + '...' if len(doc) > 200 else doc}")

        except Exception as e:
            print(f"❌ Error getting or displaying representative docs for topic {topic_id}: {e}")

def plot_within_cluster_distances(labels, keys, publication_data, template_sections):
    if publication_data is None:
        raise ValueError("publication_data must be provided for distance calculations.")

    unique_labels = sorted(set(labels))

    for label in unique_labels:
        if label == -1:
            continue

        cluster_keys = [key for i, key in enumerate(keys) if labels[i] == label]

        all_subsection_keys = set()
        for section_data in template_sections.values():
            all_subsection_keys.update(section_data["subsections"].keys())
        cluster_subsection_vectors = {subsection_key: [] for subsection_key in all_subsection_keys}

        for pub_key in cluster_keys:
            if pub_key in publication_data and "subsection_embeddings" in publication_data[pub_key]:
                for section_key, section_data in template_sections.items():
                    for subsection_key in section_data["subsections"].keys():
                        if subsection_key in publication_data[pub_key]["subsection_embeddings"]:
                            subsection_embedding = publication_data[pub_key]["subsection_embeddings"][subsection_key]
                            cluster_subsection_vectors[subsection_key].append(subsection_embedding)

        avg_subsection_vectors = {
            subsection_key: np.mean(vectors, axis=0)
            for subsection_key, vectors in cluster_subsection_vectors.items()
            if vectors
        }

        distances = []
        for pub_key in cluster_keys:
            if pub_key in publication_data and "subsection_embeddings" in publication_data[pub_key]:
                for subsection_key, avg_vector in avg_subsection_vectors.items():
                    if subsection_key in publication_data[pub_key]["subsection_embeddings"]:
                        subsection_embedding = publication_data[pub_key]["subsection_embeddings"][subsection_key]
                        distance = cosine_distances([subsection_embedding], [avg_vector])[0][0]
                        distances.append(distance)

        plt.figure()
        plt.hist(distances, bins=20, color="skyblue", edgecolor="black")
        plt.title(f"Distribution of Distances within Cluster {label}")
        plt.xlabel("Cosine Distance")
        plt.ylabel("Number of Subsections")
        plt.grid(True)
        plt.show()
