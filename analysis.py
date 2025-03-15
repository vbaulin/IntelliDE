import os
import re
import pickle
import torch
import logging
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Union, Tuple, Optional, Pattern, Any
import datetime
import uuid
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from bertopic import BERTopic
import numpy as np
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch.nn as nn  # Import the nn module

# Import utilities
from utils import load_prompt, read_description, save_markdown_report, save_json_report, save_visualization
from utils import setup_logging

# Import vector classes from vector_fns
from vector_fns import (
    Vector, TextVector, ScoreVector, ParameterVector, TableVector,
    BinaryVector, RelationshipVector, TemplateInstanceVector,
    SectionVector, SubsectionVector, embed_text
)

# --- BERTopic Analysis Functions ---
def perform_bertopic_analysis(texts: List[str],
                             embeddings: np.ndarray,
                             embedding_model,  # SciBERT model
                             vectorizer_model,
                             scibert_tokenizer,
                             umap_model=None,
                             hdbscan_model=None) -> Tuple[Optional[BERTopic], Optional[List[int]], Optional[List[float]]]:
    """
    Performs BERTopic analysis, *forcing* SciBERT embeddings.
    """
    logger = logging.getLogger(__name__)

    # 1. Create a DUMMY SentenceTransformer that uses our SciBERT function.
    class SciBERTWrapper(SentenceTransformer):
        def __init__(self, embed_fn, max_length):
            super().__init__()  # Initialize as a SentenceTransformer
            self.embed_fn = embed_fn
            # Add a dummy module (nn.Identity) so that _first_module() works.
            self.add_module("dummy_module", nn.Identity())  # CRITICAL FIX
            self.max_seq_length = max_length


        def encode(self, sentences, *args, **kwargs):
            # Override encode to use SciBERT function.
            return self.embed_fn(sentences)

    def scibert_embedding_function(text: Union[str, List[str]]):
        """Embeds text using SciBERT."""
        if isinstance(text, str):
            text = [text]
        return embed_text(text, scibert_tokenizer, embedding_model).cpu().numpy()

    try:
        # 2.  Wrap the SciBERT embedding function, passing max_seq_length.
        wrapped_model = SciBERTWrapper(scibert_embedding_function, max_length=scibert_tokenizer.model_max_length)

        # 3. Initialize BERTopic with the *wrapped* model.
        topic_model = BERTopic(
            embedding_model=wrapped_model,
            verbose=True,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
        )

        # Fit the model (use pre-computed embeddings if provided)
        topics, probs = topic_model.fit_transform(texts, embeddings=None) # embeddings = None
        logger.info(f"BERTopic model fitted. Found {len(set(topics))} topics.")
        return topic_model, topics, probs

    except Exception as e:
        logger.exception(f"Error during BERTopic analysis: {e}")
        return None, None, None


def analyze_clusters(
    topics: List[int],
    texts: List[str],
    keys: List[str],
    template_text: str,
    publication_data: Dict[str, Any],
    topic_model: BERTopic,
    probs: Optional[List[float]] = None,
    perspective_file: str = None,
    llm_provider: str = "openrouter",
    prompt_file: str = "prompts/describe_cluster.txt",
):
    """Analyzes each cluster using an LLM."""
    logger = logging.getLogger(__name__)
    logger.info("Starting cluster analysis...")

    os.makedirs("cluster_analysis_results", exist_ok=True)
    prompt_template = load_prompt(prompt_file)
    if not prompt_template:
        logger.error(f"Failed to load prompt from {prompt_file}")
        return

    for topic_id in sorted(set(topics)):
        if topic_id == -1:
            logger.info("Skipping outlier cluster (-1)")
            continue

        cluster_texts = [texts[i] for i, topic in enumerate(topics) if topic == topic_id]
        cluster_keys = [keys[i] for i, topic in enumerate(topics) if topic == topic_id]

        if not cluster_texts:
            logger.warning(f"No texts for cluster {topic_id}. Skipping.")
            continue

        example_titles = "\n".join([f"{i+1}. {title}" for i, title in enumerate(topic_model.get_topic(topic_id)[:5])])
        combined_texts = "\n".join(cluster_texts)
        prompt = prompt_template.format(
            topic_id=topic_id,
            example_titles=example_titles,
            combined_texts=combined_texts,
            num_texts=len(cluster_texts)
        )
        logger.debug(f"Generated prompt for cluster {topic_id}")

        from OpenRouter_Methods import get_llm_response
        try:
            title = get_llm_response(prompt, llm_provider)
            if not title:
                logger.warning(f"LLM failed to generate title for cluster {topic_id}.")
                continue
            logger.info(f"LLM generated title for cluster {topic_id}: {title}")
        except Exception as e:
            logger.error(f"Error getting LLM response for cluster {topic_id}: {e}")
            continue

        centroid = calculate_centroid(topic_id, topics, keys, publication_data)
        avg_distance = calculate_average_distance(centroid, topic_id, topics, keys, publication_data)

        cluster_data = {
            "label": str(topic_id),
            "title": title,
            "num_texts": len(cluster_texts),
            "example_texts": cluster_texts[:5],
            "keys": cluster_keys,
            "centroid": centroid,
            "average_distance": avg_distance
        }

        output_file = Path("cluster_analysis_results") / f"cluster_{topic_id}.json"
        try:
            save_json_report(title, output_file, cluster_data)
            logger.info(f"Saved cluster data to {output_file}")
        except Exception as e:
            logger.error(f"Error saving cluster data: {e}")

    logger.info("Cluster analysis complete.")


def calculate_centroid(topic_id: int, topics: List[int], keys: List[str], publication_data: Dict[str, Any]) -> List[float]:
    """Calculates the centroid embedding for a given cluster."""
    logger = logging.getLogger(__name__)
    embeddings = []
    for i, topic in enumerate(topics):
        if topic == topic_id:
            key = keys[i]
            parts = key.split("_")
            if len(parts) > 1:
                pub_key = parts[0]
                section_key = parts[1]

                if pub_key in publication_data:
                    template_instance = publication_data[pub_key]

                    for section_vector in template_instance.sections:
                        if section_vector.metadata.get("section_id") == section_key:
                            embeddings.append(section_vector.embedding)
                            break
                else:
                    logger.warning(f"Publication key {pub_key} not found.")
            else:
                logger.warning(f"Invalid key format: {key}")

    if not embeddings:
        logger.warning(f"No embeddings for topic {topic_id}. Returning zero vector.")
        return [0.0] * 768

    return np.mean(embeddings, axis=0).tolist()


def calculate_average_distance(centroid: List[float], topic_id: int, topics: List[int], keys: List[str], publication_data: Dict[str, Any]) -> float:
    """Calculates the average cosine distance from the centroid."""
    logger = logging.getLogger(__name__)
    distances = []
    centroid_np = np.array(centroid)

    for i, topic in enumerate(topics):
        if topic == topic_id:
            key = keys[i]
            parts = key.split("_")
            if len(parts) > 1:
                pub_key = parts[0]
                section_key = parts[1]

                if pub_key in publication_data:
                    template_instance = publication_data[pub_key]

                    for section_vector in template_instance.sections:
                        if section_vector.metadata.get("section_id") == section_key:
                            embedding_np = np.array(section_vector.embedding)
                            similarity = cosine_similarity(centroid_np.reshape(1, -1), embedding_np.reshape(1, -1))[0][0]
                            distance = 1 - similarity
                            distances.append(distance)
                            break
                else:
                    logger.warning(f"Publication key {pub_key} not found.")
            else:
                logger.warning(f"Invalid key format: {key}")

    if not distances:
        logger.warning(f"No distances calculated for topic {topic_id}. Returning 0.")
        return 0.0

    return np.mean(distances)


# --- Visualization Functions ---
def visualize_topic_similarity(topic_model: BERTopic, custom_labels: Dict[int, str] = None):
    """Visualizes the topic similarity matrix with custom labels."""
    logger = logging.getLogger(__name__)
    try:
        fig = topic_model.visualize_heatmap(custom_labels=custom_labels)
        save_visualization(fig, "topic_similarity_matrix")
        logger.info("✅ Topic Similarity Matrix saved")
    except Exception as e:
        logger.error(f"Error generating topic similarity matrix: {e}")

def visualize_intertopic_distance_map(topic_model: BERTopic, custom_labels: Dict[int, str] = None):
    """Visualizes the intertopic distance map (clustering) with custom labels."""
    logger = logging.getLogger(__name__)
    try:
        fig = topic_model.visualize_topics(custom_labels=custom_labels)
        save_visualization(fig, "intertopic_distance_map")
        logger.info("✅ Intertopic Distance Map saved")
    except Exception as e:
        logger.error(f"Error generating intertopic distance map: {e}")

def visualize_topic_hierarchy(topic_model: BERTopic, custom_labels: Dict[int, str] = None):
    """Visualizes the topic hierarchy (dendrogram) with custom labels."""
    logger = logging.getLogger(__name__)
    try:
        fig = topic_model.visualize_hierarchy(custom_labels=custom_labels)
        save_visualization(fig, "topic_hierarchy")
        logger.info("✅ Topic Hierarchy visualization saved")
    except Exception as e:
        logger.error(f"Error generating topic hierarchy: {e}")

def create_topic_visualizations(model, texts, topics, probs, topic_labels=None, scibert_model=None, scibert_tokenizer=None):
    """Create and save standard BERTopic visualizations"""
    logger = logging.getLogger(__name__)

    # 1. Intertopic Distance Map
    logger.info("Generating Intertopic Distance Map...")
    try:
        fig_distance_map = model.visualize_topics(custom_labels = topic_labels)
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

def visualize_embeddings_with_tsne(
    embeddings: List[List[float]],
    labels: List[int],
    keys: List[str],
    new_pub_key: str,
    topic_labels: Dict[int, str] = None,
    perplexity: int = 30,
    n_iter: int = 300
):
    """Visualizes embeddings using t-SNE, highlighting new publication."""
    logger = logging.getLogger(__name__)
    logger.info("Starting t-SNE visualization...")

    try:
        embeddings_array = np.array(embeddings)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=n_iter)
        reduced_embeddings = tsne.fit_transform(embeddings_array)

        df = pd.DataFrame(reduced_embeddings, columns=['x', 'y'])
        df['label'] = [str(label) for label in labels]
        df['key'] = keys
        df['is_new_pub'] = [key.startswith(new_pub_key) for key in keys]
        df['topic_name'] = df['label'].map(topic_labels if topic_labels else {str(i): f"Topic {i}" for i in set(labels)})

        fig = px.scatter(
            df,
            x='x',
            y='y',
            color='topic_name',
            hover_data=['key'],
            title=f"t-SNE Visualization (New Pub: {new_pub_key})",
            labels={'topic_name': 'Topic'},
            color_discrete_sequence=px.colors.qualitative.Set1
        )

        new_pub_df = df[df['is_new_pub']]
        fig.add_trace(
            go.Scatter(
                x=new_pub_df['x'],
                y=new_pub_df['y'],
                mode='markers',
                marker=dict(size=10, color='black', symbol='star'),
                name=f'New Publication ({new_pub_key})',
                hoverinfo='skip'
            )
        )

        save_visualization(fig, f"tsne_visualization_{new_pub_key}")
        logger.info("✅ t-SNE visualization saved.")

    except Exception as e:
        logger.error(f"Error generating t-SNE visualization: {e}")

def visualize_embeddings_static(
    embeddings: List[List[float]],
    labels: List[int],
    keys: List[str],
    new_pub_key: str,
    topic_labels: Dict[int, str] = None,
):
    """Visualizes embeddings using t-SNE in a static plot, highlighting new publication."""
    logger = logging.getLogger(__name__)
    logger.info("Starting static t-SNE visualization...")

    try:
        embeddings_array = np.array(embeddings)
        tsne = TSNE(n_components=2, random_state=42)
        reduced_embeddings = tsne.fit_transform(embeddings_array)

        df = pd.DataFrame(reduced_embeddings, columns=['x', 'y'])
        df['label'] = [str(label) for label in labels]
        df['key'] = keys
        df['is_new_pub'] = [key.startswith(new_pub_key) for key in keys]
        df['topic_name'] = df['label'].map(topic_labels if topic_labels else {str(i): f"Topic {i}" for i in set(labels)})

        plt.figure(figsize=(12, 8))
        for topic_name in df['topic_name'].unique():
            topic_df = df[df['topic_name'] == topic_name]
            plt.scatter(topic_df['x'], topic_df['y'], label=topic_name, alpha=0.6)

        new_pub_df = df[df['is_new_pub']]
        plt.scatter(new_pub_df['x'], new_pub_df['y'], color='black', marker='*', s=100, label=f'New Pub ({new_pub_key})')

        plt.title(f"t-SNE Visualization (New Pub: {new_pub_key})")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.legend(loc='best')
        plt.grid(True)

        static_filepath = os.path.join("visualization_results", f"static_tsne_visualization_{new_pub_key}.png")
        os.makedirs("visualization_results", exist_ok=True)
        plt.savefig(static_filepath)
        plt.close()
        logger.info(f"✅ Static t-SNE visualization saved to {static_filepath}")

    except Exception as e:
        logger.error(f"Error generating static t-SNE visualization: {e}")

def analyze_new_publication(new_pub_key: str, topics: List[int], texts: List[str], all_keys: List[str], topic_model: BERTopic, probs: List[float], publication_data: Dict[str, Any], template_text: str, perspective_file: str, topic_labels: Dict[int, str] = None):
    """Analyzes a new publication in the context of existing clusters.

    Args:
        new_pub_key: Key of the new publication.
        topics: Predicted topic for each text segment.
        texts: Text segments from the new publication.
        all_keys: Keys for all segments (including existing).
        topic_model: Trained BERTopic model.
        probs: Probabilities of topic assignment.
        publication_data: All publication data (including existing).
        template_text: Template text (Catechism).
        perspective_file: Path to the perspective file.
        topic_labels:  (Optional) LLM-generated topic labels.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Analyzing new publication: {new_pub_key}")

    cluster_counts = {}
    for topic in topics:
        if topic != -1:  # Exclude outlier cluster
            cluster_counts[topic] = cluster_counts.get(topic, 0) + 1

    if not cluster_counts:
        logger.warning(f"No significant cluster associations for {new_pub_key}.")
        return

    try:
        create_radar_chart(cluster_counts, new_pub_key, topic_labels)  # Pass topic_labels
    except Exception as e:
        logger.error(f"Error creating radar chart: {e}")

    try:
        summary = generate_summary_report(new_pub_key, cluster_counts, topic_model, publication_data, template_text, perspective_file, all_keys, topic_labels) #and here
        if summary:
            report_file = Path("cluster_analysis_results") / f"new_pub_summary_{new_pub_key}.md"
            save_markdown_report(summary, report_file, f"Summary Report for {new_pub_key}")
            logger.info(f"Summary report saved to {report_file}")
        else:
            logger.warning("Summary report generation failed.")
    except Exception as e:
        logger.error(f"Error generating or saving summary report: {e}")

    logger.info(f"Analysis of new publication {new_pub_key} complete.")
def create_radar_chart(cluster_counts: Dict[int, int], new_pub_key: str, topic_labels: Dict[int, str] = None):
    """
    Creates a radar chart of cluster distribution, using topic labels if available.
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating radar chart...")

    try:
        # Use topic labels if available, otherwise use topic IDs as strings
        labels = [str(topic) for topic in cluster_counts.keys()]  # Default: string IDs
        if topic_labels:
            #  try-except block: handles missing labels
            labeled_labels = []
            for topic in cluster_counts:
                try:
                    label = topic_labels.get(topic, f"Topic {topic}") #get labels
                    labeled_labels.append(label)
                except:
                    labeled_labels.append(f"Topic {topic}") #if not, just use number of topic
            labels = labeled_labels

        values = list(cluster_counts.values())
        num_vars = len(labels)

        # Close the circle
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]
        values += values[:1]

        fig = go.Figure(data=[go.Scatterpolar(
            r=values,
            theta=labels,  # Use the correctly prepared labels
            fill='toself',
            name=new_pub_key
        )])

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values)]
                )),
            showlegend=False,
            title=f"Cluster Distribution for: {new_pub_key}"
        )

        save_visualization(fig, f"radar_chart_{new_pub_key}")
        logger.info("✅ Radar chart saved.")

    except Exception as e:
        logger.error(f"Error creating radar chart: {e}")
        raise  # Re-raise for debugging

def generate_summary_report(new_pub_key: str, cluster_counts: Dict[int, int], topic_model: BERTopic, publication_data: Dict[str, Any], template_text: str, perspective_file: str, all_keys: List[str], topic_labels: Dict[int,str] = None) -> str:
    """Generates a summary report, integrating cluster info and alignment."""
    logger = logging.getLogger(__name__)
    logger.info(f"Generating summary report for {new_pub_key}...")

    try:
        report = f"# Summary Report for New Publication: {new_pub_key}\n\n"
        report += f"Analyzed on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        report += "## Top Cluster Associations\n\n"
        for topic, count in sorted(cluster_counts.items(), key=lambda item: item[1], reverse=True):
            topic_info = topic_model.get_topic(topic)
            # Use get method to avoid error, if topic labels do not exist
            topic_label = topic_labels.get(topic, f"Cluster {topic}") if topic_labels else f"Cluster {topic}"
            report += f"- **{topic_label}**: Count: {count}, Top Words: {', '.join([word for word, score in topic_info[:5]])}\n" #use topic label

        report += "\n## Cited Publications\n\n"
        for topic, count in sorted(cluster_counts.items(), key=lambda item: item[1], reverse=True):
             # Use get method to avoid error, if topic labels do not exist
            topic_label = topic_labels.get(topic, f"Cluster {topic}") if topic_labels else f"Cluster {topic}"
            cluster_keys = [key for key in all_keys if str(topic) in key]
            report += f"- **{topic_label}**: Cited in publications: {', '.join(cluster_keys)}\n"

        report += "\n## Excerpt Alignment with Perspective\n\n"
        perspective = read_description(perspective_file)
        if perspective:
            report += f"> {perspective}\n\n"
        else:
            report += "Error: Could not read perspective file.\n\n"

        return report
    except Exception as e:
        logger.error(f"Error in generate_summary_report: {e}")
        return None
