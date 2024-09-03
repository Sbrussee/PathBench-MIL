import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import kendalltau, spearmanr
import torch
import glob
import os
import random
from itertools import combinations
from sklearn.preprocessing import RobustScaler

def create_directory(path):
    """Create directory if it does not exist.
    
    Args:
        path (str): Path to directory.
        
    Returns:
        None
    """
    if not os.path.exists(path):
        os.makedirs(path)

def load_embeddings(bag_directory, tile_size, mag, normalization, feature_methods, slide):
    """Load feature extractor embeddings from the specified directory for a given slide.
    
    Args:
        bag_directory (str): Directory containing the embeddings.
        tile_size (int): Tile size used for the embeddings.
        mag (str): Magnification level.
        normalization (str): Normalization method.
        feature_methods (list): List of feature extraction methods.
        slide (str): Slide name.
        
    Returns:
        embedding_dict: Dictionary containing the embeddings for each feature extraction method.
    """
    embedding_dict = {}
    for feature_method in feature_methods:
        bag = torch.load(f"{bag_directory}/{tile_size}_{mag}_{normalization}_{feature_method}/{slide}.pt")
        embedding_dict[feature_method] = bag
    return embedding_dict

def calculate_neighbor_rankings(embedding_dict, feature_methods, k, method='knn'):
    """Calculate distance-based neigbor rankings for each embedding
    
    Args:
        embedding_dict (dict): Dictionary containing the embeddings for each feature extraction method.
        feature_methods (list): List of feature extraction methods.
        k (int): Number of neighbors to consider.
        method (str): Method to use for neighbor selection. Options: 'knn', 'cosine', 'pearson'.

    Returns:
        rankings_dict: Dictionary containing the distace rankings for each feature extraction method.
    """
    rankings_dict = {}
    for feature_method in feature_methods:
        embedding = embedding_dict[feature_method]
        num_samples = len(embedding)
        k = min(k, num_samples)  # Ensure k is not greater than the number of samples

        if method == 'knn':
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(embedding)
            distances, indices = nbrs.kneighbors(embedding)
        elif method == 'cosine':
            cos_sim_matrix = cosine_similarity(embedding)
            indices = np.argsort(-cos_sim_matrix, axis=1)[:, 1:k+1]
        elif method == 'pearson':
            pearson_sim_matrix = np.array([[spearmanr(embedding[i], embedding[j])[0] for j in range(len(embedding))] for i in range(len(embedding))])
            indices = np.argsort(-pearson_sim_matrix, axis=1)[:, 1:k+1]
        else:
            raise ValueError("Invalid method. Use 'knn', 'cosine', or 'pearson'.")
        rankings_dict[feature_method] = indices
    return rankings_dict

def calculate_shared_neighbors(rankings_dict, feature_methods, k):
    """Calculate shared neighbors between rankings of feature extractors.
    
    Args:
        rankings_dict (dict): Dictionary containing the rankings for each feature extraction method.
        feature_methods (list): List of feature extraction methods.
        k (int): Number of neighbors to consider.

    Returns:
        similarity_scores: Dictionary containing the similarity scores between pairs of feature
        extractors based on shared neighbors.
    """
    combinations = [(feature_methods[i], feature_methods[j]) for i in range(len(feature_methods)) for j in range(i+1, len(feature_methods))]
    similarity_scores = {}
    for combination in combinations:
        ranks1 = rankings_dict[combination[0]]
        ranks2 = rankings_dict[combination[1]]
        shared_count = 0
        for i in range(len(ranks1)):
            shared_neighbors = set(ranks1[i]).intersection(set(ranks2[i]))
            shared_count += len(shared_neighbors)
        similarity_scores[combination] = shared_count / (len(ranks1) * k)
    return similarity_scores

def construct_similarity_matrix(distance_scores, feature_methods, shared_neighbors=False):
    """Construct similarity matrix from distance scores or shared neighbor scores.
    
    Args:
        distance_scores (dict): Dictionary containing the distance or shared neighbor scores between pairs of feature extractors.
        feature_methods (list): List of feature extraction methods.
        shared_neighbors (bool): Whether tthe distance represents shared neighbors or not.

    Returns:
        similarity_matrix: 2D numpy array containing the similarity scores between pairs of feature extractors.
    """
    similarity_matrix = np.zeros((len(feature_methods), len(feature_methods)))
    for i in range(len(feature_methods)):
        for j in range(i+1, len(feature_methods)):
            if shared_neighbors:
                similarity = distance_scores[(feature_methods[i], feature_methods[j])]
            else:
                distance = distance_scores[(feature_methods[i], feature_methods[j])]
                similarity = 1 - distance  # Convert distance to similarity
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    np.fill_diagonal(similarity_matrix, 1)  # Similarity with itself is 1
    return similarity_matrix

def plot_similarity_matrix(similarity_matrix, feature_methods, slide, title):
    """Plot and save the similarity matrix.
    
    Args:
        similarity_matrix (np.array): 2D numpy array containing the similarity scores between pairs of feature extractors.
        feature_methods (list): List of feature extraction methods.
        slide (str): Slide name.
        title (str): Title of the plot.

    Returns:
        None
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(similarity_matrix, cmap='viridis', vmin=0, vmax=1)
    plt.xticks(range(len(feature_methods)), feature_methods, rotation=45)
    plt.yticks(range(len(feature_methods)), feature_methods)
    plt.colorbar()
    for i in range(len(feature_methods)):
        for j in range(len(feature_methods)):
            plt.text(j, i, f"{similarity_matrix[i, j]:.2f}", ha="center", va="center", color="white")
    plt.title(title)
    plt.savefig(f"similarity_results/{title}_similarity_matrix_{slide}.png")
    plt.tight_layout()
    plt.close()

def create_2d_map(similarity_matrix, feature_methods, slide, title, method='mds', n_components=2, log_transform=True, clip_value=None):
    """Create a 2D map from the similarity matrix using MDS.
    
    Args:
        similarity_matrix (np.array): 2D numpy array containing the similarity scores between pairs of feature extractors.
        feature_methods (list): List of feature extraction methods.
        slide (str): Slide name.
        title (str): Title of the plot.
        method (str): Dimensionality reduction method. Options: 'mds'.
        n_components (int): Number of components for the dimensionality reduction.
        log_transform (bool): Whether to apply log transformation to the dissimilarity matrix.
        clip_value (float): Value to clip the dissimilarity matrix.

    Returns:
        None
    """
    dissimilarity_matrix = 1 - similarity_matrix  # Convert similarity matrix to dissimilarity matrix

    # Apply log transformation to reduce the impact of large dissimilarities
    if log_transform:
        dissimilarity_matrix = np.log1p(dissimilarity_matrix)

    # Clip the dissimilarity matrix to reduce the impact of extreme outliers
    if clip_value is not None:
        dissimilarity_matrix = np.clip(dissimilarity_matrix, None, clip_value)

    # Apply robust scaling to reduce the influence of outliers
    scaler = RobustScaler()
    dissimilarity_matrix = scaler.fit_transform(dissimilarity_matrix)

    if method == 'mds':
        reducer = MDS(n_components=n_components, metric=False, n_init=10, max_iter=1000, dissimilarity="precomputed", random_state=42, eps=1e-9)
    else:
        raise ValueError("Invalid method. Use 'mds'.")

    coords = reducer.fit_transform(dissimilarity_matrix)  # Use dissimilarity matrix here
    plt.figure(figsize=(10, 10))
    plt.scatter(coords[:, 0], coords[:, 1])
    for i, feature_method in enumerate(feature_methods):
        plt.annotate(feature_method, (coords[i, 0], coords[i, 1]))
    plt.title(f"{method.upper()} map - {title}")
    plt.savefig(f"similarity_results/{method}_{title}_map_{slide}.png")
    plt.close()

def calculate_similarity(slide, bag_directory, feature_methods, mag, tile_size, normalization, k_values, neighbor_methods, ranking_methods):
    """Calculate similarity for various k values and neighbor selection methods.
    
    Args:
        slide (str): Slide name.
        bag_directory (str): Directory containing the embeddings.
        feature_methods (list): List of feature extraction methods.
        mag (str): Magnification level.
        tile_size (int): Tile size used for the embeddings.
        normalization (str): Normalization method.
        k_values (list): List of k values.
        neighbor_methods (list): List of neighbor selection methods.
        ranking_methods (list): List of ranking methods.
        
    Returns:
        similarity_matrices: Dictionary containing the similarity matrices for each neighbor selection method and ranking method.
    """
    create_directory(f"similarity_results")

    embedding_dict = load_embeddings(bag_directory, tile_size, mag, normalization, feature_methods, slide)
    
    similarity_matrices = {neighbor_method: {ranking_method: {} for ranking_method in ranking_methods} for neighbor_method in neighbor_methods}
    
    for neighbor_method in neighbor_methods:
        for ranking_method in ranking_methods:
            for k in k_values:
                rankings_dict = calculate_neighbor_rankings(embedding_dict, feature_methods, k, method=neighbor_method)
                if ranking_method == 'shared_neighbors':
                    similarity_scores = calculate_shared_neighbors(rankings_dict, feature_methods, k)
                    similarity_matrix = construct_similarity_matrix(similarity_scores, feature_methods, shared_neighbors=True)
                else:
                    raise ValueError("Invalid ranking method. Use 'shared_neighbors'.")
                similarity_matrices[neighbor_method][ranking_method][k] = similarity_matrix

    return similarity_matrices

def calculate_overall_similarity(similarity_matrices, feature_methods, k_values, neighbor_methods, ranking_methods):
    """Calculate overall similarity across all slides.
    
    Args:
        similarity_matrices: Dictionary containing the similarity matrices for each neighbor selection method and ranking method.
        feature_methods (list): List of feature extraction methods.
        k_values (list): List of k values.
        neighbor_methods (list): List of neighbor selection methods.
        ranking_methods (list): List of ranking methods.

    Returns:
        overall_similarity_matrices: Dictionary containing the overall similarity matrices for each neighbor selection method and ranking method.
    """
    overall_similarity_matrices = {neighbor_method: {ranking_method: {k: np.zeros((len(feature_methods), len(feature_methods))) for k in k_values} for ranking_method in ranking_methods} for neighbor_method in neighbor_methods}
    for neighbor_method in neighbor_methods:
        for ranking_method in ranking_methods:
            for k in k_values:
                for slide_matrix in similarity_matrices[neighbor_method][ranking_method][k]:
                    overall_similarity_matrices[neighbor_method][ranking_method][k] += slide_matrix
                overall_similarity_matrices[neighbor_method][ranking_method][k] /= len(similarity_matrices[neighbor_method][ranking_method][k])  # Average over all slides
    
    for neighbor_method in neighbor_methods:
        for ranking_method in ranking_methods:
            for k in k_values:
                plot_similarity_matrix(overall_similarity_matrices[neighbor_method][ranking_method][k], feature_methods, "overall", f"Overall Similarity Matrix ({neighbor_method}, {ranking_method}, k={k})")
                create_2d_map(overall_similarity_matrices[neighbor_method][ranking_method][k], feature_methods, "overall", f"Overall {neighbor_method}_{ranking_method}_k={k}", method='mds', n_components=2)
    
    return overall_similarity_matrices

def plot_k_similarity_change(similarity_matrices, feature_methods, neighbor_method, ranking_method):
    """Plot how similarity changes with different k values for all pairs of feature extractors and neighbor selection method.
    
    Args:
        similarity_matrices: Dictionary containing the similarity matrices for each neighbor selection method and ranking method.
        feature_methods (list): List of feature extraction methods.
        neighbor_method (str): Neighbor selection method.
        ranking_method (str): Ranking method.
        
    Returns:
        None
    """
    k_values = sorted(similarity_matrices[neighbor_method][ranking_method].keys())
    for (feature_method_1, feature_method_2) in combinations(range(len(feature_methods)), 2):
        # Access the 2D list with separate indices
        similarities = [similarity_matrices[neighbor_method][ranking_method][k][feature_method_1][feature_method_2] for k in k_values]
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, similarities, marker='o', label=f'{feature_methods[feature_method_1]} vs {feature_methods[feature_method_2]}')
        plt.xlabel("k")
        plt.ylabel("Similarity")
        plt.title(f"Similarity between {feature_methods[feature_method_1]} and {feature_methods[feature_method_2]} vs. k ({neighbor_method}, {ranking_method})")
        plt.legend()
        plt.savefig(f"similarity_results/k_similarity_change_{neighbor_method}_{ranking_method}_{feature_methods[feature_method_1]}_{feature_methods[feature_method_2]}.png")
        plt.tight_layout()
        plt.close()

def main(N=None):
    # Set the magnification and tile size
    mag = "20x"
    tile_size = 256
    # Set normalization
    normalization = "macenko"
    # Set the methods to compare, including new methods
    feature_methods = ["CTransPath", 'resnet50_imagenet', 'gigapath', 'kaiko_b8', 'kaiko_b16', 'kaiko_s8', 'kaiko_s16',
                       'kaiko_l14', 'h_optimus_0', 'phikon', 'uni', 'dino', 'virchow', 'virchow2', 'hibou_b', 'hibou_l']
    # Get some test slides from the training MF set
    slides = pd.read_csv("/exports/path-cutane-lymfomen-hpc/siemen/PathDev/Pathdev/MF_annotations_corrected.csv")['slide'].unique()
    
    if N is not None:
        slides = random.sample(list(slides), N)

    # Get the directory with the corresponding feature vectors per tile:
    bag_directory = "/exports/path-cutane-lymfomen-hpc/siemen/PathDev/Pathdev/experiments/MF_TEST/bags"

    k_values = [5, 10, 20, 50, 100]  # Example k values
    neighbor_methods = ['knn', 'cosine']  # Different neighbor selection methods
    ranking_methods = ['shared_neighbors']  # Different ranking methods

    similarity_matrices = {neighbor_method: {ranking_method: {k: [] for k in k_values} for ranking_method in ranking_methods} for neighbor_method in neighbor_methods}
    for slide in slides:
        try:
            slide_similarity_matrices = calculate_similarity(slide, bag_directory, feature_methods, mag, tile_size, normalization, k_values, neighbor_methods, ranking_methods)
            for neighbor_method in neighbor_methods:
                for ranking_method in ranking_methods:
                    for k in k_values:
                        similarity_matrices[neighbor_method][ranking_method][k].append(slide_similarity_matrices[neighbor_method][ranking_method][k])
        except Exception as e:
            print(f"Error processing slide {slide}: {e}")
    overall_similarity_matrices = calculate_overall_similarity(similarity_matrices, feature_methods, k_values, neighbor_methods, ranking_methods)
    plot_k_similarity_change(overall_similarity_matrices, feature_methods, 'knn', 'shared_neighbors')
    plot_k_similarity_change(overall_similarity_matrices, feature_methods, 'cosine', 'shared_neighbors')

if __name__ == "__main__":
    # Example usage: To use all slides, call `main()` without arguments. 
    # To use N random slides, call `main(N=<number of slides>)`.
    main(N=100)  # Use 100 random slides for example
