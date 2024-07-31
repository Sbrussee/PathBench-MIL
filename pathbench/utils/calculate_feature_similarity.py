import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import kendalltau, spearmanr
import torch
import os
from itertools import combinations

def create_directory(path):
    """Create directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def load_embeddings(bag_directory, tile_size, mag, normalization, feature_methods, slide):
    """Load embeddings."""
    embedding_dict = {}
    for feature_method in feature_methods:
        bag = torch.load(f"{bag_directory}/{tile_size}_{mag}_{normalization}_{feature_method}/{slide}.pt")
        embedding_dict[feature_method] = bag
    return embedding_dict

def calculate_neighbor_rankings(embedding_dict, feature_methods, k, method='knn'):
    """Calculate k-NN rankings for each embedding."""
    rankings_dict = {}
    for feature_method in feature_methods:
        embedding = embedding_dict[feature_method]
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

def calculate_kendall_tau(rankings_dict, feature_methods):
    """Calculate Kendall's Tau between rankings of feature extractors."""
    combinations = [(feature_methods[i], feature_methods[j]) for i in range(len(feature_methods)) for j in range(i+1, len(feature_methods))]
    distance_scores = {}
    for combination in combinations:
        ranks1 = rankings_dict[combination[0]]
        ranks2 = rankings_dict[combination[1]]
        distance = 0
        for i in range(len(ranks1)):
            tau, _ = kendalltau(ranks1[i], ranks2[i])
            distance += 1 - tau
        distance /= len(ranks1)
        distance_scores[combination] = distance
    return distance_scores

def calculate_spearman(rankings_dict, feature_methods):
    """Calculate Spearman's rank correlation between rankings of feature extractors."""
    combinations = [(feature_methods[i], feature_methods[j]) for i in range(len(feature_methods)) for j in range(i+1, len(feature_methods))]
    distance_scores = {}
    for combination in combinations:
        ranks1 = rankings_dict[combination[0]]
        ranks2 = rankings_dict[combination[1]]
        distance = 0
        for i in range(len(ranks1)):
            rho, _ = spearmanr(ranks1[i], ranks2[i])
            distance += 1 - rho
        distance /= len(ranks1)
        distance_scores[combination] = distance
    return distance_scores

def rbo_score(l1, l2, p=0.9):
    """Calculate Rank-Biased Overlap (RBO) between two rankings."""
    def helper(l1, l2, p):
        k = 1
        A_d, S_d = set(), set()
        A_size, S_size = 0, 0
        sum1, sum2, sum3 = 0, 0, 0
        while k <= len(l1):
            A_d.add(l1[k-1])
            S_d.add(l2[k-1])
            A_size += 1
            S_size += 1
            X_d = len(A_d.intersection(S_d))
            sum1 += X_d / k * p ** k
            sum2 += X_d / A_size * p ** k
            sum3 += X_d / S_size * p ** k
            k += 1
        rbo_ext = (sum1 + sum2 + sum3) * (1 - p) / p
        return rbo_ext
    return helper(l1, l2, p)

def calculate_rbo(rankings_dict, feature_methods, p=0.9):
    """Calculate RBO between rankings of feature extractors."""
    combinations = [(feature_methods[i], feature_methods[j]) for i in range(len(feature_methods)) for j in range(i+1, len(feature_methods))]
    distance_scores = {}
    for combination in combinations:
        ranks1 = rankings_dict[combination[0]]
        ranks2 = rankings_dict[combination[1]]
        distance = 0
        for i in range(len(ranks1)):
            distance += 1 - rbo_score(list(ranks1[i]), list(ranks2[i]), p)
        distance /= len(ranks1)
        distance_scores[combination] = distance
    return distance_scores

def calculate_shared_neighbors(rankings_dict, feature_methods, k):
    """Calculate shared neighbors between rankings of feature extractors."""
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
    """Construct similarity matrix from distance scores or shared neighbor scores."""
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
    """Plot and save the similarity matrix."""
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

def create_2d_map(similarity_matrix, feature_methods, slide, title, method='mds', n_components=2):
    """Create a 2D map from the similarity matrix using MDS."""
    if method == 'mds':
        reducer = MDS(n_components=n_components, metric=False, n_init=10, max_iter=1000, dissimilarity="precomputed", random_state=42, eps=1e-9)
    else:
        raise ValueError("Invalid method. Use 'mds'.")

    coords = reducer.fit_transform(1 - similarity_matrix)  # Use 1 - similarity_matrix to create dissimilarity matrix
    plt.figure(figsize=(10, 10))
    plt.scatter(coords[:, 0], coords[:, 1])
    for i, feature_method in enumerate(feature_methods):
        plt.annotate(feature_method, (coords[i, 0], coords[i, 1]))
    plt.title(f"{method.upper()} map - {title}")
    plt.savefig(f"similarity_results/{method}_{title}_map_{slide}.png")
    plt.close()

def calculate_similarity(slide, bag_directory, feature_methods, mag, tile_size, normalization, k_values, neighbor_methods, ranking_methods):
    """Calculate similarity for various k values and neighbor selection methods."""
    create_directory(f"similarity_results")

    embedding_dict = load_embeddings(bag_directory, tile_size, mag, normalization, feature_methods, slide)
    
    similarity_matrices = {neighbor_method: {ranking_method: {} for ranking_method in ranking_methods} for neighbor_method in neighbor_methods}
    
    for neighbor_method in neighbor_methods:
        for ranking_method in ranking_methods:
            for k in k_values:
                rankings_dict = calculate_neighbor_rankings(embedding_dict, feature_methods, k, method=neighbor_method)
                if ranking_method == 'kendall_tau':
                    distance_scores = calculate_kendall_tau(rankings_dict, feature_methods)
                    similarity_matrix = construct_similarity_matrix(distance_scores, feature_methods)
                elif ranking_method == 'spearman':
                    distance_scores = calculate_spearman(rankings_dict, feature_methods)
                    similarity_matrix = construct_similarity_matrix(distance_scores, feature_methods)
                elif ranking_method == 'rbo':
                    distance_scores = calculate_rbo(rankings_dict, feature_methods)
                    similarity_matrix = construct_similarity_matrix(distance_scores, feature_methods)
                elif ranking_method == 'shared_neighbors':
                    similarity_scores = calculate_shared_neighbors(rankings_dict, feature_methods, k)
                    similarity_matrix = construct_similarity_matrix(similarity_scores, feature_methods, shared_neighbors=True)
                else:
                    raise ValueError("Invalid ranking method. Use 'kendall_tau', 'spearman', 'rbo', or 'shared_neighbors'.")
                similarity_matrices[neighbor_method][ranking_method][k] = similarity_matrix
                plot_similarity_matrix(similarity_matrix, feature_methods, slide, f"Similarity Matrix ({neighbor_method}, {ranking_method}, k={k})")
                create_2d_map(similarity_matrix, feature_methods, slide, f"{neighbor_method}_{ranking_method}_k={k}", method='mds', n_components=2)

    return similarity_matrices

def plot_k_similarity_change(similarity_matrices, feature_methods, neighbor_method, ranking_method):
    """Plot how similarity changes with different k values for all pairs of feature extractors and neighbor selection method."""
    k_values = sorted(similarity_matrices[neighbor_method][ranking_method].keys())
    for (feature_method_1, feature_method_2) in combinations(feature_methods, 2):
        similarities = [similarity_matrices[neighbor_method][ranking_method][k][feature_method_1, feature_method_2] for k in k_values]
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, similarities, marker='o', label=f'{feature_method_1} vs {feature_method_2}')
        plt.xlabel("k")
        plt.ylabel("Similarity")
        plt.title(f"Similarity between {feature_method_1} and {feature_method_2} vs. k ({neighbor_method}, {ranking_method})")
        plt.legend()
        plt.savefig(f"similarity_results/k_similarity_change_{neighbor_method}_{ranking_method}_{feature_method_1}_{feature_method_2}.png")
        plt.tight_layout()
        plt.close()

def calculate_overall_similarity(similarity_matrices, feature_methods, k_values, neighbor_methods, ranking_methods):
    """Calculate overall similarity across all slides."""
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

def main():
    # Set the magnification and tile size
    mag = "20x"
    tile_size = 256
    # Set normalization
    normalization = "macenko"
    # Set the methods to compare, including new methods
    feature_methods = ["CTransPath", 'resnet50_imagenet', 'RetCCL', "HistoSSL", "PLIP", 'uni', 'gigapath', 'kaiko_b8', 'kaiko_b16', 'kaiko_s8', 'kaiko_s16',
    'kaiko_l14', 'h_optimus_0', 'swav', 'mocov2', 'barlow_twins', 'dino']
    # Get some test slides from the training MF set
    slides = pd.read_csv("/exports/path-cutane-lymfomen-hpc/siemen/MF_annotations.csv")['slide'].unique()
    # Get the directory with the corresponding feature vectors per tile:
    bag_directory = "/exports/path-cutane-lymfomen-hpc/siemen/PathDev/Pathdev/MF_TEST/bags"

    k_values = [5, 10, 20, 50, 100]  # Example k values
    neighbor_methods = ['knn', 'cosine', 'pearson']  # Different neighbor selection methods
    ranking_methods = [ 'rbo', 'shared_neighbors']  # Different ranking methods

    similarity_matrices = {neighbor_method: {ranking_method: {k: [] for k in k_values} for ranking_method in ranking_methods} for neighbor_method in neighbor_methods}
    for slide in slides:
        try:
            slide_similarity_matrices = calculate_similarity(slide, bag_directory, feature_methods, mag, tile_size, normalization, k_values, neighbor_methods, ranking_methods)
            for neighbor_method in neighbor_methods:
                for ranking_method in ranking_methods:
                    for k in k_values:
                        similarity_matrices[neighbor_method][ranking_method][k].append(slide_similarity_matrices[neighbor_method][ranking_method][k])
        except:
            print(f"Error processing slide {slide}")
            pass
    overall_similarity_matrices = calculate_overall_similarity(similarity_matrices, feature_methods, k_values, neighbor_methods, ranking_methods)
    
    # Plot similarity change with k for all pairs of feature extractors and neighbor methods
    for neighbor_method in neighbor_methods:
        for ranking_method in ranking_methods:
            plot_k_similarity_change(overall_similarity_matrices, feature_methods, neighbor_method, ranking_method)

if __name__ == "__main__":
    main()
