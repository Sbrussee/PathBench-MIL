def calculate_similarity_matrix(
    bag_directory: str,
    slides: list,
    feature_methods: list,
    normalizations: list,
    tile_size: int,
    magnification: int,
    k=20,
    method='cosine',
    num_samples=100
):
    """
    Calculate the similarity matrix based on random samples of slides, comparing only feature extraction 
    methods and normalizations for a fixed tile size and magnification.
    
    Args:
        bag_directory: The bag directory
        slides: List of slides
        feature_methods: List of feature methods
        normalizations: List of normalization methods
        tile_size: The tile size to be used
        magnification: The magnification to be used
        k: Number of neighbors for similarity calculation
        method: Similarity method (e.g., 'cosine')
        num_samples: Sample size for slides

    Returns:
        similarity_matrix: The similarity matrix of all combinations of feature methods and normalizations
        combinations: The list of combinations of parameters (feature method, normalization)
    """
    # Generate combinations of feature methods and normalizations
    combinations = list(product(feature_methods, normalizations))

    # Sample slides if there are more slides than num_samples
    if len(slides) > num_samples:
        slides = random.sample(list(slides), num_samples)

    similarity_scores = []
    valid_combinations = set()  # Track valid combinations across all slides

    for slide in slides:
        slide_similarity_scores = {}
        embeddings = {}
        slide_valid_combinations = set()  # Track valid combinations for this slide

        # Load embeddings for each combination (with the specified tile size and magnification)
        for (feature_method, normalization) in combinations:
            try:
                embedding = load_embeddings(
                    bag_directory, tile_size, magnification, normalization, feature_method, slide
                )
                embeddings[(feature_method, normalization)] = embedding
                slide_valid_combinations.add((feature_method, normalization))
            except Exception as e:
                logging.warning(f"Error loading embeddings for slide {slide}, combination {feature_method}, {normalization}: {e}")
                continue

        # If no valid embeddings were loaded for this slide, skip it
        if not slide_valid_combinations:
            logging.warning(f"Skipping slide {slide} as it has no valid embeddings.")
            continue

        # Update valid combinations set with the combinations that worked for this slide
        valid_combinations.update(slide_valid_combinations)
        logging.info(f"Valid combinations for slide {slide}: {slide_valid_combinations}")

        # Calculate neighbor rankings and shared neighbors for each valid combination pair
        for i, combo1 in enumerate(slide_valid_combinations):
            for j, combo2 in enumerate(slide_valid_combinations):
                if combo1 != combo2:  # Only compare different combinations
                    embedding1 = embeddings.get(combo1)
                    embedding2 = embeddings.get(combo2)

                    if embedding1 is not None and embedding2 is not None:
                        # Calculate rankings for both combinations
                        rankings_dict1 = calculate_neighbor_rankings({combo1: embedding1}, [combo1], k, method=method)
                        rankings_dict2 = calculate_neighbor_rankings({combo2: embedding2}, [combo2], k, method=method)

                        # Calculate shared neighbors between the two combinations
                        shared_neighbors = calculate_shared_neighbors(
                            {combo1: rankings_dict1[combo1], combo2: rankings_dict2[combo2]},
                            [combo1, combo2], k
                        )

                        # Store the similarity score between these two combinations
                        slide_similarity_scores[(combo1, combo2)] = shared_neighbors[(combo1, combo2)]

        # Append the similarity scores for this slide to the list of all slides
        if slide_similarity_scores:
            similarity_scores.append(slide_similarity_scores)

        logging.info(f"slide {slide}, embedding {embedding}, slide_similarity_score {slide_similarity_scores}")

    # Average similarity scores across all slides
    avg_similarity_scores = {}
    if similarity_scores:
        for key in similarity_scores[0].keys():
            avg_similarity_scores[key] = np.mean([score[key] for score in similarity_scores if key in score])

    # Convert valid_combinations set to list
    valid_combinations = list(valid_combinations)

    # Construct similarity matrix using only valid combinations
    return construct_similarity_matrix(avg_similarity_scores, valid_combinations, shared_neighbors=True), valid_combinations

def load_embeddings(bag_directory, tile_size, magnification, normalization, feature_method, slide):
    """Load feature extractor embeddings for the given combination of parameters."""
    file_path = f"{bag_directory}/{tile_size}_{magnification}_{normalization}_{feature_method}/{slide}.pt"
    if os.path.exists(file_path):
        return torch.load(file_path)
    else:
        raise FileNotFoundError(f"Embedding file not found for: {file_path}")

def calculate_similarity_between_combinations(slide, bag_directory, feature_methods, normalizations, tile_size, magnification):
    """
    Calculate the similarity matrix for combinations of normalization and feature extraction methods,
    fixing the tile size and magnification.
    
    Args:
        slide (str): Slide name.
        bag_directory (str): Directory containing the embeddings.
        feature_methods (list): List of feature extraction methods.
        normalizations (list): List of normalization methods.
        tile_size (int): The tile size to be used.
        magnification (int): The magnification to be used.
    
    Returns:
        similarity_matrix: 2D numpy array representing the similarity between combinations of normalization and feature methods.
        combinations: List of all combinations of parameters used as labels for the matrix.
    """
    # Generate combinations of normalization and feature extraction method
    combinations = list(product(normalizations, feature_methods))
    
    # Initialize the similarity matrix
    num_combinations = len(combinations)
    similarity_matrix = np.zeros((num_combinations, num_combinations))
    
    # Load embeddings for all combinations
    embeddings_dict = {}
    for i, (norm, feature_method) in enumerate(combinations):
        try:
            embeddings_dict[i] = load_embeddings(bag_directory, tile_size, magnification, norm, feature_method, slide)
        except FileNotFoundError as e:
            print(e)
            embeddings_dict[i] = None

    # Calculate similarity between all pairs of combinations
    for i in range(num_combinations):
        for j in range(i, num_combinations):
            if embeddings_dict[i] is not None and embeddings_dict[j] is not None:
                # Calculate cosine similarity between embeddings
                sim = cosine_similarity(embeddings_dict[i], embeddings_dict[j])
                avg_similarity = np.mean(sim)
                similarity_matrix[i, j] = avg_similarity
                similarity_matrix[j, i] = avg_similarity  # Symmetric matrix
    
    return similarity_matrix, combinations


def optimize_ensemble_model(
    config: dict, 
    project: sf.Project, 
    all_bags: list, 
    train: sf.Dataset, 
    val: sf.Dataset, 
    test: sf.Dataset = None,
    results_df: pd.DataFrame = None,
    mil_conf = None,
    target = None
):
    """
    Optimized version of ensemble model selection by dynamically calculating and caching 
    a similarity matrix for each magnification and tile size combination.

    Args:
        config: The pathbench configuration dictionary.
        project: The Slideflow project.
        all_bags: List of all bags to sample from.
        train: The training dataset.
        val: The validation dataset.
        test: The test dataset (optional).
        results_df: The benchmarking results DataFrame, sorted by performance.

    Returns:
        best_bags: Best performing bag ensemble.
        best_performance: Best performance of the ensemble.
    """
    if results_df is None or results_df.empty:
        raise ValueError("results_df must not be empty. It should contain model performance data.")
    
    # Directory setup for bag files and cache
    project_name = config['experiment']['project_name']
    bag_directory = f"experiments/{project_name}/bags"
    ensemble_directory = f"experiments/{project_name}/ensemble_model"
    os.makedirs(ensemble_directory, exist_ok=True)  # Ensure the directory exists
    
    bag_paths = [os.path.join(bag_directory, d) for d in os.listdir(bag_directory) if os.path.isdir(os.path.join(bag_directory, d))]
    bag_names = {os.path.basename(d): d for d in bag_paths}
    
    # Get list of magnifications and tile sizes
    magnifications = config['benchmark_parameters']['tile_um']
    tile_sizes = config['benchmark_parameters']['tile_px']

    best_bags = []
    best_performance = 0  # Initialize best performance
    max_n_models = config['experiment']['ensemble']['max_n_models']
    use_similarity_matrix = config['experiment']['ensemble']['use_similarity_matrix']

    # Step 1: Add one model for every available combination of tile sizes and magnifications
    for mag in magnifications:
        for tile_size in tile_sizes:
            # Filter the results_df for the current magnification and tile size
            mag_tile_results = results_df[(results_df['tile_um'] == mag) & (results_df['tile_px'] == tile_size)]
            if mag_tile_results.empty:
                continue
            best_model_row = mag_tile_results.iloc[0]  # Select the top row for this combination
            
            normalization = best_model_row['normalization']
            feature_extraction = best_model_row['feature_extraction']
            
            # Construct the bag name for the best model
            best_bag = f"{tile_size}_{mag}_{normalization}_{feature_extraction}"
            
            if best_bag in bag_names:
                best_bags.append(best_bag)
                logging.info(f"Selected best bag for magnification {mag} and tile size {tile_size}: {best_bag}")
            else:
                logging.warning(f"Bag not found for {best_bag}. Skipping.")
    
    # Initial ensemble performance test (without additional dissimilar models)
    if best_bags:
        best_performance = evaluate_ensemble(best_bags, mil_conf, train, val, target, config, test, bag_names, project)
        logging.info(f"Initial ensemble performance: {best_performance} with bags: {best_bags}")
    else:
        logging.warning("No valid bags selected for the initial ensemble.")
    
    # Step 2: If similarity matrix is enabled, construct the matrix and add dissimilar models iteratively
    if use_similarity_matrix:
        for mag in magnifications:
            for tile_size in tile_sizes:
                # Filter the results_df for the current magnification and tile size
                mag_tile_results = results_df[(results_df['tile_um'] == mag) & (results_df['tile_px'] == tile_size)]
                if mag_tile_results.empty:
                    continue
                
                #Get bags associated with the chosen mag and tile size
                bag_directories = [bag_path for bag_name, bag_path in bag_names.items() if bag_name.startswith(f"{tile_size}_{mag}_")]
                # Calculate or load the cached similarity matrix for this magnification and tile size
                similarity_matrix, combinations = calculate_similarity_matrix_for_mag_tile(
                    bag_directories, mag_tile_results, mag, tile_size, ensemble_directory, bag_directory
                )
                
                # Find the best bag for this magnification and tile size (already added)
                best_model_row = mag_tile_results.iloc[0]
                best_bag = f"{best_model_row['tile_px']}_{mag}_{best_model_row['normalization']}_{best_model_row['feature_extraction']}"
                
                #Log the columns of the similarity matrix
                logging.info(f"Similarity matrix columns: {combinations}")
                #Get the relevant key
                key_parts = best_bag.split('_')

                color_method = key_parts[2]  
                model_name = '_'.join(key_parts[3:]) 

                best_bag_key = (color_method, model_name)
                # Ensure both the key and the columns are tuples of strings
                best_bag_key = tuple(map(str, best_bag_key))  # Convert the key elements to strings

                # Convert the column names to tuples of strings as well
                similarity_matrix.columns = similarity_matrix.columns.map(lambda x: tuple(map(str, x)))

                if best_bag_key in similarity_matrix.columns:
                    dissimilar_bags = similarity_matrix.loc[best_bag_key].sort_values().index.tolist()
                else:
                    logging.warning(f"Key {best_bag_key} not found in similarity matrix columns.")
                
                for dissimilar_bag in dissimilar_bags:
                    if dissimilar_bag in bag_names and dissimilar_bag in mag_tile_results['bag_name'].tolist():
                        # Test performance by adding this dissimilar model
                        current_bags = best_bags + [dissimilar_bag]
                        new_performance = evaluate_ensemble(current_bags, mil_conf, train, val, target, config, test, bag_names, project)
                        
                        if new_performance > best_performance:
                            best_bags.append(dissimilar_bag)
                            best_performance = new_performance
                            logging.info(f"Added dissimilar bag for magnification {mag} and tile size {tile_size}: {dissimilar_bag}")
                            if len(best_bags) >= max_n_models:
                                logging.info(f"Reached max number of models: {max_n_models}")
                                break  # Stop adding more models if the limit is reached
                        else:
                            logging.info(f"Discarded dissimilar bag {dissimilar_bag} as it did not improve performance.")
    
    # Final performance logging
    if best_bags:
        logging.info(f"Final ensemble performance: {best_performance} with bags: {best_bags}")
    else:
        logging.warning("No valid bags selected for the final ensemble.")

    return best_bags, best_performance


def calculate_similarity_matrix_for_mag_tile(bag_directories, mag_tile_results, mag, tile_size, ensemble_directory,
    global_bag_directory):
    """
    Dynamically calculate or load a cached similarity matrix for a specific magnification and tile size combination.

    Args:
        bag_directory: List of relevant bag directories.
        mag_tile_results: Filtered DataFrame for the specific magnification and tile size.
        mag: The magnification for the current tile size.
        tile_size: The tile size for the current magnification.
        ensemble_directory: Directory to cache the similarity matrix.
        global_bag_directory: Global bag directory (x/x/bags)

    Returns:
        similarity_matrix: The dynamically calculated or loaded similarity matrix.
        combinations: The combinations of feature extractor and normalization methods used.
    """
    # Define cache file path
    cache_file = os.path.join(ensemble_directory, f"similarity_matrix_{mag}_{tile_size}.csv")
    
    # If the similarity matrix is already cached, load it from the .csv file
    if os.path.exists(cache_file):
        logging.info(f"Loading cached similarity matrix for magnification {mag} and tile size {tile_size} from {cache_file}")
        similarity_matrix = pd.read_csv(cache_file, index_col=0)
        combinations = similarity_matrix.index.tolist()  # Row/column names represent combinations
        return similarity_matrix, combinations
    
    # Otherwise, calculate the similarity matrix
    feature_methods = mag_tile_results['feature_extraction'].unique().tolist()
    normalizations = mag_tile_results['normalization'].unique().tolist()

    #Get all bag directories where with the specified magnification and tile size

    slides = list(set([f.split('.')[0] for bag_directory in bag_directories for f in os.listdir(bag_directory) if f.endswith('.pt')]))

    logging.info(f"{feature_methods}, {normalizations}, {slides}")

    # Calculate the similarity matrix for this specific magnification and tile size
    similarity_matrix, combinations = calculate_similarity_matrix(
        bag_directory=global_bag_directory,
        slides=slides,
        feature_methods=feature_methods,
        normalizations=normalizations,
        tile_size=tile_size,
        magnification=mag,
        k=20,  # Number of neighbors
        method='cosine'  # Cosine similarity
    )
    
    logging.info(f"Generated matrix {similarity_matrix} for magnification {mag} and tile size {tile_size}")
    # Cache the similarity matrix to a CSV file with row/column names
    similarity_matrix_df = pd.DataFrame(similarity_matrix, index=combinations, columns=combinations)
    similarity_matrix_df.to_csv(cache_file)
    logging.info(f"Cached similarity matrix for magnification {mag} and tile size {tile_size} to {cache_file}")
    
    return similarity_matrix_df, combinations

def evaluate_ensemble(ensemble, mil_conf, train, val, target, config, test, bag_names, project):
    """Function to train and evaluate the ensemble on the validation set, and optionally on the test set.
    
    Args:
        ensemble (list): The ensemble of bags
        mil_conf (mil-config): The MIL configuration
        train (sf.Dataset): The training dataset
        val (sf.Dataset): The validation dataset
        target (str): The target variable
        config (dict): The configuration dictionary
        test (sf.Dataset): The test dataset
        bag_names (dict): Dictionary mapping bag names to directories
        project (sf.Project): The Slideflow project

    Returns:
        performance (float): The performance metric of the ensemble
    """
    # Train the MIL model with the new ensemble
    train_result = project.train_mil(
        config=mil_conf,
        outcomes=target,
        train_dataset=train,
        val_dataset=val,
        bags=[bag_names[bag] for bag in ensemble],
        exp_label=f"ensemble_{len(ensemble)}",
        pb_config=config
    )

    # Validate the model
    number = get_highest_numbered_filename(f"experiments/{config['experiment']['project_name']}/mil/")
    val_result = pd.read_parquet(f"experiments/{config['experiment']['project_name']}/mil/{number}-ensemble_{len(ensemble)}/predictions.parquet")

    # Evaluate validation performance
    if config['experiment']['task'] == 'survival':
        metrics, durations, events, predictions = calculate_survival_results(val_result)
    elif config['experiment']['task'] == 'regression':
        metrics, tpr, fpr, val_pr_curves = calculate_results(val_result, config, "ensemble", "val")
    else:  # Classification
        metrics, tpr, fpr, val_pr_curves = calculate_results(val_result, config, "ensemble", "val")

    # Calculate validation performance metric
    if config['experiment']['ensemble']['dataset'] == 'val':
        performance = np.mean(metrics[config['experiment']['ensemble']['performance_column']])

    # If test set is provided, evaluate on the test set as well
    if test is not None:
        test_result = eval_mil(
            weights=f"experiments/{config['experiment']['project_name']}/mil/{number}-ensemble_{len(ensemble)}",
            outcomes=target,
            dataset=test,
            bags=[bag_names[bag] for bag in ensemble],
            config=mil_conf,
            outdir=f"experiments/{config['experiment']['project_name']}/mil_eval/{number}-ensemble_{len(ensemble)}",
            pb_config=config,
        )
        test_result_df = pd.read_parquet(f"experiments/{config['experiment']['project_name']}/mil_eval/{number}-ensemble_{len(ensemble)}/00000-mm_attention_mil/predictions.parquet")

        if config['experiment']['task'] == 'survival':
            test_metrics, test_durations, test_events, test_predictions = calculate_survival_results(test_result_df)
        elif config['experiment']['task'] == 'regression':
            test_metrics, test_tpr, test_fpr, test_pr_curves = calculate_results(test_result_df, config, "ensemble", "test")
        else:  # Classification
            test_metrics, test_tpr, test_fpr, test_pr_curves = calculate_results(test_result_df, config, "ensemble", "test")

        logging.info(f"Test results for ensemble: {test_metrics}")

    if config['experiment']['ensemble']['dataset'] == 'test':
        performance = np.mean(test_metrics[config['experiment']['ensemble']['performance_column']]) if test_metrics else 0

    if performance == 0:
        logging.warning("Performance metric is 0. Did you provide the relevant results dataframe?")
    else:
        logging.info(f"Ensemble with {ensemble}, Performance metric: {performance} on dataset {config['experiment']['ensemble']['dataset']}")
    return performance

def ensemble(project, config):
    """
    Main ensemble construction function. This function optimizes the ensemble model by selecting 
    the best-performing model for each magnification based on prior benchmarking results. 
    The ensemble is constructed by picking the top model for each magnification and evaluating 
    the performance on the validation set.
    
    Args:
        project: The Slideflow project.
        config: The PathBench configuration dictionary.
    """
    # Get slides and bag directories
    slides = pd.read_csv(project.annotations)['slide'].unique()
    bag_directory = f"experiments/{config['experiment']['project_name']}/bags"
    bag_paths = [os.path.join(bag_directory, d) for d in os.listdir(bag_directory) if os.path.isdir(os.path.join(bag_directory, d))]
    bag_names = {os.path.basename(d): d for d in bag_paths}
    logging.info(f"Found bags: {bag_names}")

    # Get feature extractor models, magnifications, tile sizes, and normalizations from the config
    feature_methods = config['benchmark_parameters']['feature_extraction']
    magnifications = config['benchmark_parameters']['tile_um']
    tile_sizes = config['benchmark_parameters']['tile_px']
    normalizations = config['benchmark_parameters']['normalization']

    # Determine the target variable based on task
    target = determine_target_variable(config['experiment']['task'], config)
    logging.info(f"Target variable: {target}")

    # Configure MIL model
    mil_conf = mil_config("mm_attention_mil")

    # Load the results dataframe
    results_df = pd.read_csv(config['experiment']['ensemble']['results_df']) if config['experiment']['ensemble']['results_df'] is not None else None
    if results_df is not None:
        logging.info(f"Results dataframe loaded with {len(results_df)} rows.")

    # Create directory for ensemble models
    ensemble_model_directory = f"experiments/{config['experiment']['project_name']}/ensemble_model"
    os.makedirs(ensemble_model_directory, exist_ok=True)

    # Load the dataset
    tile_px, tile_um = tile_sizes[0], magnifications[0]
    all_data = project.dataset(tile_px=tile_px, tile_um=tile_um)
    logging.info("Dataset loaded.")
    
    # Split the dataset into training and validation sets
    train = all_data.filter(filters={'dataset': 'train'})
    train = balance_dataset(train, config['experiment']['task'], config)
    test_set = all_data.filter(filters={'dataset': 'validate'})
    
    # Split into training and validation sets
    logging.info("Splitting datasets...")
    splits = split_datasets(config, project, determine_splits_file(config, f"experiments/{config['experiment']['project_name']}"), target, f"experiments/{config['experiment']['project_name']}", train)
    train, val = splits[0]
    
    # Optimize the ensemble model by selecting the top model for each magnification
    logging.info("Optimizing ensemble model based on top models for each magnification...")
    best_bags, best_performance = optimize_ensemble_model(
        config, 
        project, 
        bag_names, 
        train, 
        val, 
        test_set, 
        results_df,
        mil_conf,
        target
    )
    
    logging.info(f"Best bags: {best_bags}, Best performance: {best_performance}")
