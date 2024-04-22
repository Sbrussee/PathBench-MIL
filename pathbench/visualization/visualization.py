import matplotlib.pyplot as plt
import os

def visualize_activations(features, config, dataset, save_string):
    os.makedirs(f"{config['experiment']['project_name']}/visualizations", exist_ok=True)
    slide_map = features.map_activations(
        n_neighbors=10,
        min_dist=0.2
    )

    labels, unique_labels = dataset.labels('category', format='name')
    slide_map.label_by_slide(labels)
    slide_map.plot(s=10)
    plt.savefig(f"{config['experiment']['project_name']}/visualizations/{save_string}.png")
    plt.close()
