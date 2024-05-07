import numpy as np
import matplotlib.pyplot
import umap
import umap.plot
from sklearn.datasets import load_digits
from fire import Fire
import os

from ginv.istanbul_ein_dataset import IstanbulEinDataset
from ginv.transform_graph import transform_graph_for_umap_node_level

# digits = load_digits()

# mapper = umap.UMAP().fit(digits.data)
# # umap.plot.output_file("plot.png")
# umap.plot.points(mapper, labels=digits.target)
# matplotlib.pyplot.savefig("plot.png")


def main(node_count=0, plots_dir="./plots/umap"):

    os.makedirs(plots_dir, exist_ok=True)

    dataset = IstanbulEinDataset("./data/istanbul")

    if node_count > 0:
        print("Reducing the dataset")
        dataset.reduce(node_count)
        print(dataset)

    print("Creating umap_data")
    umap_data = transform_graph_for_umap_node_level(dataset)
    labels = np.copy(dataset.node_profits)
    logs = np.log10(np.abs(labels))
    positive_labels = labels > 0
    labels[positive_labels] = np.floor(logs[positive_labels])
    labels[~positive_labels] = -np.floor(logs[~positive_labels])

    for n_neighbours in [2, 4, 8, 16]:
        for min_dist in [0.05, 0.1, 0.2, 0.4]:

            name = f"umap_istanbul_{dataset.node_count}n_{n_neighbours}nb_{str(min_dist).replace('.', '')}d"

            print(f"Fitting umap for {name}")
            reducer = umap.UMAP(
                n_neighbors=n_neighbours,
                min_dist=min_dist,
            )
            mapper = reducer.fit(umap_data)

            print(f"Plotting for {name}")
            umap.plot.points(mapper, labels=labels)

            matplotlib.pyplot.savefig(f"{plots_dir}/{name}.png")

            print(f"Saving embeddings for {name}")
            embeddings = reducer.transform(umap_data)

            np.save(f"{plots_dir}/embeddings_{name}", embeddings)

            print()


if __name__ == '__main__':
    Fire(main)
