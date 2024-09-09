import os
from pathlib import Path
from typing import List
from fire import Fire

from ginv.insider_network_dynamic_dataset import InsiderNetworkDynamicDataset
from run_umap import parse_insider_days


def prepare_insiders_for_glodyne(
    days: List[int] = [0, 501],
    save_path = "./data/prepared/insider_network.pkl",
    dataset_path = "./data/insider-network",
    prune=True,
):
    
    save_path = Path(save_path)

    os.makedirs(save_path.parent, exist_ok=True)

    days = parse_insider_days(days)

    dataset = InsiderNetworkDynamicDataset(
        dataset_path,
        days,
    )

    dataset.to_networkx_pickle(save_path, prune=prune)

    print("Done")


if __name__ == "__main__":
    Fire()
