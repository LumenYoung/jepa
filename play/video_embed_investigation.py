import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import os
from tqdm import tqdm
from typing import List

data_root = "cliport_vid_embed"
# Assuming `first_embed` is your tensor with shape [1, 12544, 1024]

# Step 1: Flatten the tensor to 2D (discarding the first dimension because it's size 1)
# We reshape the tensor from [1, 12544, 1024] to [12544, 1024] for t-SNE


def get_embed(data_root: str) -> List[torch.Tensor]:
    epochs = [f.path for f in os.scandir(data_root) if f.is_dir()]
    tensors = []

    for epoch in epochs:
        embeddings = [
            f.path for f in os.scandir(epoch) if f.is_file() and f.name.endswith(".pt")
        ]

        for embed in embeddings:
            embed_tensor = torch.load(embed, map_location=torch.device("cpu"))
            tensors.append(embed_tensor)

    return tensors


def compute_tsne_vis(embed: torch.Tensor) -> None:
    vectors = embed.squeeze(0).detach().numpy()

    # Step 2: Instantiate and run t-SNE to reduce to 2 dimensions for features
    tsne = TSNE(n_components=2, random_state=42)  # Using 2 components for a 2D plot
    reduced_vectors = tsne.fit_transform(vectors)

    # Step 3: Prepare the x, y coordinates for plotting
    x = reduced_vectors[:, 0]  # The first dimension from t-SNE (feature 1)
    y = reduced_vectors[:, 1]  # The second dimension from t-SNE (feature 2)
    z = np.arange(len(x))  # Representing individual instance index

    # Step 4: Plot using a 3D plot but with the third axis as the instance index
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for i in tqdm(range(len(z) - 1), desc="Plotting"):
        ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], [z[i], z[i + 1]], color="blue")

    ax.scatter(x, y, z)  # Points in red

    # Labels for axes
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Frame")

    # Show plot
    plt.show()


# Assuming 'tensor' is the given tensor with shape [1, 12544, 1024]
# We initialize it with random data for simulation purposes.

# Step 1: get a tensor excluding the first entry along the second dimension


def compute_l1_distances_between(embed1: torch.Tensor, embed2: torch.Tensor) -> None:
    l1_distances = torch.abs(embed1 - embed2).sum(dim=2)

    l1_distances_flat = l1_distances.flatten().numpy()

    plt.figure(figsize=(10, 5))
    plt.plot(l1_distances_flat)
    plt.title("L1 Distances Between Consecutive Entries")
    plt.xlabel("Entry Index")
    plt.ylabel("L1 Distance")
    plt.grid(True)
    plt.show()


def compute_l1_distances_within(embed: torch.Tensor) -> None:
    tensor_ex_first = embed[:, 1:, :]
    # Step 2: get a tensor excluding the last entry along the second dimension
    tensor_ex_last = embed[:, :-1, :]
    # Step 3: compute the L1 distance between tensor_ex_first and tensor_ex_last
    l1_distances = torch.abs(tensor_ex_first - tensor_ex_last).sum(dim=2)
    # Step 4: prepend a zero for the distance of the first entry
    first_zero = torch.zeros((1, 1))
    l1_distances = torch.cat((first_zero, l1_distances), dim=1)
    # Verify the final shape of the tensor (should be [1, 12544])
    # Prepare the data for plotting by flattening it
    l1_distances_flat = l1_distances.flatten().numpy()
    # Step 5: plot the distances using matplotlib
    plt.figure(figsize=(10, 5))
    plt.plot(l1_distances_flat)
    plt.title("L1 Distances Between Consecutive Entries")
    plt.xlabel("Entry Index")
    plt.ylabel("L1 Distance")
    plt.grid(True)
    plt.show()


def compare_diverg_vids():
    dir1 = "diverging_vid1_embed"
    dir2 = "diverging_vid2_embed"

    tensors1 = get_embed(dir1)
    tensors2 = get_embed(dir2)

    for embed1, embed2 in zip(tensors1, tensors2):
        compute_l1_distances_between(embed1, embed2)

def compare_converg_vids():
    dir1 = "converging_vid1_embed"
    dir2 = "converging_vid2_embed"

    tensors1 = get_embed(dir1)
    tensors2 = get_embed(dir2)

    for embed1, embed2 in zip(tensors1, tensors2):
        compute_l1_distances_between(embed1, embed2)

if __name__ == "__main__":
    # compute_tsne_vis(first_embed)
    # compute_l1_distances_within(first_embed)

    # compare_diverg_vids()
    compare_converg_vids()

