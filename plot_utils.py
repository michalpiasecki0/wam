from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from src.viewers import plot_wam


def plot_diagonal(diagonals: dict[np.ndarray], cmap="viridis", figsize=(14, 4)) -> plt.Figure:
    """
    Rysuje wszystkie poziomy diagonalne + approx obok siebie
    """
    keys = list(diagonals)
    n = len(keys)

    fig, axes = plt.subplots(1, n, figsize=figsize)

    # Ensure axes is always iterable
    for ax, key in zip(axes, keys):
        im = ax.imshow(diagonals[key], cmap=cmap)
        ax.set_title(str(key))
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

def visualize_explanations_basic(
    explanations,
    images,
    levels,
    cmap='viridis',
    smooth=True,
    which=0
):
    """
    Visualizes original images and corresponding explanations (WAM).

    Parameters:
    - explanations : list or tensor of explanation maps
    - images       : tensor of images (N, C, H, W)
    - levels       : contour levels for plot_wam
    - cmap         : colormap
    - smooth       : whether to smooth WAM
    - which        : index of image to show or 'all'
    """

    if which == 'all':
        indices = range(len(explanations))
    else:
        if which < 0 or which >= len(explanations):
            raise IndexError("Parameter 'which' is out of range.")
        indices = [which]

    for i in indices:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # --- Original image ---
        ax1.imshow(images[i])

        ax1.set_title('Original Image')
        ax1.axis('off')

        # --- WAM ---
        plot_wam(
            ax2,
            explanations[i],
            levels=levels,
            cmap=cmap,
            smooth=smooth
        )
        ax2.set_title('WAM')
        ax2.axis('off')

        plt.tight_layout()
        plt.show()

    return None


def visualize_gradients_at_levels(gradients_at_levels, title, names=None):
    """
    Visualizes gradients at levels as a grouped bar plot.
    For each scale level creates as many bars as there are values
    corresponding to that level.
    
    gradients_at_levels shape: (num_samples, num_levels)
    """
    gradients_at_levels = np.array(gradients_at_levels)
    num_samples, num_levels = gradients_at_levels.shape
    
    if names is None:
        names = [f"Sample {i+1}" for i in range(num_samples)]

    levels = np.arange(num_levels)
    bar_width = 0.8 / num_samples

    plt.figure(figsize=(10, 6))

    for i in range(num_samples):
        offsets = levels + i * bar_width
        plt.bar(
            offsets,
            gradients_at_levels[i],
            width=bar_width,
            label=names[i]
        )

    plt.xlabel("Scale level")
    plt.ylabel("Attribution")
    plt.title(title)

    plt.xticks(levels + 0.4, levels + 1)
    plt.legend()
    plt.tight_layout()
    plt.show()
