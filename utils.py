import os
import numpy as np
from scipy.ndimage import zoom
import torch
from PIL import Image


def get_explanations_for_image(
    image: Image, model, explainer, transform, device, J
):
    Y = []
    X = torch.empty(0, 3, 224, 224)
    x = transform(image).unsqueeze(0)  # noqa: F821
    y = model(x.to(device)).argmax().item()
    Y.append(y)
    X = torch.cat((X, x), dim=0)
    explanation = explainer(X, Y)
    return explanation.squeeze()



def get_diagonal(grad_wam: np.ndarray, J: int) -> dict[np.ndarray]:
    """
    Get diagonal blocks from WAM:
    - level_0 ... level_{J-1}  (szczegóły, od najdrobniejszych)
    - approx                  (najgrubsza skala)
    """
    H, W = grad_wam.shape
    assert H == W, "grad_wam must be square"

    diagonals = {}

    for j in range(J):
        start = H // (2 ** (j + 1))
        end = H // (2**j)
        diagonals[f"level_{j}"] = grad_wam[start:end, start:end]

    approx_size = H // (2**J)
    diagonals["approx"] = grad_wam[:approx_size, :approx_size]

    return diagonals


def get_mean_pixelwise_variance(grad_wam, J, size="maximal"):
    """
    Liczy pixel-wise wariancję pomiędzy poziomami szczegółów.

    Parametry:
    - grad_wam : 2D ndarray
    - J        : liczba poziomów
    - size     : "maximal" | "minimal"
        - maximal -> skaluj do największego poziomu
        - minimal -> skaluj do najmniejszego poziomu

    Zwraca:
    - mean_variance : float
    - variance_map  : 2D ndarray
    """
    diagonals = get_diagonal(grad_wam, J)

    # tylko szczegóły
    detail_levels = [diagonals[f"level_{j}"] for j in range(J)]

    sizes = [lvl.shape[0] for lvl in detail_levels]

    if size == "maximal":
        target_size = max(sizes)
    elif size == "minimal":
        target_size = min(sizes)
    else:
        raise ValueError("size must be 'maximal' or 'minimal'")

    resized = []
    for lvl in detail_levels:
        scale = target_size / lvl.shape[0]
        lvl_resized = zoom(lvl, scale, order=1)
        resized.append(lvl_resized[:target_size, :target_size])

    stack = np.stack(resized, axis=0)  # (J, H, W)

    variance_map = np.var(stack, axis=0)
    mean_variance = float(np.mean(variance_map))

    return mean_variance, variance_map


def rank_images(explanations, J, size="maximal"):
    """
    Rankuje obrazy według cross-level pixel-wise wariancji.

    Zwraca:
    lista słowników:
    [
        {
            "image_index": int,
            "mean_pixelwise_variance": float
        },
        ...
    ]
    """
    ranking = []

    for idx, grad_wam in enumerate(explanations):
        mean_var, _ = get_mean_pixelwise_variance(grad_wam, J, size=size)
        ranking.append({"image_index": idx, "mean_pixelwise_variance": mean_var})

    ranking.sort(key=lambda x: x["mean_pixelwise_variance"], reverse=True)

    return ranking

def get_gradients_attribution_on_levels(images, model, explainer, transform, device, LEVELS):
    """
    For each image in list of images, gets diagonal, sums gradients at each level and returns normalized values.
    get_explanations_for_image(img, resnet, resnet_explainer, transform, device, LEVELS)
    """
    explanations = [get_explanations_for_image(img, model, explainer, transform, device, LEVELS) for img in images]
    diagonals_list = [get_diagonal(expl, LEVELS) for expl in explanations]
    gradients_at_levels = []

    for diag in diagonals_list:
        gradient_sums_per_level = []
        for k, v in diag.items():
            gradient_sum = np.sum(np.abs(v))
            gradient_sums_per_level.append(gradient_sum)
        # Normalize
        gradients_sum_per_level = np.array(gradient_sums_per_level)
        gradients_sum_per_level /= np.sum(gradients_sum_per_level)
        gradients_at_levels.append(gradients_sum_per_level)
    

    return gradients_at_levels

def get_multiple_grad_attr(images, models, explainers, transform, device, LEVELS):
    all_gradients = []
    for (model, explainer) in zip(models, explainers):
        gradients = get_gradients_attribution_on_levels(images, model, explainer, transform, device, LEVELS)
        all_gradients.append(gradients)
    return all_gradients

def get_mean_across_images(all_grads):
    """
    Gets mean gradients across images for each model. Returns mean gradient for each level for each model"""
    
    mean_grads = []
    for grads in all_grads:
        mean_grad = np.mean(np.array(grads), axis=0)
        mean_grads.append(mean_grad)
    return mean_grads
