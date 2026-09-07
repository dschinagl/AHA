from pathlib import Path

import matplotlib.pyplot as plt
import torch


def save_input_heatmap_overlay(
    image: torch.Tensor,
    heatmap: torch.Tensor,
    path: Path,
    title: str | None = None,
    title_color: str = "black",
) -> None:
    """Input | heatmap | overlay, (turbo colormap, alpha=0.5 overlay)."""
    image_np = image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    heat_np = heatmap.detach().cpu().float().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    ax[0].imshow(image_np)
    ax[0].axis("off")
    ax[0].set_title("Image")

    ax[1].imshow(heat_np, cmap="turbo")
    ax[1].axis("off")
    ax[1].set_title("Heatmap")

    ax[2].imshow(image_np)
    ax[2].imshow(heat_np, cmap="turbo", alpha=0.5)
    ax[2].axis("off")
    ax[2].set_title("Overlay")

    if title is not None:
        fig.suptitle(title, color=title_color)

    fig.tight_layout()
    fig.savefig(str(path))
    plt.close(fig)
