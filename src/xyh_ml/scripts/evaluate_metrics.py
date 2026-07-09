from pathlib import Path
import pandas as pd
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set seaborn plot style
sns.set_style("white")
sns.set_theme(context="talk")
sns.set_palette("colorblind")


THIS_DIR = Path(__file__).resolve().parent
INPUT_DIR = THIS_DIR.parent.parent.parent / "test_output"
OUTPUT_DIR = INPUT_DIR

EXTENSIONS = ["pdf", "png"]


def _prepare_output_dir(output_dir: Path):
    # Create the output directory if it does not exist
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        logger.debug(f"Created output directory {output_dir}")


def _load_metrics(input_file: Path):
    # Deserialize tracked metrics from a JSON file
    with input_file.open(mode="r") as f:
        metrics = json.load(f)
    logger.debug(
        f"Loaded metrics from {input_file} in JSON format"
    )
    return metrics


def _plot_loss(metrics: dict):
    # Create figure
    fig, ax = plt.subplots()

    # Plot training and validation loss
    ax.plot(
        metrics["train_step"],
        metrics["train_loss"],
        label="Training",
    )
    ax.plot(
        metrics["val_step"],
        metrics["val_loss"],
        label="Validation",
    )

    # Set
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")

    return fig, ax


def main():
    # Prepare the output directory
    _prepare_output_dir(OUTPUT_DIR)

    # Load the input file
    input_file = OUTPUT_DIR / "metrics.json"
    metrics = _load_metrics(input_file)

    # Plot loss
    fig, ax = _plot_loss(metrics)
    for ext in EXTENSIONS:
        fig.savefig(OUTPUT_DIR / f"loss.{ext}")

    # Show all plots
    plt.show()


if __name__ == "__main__":
    main()

