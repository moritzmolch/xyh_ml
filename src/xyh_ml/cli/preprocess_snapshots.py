"""
Preprocess snapshots and store them as data frames for training and evaluation.
"""

import logging
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore

from xyh_ml import PROJECT_DIR
from xyh_ml.config import Config, yield_base_dirs
from xyh_ml.data.snapshots import group_snapshots, save_data_frames

# Set up logging
logger = logging.getLogger(__name__)

# Enable hydra to find the configuration file and register it with the
# ConfigStore
cs = ConfigStore.instance()
cs.store(name="preprocess_snapshots", node=Config)


@hydra.main(
    config_path=str(PROJECT_DIR / "config"),
    config_name="preprocess_snapshots",
    version_base="1.3",
)
def main(cfg: Config):
    # Get snapshot groups for all campaigns and channels
    snapshot_groups: dict[tuple[str, str, str], list[Path]] = {}

    for (campaign, channel), base_dir in yield_base_dirs(cfg.data_source):
        # Construct snapshot groups for this era and channel
        _snapshot_groups = group_snapshots(
            base_dir, campaign, channel, cfg.outputs.classes
        )

        # Check if any keys are overlapping with the existing groups
        if not snapshot_groups.keys().isdisjoint(_snapshot_groups.keys()):
            raise ValueError(
                "Overlapping keys (campaign, channel, output_class) found in "
                + "snapshot groups."
            )

        # Append current group to the global list of snapshot groups
        snapshot_groups.update(_snapshot_groups)

    # Store merged data frames to disk
    save_data_frames(
        snapshot_groups,
        cfg.inputs.variables,
        cfg.inputs.parameters,
        cfg.outputs.variables[0],
        cfg.outputs.classes,
        Path(cfg.artifacts.data_frame_dir),
    )


if __name__ == "__main__":
    main()
