"""
Data Preprocessing Module for Dead Tree Segmentation Project

Features:
- 4-channel image construction (RGB + NIR)
- File alignment verification
- Stratified cross-validation based on dead tree area ratio
- Data augmentation with synchronized transformations
- Statistical analysis and visualization
- Reproducible random seed management

Author: Peng Cui(z5557532)
Date: July 2025
"""

import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import shuffle
import pandas as pd
import logging
from dataclasses import dataclass
import pickle
import json
from collections import defaultdict
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatasetStats:
    """Container for dataset statistics"""

    total_samples: int
    valid_samples: int
    invalid_samples: int
    dead_tree_ratios: List[float]
    stratification_distribution: Dict[str, int]
    mean_dead_tree_ratio: float
    std_dead_tree_ratio: float


class DeadTreeDataPreprocessor:
    """
    Comprehensive data preprocessing class for dead tree segmentation project.

    This class handles loading, validation, augmentation, and stratified splitting
    of aerial imagery data for dead tree detection.
    """

    def __init__(
        self,
        dataset_root: str,
        random_seed: int = 42,
        stratification_bins: int = 4,
        test_split_ratio: float = 0.2,
        k_folds: int = 5,
    ):
        """
        Initialize the data preprocessor.

        Args:
            dataset_root: Path to USA_segmentation folder
            random_seed: Random seed for reproducibility
            stratification_bins: Number of stratification bins
            test_split_ratio: Ratio for final test set split
            k_folds: Number of folds for cross-validation
        """
        self.dataset_root = Path(dataset_root)
        self.random_seed = random_seed
        self.stratification_bins = stratification_bins
        self.test_split_ratio = test_split_ratio
        self.k_folds = k_folds

        # Set random seeds for reproducibility
        np.random.seed(random_seed)

        # Initialize paths
        self.rgb_path = self.dataset_root / "RGB_images"
        self.nrg_path = self.dataset_root / "NRG_images"
        self.mask_path = self.dataset_root / "masks"

        # Initialize data containers
        self.sample_list = []
        self.dead_tree_ratios = []
        self.stratification_labels = []
        self.stats = None

        logger.info(f"Initialized DeadTreeDataPreprocessor with:")
        logger.info(f"  Dataset root: {self.dataset_root}")
        logger.info(f"  Random seed: {self.random_seed}")
        logger.info(f"  Stratification bins: {self.stratification_bins}")

    def verify_dataset_structure(self) -> bool:
        """
        Verify that all required directories exist.

        Returns:
            bool: True if all directories exist
        """
        required_dirs = [self.rgb_path, self.nrg_path, self.mask_path]
        missing_dirs = [d for d in required_dirs if not d.exists()]

        if missing_dirs:
            logger.error(f"Missing directories: {missing_dirs}")
            return False

        logger.info("Dataset structure verification passed")
        return True

    def get_matching_files(self) -> List[Dict[str, str]]:
        """
        Find all matching RGB, NRG, and mask files.

        Returns:
            List of dictionaries containing file paths for each sample
        """
        # Get all RGB files
        rgb_files = list(self.rgb_path.glob("*.png"))
        logger.info(f"Found {len(rgb_files)} RGB files")

        matching_files = []
        missing_files = []

        for rgb_file in rgb_files:
            # Extract base name (remove RGB_ prefix and extension)
            base_name = rgb_file.stem.replace("RGB_", "")

            # Construct corresponding NRG and mask file paths
            nrg_file = self.nrg_path / f"NRG_{base_name}.png"
            mask_file = self.mask_path / f"mask_{base_name}.png"

            # Check if all files exist
            if nrg_file.exists() and mask_file.exists():
                matching_files.append(
                    {
                        "base_name": base_name,
                        "rgb_path": str(rgb_file),
                        "nrg_path": str(nrg_file),
                        "mask_path": str(mask_file),
                    }
                )
            else:
                missing_files.append(
                    {
                        "base_name": base_name,
                        "rgb_exists": rgb_file.exists(),
                        "nrg_exists": nrg_file.exists(),
                        "mask_exists": mask_file.exists(),
                    }
                )

        logger.info(f"Found {len(matching_files)} complete sample sets")
        if missing_files:
            logger.warning(f"Found {len(missing_files)} incomplete sample sets")

        return matching_files

    def load_and_validate_image(
        self, rgb_path: str, nrg_path: str, mask_path: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Load and validate a single sample (RGB, NRG, mask).

        Args:
            rgb_path: Path to RGB image
            nrg_path: Path to NRG image
            mask_path: Path to mask image

        Returns:
            Tuple of (4-channel image, mask) or None if validation fails
        """
        try:
            # Load images
            rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            nrg_img = cv2.imread(nrg_path, cv2.IMREAD_COLOR)
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Check if loading was successful
            if rgb_img is None or nrg_img is None or mask_img is None:
                logger.warning(f"Failed to load images for {Path(rgb_path).stem}")
                return None

            # Convert BGR to RGB for RGB image
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            nrg_img = cv2.cvtColor(nrg_img, cv2.COLOR_BGR2RGB)

            # Verify shapes match
            if (
                rgb_img.shape[:2] != nrg_img.shape[:2]
                or rgb_img.shape[:2] != mask_img.shape
            ):
                logger.warning(f"Shape mismatch for {Path(rgb_path).stem}")
                return None

            # Extract NIR channel (assuming it's the first channel in NRG)
            nir_channel = nrg_img[:, :, 0]

            # Construct 4-channel image (RGB + NIR)
            four_channel_img = np.dstack(
                [
                    rgb_img[:, :, 0],  # Red
                    rgb_img[:, :, 1],  # Green
                    rgb_img[:, :, 2],  # Blue
                    nir_channel,  # NIR
                ]
            )

            # Normalize mask to binary (0, 1)
            mask_binary = (mask_img > 127).astype(np.uint8)

            return four_channel_img, mask_binary

        except Exception as e:
            logger.error(f"Error processing {Path(rgb_path).stem}: {str(e)}")
            return None

    def calculate_dead_tree_ratio(self, mask: np.ndarray) -> float:
        """
        Calculate the ratio of dead tree pixels in the mask.

        Args:
            mask: Binary mask image

        Returns:
            Ratio of dead tree pixels (0.0 to 1.0)
        """
        total_pixels = mask.size
        dead_tree_pixels = np.sum(mask)
        return dead_tree_pixels / total_pixels if total_pixels > 0 else 0.0

    def create_stratification_labels(self, ratios: List[float]) -> List[str]:
        """
        Create stratification labels based on dead tree ratios.
        Uses actual data range (0 to max_ratio) instead of theoretical range (0 to 1).

        Args:
            ratios: List of dead tree ratios

        Returns:
            List of stratification labels
        """
        # Get actual data range
        min_ratio = min(ratios)
        max_ratio = max(ratios)

        logger.info(f"Dead tree ratio range: {min_ratio:.4f} to {max_ratio:.4f}")

        # Create bins based on actual data range
        bins = np.linspace(min_ratio, max_ratio, self.stratification_bins + 1)

        # Special handling for exactly zero ratios (all background)
        labels = []
        for ratio in ratios:
            if ratio == 0.0:
                labels.append("bin_0")  # All background
            else:
                # Use actual data range for binning
                bin_idx = np.digitize(ratio, bins) - 1
                # Ensure bin index is within valid range
                bin_idx = max(1, min(bin_idx, self.stratification_bins - 1))
                labels.append(f"bin_{bin_idx}")

        # Log bin ranges for verification
        for i in range(self.stratification_bins):
            if i == 0:
                logger.info(f"bin_0: exactly 0.0 (pure background)")
            else:
                lower = (
                    bins[i] if i > 1 else bins[1]
                )  # bin_1 starts from first non-zero bin
                upper = bins[i + 1] if i < self.stratification_bins - 1 else max_ratio
                logger.info(f"bin_{i}: {lower:.4f} to {upper:.4f}")

        return labels

    def load_and_preprocess_dataset(self) -> bool:
        """
        Load and preprocess the entire dataset.

        Returns:
            bool: Success status
        """
        logger.info("Starting dataset loading and preprocessing...")

        # Verify dataset structure
        if not self.verify_dataset_structure():
            return False

        # Get matching files
        matching_files = self.get_matching_files()
        if not matching_files:
            logger.error("No matching files found")
            return False

        # Process each sample
        valid_samples = []
        dead_tree_ratios = []

        for file_info in matching_files:
            result = self.load_and_validate_image(
                file_info["rgb_path"], file_info["nrg_path"], file_info["mask_path"]
            )

            if result is not None:
                four_channel_img, mask = result
                ratio = self.calculate_dead_tree_ratio(mask)

                valid_samples.append(
                    {
                        "base_name": file_info["base_name"],
                        "image_shape": four_channel_img.shape,
                        "mask_shape": mask.shape,
                        "dead_tree_ratio": ratio,
                        "rgb_path": file_info["rgb_path"],
                        "nrg_path": file_info["nrg_path"],
                        "mask_path": file_info["mask_path"],
                    }
                )
                dead_tree_ratios.append(ratio)

        # Create stratification labels
        stratification_labels = self.create_stratification_labels(dead_tree_ratios)

        # Store results
        self.sample_list = valid_samples
        self.dead_tree_ratios = dead_tree_ratios
        self.stratification_labels = stratification_labels

        # Calculate statistics
        self._calculate_statistics(len(matching_files), len(valid_samples))

        logger.info(f"Successfully loaded {len(valid_samples)} valid samples")
        return True

    def _calculate_statistics(self, total_files: int, valid_files: int):
        """Calculate and store dataset statistics."""
        stratification_dist = {}
        for label in self.stratification_labels:
            stratification_dist[label] = stratification_dist.get(label, 0) + 1

        self.stats = DatasetStats(
            total_samples=total_files,
            valid_samples=valid_files,
            invalid_samples=total_files - valid_files,
            dead_tree_ratios=self.dead_tree_ratios.copy(),
            stratification_distribution=stratification_dist,
            mean_dead_tree_ratio=np.mean(self.dead_tree_ratios),
            std_dead_tree_ratio=np.std(self.dead_tree_ratios),
        )

    def create_stratified_splits(self, train_val_ratio: float = 0.8) -> Dict[str, Any]:
        """
        Create stratified train/validation/test splits.

        Args:
            train_val_ratio: Ratio for train within train+val set

        Returns:
            Dictionary containing split information
        """
        if not self.sample_list:
            raise ValueError(
                "Dataset not loaded. Call load_and_preprocess_dataset() first."
            )

        # Create indices
        indices = np.arange(len(self.sample_list))

        # First split: separate test set (20%)
        train_val_indices, test_indices, train_val_labels, test_labels = (
            train_test_split(
                indices,
                self.stratification_labels,
                test_size=self.test_split_ratio,
                stratify=self.stratification_labels,
                random_state=self.random_seed,
            )
        )

        # Create K-fold splits for training/validation
        skf = StratifiedKFold(
            n_splits=self.k_folds, shuffle=True, random_state=self.random_seed
        )

        cv_splits = []
        for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(train_val_indices, train_val_labels)
        ):
            # Convert to original indices
            train_indices = train_val_indices[train_idx]
            val_indices = train_val_indices[val_idx]

            cv_splits.append(
                {
                    "fold": fold_idx,
                    "train_indices": train_indices.tolist(),
                    "val_indices": val_indices.tolist(),
                    "train_size": len(train_indices),
                    "val_size": len(val_indices),
                }
            )

        splits_info = {
            "test_indices": test_indices.tolist(),
            "test_size": len(test_indices),
            "cv_splits": cv_splits,
            "stratification_labels": self.stratification_labels,
            "random_seed": self.random_seed,
        }

        logger.info(f"Created {self.k_folds}-fold cross-validation splits")
        logger.info(
            f"Test set size: {len(test_indices)} ({self.test_split_ratio*100:.1f}%)"
        )

        return splits_info

    def apply_augmentation(
        self, image: np.ndarray, mask: np.ndarray, augmentation_type: str = "basic"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply synchronized augmentation to image and mask.

        Args:
            image: 4-channel input image
            mask: Binary mask
            augmentation_type: Type of augmentation to apply

        Returns:
            Tuple of (augmented_image, augmented_mask)
        """
        aug_image = image.copy()
        aug_mask = mask.copy()

        if augmentation_type == "basic":
            # Random horizontal flip
            if np.random.random() > 0.5:
                aug_image = np.fliplr(aug_image)
                aug_mask = np.fliplr(aug_mask)

            # Random vertical flip
            if np.random.random() > 0.5:
                aug_image = np.flipud(aug_image)
                aug_mask = np.flipud(aug_mask)

            # Random rotation (90, 180, 270 degrees)
            k = np.random.choice([0, 1, 2, 3])
            if k > 0:
                aug_image = np.rot90(aug_image, k)
                aug_mask = np.rot90(aug_mask, k)

        elif augmentation_type == "advanced":
            # Include basic augmentations plus additional ones
            aug_image, aug_mask = self.apply_augmentation(aug_image, aug_mask, "basic")

            # Random brightness adjustment (only for RGB channels)
            if np.random.random() > 0.5:
                brightness_factor = np.random.uniform(0.8, 1.2)
                aug_image[:, :, :3] = np.clip(
                    aug_image[:, :, :3] * brightness_factor, 0, 255
                ).astype(np.uint8)

        return aug_image, aug_mask

    def normalize_channels(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize each channel individually.

        Args:
            image: 4-channel input image

        Returns:
            Normalized image (float32, 0-1 range)
        """
        normalized = image.astype(np.float32)

        for channel in range(image.shape[2]):
            channel_data = normalized[:, :, channel]
            ch_min = channel_data.min()
            ch_max = channel_data.max()

            if ch_max > ch_min:
                normalized[:, :, channel] = (channel_data - ch_min) / (ch_max - ch_min)
            else:
                normalized[:, :, channel] = 0.0

        return normalized

    def generate_statistics_report(self) -> str:
        """
        Generate a comprehensive statistics report.

        Returns:
            Formatted statistics report string
        """
        if not self.stats:
            return "No statistics available. Run load_and_preprocess_dataset() first."

        report = f"""
        ===== Dataset Statistics Report =====
        
        Total Samples: {self.stats.total_samples}
        Valid Samples: {self.stats.valid_samples}
        Invalid Samples: {self.stats.invalid_samples}
        Success Rate: {self.stats.valid_samples/self.stats.total_samples*100:.2f}%
        
        Dead Tree Ratio Statistics:
        - Mean: {self.stats.mean_dead_tree_ratio:.4f}
        - Standard Deviation: {self.stats.std_dead_tree_ratio:.4f}
        - Min: {min(self.stats.dead_tree_ratios):.4f}
        - Max: {max(self.stats.dead_tree_ratios):.4f}
        
        Stratification Distribution:
        """

        for bin_name, count in self.stats.stratification_distribution.items():
            percentage = count / self.stats.valid_samples * 100
            report += f"- {bin_name}: {count} samples ({percentage:.1f}%)\n        "

        return report

    def visualize_statistics(self, save_path: Optional[str] = None):
        """
        Create visualization plots for dataset statistics.

        Args:
            save_path: Optional path to save the plots
        """
        if not self.stats:
            logger.error("No statistics available for visualization")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Dataset Statistics Visualization", fontsize=16)

        # 1. Dead tree ratio histogram
        axes[0, 0].hist(
            self.stats.dead_tree_ratios, bins=50, alpha=0.7, color="forestgreen"
        )
        axes[0, 0].set_xlabel("Dead Tree Ratio")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("Distribution of Dead Tree Ratios")
        axes[0, 0].axvline(
            self.stats.mean_dead_tree_ratio,
            color="red",
            linestyle="--",
            label=f"Mean: {self.stats.mean_dead_tree_ratio:.3f}",
        )
        axes[0, 0].legend()

        # 2. Stratification distribution pie chart
        labels = list(self.stats.stratification_distribution.keys())
        sizes = list(self.stats.stratification_distribution.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

        axes[0, 1].pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors)
        axes[0, 1].set_title("Stratification Distribution")

        # 3. Sample validity breakdown
        validity_data = ["Valid", "Invalid"]
        validity_counts = [self.stats.valid_samples, self.stats.invalid_samples]
        axes[1, 0].bar(
            validity_data, validity_counts, color=["green", "red"], alpha=0.7
        )
        axes[1, 0].set_ylabel("Number of Samples")
        axes[1, 0].set_title("Sample Validity Breakdown")

        # 4. Dead tree ratio box plot by stratification bin
        stratified_ratios = defaultdict(list)
        for ratio, label in zip(
            self.stats.dead_tree_ratios, self.stratification_labels
        ):
            stratified_ratios[label].append(ratio)

        box_data = [
            stratified_ratios[label] for label in sorted(stratified_ratios.keys())
        ]
        box_labels = sorted(stratified_ratios.keys())

        axes[1, 1].boxplot(box_data, labels=box_labels)
        axes[1, 1].set_xlabel("Stratification Bin")
        axes[1, 1].set_ylabel("Dead Tree Ratio")
        axes[1, 1].set_title("Dead Tree Ratio by Stratification Bin")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Statistics visualization saved to {save_path}")
        else:
            plt.show()

    def visualize_random_samples(
        self, num_samples: int = 8, save_path: Optional[str] = None
    ):
        """
        Visualize random samples from each stratification bin.

        Args:
            num_samples: Number of samples to visualize
            save_path: Optional path to save the visualization
        """
        if not self.sample_list:
            logger.error("No samples available for visualization")
            return

        # Group samples by stratification bin
        binned_samples = defaultdict(list)
        for i, label in enumerate(self.stratification_labels):
            binned_samples[label].append(i)

        # Select random samples from each bin
        selected_samples = []
        bin_names = sorted(binned_samples.keys())

        for bin_name in bin_names:
            bin_indices = binned_samples[bin_name]
            if bin_indices:
                # Select up to 2 samples per bin
                n_select = min(2, len(bin_indices))
                selected_indices = np.random.choice(
                    bin_indices, n_select, replace=False
                )
                for idx in selected_indices:
                    selected_samples.append((idx, bin_name))

        # Limit total samples
        selected_samples = selected_samples[:num_samples]

        # Create visualization
        n_cols = 4  # RGB, NIR, Mask, Overlay
        n_rows = len(selected_samples)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for row, (sample_idx, bin_name) in enumerate(selected_samples):
            sample_info = self.sample_list[sample_idx]

            # Load the actual image data
            result = self.load_and_validate_image(
                sample_info["rgb_path"],
                sample_info["nrg_path"],
                sample_info["mask_path"],
            )

            if result is not None:
                four_channel_img, mask = result

                # RGB image
                rgb_img = four_channel_img[:, :, :3]
                axes[row, 0].imshow(rgb_img)
                axes[row, 0].set_title(f"RGB - {bin_name}")
                axes[row, 0].axis("off")

                # NIR image
                nir_img = four_channel_img[:, :, 3]
                axes[row, 1].imshow(nir_img, cmap="gray")
                axes[row, 1].set_title(
                    f'NIR - Ratio: {sample_info["dead_tree_ratio"]:.3f}'
                )
                axes[row, 1].axis("off")

                # Mask
                axes[row, 2].imshow(mask, cmap="gray")
                axes[row, 2].set_title("Dead Tree Mask")
                axes[row, 2].axis("off")

                # Overlay
                overlay = rgb_img.copy()
                red_mask = np.zeros_like(overlay)
                red_mask[:, :, 0] = mask * 255
                overlay = cv2.addWeighted(
                    overlay.astype(np.uint8), 0.7, red_mask.astype(np.uint8), 0.3, 0
                )
                axes[row, 3].imshow(overlay)
                axes[row, 3].set_title("RGB + Mask Overlay")
                axes[row, 3].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Sample visualization saved to {save_path}")
        else:
            plt.show()

    def save_preprocessing_config(self, save_path: str):
        """
        Save preprocessing configuration and statistics.

        Args:
            save_path: Path to save the configuration
        """
        config = {
            "dataset_root": str(self.dataset_root),
            "random_seed": self.random_seed,
            "stratification_bins": self.stratification_bins,
            "test_split_ratio": self.test_split_ratio,
            "k_folds": self.k_folds,
            "statistics": self.stats.__dict__ if self.stats else None,
            "sample_count": len(self.sample_list),
        }

        with open(save_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Preprocessing configuration saved to {save_path}")

    def load_sample_by_index(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a specific sample by index.

        Args:
            index: Sample index

        Returns:
            Tuple of (4-channel image, mask)
        """
        if index >= len(self.sample_list):
            raise IndexError(
                f"Index {index} out of range for {len(self.sample_list)} samples"
            )

        sample_info = self.sample_list[index]
        result = self.load_and_validate_image(
            sample_info["rgb_path"], sample_info["nrg_path"], sample_info["mask_path"]
        )

        if result is None:
            raise ValueError(f"Failed to load sample at index {index}")

        return result


# Utility functions for easy access
def create_preprocessor(dataset_root: str, **kwargs) -> DeadTreeDataPreprocessor:
    """
    Factory function to create a preprocessor instance.

    Args:
        dataset_root: Path to USA_segmentation folder
        **kwargs: Additional parameters for DeadTreeDataPreprocessor

    Returns:
        Configured DeadTreeDataPreprocessor instance
    """
    return DeadTreeDataPreprocessor(dataset_root, **kwargs)


def quick_dataset_analysis(
    dataset_root: str,
    save_visualizations: bool = True,
    output_dir: str = "analysis_output",
) -> DeadTreeDataPreprocessor:
    """
    Perform quick dataset analysis with default settings.

    Args:
        dataset_root: Path to USA_segmentation folder
        save_visualizations: Whether to save visualization plots
        output_dir: Directory to save outputs

    Returns:
        Configured and analyzed DeadTreeDataPreprocessor instance
    """
    # Create output directory
    if save_visualizations:
        os.makedirs(output_dir, exist_ok=True)

    # Initialize preprocessor
    preprocessor = create_preprocessor(dataset_root)

    # Load and analyze dataset
    if preprocessor.load_and_preprocess_dataset():
        # Generate report
        report = preprocessor.generate_statistics_report()
        print(report)

        if save_visualizations:
            # Save statistics plots
            stats_path = os.path.join(output_dir, "dataset_statistics.png")
            preprocessor.visualize_statistics(stats_path)

            # Save sample visualizations
            samples_path = os.path.join(output_dir, "sample_examples.png")
            preprocessor.visualize_random_samples(save_path=samples_path)

            # Save configuration
            config_path = os.path.join(output_dir, "preprocessing_config.json")
            preprocessor.save_preprocessing_config(config_path)

            logger.info(f"Analysis outputs saved to {output_dir}")

    return preprocessor


# if __name__ == "__main__":
#     dataset_path = "USA_segmentation"

#     # Quick analysis
#     preprocessor = quick_dataset_analysis(dataset_path)

#     # Create stratified splits
#     if preprocessor.sample_list:
#         splits = preprocessor.create_stratified_splits()
#         print(f"\nCreated {len(splits['cv_splits'])} CV folds")
#         print(f"Test set size: {splits['test_size']}")
