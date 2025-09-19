import argparse
import os
import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import sys
import gc
from torchvision.utils import save_image


class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from discrete_mbrl.data_helpers import prepare_dataloaders
from discrete_mbrl.model_construction import construct_ae_model
from discrete_mbrl.visualization import states_to_imgs  # Import proper image conversion


def analyze_codebook_usage(encoder, data_loader, device, num_batches=10):
    """Analyze how many codebook entries are actually being used"""
    print("\n=== CODEBOOK USAGE ANALYSIS ===")

    all_codes = []
    spatial_code_maps = []  # Store spatial distribution of codes

    with torch.no_grad():
        for batch_idx, (batch, _, _, _, _) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break

            batch = batch.to(device)
            try:
                _, _, _, encodings = encoder(batch)

                # Convert to indices if needed
                if encodings.dtype == torch.float32:
                    encoding_indices = encodings.argmax(dim=1)
                else:
                    encoding_indices = encodings.long()

                # Store first few spatial maps for analysis
                if batch_idx < 3:
                    spatial_code_maps.extend(encoding_indices[:2].cpu().numpy())  # Take 2 samples per batch

                all_codes.extend(encoding_indices.flatten().cpu().numpy())

            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

    if all_codes:
        unique_codes = np.unique(all_codes)
        print(f"Total codes used: {len(unique_codes)} out of {encoder.quantizer._num_embeddings}")
        print(f"Utilization: {len(unique_codes) / encoder.quantizer._num_embeddings * 100:.1f}%")
        print(f"Code range: {unique_codes.min()} to {unique_codes.max()}")

        # Show frequency distribution
        codes, counts = np.unique(all_codes, return_counts=True)
        total_positions = len(all_codes)
        top_codes = sorted(zip(codes, counts), key=lambda x: x[1], reverse=True)[:10]
        print(f"Most frequent codes (code: count, percentage):")
        for code, count in top_codes:
            percentage = (count / total_positions) * 100
            print(f"  Code {code}: {count} positions ({percentage:.1f}%)")

        # Analyze spatial distribution of codes
        if spatial_code_maps:
            print(f"\n=== SPATIAL CODE DISTRIBUTION ===")
            for i, code_map in enumerate(spatial_code_maps[:3]):  # Show first 3 samples
                print(f"Sample {i + 1} code map:")
                print(f"  Shape: {code_map.shape}")
                sample_codes, sample_counts = np.unique(code_map, return_counts=True)
                for code, count in zip(sample_codes, sample_counts):
                    percentage = (count / code_map.size) * 100
                    print(f"    Code {code}: {count}/{code_map.size} positions ({percentage:.1f}%)")

        return unique_codes
    else:
        print("No codes found!")
        return np.array([])


def main(args):
    """Main function to run the semantic visualization experiment."""
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    train_loader, _, _ = prepare_dataloaders(
        env_name=args.env_name, batch_size=args.batch_size, n=args.max_transitions,
        preprocess=True, preload_all=False, n_preload=0
    )
    print("Data loaded successfully.")

    # Get the reverse transformation function
    rev_transform = train_loader.dataset.flat_rev_obs_transform
    print(f"Reverse transform function: {rev_transform}")

    print("Determining input shape from data...")
    first_obs_batch = next(iter(train_loader))[0]
    input_dim = first_obs_batch.shape[1:]
    print(f"Detected input shape: {input_dim}")
    del first_obs_batch
    gc.collect()

    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_args = Dotdict({
        "env_name": args.env_name, "device": device, "ae_model_type": "vqvae",
        "embedding_dim": args.embedding_dim, "filter_size": args.filter_size,
        "ae_model_version": args.ae_model_version, "codebook_size": args.codebook_size,
        "latent_dim": args.latent_dim, "ae_grad_clip": 1.0, "learning_rate": 0.0003, "wandb": False
    })

    try:
        encoder, _ = construct_ae_model(input_dim=input_dim, args=model_args, load=False)
        encoder.load_state_dict(torch.load(args.model_path, map_location=device))
        encoder.to(device)
        encoder.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # First, analyze codebook usage
    used_codes = analyze_codebook_usage(encoder, train_loader, device)

    if len(used_codes) < 10:
        print(
            f"\n⚠️  WARNING: Your model is only using {len(used_codes)} codes out of {encoder.quantizer._num_embeddings}!")
        print("This indicates severe codebook collapse. Consider:")
        print("1. Training for more epochs")
        print("2. Adjusting the commitment cost")
        print("3. Using codebook initialization techniques")
        print("4. Checking if the model architecture is appropriate")

    print("\nCreating semantic map...")
    semantic_map = create_semantic_map_optimized(encoder, train_loader, device, max_samples_per_code=50)
    print("Semantic map created.")

    if len(semantic_map) > 0:
        print("Generating visualizations...")
        visualize_prototypes(semantic_map, args.output_dir, args.env_name, rev_transform)
        visualize_samples(semantic_map, args.output_dir, args.env_name, rev_transform)
        visualize_originals(semantic_map, args.output_dir, args.env_name, rev_transform)
        print(f"Visualizations saved to {args.output_dir}")
    else:
        print("⚠️  No semantic map generated - no codes found with samples!")


def create_semantic_map_optimized(encoder, data_loader, device, max_samples_per_code=50):
    """
    Optimized function to create a semantic map using efficient tensor operations.
    """
    num_codes = encoder.quantizer._num_embeddings
    semantic_map = {i: {'samples': [], 'originals': [], 'spatial_positions': []} for i in range(num_codes)}

    print(f"Processing data with max {max_samples_per_code} samples per code...")
    print(f"Codebook size: {num_codes}")

    # Track which codes are actually used
    codes_found = set()

    with torch.no_grad():
        for batch_idx, (batch, _, _, _, _) in enumerate(data_loader):
            batch = batch.to(device)
            try:
                recon_batch, _, _, encodings = encoder(batch)

                # Debug: Check the shape and type of encodings
                if batch_idx == 0:
                    print(f"Encodings shape: {encodings.shape}")
                    print(f"Encodings dtype: {encodings.dtype}")
                    print(f"Encodings min/max: {encodings.min().item():.3f}/{encodings.max().item():.3f}")

                # Convert one-hot encodings to indices if needed
                if encodings.dtype == torch.float32:
                    # For soft/one-hot encodings: [batch, n_embeddings, spatial_positions]
                    encoding_indices = encodings.argmax(dim=1)  # [batch, spatial_positions]
                    print(f"Converted one-hot to indices, shape: {encoding_indices.shape}")
                else:
                    # For discrete encodings: [batch, spatial_positions]
                    encoding_indices = encodings.long()

                # Reshape to [batch, height, width] if needed
                if len(encoding_indices.shape) == 2:
                    spatial_dim = int(encoding_indices.shape[1] ** 0.5)
                    if spatial_dim * spatial_dim == encoding_indices.shape[1]:
                        encoding_indices = encoding_indices.reshape(encoding_indices.shape[0], spatial_dim, spatial_dim)

                # Find unique codes in this batch
                unique_codes_in_batch = torch.unique(encoding_indices).cpu().numpy()
                codes_found.update(unique_codes_in_batch)

                if batch_idx == 0:
                    print(f"Unique codes in first batch: {len(unique_codes_in_batch)}")
                    print(f"Code range: {unique_codes_in_batch.min()} to {unique_codes_in_batch.max()}")

                # For each image in the batch, find the most common code
                for img_idx in range(batch.shape[0]):
                    img_codes = encoding_indices[img_idx].flatten()

                    # Find the most frequent code for this image
                    unique_codes, counts = torch.unique(img_codes, return_counts=True)
                    most_frequent_idx = counts.argmax()
                    dominant_code = unique_codes[most_frequent_idx].item()

                    # Add this image to the dominant code's collection
                    if len(semantic_map[dominant_code]['samples']) < max_samples_per_code:
                        semantic_map[dominant_code]['samples'].append(recon_batch[img_idx].cpu())
                        semantic_map[dominant_code]['originals'].append(batch[img_idx].cpu())

                        # Store spatial information
                        code_positions = (img_codes == dominant_code).nonzero().cpu().numpy().flatten()
                        semantic_map[dominant_code]['spatial_positions'].append(code_positions)

                if batch_idx % 5 == 0:
                    active_codes = len([k for k, v in semantic_map.items() if v['samples']])
                    total_samples = sum(len(v['samples']) for v in semantic_map.values())
                    print(
                        f"Batch {batch_idx}: Found {active_codes} unique codes so far, collected {total_samples} total samples.")
                    print(f"Total unique codes discovered: {len(codes_found)}")

                # Stop if we've processed enough data
                if batch_idx >= 20:  # Process more batches to find more codes
                    break

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Remove empty entries and report statistics
    final_map = {k: v for k, v in semantic_map.items() if v['samples']}

    print(f"\n=== CODEBOOK USAGE ANALYSIS ===")
    print(f"Total codes found: {len(codes_found)}")
    print(f"Codes with samples: {len(final_map)}")
    print(f"Codebook utilization: {len(codes_found) / num_codes * 100:.1f}%")

    if len(codes_found) < 10:
        print(f"⚠️  WARNING: Very low codebook utilization! Only {len(codes_found)} out of {num_codes} codes are used.")
        print("This suggests the model may be undertrained or suffering from codebook collapse.")

    # Show sample counts per code
    sample_counts = [(k, len(v['samples'])) for k, v in final_map.items()]
    sample_counts.sort(key=lambda x: x[1], reverse=True)
    print(f"Top codes by sample count: {sample_counts[:10]}")

    return final_map


def tensor_to_proper_image(tensor_data, env_name, rev_transform):
    """Convert tensor data to properly formatted images using the same pipeline as evaluate_model.py"""
    if isinstance(tensor_data, list):
        tensor_data = torch.stack(tensor_data)

    # Use the same image conversion pipeline as evaluate_model.py
    images = states_to_imgs(tensor_data, env_name, transform=rev_transform)

    # Convert to numpy and ensure proper format
    if isinstance(images, torch.Tensor):
        images = images.numpy()

    # Handle different image formats
    if images.ndim == 4:  # [batch, channels, height, width]
        images = np.transpose(images, (0, 2, 3, 1))  # [batch, height, width, channels]
    elif images.ndim == 3:  # [channels, height, width]
        images = np.transpose(images, (1, 2, 0))  # [height, width, channels]

    # Handle grayscale images
    if images.shape[-1] == 1:
        images = images.squeeze(-1)

    # Ensure values are in [0, 1] range
    images = np.clip(images, 0, 1)

    return images


def visualize_prototypes(semantic_map, output_dir, env_name, rev_transform):
    """Visualizes the 'prototype' (average) observation for each discrete code."""
    print("  - Visualizing prototypes...")
    proto_dir = os.path.join(output_dir, 'prototypes')
    os.makedirs(proto_dir, exist_ok=True)

    for code, data in semantic_map.items():
        obs_list = data['samples']
        if not obs_list:
            continue
        try:
            # Average the observations in tensor space
            avg_obs_tensor = torch.stack(obs_list).mean(dim=0, keepdim=True)

            # Convert to proper image format using the same pipeline as evaluate_model.py
            avg_obs = tensor_to_proper_image(avg_obs_tensor, env_name, rev_transform)

            # If batch dimension exists, take the first (and only) image
            if avg_obs.ndim == 4:
                avg_obs = avg_obs[0]
            elif avg_obs.ndim == 3 and avg_obs.shape[0] == 1:
                avg_obs = avg_obs[0]

            plt.imsave(os.path.join(proto_dir, f'code_{code}.png'), avg_obs)

        except Exception as e:
            print(f"Error visualizing prototype for code {code}: {e}")


def visualize_samples(semantic_map, output_dir, env_name, rev_transform, num_samples=16):
    """Visualizes a grid of random sample observations for each discrete code."""
    print("  - Visualizing samples...")
    samples_dir = os.path.join(output_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)

    for code, data in semantic_map.items():
        obs_list = data['samples']
        if not obs_list:
            continue
        try:
            count = min(len(obs_list), num_samples)
            obs_tensor = torch.stack(obs_list[:count])

            # Convert to proper image format
            images = tensor_to_proper_image(obs_tensor, env_name, rev_transform)

            # Convert back to tensor for save_image
            if images.ndim == 3:  # Single image [H, W, C]
                images = torch.from_numpy(images).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
            elif images.ndim == 4:  # Multiple images [B, H, W, C]
                images = torch.from_numpy(images).permute(0, 3, 1, 2)  # [B, C, H, W]

            # Handle grayscale images
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)  # Convert to RGB for save_image

            save_image(images, os.path.join(samples_dir, f'code_{code}.png'),
                       normalize=False, nrow=int(np.sqrt(count)))

        except Exception as e:
            print(f"Error visualizing samples for code {code}: {e}")


def visualize_originals(semantic_map, output_dir, env_name, rev_transform, num_samples=16):
    """Visualizes a grid of original input images for each discrete code."""
    print("  - Visualizing original inputs...")
    originals_dir = os.path.join(output_dir, 'originals')
    os.makedirs(originals_dir, exist_ok=True)

    for code, data in semantic_map.items():
        obs_list = data['originals']
        if not obs_list:
            continue
        try:
            count = min(len(obs_list), num_samples)
            obs_tensor = torch.stack(obs_list[:count])

            # Convert to proper image format
            images = tensor_to_proper_image(obs_tensor, env_name, rev_transform)

            # Convert back to tensor for save_image
            if images.ndim == 3:  # Single image [H, W, C]
                images = torch.from_numpy(images).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
            elif images.ndim == 4:  # Multiple images [B, H, W, C]
                images = torch.from_numpy(images).permute(0, 3, 1, 2)  # [B, C, H, W]

            # Handle grayscale images
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)  # Convert to RGB for save_image

            save_image(images, os.path.join(originals_dir, f'code_{code}.png'),
                       normalize=False, nrow=int(np.sqrt(count)))

        except Exception as e:
            print(f"Error visualizing originals for code {code}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained VQ-VAE model.')
    parser.add_argument('--env_name', type=str, default='minigrid-crossing-stochastic', help='Name of the environment.')
    parser.add_argument('--output_dir', type=str, default='semantic_visuals', help='Directory to save visualizations.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for processing data.')
    parser.add_argument('--max_transitions', type=int, default=10000, help='Max transitions to load.')
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--ae_model_version', type=str, default='2')
    parser.add_argument('--filter_size', type=int, default=3)
    parser.add_argument('--codebook_size', type=int, default=1024)
    parser.add_argument('--latent_dim', type=int, default=81)
    args = parser.parse_args()
    main(args)