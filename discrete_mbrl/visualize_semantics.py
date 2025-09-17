import argparse
import os
import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import sys
import gc


class Dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# Add the project's root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from discrete_mbrl.data_helpers import prepare_dataloaders
from discrete_mbrl.model_construction import construct_ae_model


def main(args):
    """
    Main function to run the semantic visualization experiment.
    """
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    # REDUCE batch size and max transitions to prevent memory issues
    train_loader, _, _ = prepare_dataloaders(
        env_name=args.env_name,
        batch_size=min(args.batch_size, 64),  # Cap batch size
        n=min(args.max_transitions, 10000),  # Cap transitions
        preprocess=True,
        preload_all=False,
        n_preload=0  # Disable multiprocessing to avoid issues
    )
    print("Data loaded successfully.")

    # Get input_dim from the first batch
    print("Determining input shape from data...")
    first_batch = next(iter(train_loader))
    first_obs_batch = first_batch[0]
    input_dim = first_obs_batch.shape[1:]
    print(f"Detected input shape: {input_dim}")

    # Clear the batch from memory
    del first_batch, first_obs_batch
    gc.collect()

    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_args = Dotdict({
        "env_name": args.env_name,
        "device": device,
        "ae_model_type": "vqvae",
        "embedding_dim": args.embedding_dim,
        "filter_size": args.filter_size,
        "ae_model_version": args.ae_model_version,
        "codebook_size": args.codebook_size,
        "latent_dim": args.latent_dim,
        "ae_grad_clip": 1.0,
        "learning_rate": 0.0003,
        "wandb": False
    })

    try:
        encoder, _ = construct_ae_model(
            input_dim=input_dim,
            args=model_args,
            load=False
        )
        encoder.load_state_dict(torch.load(args.model_path, map_location=device))
        encoder.to(device)
        encoder.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Create the semantic map with memory limits
    print("Creating semantic map...")
    semantic_map = create_semantic_map_safe(encoder, train_loader, device, max_samples_per_code=50)
    print("Semantic map created.")

    # Generate and save visualizations
    print("Generating visualizations...")
    visualize_prototypes(semantic_map, args.output_dir)
    visualize_samples(semantic_map, args.output_dir, num_samples=min(16, 25))
    print(f"Visualizations saved to {args.output_dir}")


def create_semantic_map_safe(encoder, data_loader, device, max_samples_per_code=50):
    """
    Creates a map from discrete code indices to observations with MEMORY LIMITS.
    """
    semantic_map = defaultdict(list)
    code_counts = defaultdict(int)

    print(f"Processing data with max {max_samples_per_code} samples per code...")

    with torch.no_grad():
        for batch_idx, (batch, _, _, _, _) in enumerate(data_loader):
            # Process smaller chunks to avoid memory issues
            batch = batch.to(device)

            try:
                # Get encodings - this might return different things depending on model
                result = encoder(batch)
                if isinstance(result, tuple):
                    # Handle different return formats
                    if len(result) == 4:  # (recon, codes, losses, quantized)
                        _, _, _, encodings = result
                    else:
                        encodings = result[-1]  # Take last element
                else:
                    encodings = result

                if encodings is None:
                    print(f"Warning: No encodings from batch {batch_idx}")
                    continue

                # Flatten encodings to get discrete codes
                if len(encodings.shape) > 2:
                    encodings = encodings.reshape(batch.size(0), -1)

                # Sample observations for each code, but limit memory usage
                for obs_idx in range(min(encodings.size(0), 32)):  # Limit observations per batch
                    obs = batch[obs_idx].cpu().numpy()

                    for latent_idx in range(min(encodings.size(1), 64)):  # Limit latent dimensions
                        code = int(encodings[obs_idx, latent_idx].item())

                        # Only store if we haven't reached the limit for this code
                        if code_counts[code] < max_samples_per_code:
                            semantic_map[code].append(obs.copy())
                            code_counts[code] += 1

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue

            # Clear GPU memory periodically
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()

                # Print progress
                total_samples = sum(len(samples) for samples in semantic_map.values())
                print(f"Batch {batch_idx}: {len(semantic_map)} unique codes, {total_samples} total samples")

                # Stop if we have enough data
                if len(semantic_map) > 100 or total_samples > 5000:
                    print("Stopping early to prevent memory issues")
                    break

    print(f"Final semantic map: {len(semantic_map)} unique codes")
    return semantic_map


def visualize_prototypes(semantic_map, output_dir):
    """
    Visualizes the 'prototype' (average) observation for each discrete code.
    """
    print("  - Visualizing prototypes...")
    proto_dir = os.path.join(output_dir, 'prototypes')
    os.makedirs(proto_dir, exist_ok=True)

    for code, obs_list in list(semantic_map.items())[:50]:  # Limit to first 50 codes
        if len(obs_list) == 0:
            continue

        try:
            avg_obs = np.mean(obs_list, axis=0)

            # Handle different image formats
            if len(avg_obs.shape) == 3:
                if avg_obs.shape[0] in [1, 3]:  # Channels first
                    avg_obs = np.transpose(avg_obs, (1, 2, 0))
                if avg_obs.shape[2] == 1:  # Grayscale
                    avg_obs = avg_obs.squeeze(2)

            avg_obs = np.clip(avg_obs, 0, 1)

            plt.figure(figsize=(6, 6))
            plt.imshow(avg_obs, cmap='gray' if len(avg_obs.shape) == 2 else None)
            plt.title(f'Prototype for Code {code} ({len(obs_list)} samples)')
            plt.axis('off')
            plt.savefig(os.path.join(proto_dir, f'code_{code}.png'), bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Error visualizing prototype for code {code}: {e}")
            continue


def visualize_samples(semantic_map, output_dir, num_samples=16):
    """
    Visualizes a grid of random sample observations for each discrete code.
    """
    print("  - Visualizing samples...")
    samples_dir = os.path.join(output_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)

    for code, obs_list in list(semantic_map.items())[:20]:  # Limit to first 20 codes
        if len(obs_list) < 4:  # Need at least 4 samples
            continue

        try:
            sample_count = min(len(obs_list), num_samples)
            sample_indices = np.random.choice(len(obs_list), sample_count, replace=False)
            samples = [obs_list[i] for i in sample_indices]

            grid_size = int(np.ceil(np.sqrt(sample_count)))
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
            fig.suptitle(f'Samples for Code {code}', fontsize=16)

            if grid_size == 1:
                axes = [axes]
            else:
                axes = axes.flat

            for i, ax in enumerate(axes):
                if i < len(samples):
                    img = samples[i]
                    if len(img.shape) == 3 and img.shape[0] in [1, 3]:
                        img = np.transpose(img, (1, 2, 0))
                    if len(img.shape) == 3 and img.shape[2] == 1:
                        img = img.squeeze(2)
                    img = np.clip(img, 0, 1)
                    ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
                ax.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(samples_dir, f'code_{code}.png'), bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Error visualizing samples for code {code}: {e}")
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained VQ-VAE model.')
    parser.add_argument('--env_name', type=str, default='minigrid-crossing-stochastic', help='Name of the environment.')
    parser.add_argument('--output_dir', type=str, default='semantic_visuals', help='Directory to save visualizations.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing data.')  # Reduced default
    parser.add_argument('--max_transitions', type=int, default=5000,  # Reduced default
                        help='Max transitions to load from dataset for visualization.')

    # Model Hyperparameters
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--ae_model_version', type=str, default='2')
    parser.add_argument('--filter_size', type=int, default=3)
    parser.add_argument('--codebook_size', type=int, default=512)
    parser.add_argument('--latent_dim', type=int, default=None)

    args = parser.parse_args()
    main(args)