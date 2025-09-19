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


def main(args):
    """Main function to run the semantic visualization experiment."""
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    train_loader, _, _ = prepare_dataloaders(
        env_name=args.env_name, batch_size=args.batch_size, n=args.max_transitions,
        preprocess=True, preload_all=False, n_preload=0
    )
    print("Data loaded successfully.")

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

    print("Creating semantic map...")
    semantic_map = create_semantic_map_optimized(encoder, train_loader, device, max_samples_per_code=50)
    print("Semantic map created.")

    print("Generating visualizations...")
    visualize_prototypes(semantic_map, args.output_dir)
    visualize_samples(semantic_map, args.output_dir)
    visualize_originals(semantic_map, args.output_dir)
    print(f"Visualizations saved to {args.output_dir}")


def create_semantic_map_optimized(encoder, data_loader, device, max_samples_per_code=50):
    """
    Optimized function to create a semantic map using efficient tensor operations.
    """
    num_codes = encoder.quantizer._num_embeddings
    semantic_map = {i: {'samples': [], 'originals': []} for i in range(num_codes)}
    needed_codes = set(range(num_codes))

    print(f"Processing data with max {max_samples_per_code} samples per code...")

    with torch.no_grad():
        for batch_idx, (batch, _, _, _, _) in enumerate(data_loader):
            if not needed_codes:
                print("All necessary samples collected. Stopping early.")
                break

            batch = batch.to(device)
            try:
                recon_batch, _, _, encodings = encoder(batch)
                encodings_flat = encodings.reshape(-1)

                unique_codes_in_batch = torch.unique(encodings_flat).cpu().numpy()

                codes_to_process = needed_codes.intersection(unique_codes_in_batch)

                for code in codes_to_process:
                    # Find where this code appears in the flattened encodings
                    indices = (encodings_flat == code).nonzero(as_tuple=True)[0]

                    # Determine how many samples we still need for this code
                    needed_now = max_samples_per_code - len(semantic_map[code]['samples'])

                    # Take the minimum of what's needed and what's available
                    indices_to_take = indices[:needed_now]

                    # Map flat indices back to batch indices
                    spatial_size = encodings.shape[1] * encodings.shape[2]  # 81
                    batch_indices = (indices_to_take // spatial_size).long()

                    if batch_indices.numel() > 0:
                        semantic_map[code]['samples'].extend(list(recon_batch[batch_indices].cpu()))
                        semantic_map[code]['originals'].extend(list(batch[batch_indices].cpu()))

                    # If we have enough for this code, we don't need to look for it anymore
                    if len(semantic_map[code]['samples']) >= max_samples_per_code:
                        needed_codes.discard(code)

                if batch_idx % 5 == 0:
                    active_codes = len([k for k, v in semantic_map.items() if v['samples']])
                    total_samples = sum(len(v['samples']) for v in semantic_map.values())
                    print(
                        f"Batch {batch_idx}: Found {active_codes} unique codes so far, collected {total_samples} total samples.")

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue

    final_map = {k: v for k, v in semantic_map.items() if v['samples']}
    print(f"Final semantic map: {len(final_map)} unique codes")
    return final_map


def visualize_prototypes(semantic_map, output_dir):
    """Visualizes the 'prototype' (average) observation for each discrete code."""
    print("  - Visualizing prototypes...")
    proto_dir = os.path.join(output_dir, 'prototypes')
    os.makedirs(proto_dir, exist_ok=True)
    for code, data in semantic_map.items():
        obs_list = data['samples']
        if not obs_list: continue
        try:
            avg_obs = torch.stack(obs_list).mean(dim=0).numpy()
            avg_obs = np.transpose(avg_obs, (1, 2, 0))
            avg_obs = np.clip(avg_obs, 0, 1)
            plt.imsave(os.path.join(proto_dir, f'code_{code}.png'), avg_obs)
        except Exception as e:
            print(f"Error visualizing prototype for code {code}: {e}")


def visualize_samples(semantic_map, output_dir, num_samples=16):
    """Visualizes a grid of random sample observations for each discrete code."""
    print("  - Visualizing samples...")
    samples_dir = os.path.join(output_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    for code, data in semantic_map.items():
        obs_list = data['samples']
        if not obs_list: continue
        try:
            count = min(len(obs_list), num_samples)
            grid = torch.stack(obs_list[:count])
            save_image(grid, os.path.join(samples_dir, f'code_{code}.png'), normalize=True, nrow=int(np.sqrt(count)))
        except Exception as e:
            print(f"Error visualizing samples for code {code}: {e}")


def visualize_originals(semantic_map, output_dir, num_samples=16):
    """Visualizes a grid of original input images for each discrete code."""
    print("  - Visualizing original inputs...")
    originals_dir = os.path.join(output_dir, 'originals')
    os.makedirs(originals_dir, exist_ok=True)
    for code, data in semantic_map.items():
        obs_list = data['originals']
        if not obs_list: continue
        try:
            count = min(len(obs_list), num_samples)
            grid = torch.stack(obs_list[:count])
            save_image(grid, os.path.join(originals_dir, f'code_{code}.png'), normalize=True, nrow=int(np.sqrt(count)))
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