from shared.models import VQVAEModel
from discrete_mbrl.model_construction import construct_ae_model, load_model
from discrete_mbrl.training_helpers import get_args
from discrete_mbrl.env_helpers import make_env

def get_observation_shape(env_name):
    """Get observation shape from environment"""
    env = make_env(env_name)
    obs = env.reset()
    if isinstance(obs, tuple):
        obs, _ = obs
    shape = obs.shape
    env.close()
    return shape

# Parse arguments
args = get_args(apply_optimizations=False)  # Disable auto-optimizations

# Get observation shape
obs_shape = get_observation_shape(args.env_name)

# Construct the encoder model
encoder_model, trainer = construct_ae_model(obs_shape, args)

# Load the model weights
try:
    loaded_model = load_model(encoder_model, args, model_hash=args.ae_model_hash)
    if loaded_model is None:
        print("❌ Failed to load model weights")
    else:
        print("✅ Model weights loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Check if the loaded model is a VQVAEModel
# Method 1: Check by instance type
if isinstance(encoder_model, VQVAEModel):
    print("✅ The loaded model is a VQ-VAE model (by instance check).")
else:
    print("❌ The loaded model is NOT a VQ-VAE model (by instance check).")

# Method 2: Check by model type argument (more reliable in this codebase)
if args.ae_model_type == 'vqvae':
    print("✅ The model type is VQ-VAE (by args check).")
else:
    print(f"❌ The model type is {args.ae_model_type}, not VQ-VAE (by args check).")

# Method 3: Check by model attributes (most reliable for actual capabilities)
has_quantizer = hasattr(encoder_model, 'quantizer')
has_embeddings = hasattr(encoder_model, 'n_embeddings')
has_codebook = hasattr(encoder_model, 'codebook_size')

if has_quantizer and has_embeddings:
    print("✅ The model has VQ-VAE characteristics (quantizer + embeddings).")
    if has_codebook:
        print(f"   Codebook size: {encoder_model.codebook_size}")
    if hasattr(encoder_model, 'n_latent_embeds'):
        print(f"   Number of latent embeds: {encoder_model.n_latent_embeds}")
else:
    print("❌ The model does NOT have VQ-VAE characteristics.")

# Additional VQ-VAE specific information
if hasattr(encoder_model, 'quantizer'):
    print(f"📊 VQ-VAE Details:")

    print(f"   Model class: {type(encoder_model).__name__}")
    print(f"   Quantizer class: {type(encoder_model.quantizer).__name__}")
    if hasattr(encoder_model, 'n_embeddings'):
        print(f"   Vocabulary size: {encoder_model.n_embeddings}")
    if hasattr(encoder_model, 'embedding_dim'):
        print(f"   Embedding dimension: {encoder_model.embedding_dim}")