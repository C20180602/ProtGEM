import os
import json
import torch
from pathlib import Path
from model.enhanced_predictor import EnhancedProteinGoModel, EnhancedProteinGoModelv2
from model.enhanced_seq_model import ImprovedSeqEncoder
from utils import load_config_clone

def load_enhanced_model(model_type='enhanced_v1', workspace_dir=None, config_dir=None, device='cuda:0'):
    """
    Load enhanced model with improved configurations

    Args:
        model_type: 'enhanced_v1' or 'enhanced_v2'
        workspace_dir: Path to workspace directory
        config_dir: Path to config directory (optional, will use default if None)
        device: Device to load model on

    Returns:
        model: Enhanced model instance
        configs: Configuration dictionaries
    """

    # Default config paths
    if config_dir is None:
        config_dir = Path("config")

    enhanced_pm_conf = config_dir / "EnhancedProteinEncoder.conf"
    enhanced_gm_conf = config_dir / "EnhancedGoEncoder.conf"

    # Load configurations
    with open(enhanced_pm_conf, 'r') as f:
        pm_param = json.load(f)

    with open(enhanced_gm_conf, 'r') as f:
        gm_param = json.load(f)

    # Extract enhanced config
    enhanced_config = pm_param.pop('enhanced_config', None)

    # Initialize model
    if model_type == 'enhanced_v1':
        model = EnhancedProteinGoModel(pm_param, gm_param, enhanced_config)
    elif model_type == 'enhanced_v2':
        model = EnhancedProteinGoModelv2(pm_param, gm_param, enhanced_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)

    # Load pre-trained weights if available
    if workspace_dir:
        model_param_path = Path(workspace_dir) / "model_param"
        if model_param_path.exists():
            # Look for saved model files
            for model_file in model_param_path.glob("*.pth"):
                print(f"Found pre-trained model: {model_file}")
                try:
                    model.load_state_dict(torch.load(model_file, map_location=device))
                    print(f"Loaded pre-trained weights from {model_file}")
                    break
                except Exception as e:
                    print(f"Could not load {model_file}: {e}")

    return model, {'pm_param': pm_param, 'gm_param': gm_param, 'enhanced_config': enhanced_config}

def create_enhanced_training_config():
    """Create enhanced training configuration with improved hyperparameters"""
    config = {
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "batch_size": 32,
        "epochs": 100,
        "warmup_epochs": 10,
        "scheduler": "cosine_with_warmup",
        "gradient_clip": 1.0,

        # Enhanced loss configuration
        "loss_config": {
            "focal_loss": True,
            "focal_alpha": 0.25,
            "focal_gamma": 2.0,
            "label_smoothing": 0.1,
            "class_weights": True
        },

        # Optimization improvements
        "optimizer": "adamw",
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,

        # Training enhancements
        "mixed_precision": True,
        "gradient_accumulation_steps": 2,
        "early_stopping": {
            "patience": 15,
            "min_delta": 1e-4,
            "monitor": "val_f1"
        },

        # Regularization
        "dropout_schedule": True,
        "stochastic_depth": 0.1,
        "layer_decay": 0.8
    }
    return config

def benchmark_models(original_model, enhanced_model, test_loader, device='cuda:0'):
    """
    Benchmark original vs enhanced models

    Args:
        original_model: Original model instance
        enhanced_model: Enhanced model instance
        test_loader: Test data loader
        device: Device for computation

    Returns:
        dict: Performance comparison results
    """

    results = {
        'original': {'inference_time': [], 'memory_usage': []},
        'enhanced': {'inference_time': [], 'memory_usage': []}
    }

    # Benchmark function
    def benchmark_model(model, name):
        model.eval()
        torch.cuda.empty_cache()

        with torch.no_grad():
            for batch_idx, (seq_embs, batched_graph, domain_info, go_idx, labels) in enumerate(test_loader):
                if batch_idx >= 10:  # Limit to 10 batches for benchmarking
                    break

                # Move data to device
                seq_embs = seq_embs.to(device)
                batched_graph = batched_graph.to(device)
                domain_info = [d.to(device) for d in domain_info]
                go_idx = go_idx.to(device)

                # Measure inference time
                torch.cuda.synchronize()
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)

                start_time.record()
                outputs = model(seq_embs, batched_graph, domain_info, go_idx)
                end_time.record()

                torch.cuda.synchronize()
                inference_time = start_time.elapsed_time(end_time)

                # Measure memory usage
                memory_usage = torch.cuda.max_memory_allocated() / 1024**3  # GB

                results[name]['inference_time'].append(inference_time)
                results[name]['memory_usage'].append(memory_usage)

                torch.cuda.reset_peak_memory_stats()

    # Run benchmarks
    benchmark_model(original_model, 'original')
    benchmark_model(enhanced_model, 'enhanced')

    # Calculate statistics
    for model_name in results:
        results[model_name]['avg_inference_time'] = sum(results[model_name]['inference_time']) / len(results[model_name]['inference_time'])
        results[model_name]['avg_memory_usage'] = sum(results[model_name]['memory_usage']) / len(results[model_name]['memory_usage'])

    return results

# Example usage and testing
if __name__ == "__main__":
    # Set working directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.basename(os.getcwd()) == "ProtGEM":
        os.chdir("..")

    print("Loading enhanced model...")
    model, configs = load_enhanced_model('enhanced_v1')

    print("Model architecture:")
    print(f"- Cross attention: {configs['enhanced_config']['use_cross_attention']}")
    print(f"- Adaptive GO: {configs['enhanced_config']['use_adaptive_go']}")
    print(f"- Hierarchical pooling: {configs['enhanced_config']['use_hierarchical_pooling']}")
    print(f"- Interaction method: {configs['enhanced_config']['interaction_method']}")

    print("\nEnhanced training config:")
    training_config = create_enhanced_training_config()
    for key, value in training_config.items():
        print(f"- {key}: {value}")