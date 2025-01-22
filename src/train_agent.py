import pandas as pd
import numpy as np
import torch
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
import wandb
from wandb.integration.sb3 import WandbCallback
from typing import Dict, Any, Optional, List, Tuple
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import logging
from datetime import datetime
import multiprocessing
import os
from env import DEXTradingEnv

# Set environment variables for PyTorch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# Configure GPU settings
torch.backends.cudnn.benchmark = True



# GPU Initialization and Monitoring
def setup_gpu():
    """Initialize GPU settings."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return device
    else:
        logger.warning("CUDA not available, using CPU")
        return torch.device("cpu")

def print_gpu_utilization():
    """Print current GPU utilization stats."""
    if torch.cuda.is_available():
        # Get utilization for the first GPU
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        try:
            import nvidia_smi
            nvidia_smi.nvmlInit()
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            info = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            print(f"GPU Utilization: {info.gpu}%")
            print(f"Memory Utilization: {info.memory}%")
        except:
            pass

# Initialize GPU
device = setup_gpu()

class TradingCallback(BaseCallback):
    """Enhanced callback for logging trading metrics with GPU monitoring."""
    
    def __init__(self, eval_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.best_reward = -np.inf
        self.best_model_path = None
        self.episode_rewards: List[float] = []
        self.portfolio_values: List[float] = []
        
    def _on_step(self) -> bool:
        try:
            if len(self.model.ep_info_buffer) > 0:
                episode_info = self.model.ep_info_buffer[-1]
                episode_reward = episode_info["r"]
                episode_length = episode_info["l"]
                
                env_info = self.training_env.get_attr("last_info")[0]
                if env_info is None:
                    return True
                
                self.episode_rewards.append(episode_reward)
                self.portfolio_values.append(env_info["portfolio_value"])
                
                window_size = min(len(self.episode_rewards), 100)
                avg_reward = np.mean(self.episode_rewards[-window_size:])
                avg_portfolio = np.mean(self.portfolio_values[-window_size:])
                
                metrics = {
                    "train/episode_reward": episode_reward,
                    "train/episode_length": episode_length,
                    "train/portfolio_value": env_info["portfolio_value"],
                    "train/avg_reward_100": avg_reward,
                    "train/avg_portfolio_100": avg_portfolio,
                }
                
                # Add GPU metrics if available
                if torch.cuda.is_available():
                    metrics.update({
                        "system/gpu_utilization": torch.cuda.utilization(),
                        "system/gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3,
                        "system/gpu_memory_cached": torch.cuda.memory_reserved() / 1024**3
                    })
                
                wandb.log(metrics)
                
                if episode_reward > self.best_reward:
                    self.best_reward = episode_reward
                    if self.best_model_path:
                        self.model.save(self.best_model_path)
                
            return True
            
        except Exception as e:
            logger.error(f"Error in callback: {str(e)}")
            return False

def create_parallel_envs(data: pd.DataFrame, env_config: Dict[str, Any], n_envs: int = 16) -> VecNormalize:
    """Create multiple environments to run in parallel."""
    
    def make_env(rank: int):
        def _init() -> DEXTradingEnv:
            # Don't initialize GPU settings in subprocesses
            env = DEXTradingEnv(
                data=data,
                initial_balance=env_config.get('initial_balance', 10.0),
                transaction_fee=env_config.get('transaction_fee', 0.002),
                max_position=env_config.get('max_position', 25.0),
                reward_scaling=env_config.get('reward_scaling', 1.0),
                risk_free_rate=env_config.get('risk_free_rate', 0.02/365)
            )
            return env
        return _init

    # Create process pool with specified number of environments
    n_envs = min(n_envs, multiprocessing.cpu_count())  # Don't exceed CPU count
    env = SubprocVecEnv(
        [make_env(i) for i in range(n_envs)], 
        start_method='spawn'  # Use 'spawn' for better GPU compatibility
    )
    
    # Apply normalization wrapper
    env = VecNormalize(
        env,
        norm_obs=env_config.get('normalize_observations', True),
        norm_reward=env_config.get('normalize_rewards', True),
        clip_obs=env_config.get('clip_obs', 5.0),
        clip_reward=env_config.get('clip_reward', 5.0),
        gamma=env_config.get('gamma', 0.99),
        epsilon=1e-08
    )
    
    return env

def create_optimized_agent(env: VecNormalize, config: Dict[str, Any]) -> PPO:
    """Create a PPO agent optimized for GPU usage."""
    
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],
            vf=[256, 256]
        ),
        activation_fn=torch.nn.ReLU,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=dict(
            eps=1e-5,
            weight_decay=1e-5
        )
    )

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config["learning_rate"],
        n_steps=2048,
        batch_size=1024,
        n_epochs=10,
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        max_grad_norm=config["max_grad_norm"],
        vf_coef=config["vf_coef"],
        policy_kwargs=policy_kwargs,
        tensorboard_log=f"./runs/{int(datetime.now().timestamp())}",
        device=device,
        verbose=1
    )
    
    return model

def evaluate_agent(model: PPO, env: VecNormalize, num_episodes: int = 10) -> Dict[str, float]:
    """Evaluate the trained agent with comprehensive metrics."""
    episode_rewards = []
    portfolio_values = []
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            # Handle the case where action is a numpy array
            action_value = action.item() if hasattr(action, 'item') else action[0]
            action_counts[action_value] += 1
            
            obs, reward, done_array, info = env.step(action)
            # Sum reward across all environments if using vectorized env
            episode_reward += reward.sum() if isinstance(reward, np.ndarray) else reward
            
            # Check if any environment is done
            done = done_array.any() if isinstance(done_array, np.ndarray) else done_array
            
            if done:
                if isinstance(info, list) and len(info) > 0:
                    portfolio_value = info[0].get("portfolio_value", 0.0)
                else:
                    portfolio_value = info.get("portfolio_value", 0.0)
                portfolio_values.append(portfolio_value)
                episode_rewards.append(episode_reward)
                break
    
    # Calculate metrics
    total_actions = sum(action_counts.values())
    action_distribution = {k: v/total_actions if total_actions > 0 else 0 
                         for k, v in action_counts.items()}
    
    return {
        "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "std_reward": float(np.std(episode_rewards)) if len(episode_rewards) > 1 else 0.0,
        "mean_portfolio_value": float(np.mean(portfolio_values)) if portfolio_values else 0.0,
        "best_portfolio_value": float(max(portfolio_values)) if portfolio_values else 0.0,
        "worst_portfolio_value": float(min(portfolio_values)) if portfolio_values else 0.0,
        "action_distribution": action_distribution
    }

def train_agent(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Main training function with improved error handling."""
    try:
        run = wandb.init(
            project="dex-trading-rl",
            config=config,
            sync_tensorboard=True,
            group=f"training_{datetime.now().strftime('%Y%m%d')}",
            job_type="training"
        )
        
        save_dir = Path(config["save_dir"])
        save_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Using device: {device}")
        print_gpu_utilization()
        
        # Load data with error handling
        try:
            train_data = pd.read_pickle(Path(config["data_dir"]) / "train_data.pkl")
            val_data = pd.read_pickle(Path(config["data_dir"]) / "val_data.pkl")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
        
        logger.info(f"Training data shape: {train_data.shape}")
        logger.info(f"Validation data shape: {val_data.shape}")
        
        # Create environments with error handling
        try:
            train_env = create_parallel_envs(train_data, config, n_envs=config["n_envs"])
            val_env = create_parallel_envs(val_data, config, n_envs=config["n_envs"])
        except Exception as e:
            logger.error(f"Error creating environments: {e}")
            raise
        
        # Create and train agent
        try:
            model = create_optimized_agent(train_env, config)
            
            eval_callback = EvalCallback(
                val_env,
                best_model_save_path=str(save_dir / "best_model"),
                log_path=str(save_dir / "logs"),
                eval_freq=config["eval_freq"],
                deterministic=True,
                render=False,
                n_eval_episodes=config["eval_episodes"]
            )
            
            trading_callback = TradingCallback(eval_freq=config["eval_freq"])
            wandb_callback = WandbCallback(
                gradient_save_freq=100,
                model_save_path=str(save_dir / "wandb_models"),
                verbose=2
            )
            
            logger.info("Starting training...")
            model.learn(
                total_timesteps=config["total_timesteps"],
                callback=[eval_callback, trading_callback, wandb_callback],
                progress_bar=True
            )
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
            
        # Evaluate with error handling
        try:
            logger.info("Evaluating final model...")
            eval_metrics = evaluate_agent(model, val_env, num_episodes=config["eval_episodes"])
            wandb.log({"eval/" + k: v for k, v in eval_metrics.items()})
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise
            
        # Save final model
        try:
            final_model_path = save_dir / "final_model.zip"
            model.save(str(final_model_path))
            train_env.save(str(save_dir / "vec_normalize.pkl"))
            logger.info(f"Saved final model to {final_model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
            
        print_gpu_utilization()
        
        return {
            "model": model,
            "train_env": train_env,
            "val_env": val_env,
            "metrics": eval_metrics
        }
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        return None
        
    finally:
        wandb.finish()

def main():
    """Main entry point with GPU-optimized configuration."""
    config = {
        # Data and saving parameters
        "data_dir": "processed_data",
        "save_dir": "models",
        
        # Environment parameters
        "initial_balance": 10.0,
        "max_position": 15.0,
        "transaction_fee": 0.002,
        "reward_scaling": 1.0,
        "risk_free_rate": 0.02/365,
        
        # Parallel processing
        "n_envs": 16,
        
        # Training parameters
        "total_timesteps": 1_000_000,
        "eval_freq": 10000,
        "eval_episodes": 10,
        
        # PPO parameters
        "learning_rate": 3e-4,
        "n_steps": 4096,
        "batch_size": 2048,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.005,
        "max_grad_norm": 0.5,
        "vf_coef": 0.5,
        
        # Normalization parameters
        "normalize_observations": True,
        "normalize_rewards": True,
        "clip_obs": 5.0,
        "clip_reward": 5.0,
        
        # Network architecture
        "net_arch": {
            "pi": [256, 256],
            "vf": [256, 256]
        }
    }
    
    results = train_agent(config)
    
    if results:
        logger.info("Training completed successfully")
        logger.info("\nFinal Evaluation Metrics:")
        for metric, value in results["metrics"].items():
            if isinstance(value, dict):
                logger.info(f"\n{metric}:")
                for k, v in value.items():
                    logger.info(f"  {k}: {v:.4f}")
            else:
                logger.info(f"{metric}: {value:.4f}")
    else:
        logger.error("Training failed")

if __name__ == "__main__":
    main()