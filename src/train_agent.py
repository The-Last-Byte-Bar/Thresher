import pandas as pd
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import wandb
from wandb.integration.sb3 import WandbCallback
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import logging
from datetime import datetime
from env import DEXTradingEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingCallback(BaseCallback):
    """Custom callback for logging trading-specific metrics."""
    
    def __init__(self, eval_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.best_reward = -np.inf
        self.best_model_path = None
        
    def _on_step(self) -> bool:
        """Log metrics at each step."""
        try:
            # Get the most recent episode info
            if len(self.model.ep_info_buffer) > 0:
                episode_info = self.model.ep_info_buffer[-1]
                episode_reward = episode_info["r"]
                episode_length = episode_info["l"]
                
                # Get environment info
                env_info = self.training_env.get_attr("last_info")[0]
                
                # Log episode metrics
                wandb.log({
                    "train/episode_reward": episode_reward,
                    "train/episode_length": episode_length,
                    "train/portfolio_value": env_info["portfolio_value"],
                    "train/sharpe_ratio": env_info["sharpe_ratio"],
                    "train/max_drawdown": env_info["max_drawdown"],
                    "train/current_price": env_info["current_price"],
                    "train/last_action": env_info["last_action"]
                })
                
                # Save best model
                if episode_reward > self.best_reward:
                    self.best_reward = episode_reward
                    if self.best_model_path:
                        self.model.save(self.best_model_path)
                        
            return True
            
        except Exception as e:
            logger.error(f"Error in callback: {str(e)}")
            return False

def create_env(
    train_data: pd.DataFrame,
    initial_balance: float = 10.0,
    max_position: float = 50.0,
    transaction_fee: float = 0.002
) -> DummyVecEnv:
    """Create and configure the trading environment."""
    
    def make_env():
        env = DEXTradingEnv(
            data=train_data,
            initial_balance=initial_balance,
            max_position=max_position,
            transaction_fee=transaction_fee,
            reward_window=5,
            risk_free_rate=0.02,
            volatility_penalty_factor=0.5
        )
        return env
    
    # Create vectorized environment
    env = DummyVecEnv([make_env])
    
    # Normalize observations and rewards
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
        epsilon=1e-08
    )
    
    return env

def create_agent(env: DummyVecEnv, config: Dict[str, Any]) -> PPO:
    """Create and configure the PPO agent."""
    policy_kwargs = dict(
        net_arch=dict(
            pi=[128, 128],  # Policy network
            vf=[128, 128]   # Value function network
        )
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        policy_kwargs=policy_kwargs,
        tensorboard_log=f"./runs/{int(datetime.now().timestamp())}",
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=1
    )
    
    return model

def evaluate_agent(
    model: PPO,
    env: DummyVecEnv,
    num_episodes: int = 10
) -> Dict[str, float]:
    """Evaluate the trained agent."""
    episode_rewards = []
    portfolio_values = []
    sharpe_ratios = []
    max_drawdowns = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            
            if done:
                episode_rewards.append(episode_reward)
                portfolio_values.append(info[0]["portfolio_value"])
                sharpe_ratios.append(info[0]["sharpe_ratio"])
                max_drawdowns.append(info[0]["max_drawdown"])
                
    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_portfolio_value": np.mean(portfolio_values),
        "mean_sharpe_ratio": np.mean(sharpe_ratios),
        "mean_max_drawdown": np.mean(max_drawdowns),
        "best_portfolio_value": max(portfolio_values),
        "worst_portfolio_value": min(portfolio_values)
    }

def plot_results(metrics: Dict[str, float], save_dir: Path):
    """Plot and save training results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot portfolio values
    axes[0, 0].hist(metrics.get("portfolio_values", []), bins=20)
    axes[0, 0].set_title("Portfolio Value Distribution")
    axes[0, 0].set_xlabel("Portfolio Value (ERG)")
    axes[0, 0].set_ylabel("Frequency")
    
    # Plot Sharpe ratios
    axes[0, 1].plot(metrics.get("sharpe_ratios", []))
    axes[0, 1].set_title("Sharpe Ratio Evolution")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Sharpe Ratio")
    
    # Plot drawdowns
    axes[1, 0].plot(metrics.get("max_drawdowns", []))
    axes[1, 0].set_title("Maximum Drawdown Evolution")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Max Drawdown")
    
    # Plot rewards
    axes[1, 1].plot(metrics.get("episode_rewards", []))
    axes[1, 1].set_title("Episode Rewards")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Total Reward")
    
    plt.tight_layout()
    plt.savefig(save_dir / "training_results.png")
    plt.close()

def train_agent(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Main training function."""
    try:
        # Initialize wandb
        run = wandb.init(
            project="dex-trading-rl",
            config=config,
            sync_tensorboard=True
        )
        
        # Create directories
        save_dir = Path(config["save_dir"])
        save_dir.mkdir(exist_ok=True)
        
        # Load preprocessed data
        train_data = pd.read_pickle(Path(config["data_dir"]) / "train_data.pkl")
        val_data = pd.read_pickle(Path(config["data_dir"]) / "val_data.pkl")
        
        logger.info(f"Loaded training data shape: {train_data.shape}")
        logger.info(f"Loaded validation data shape: {val_data.shape}")
        
        # Create environments
        train_env = create_env(
            train_data,
            initial_balance=config["initial_balance"],
            max_position=config["max_position"],
            transaction_fee=config["transaction_fee"]
        )
        
        val_env = create_env(
            val_data,
            initial_balance=config["initial_balance"],
            max_position=config["max_position"],
            transaction_fee=config["transaction_fee"]
        )
        
        # Create agent
        model = create_agent(train_env, config)
        
        # Setup callbacks
        eval_callback = EvalCallback(
            val_env,
            best_model_save_path=str(save_dir / "best_model"),
            log_path="./logs/",
            eval_freq=config["eval_freq"],
            deterministic=True,
            render=False
        )
        
        trading_callback = TradingCallback(
            eval_freq=config["eval_freq"]
        )
        
        wandb_callback = WandbCallback()
        
        # Train the agent
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=[eval_callback, trading_callback, wandb_callback]
        )
        
        # Evaluate the trained agent
        eval_metrics = evaluate_agent(
            model,
            val_env,
            num_episodes=config["eval_episodes"]
        )
        
        # Log final metrics
        wandb.log({"eval/" + k: v for k, v in eval_metrics.items()})
        
        # Plot and save results
        plot_results(eval_metrics, save_dir)
        
        # Save the final model and environment
        model.save(str(save_dir / "final_model"))
        train_env.save(str(save_dir / "vec_normalize.pkl"))
        
        return {
            "model": model,
            "train_env": train_env,
            "val_env": val_env,
            "metrics": eval_metrics
        }
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return None
        
    finally:
        wandb.finish()

def main():
    # Training configuration
    config = {
        "data_dir": "processed_data",
        "save_dir": "models",
        
        # Environment parameters
        "initial_balance": 10.0,
        "max_position": 50.0,
        "transaction_fee": 0.002,
        
        # Training parameters
        "total_timesteps": 1_000_000,
        "eval_freq": 10000,
        "eval_episodes": 10,
        
        # PPO parameters
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01
    }
    
    # Train the agent
    results = train_agent(config)
    
    if results:
        logger.info("Training completed successfully")
        logger.info("\nEvaluation Metrics:")
        for metric, value in results["metrics"].items():
            logger.info(f"{metric}: {value:.4f}")
    else:
        logger.error("Training failed")

if __name__ == "__main__":
    main()