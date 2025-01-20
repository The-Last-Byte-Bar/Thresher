import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import wandb
from wandb.integration.sb3 import WandbCallback
from typing import Dict, Any
import matplotlib.pyplot as plt
from pathlib import Path
import time
import pickle
import logging
from env import DEXTradingEnv
from process_data import DEXDataPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CustomCallback(BaseCallback):
    """Enhanced callback for logging trading-specific metrics"""
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.model_save_path = f"models/{int(time.time())}"
        
    def _on_step(self) -> bool:
        """Log custom metrics at each step"""
        # Get the most recent episode info
        if len(self.model.ep_info_buffer) > 0:
            last_episode = self.model.ep_info_buffer[-1]
            episode_reward = last_episode["r"]
            episode_length = last_episode["l"]
            
            # Get environment info from the last step
            if hasattr(self.training_env, "get_attr"):
                env_info = self.training_env.get_attr("last_info")[0]
                if env_info is not None:
                    # Log episode metrics
                    wandb.log({
                        "episode_reward": episode_reward,
                        "episode_length": episode_length,
                        "portfolio_value": env_info["portfolio_value"],
                        "sharpe_ratio": env_info["sharpe_ratio"],
                        "max_drawdown": env_info["max_drawdown"],
                        "volatility_penalty": env_info["volatility_penalty"]
                    })
            
        return True

def prepare_env(data: pd.DataFrame, 
                initial_balance: float = 10.0,
                max_position: float = 50.0) -> DummyVecEnv:
    """Prepare and vectorize the trading environment"""
    def make_env():
        env = DEXTradingEnv(
            data=data,
            initial_balance=initial_balance,
            max_position=max_position
        )
        return Monitor(env)
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,  # Increased from 5.0 for more stability
        clip_reward=10.0,
        gamma=0.99,
        epsilon=1e-08
    )
    
    return env

def check_env_observations(env: DummyVecEnv, num_steps: int = 100) -> bool:
    """Debug function to check for NaN values in observations"""
    logger = logging.getLogger(__name__)
    
    try:
        obs = env.reset()
        logger.info(f"Initial observation shape: {obs.shape}")
        logger.info(f"Initial observation range: min={obs.min():.4f}, max={obs.max():.4f}")
        
        for step in range(num_steps):
            # Create a proper action array for vectorized environment
            action = np.array([env.action_space.sample()])
            
            # Check action
            if np.any(np.isnan(action)):
                logger.error(f"Step {step}: NaN found in action: {action}")
                return False
                
            # Take step and check results
            obs, rewards, dones, infos = env.step(action)
            
            # Check for NaN or infinite values
            if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
                logger.error(f"Step {step}: Invalid values in observation:")
                logger.error(f"NaN positions: {np.where(np.isnan(obs))}")
                logger.error(f"Inf positions: {np.where(np.isinf(obs))}")
                return False
                
            # Check reward validity
            if np.any(np.isnan(rewards)) or np.any(np.isinf(rewards)):
                logger.error(f"Step {step}: Invalid reward value: {rewards}")
                return False
                
            # Log periodic debug info
            if step % 10 == 0:
                logger.info(f"\nStep {step}:")
                logger.info(f"Action taken: {action}")
                logger.info(f"Observation shape: {obs.shape}")
                logger.info(f"Observation range: min={obs.min():.4f}, max={obs.max():.4f}")
                logger.info(f"Rewards: {rewards}")
                logger.info(f"Done: {dones}")
                
                # Log portfolio info if available
                if infos and 'portfolio_value' in infos[0]:
                    logger.info(f"Portfolio value: {infos[0]['portfolio_value']:.4f}")
                    
        return True
        
    except Exception as e:
        logger.error(f"Error during environment check: {str(e)}", exc_info=True)
        return False

def create_ppo_model(env: DummyVecEnv, 
                    learning_rate: float = 3e-4,
                    batch_size: int = 64) -> PPO:
    """Create and configure PPO model"""
    policy_kwargs = dict(
        net_arch=dict(
            pi=[128, 128],  # Increased network size
            vf=[128, 128]
        )
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        device='cpu',
        verbose=1
    )
    
    return model

def train_agent(data: pd.DataFrame,
                total_timesteps: int = 1_000_000,
                eval_freq: int = 10000,
                save_dir: str = "models") -> Dict[str, Any]:
    """Train the trading agent with enhanced data preprocessing"""
    # Initialize wandb
    run = wandb.init(
        project="dex-trading-rl",
        config={
            "algorithm": "PPO",
            "total_timesteps": total_timesteps,
            "eval_freq": eval_freq,
            "initial_balance": 10.0,
            "max_position": 50.0
        }
    )
    
    # Prepare directories
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = DEXDataPreprocessor()
    
    # Prepare and split data
    try:
        train_data, val_data = preprocessor.prepare_train_val_data(data)
        logger.info("Data preprocessing successful")
        logger.info(f"Training data shape: {train_data.shape}")
        logger.info(f"Validation data shape: {val_data.shape}")
    except Exception as e:
        logger.error(f"Error during data preprocessing: {str(e)}")
        raise
    
    # Create environments with processed data
    train_env = prepare_env(train_data)
    eval_env = prepare_env(val_data)
    
    # Check environments for NaN values
    logger.info("\nChecking training environment...")
    if not check_env_observations(train_env):
        raise ValueError("Training environment producing NaN values")
    
    logger.info("\nChecking evaluation environment...")
    if not check_env_observations(eval_env):
        raise ValueError("Evaluation environment producing NaN values")
    
    # Create model
    model = create_ppo_model(train_env)
    
    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir / "best_model"),
        log_path="./logs/",
        eval_freq=eval_freq,
        deterministic=True,
        render=False
    )
    
    custom_callback = CustomCallback()
    
    # Initialize callbacks
    callbacks = CallbackList([eval_callback, custom_callback])
    callbacks.init_callback(model)
    
    # Train the model
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks
        )
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise
    
    # Save final model and environment statistics
    model.save(str(save_dir / "final_model"))
    train_env.save(str(save_dir / "vec_normalize.pkl"))
    
    # Save preprocessor for later use
    with open(save_dir / "preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)
    
    return {
        "model": model,
        "train_env": train_env,
        "eval_env": eval_env,
        "save_dir": save_dir,
        "preprocessor": preprocessor
    }

def evaluate_agent(model: PPO, 
                  env: DummyVecEnv, 
                  num_episodes: int = 10) -> Dict[str, float]:
    """Evaluate the trained agent"""
    episode_rewards = []
    episode_lengths = []
    portfolio_values = []
    sharpe_ratios = []
    max_drawdowns = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            done = dones.any()
            
            episode_reward += reward[0]
            episode_length += 1
            
            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                portfolio_values.append(infos[0]["portfolio_value"])
                sharpe_ratios.append(infos[0]["sharpe_ratio"])
                max_drawdowns.append(infos[0]["max_drawdown"])
    
    metrics = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "mean_portfolio_value": np.mean(portfolio_values),
        "mean_sharpe_ratio": np.mean(sharpe_ratios),
        "mean_max_drawdown": np.mean(max_drawdowns),
        "best_portfolio_value": max(portfolio_values),
        "worst_portfolio_value": min(portfolio_values)
    }
    
    return metrics

def plot_training_results(metrics: Dict[str, float], save_dir: Path):
    """Plot and save training results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Portfolio Value Distribution
    axes[0, 0].hist(metrics.get("portfolio_values", []), bins=20)
    axes[0, 0].set_title("Portfolio Value Distribution")
    axes[0, 0].set_xlabel("Portfolio Value (ERG)")
    axes[0, 0].set_ylabel("Frequency")
    
    # Sharpe Ratio Over Time
    axes[0, 1].plot(metrics.get("sharpe_ratios", []))
    axes[0, 1].set_title("Sharpe Ratio Over Time")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Sharpe Ratio")
    
    # Drawdown Over Time
    axes[1, 0].plot(metrics.get("max_drawdowns", []))
    axes[1, 0].set_title("Maximum Drawdown Over Time")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Max Drawdown")
    
    # Rewards Over Time
    axes[1, 1].plot(metrics.get("episode_rewards", []))
    axes[1, 1].set_title("Episode Rewards Over Time")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Total Reward")
    
    plt.tight_layout()
    plt.savefig(save_dir / "training_results.png")
    plt.close()

if __name__ == "__main__":
    try:
        # Load data
        data = pd.read_csv("data/metrics/CYPX_swaps_20250116_150544.csv")
        data = data.dropna()
        logger.info(f"Loaded data shape: {data.shape}")
        
        # Train the agent with enhanced preprocessing
        training_results = train_agent(
            data=data,
            total_timesteps=1_000_000,
            eval_freq=10000
        )
        
        # Evaluate the trained agent
        eval_metrics = evaluate_agent(
            model=training_results["model"],
            env=training_results["eval_env"]
        )
        
        # Plot and save results
        plot_training_results(eval_metrics, training_results["save_dir"])
        
        # Print final metrics
        logger.info("\nEvaluation Metrics:")
        for metric, value in eval_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
            
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise