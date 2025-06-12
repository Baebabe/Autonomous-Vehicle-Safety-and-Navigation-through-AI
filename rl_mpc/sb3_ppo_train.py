import os
import time
import numpy as np
import gymnasium as gym 
from gymnasium import spaces 
import torch
import optuna #used for hyperparameter optimization
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import json
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy
from rl_mpc.Carla_environment import CarEnv

# Custom wrapper for your CarEnv to make it gym-compatible
class CarlaGymEnv(gym.Env):
    """Custom Environment that follows gym interface for CARLA environment"""
    metadata = {'render.modes': ['human']}

    def __init__(self, carla_env=None):
        super(CarlaGymEnv, self).__init__()
        
        # Create or use provided CARLA environment
        if carla_env is None:
            self.env = CarEnv()
        else:
            self.env = carla_env
            
        # Define action and observation space
        # They must be gym.spaces objects
        
        # Action space: [steering, throttle/brake]
        # Both ranging from -1 to 1
        # In CarlaGymEnv.__init__, replace action space definition with:
        self.action_space = spaces.Box(
            low=np.array([-1.0]),  # -1 for maximum braking
            high=np.array([1.0]),  # +1 for maximum throttle
            dtype=np.float32
        )
        
        # Observation space:
        # - object detections: max_objects * 3 features (x, y, depth)
        # - lane info: 2 features (distance, angle)
        # - vehicle state: 2 features (speed, steering)

        # obs_dim = self.env.max_objects * 3 + 2 + 2
        obs_dim = self.env.max_objects * 6 + 1

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        self.episode_reward = 0
        self.episode_length = 0
        self.metrics = {
            'speeds': [],
            'collisions': 0,
            'lane_deviations': []
        }

        self.max_episode_steps = 4000

    def step(self, action):
        # Execute action and get results
        obs, reward, done, info = self.env.step(action)
        self.episode_reward += reward
        self.episode_length += 1

        if self.episode_length >= self.max_episode_steps:
            done = True

        # Update episode metrics
        if 'speed' in info:
            self.metrics['speeds'].append(info['speed'])
        if 'collision' in info:
            self.metrics['collisions'] += 1
        if 'distance_from_center' in info:
            self.metrics['lane_deviations'].append(info['distance_from_center'])
        
        # Update info with episode metrics
        info.update({
            'episode': {
                'r': self.episode_reward,
                'l': self.episode_length
            } if done else None
        })
        
        # Convert to gym format
        return obs.astype(np.float32), reward, done, False, info

    def reset(self, **kwargs):
        # Reset the environment
        obs = self.env.reset()
        
        # Reset episode metrics
        self.episode_reward = 0
        self.episode_length = 0
        self.metrics = {'speeds': [], 'collisions': 0, 'lane_deviations': []}
        
        # Convert to gym format
        return obs.astype(np.float32), {}

    def render(self, mode='human'):
        # The environment already renders via pygame in _process_image
        pass

    def close(self):
        if hasattr(self.env, 'cleanup_actors'):
            self.env.cleanup_actors()
        if hasattr(self.env, 'cleanup_npcs'):
            self.env.cleanup_npcs()

# Callback for training monitoring
class TrainingMonitorCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0,log_freq=100):
        super(TrainingMonitorCallback, self).__init__(verbose)
        self.log_freq = log_freq
        self.training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'avg_speeds': [],
            'collisions': [],
            'lane_deviations': []
        }
        self.start_time = time.time()
    
    def _on_step(self):
        if self.num_timesteps % self.log_freq == 0:
            env = self.training_env.envs[0].env
            self.logger.record('env/speed', env.get_vehicle_state()['speed'])
            self.logger.dump(step=self.num_timesteps)
        # Log scalar metrics for tensorboard
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.model.env.close()
                raise Exception("User closed the window")
        for i in range(len(self.model.env.buf_dones)):
            if self.model.env.buf_dones[i]:
                # Extract metrics from the environment
                env_idx = i if self.model.env.num_envs > 1 else 0
                env = self.training_env.envs[env_idx].env

                if hasattr(env, 'metrics'):
                    # Calculate episode metrics
                    avg_speed = np.mean(env.metrics['speeds']) if env.metrics['speeds'] else 0
                    max_lane_dev = max(env.metrics['lane_deviations']) if env.metrics['lane_deviations'] else 0

                    # Store metrics
                    self.training_metrics['episode_rewards'].append(env.episode_reward)
                    self.training_metrics['episode_lengths'].append(env.episode_length)
                    self.training_metrics['avg_speeds'].append(avg_speed)
                    self.training_metrics['collisions'].append(env.metrics['collisions'])
                    self.training_metrics['lane_deviations'].append(max_lane_dev)

                    # Log to tensorboard
                    self.logger.record('env/episode_reward', env.episode_reward)
                    
                    self.logger.record('env/episode_length', env.episode_length)
                    self.logger.record('env/average_speed', avg_speed)
                    self.logger.record('env/collisions', env.metrics['collisions'])
                    self.logger.record('env/max_lane_deviation', max_lane_dev)
                    
                    # Log training speed
                    elapsed_time = time.time() - self.start_time
                    fps = int(self.num_timesteps / elapsed_time)
                    self.logger.record('time/fps', fps)
                    self.logger.dump(step=self.num_timesteps)
                    # Print episode summary to console
                    print(f"\n=== Episode Complete ===")
                    print(f"Episode Reward: {env.episode_reward:.2f}")
                    print(f"Episode Length: {env.episode_length}")
                    print(f"Average Speed: {avg_speed:.2f} km/h")
                    print(f"Collisions: {env.metrics['collisions']}")
                    print(f"Max Lane Deviation: {max_lane_dev:.2f} m")
                    print(f"Training FPS: {fps}")
                    print(f"========================\n")

        return True

    
    def save_metrics(self, path):
        """Save training metrics to a file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.training_metrics, f)

# Custom feature extractor (optional, for more complex inputs)
class CustomFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for more complex processing of observations
    """
    def __init__(self, observation_space, features_dim=128):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # Neural network layers
        self.net = torch.nn.Sequential(
            torch.nn.Linear(observation_space.shape[0], 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, features_dim),
            torch.nn.ReLU()
        )
    
    def forward(self, observations):
        return self.net(observations)

# Create a vectorized environment
def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param rank: (int) index of the subprocess
    :param seed: (int) the initial seed for RNG
    """
    def _init():
        env = CarlaGymEnv()
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

# Function to perform hyperparameter optimization
def optimize_ppo(trial):
    """
    Optimization function for Optuna
    """
    # Define the range of hyperparameters to search
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_int("n_steps", 256, 2048, log=True)
    batch_size = trial.suggest_int("batch_size", 32, 256, log=True)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.999)
    ent_coef = trial.suggest_float("ent_coef", 0.00001, 0.1, log=True)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    
    # Setup environment for evaluation
    env = CarlaGymEnv()
    env = Monitor(env)
    
    # Create the model with current hyperparameters
    model = PPO("MlpPolicy", 
                env, 
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                gamma=gamma,
                gae_lambda=gae_lambda,
                ent_coef=ent_coef,
                clip_range=clip_range,
                verbose=0)
    
    # Train model for a small number of steps
    try:
        model.learn(total_timesteps=10000)
        
        # Evaluate model 
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=3)
        env.close()
        
        return mean_reward
    
    except Exception as e:
        print(f"Optimization trial failed: {e}")
        env.close()
        return -1000  # Return a poor score on failure

# In the train_ppo function, after your checkpoint_callback definition, add this:
class SaveLatestModelCallback(BaseCallback):
    """
    Callback for saving the latest model to a fixed filename
    """
    def __init__(self, save_path, save_freq=500, save_vecnormalize=True, verbose=0):
        super(SaveLatestModelCallback, self).__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        self.save_vecnormalize = save_vecnormalize
        
    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            # Save the model
            latest_model_path = os.path.join(self.save_path, "latest_model")
            self.model.save(latest_model_path)
            
            # Save the normalization parameters
            if self.save_vecnormalize and self.model.get_env() is not None:
                self.model.get_env().save(os.path.join(self.save_path, "latest_vec_normalize.pkl"))
                
            if self.verbose > 0:
                print(f"Saving latest model to {latest_model_path}")
        return True

def train_ppo(env=None, num_envs=1, total_timesteps=1000000, hyperparameter_tuning=False, 
              continue_training=False, checkpoint_path=None):
    """
    Train the PPO agent with all the advanced features

    Parameters:
        env: Pre-initialized CarlaEnv instance (optional)
        num_envs: Number of parallel environments
        total_timesteps: Total timesteps to train for
        hyperparameter_tuning: Whether to run hyperparameter optimization
        continue_training: Whether to continue from a previous checkpoint
        checkpoint_path: Path to the previous model checkpoint
    """
    # Set paths
    log_dir = "logs"
    save_dir = "models"
    tensorboard_dir = os.path.join(log_dir, f"ppo_carla_{int(time.time())}")
    metrics_path = os.path.join(log_dir, "training_metrics.json")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # Use provided environment or create a new one
    if env is None:
        print("Creating new CarlaEnv instance...")
        base_env = CarlaGymEnv()
    else:
        print("Using provided CarlaEnv instance...")
        base_env = CarlaGymEnv(carla_env=env)

    # Create vectorized environments
    if num_envs > 1:
        print(f"Creating {num_envs} parallel environments...")
        env = SubprocVecEnv([make_env("CarlaGymEnv", i) for i in range(num_envs)])
    else:
        print("Creating single vectorized environment...")
        env = DummyVecEnv([lambda: base_env])

    # Define default hyperparameters first
    hp = {
        "learning_rate": 3e-4,
        "n_steps": 1024,
        "batch_size": 32,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "clip_range": 0.2
    }

    # Add normalization wrapper for observations and rewards
    if continue_training and os.path.exists(f"{os.path.dirname(checkpoint_path)}/vec_normalize.pkl"):
        print(f"Loading normalization statistics from {os.path.dirname(checkpoint_path)}/vec_normalize.pkl")
        env = VecNormalize.load(f"{os.path.dirname(checkpoint_path)}/vec_normalize.pkl", env)
        # Don't update normalization statistics if not needed
        env.training = True
        env.norm_reward = True
    else:
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=hp["gamma"]
        )

    # Run hyperparameter optimization if enabled and not continuing training
    if hyperparameter_tuning and not continue_training:
        print("Running hyperparameter optimization...")
        study = optuna.create_study(direction="maximize")
        study.optimize(optimize_ppo, n_trials=20)
        
        print("Best hyperparameters:", study.best_params)
        # Update hp with optimized values
        hp.update(study.best_params)

    # Configure logger
    new_logger = configure(tensorboard_dir, ["tensorboard", "stdout"])

    # Set up callbacks
    monitor_callback = TrainingMonitorCallback()

    checkpoint_callback = CheckpointCallback(
        save_freq=500,
        save_path=save_dir,
        name_prefix="ppo_carla_model",
        save_vecnormalize=True
    )

    # Separate environment for evaluation
    if env is None:
        eval_base_env = CarlaGymEnv()
    else:
        # eval_base_env = CarlaGymEnv(carla_env=env.env)
        eval_base_env = CarlaGymEnv(carla_env=env.envs[0].env)
        
    eval_env = DummyVecEnv([lambda: eval_base_env])
    eval_env = VecNormalize.load(f"{save_dir}/vec_normalize.pkl", eval_env) if os.path.exists(f"{save_dir}/vec_normalize.pkl") else VecNormalize(eval_env)
    # Don't update the normalization statistics during evaluation
    eval_env.training = False
    eval_env.norm_reward = False

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_dir}/best_model",
        log_path=log_dir,
        eval_freq=100000,
        n_eval_episodes=10,
        deterministic=True
    )

    callback_list = CallbackList([checkpoint_callback, monitor_callback, eval_callback])

    # In train_ppo function, update the policy architecture:
    policy_kwargs = {
        "net_arch": [dict(pi=[128, 64], vf=[128, 64])]  # Simplified network for speed control
    }

    # Create or load the PPO model
    if continue_training and checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path} to continue training...")
        model = PPO.load(
            checkpoint_path,
            env=env,
            tensorboard_log=tensorboard_dir,
            verbose=1
        )
        # Optionally reset the learning rate schedule
        model.learning_rate = lr_schedule if 'lr_schedule' in locals() else hp["learning_rate"]
    else:
        # Create a custom learning rate schedule
        def lr_schedule(remaining_progress):
            """Linear learning rate schedule"""
            return hp["learning_rate"] * remaining_progress
        
        # Create a new model
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=lr_schedule,
            n_steps=hp["n_steps"],
            batch_size=hp["batch_size"],
            gamma=hp["gamma"],
            gae_lambda=hp["gae_lambda"],
            ent_coef=hp["ent_coef"],
            clip_range=hp["clip_range"],
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_dir,
            verbose=1
        )

    # Set logger
    model.set_logger(new_logger)

    try:
        print("Starting training...")
        start_time = time.time()
        
        # Train the model
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            reset_num_timesteps=not continue_training  # Don't reset timestep count if continuing training
        )
        
        # Save the final model
        model.save(f"{save_dir}/final_model")
        # Save VecNormalize statistics
        env.save(f"{save_dir}/vec_normalize.pkl")
        
        # Save training metrics
        monitor_callback.save_metrics(metrics_path)
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time / 3600:.2f} hours")
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
        # Save model on interrupt
        model.save(f"{save_dir}/interrupted_model")
        env.save(f"{save_dir}/vec_normalize_interrupted.pkl")
        print(f"Model saved to {save_dir}/interrupted_model.zip")
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        env.close()
        eval_env.close()
        
    return model

def evaluate_model(model_path_or_params, env=None, num_episodes=10):
    """
    Evaluate a trained PPO model with focus on speed control metrics
    
    Parameters:
        model_path_or_params: Path to model file or model parameters
        env: Pre-initialized CarlaEnv instance (optional)
        num_episodes: Number of episodes to evaluate
    """
    try:
        # Load or create the environment
        if env is None:
            print("Creating new environment for evaluation...")
            eval_env = CarlaGymEnv()
        else:
            print("Using provided environment for evaluation...")
            eval_env = CarlaGymEnv(carla_env=env)
            
        eval_env = Monitor(eval_env)
        
        # Load the model
        if isinstance(model_path_or_params, str):
            print(f"Loading model from {model_path_or_params}")
            model = PPO.load(model_path_or_params)
        else:
            print("Using provided model parameters")
            model = model_path_or_params
        
        print(f"Evaluating model for {num_episodes} episodes...")
        mean_reward, std_reward = evaluate_policy(
            model, 
            eval_env, 
            n_eval_episodes=num_episodes,
            deterministic=True
        )
        
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        
        # Get additional metrics with focus on speed control
        episode_lengths = []
        speeds = []
        target_speeds = []
        speed_diffs = []
        collisions = 0
        lane_deviations = []
        
        for i in range(num_episodes):
            obs, _ = eval_env.reset()
            done = False
            episode_length = 0
            episode_speeds = []
            episode_target_speeds = []
            episode_speed_diffs = []
            episode_lane_devs = []
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = eval_env.step(action)
                episode_length += 1
                
                # Collect speed metrics
                if 'speed' in info:
                    episode_speeds.append(info['speed'])
                
                # Target speed and deviation
                if 'target_speed' in info:
                    episode_target_speeds.append(info['target_speed'])
                    if 'speed' in info:
                        speed_diff = abs(info['speed'] - info['target_speed'])
                        episode_speed_diffs.append(speed_diff)
                
                # Collision tracking
                if 'collision' in info and info['collision']:
                    collisions += 1
                
                # Lane deviation tracking (for info only)
                if 'distance_from_center' in info:
                    episode_lane_devs.append(info['distance_from_center'])
            
            # Record episode statistics
            episode_lengths.append(episode_length)
            
            if episode_speeds:
                speeds.append(np.mean(episode_speeds))
            
            if episode_target_speeds:
                target_speeds.append(np.mean(episode_target_speeds))
            
            if episode_speed_diffs:
                speed_diffs.append(np.mean(episode_speed_diffs))
            
            if episode_lane_devs:
                lane_deviations.append(np.mean(episode_lane_devs))
                
            print(f"Episode {i+1}: Length = {episode_length}, " +
                  f"Avg Speed = {np.mean(episode_speeds) if episode_speeds else 0:.2f} km/h, " +
                  f"Avg Target = {np.mean(episode_target_speeds) if episode_target_speeds else 0:.2f} km/h, " +
                  f"Avg Speed Diff = {np.mean(episode_speed_diffs) if episode_speed_diffs else 0:.2f} km/h")
        
        # Print summary statistics with focus on speed control
        print("\nEvaluation Summary:")
        print(f"Average Episode Length: {np.mean(episode_lengths):.2f}")
        print(f"Average Speed: {np.mean(speeds):.2f} km/h")
        print(f"Average Target Speed: {np.mean(target_speeds):.2f} km/h")
        print(f"Average Speed Deviation: {np.mean(speed_diffs):.2f} km/h")
        print(f"Speed Control Accuracy: {100 - (np.mean(speed_diffs) / np.mean(target_speeds) * 100):.2f}%")
        print(f"Average Lane Deviation: {np.mean(lane_deviations):.2f} m")
        print(f"Total Collisions: {collisions}")
        
        return mean_reward, std_reward
        
    except Exception as e:
        print(f"Evaluation error: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    finally:
        if 'eval_env' in locals():
            eval_env.close()


import glob
import re

def get_latest_model():
    # Find all .zip files in the models directory
    model_files = glob.glob("models/ppo_carla_model_*.zip")
    
    if not model_files:
        raise FileNotFoundError("No model files found in models directory")
    
    # Extract the step count from each filename using regex
    steps_pattern = re.compile(r'ppo_carla_model_(\d+)_steps\.zip')
    
    # Create a list of (filename, step_count) tuples
    models_with_steps = []
    for model_file in model_files:
        match = steps_pattern.search(os.path.basename(model_file))
        if match:
            step_count = int(match.group(1))
            models_with_steps.append((model_file, step_count))
    
    # Sort by step count (descending) and get the highest one
    models_with_steps.sort(key=lambda x: x[1], reverse=True)
    
    if not models_with_steps:
        raise ValueError("Found model files but couldn't parse step counts")
    
    return models_with_steps[0][0]

def main():
    """Main function to run CARLA environment with MPC controller integration"""
    # Set random seeds for reproducibility
    set_random_seed(16)
    
    # Configuration parameters
    TRAIN_RL = True # Set to False to run only with MPC controller
    EVALUATE_ONLY = False  # Set to True to only evaluate without training
    TUNE_HYPERPARAMETERS = False
    CONTINUE_TRAINING = True
    CHECKPOINT_PATH = get_latest_model()  # Your checkpoint path
    # CHECKPOINT_PATH = None
    try:
        print("Starting CARLA with hybrid MPC-RL control system")
        print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
        
        # Initialize the CarlaEnv with MPC controller
        env = CarEnv()
        if not env.setup_vehicle():
            print("Failed to set up CARLA environment")
            return
        
        # If we're just running with the MPC controller
        if not TRAIN_RL:
            print("Running with MPC controller only...")
            env.run()
            return
            
        # If we're only evaluating a trained model
        if EVALUATE_ONLY:
            print("Evaluating model with MPC controller...")
            if os.path.exists(CHECKPOINT_PATH):
                evaluate_model(CHECKPOINT_PATH, env=env)
            else:
                print(f"Checkpoint not found at {CHECKPOINT_PATH}")
            return
        
        # Train the PPO model with MPC controller for lane keeping and path following
        model = train_ppo(
            env=env,  # Pass the CARLA environment with MPC
            num_envs=1,
            total_timesteps=1_000_000,
            hyperparameter_tuning=TUNE_HYPERPARAMETERS,
            continue_training=CONTINUE_TRAINING,
            checkpoint_path=CHECKPOINT_PATH if CONTINUE_TRAINING else None
        )
        
        # Evaluate the best model
        best_model_path = "models/best_model/best_model.zip"
        if os.path.exists(best_model_path):
            print("\nEvaluating best model...")
            evaluate_model(best_model_path, env=env)
        
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Program terminated")
        # Make sure to clean up CARLA actors
        if 'env' in locals() and hasattr(env, 'cleanup_actors'):
            env.cleanup_actors()
        if 'env' in locals() and hasattr(env, 'cleanup_npcs'):
            env.cleanup_npcs()

if __name__ == '__main__':
    main()



