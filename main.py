# main.py
import sys
import argparse
from pathlib import Path
framework_path = Path(__file__).parent / 'marl-framework'
sys.path.append(str(framework_path))

import numpy as np
from src.environments.grid_world_env import GridWorldEnv
from src.agents.dqn_agent import DQNAgent
from src.configs.training_config import TrainingConfig
from src.utils.device_utils import get_device

import torch

def evaluate_agents(env, agents, config):
    total_rewards = []
    for _ in range(config.eval_episodes):
        observations, _ = env.reset()
        episode_rewards = {i: 0 for i in range(len(agents))}
        done = False
        
        while not done:
            actions = {}
            for i, agent in enumerate(agents):
                actions[i] = agent.select_action(observations[i])
                
            next_obs, rewards, done, _, _ = env.step(actions)
            
            for i in range(len(agents)):
                episode_rewards[i] += rewards[i]
                
            observations = next_obs
            
        total_rewards.append(sum(episode_rewards.values()) / len(agents))
    
    return np.mean(total_rewards)

def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description='Train DQN agents')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'mps'], 
                      help='Device to use (cuda/cpu/mps)')
    args = parser.parse_args()
    
    # Device 설정
    device = get_device(args.device)
    
    # 설정 및 환경 생성
    config = TrainingConfig()
    env = GridWorldEnv(config)
    
    # 에이전트 생성
    agents = [
        DQNAgent(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.n,
            config=config,
            device=device
        )
        for _ in range(env.n_agents)
    ]
    
    # 학습 루프
    best_eval_reward = float('-inf')
    
    for episode in range(config.max_episodes):
        observations, _ = env.reset()
        episode_rewards = {i: 0 for i in range(len(agents))}
        done = False
        
        while not done:
            actions = {}
            for i, agent in enumerate(agents):
                actions[i] = agent.select_action(observations[i])
                
            next_obs, rewards, done, _, _ = env.step(actions)
            
            # Store transitions and train
            for i, agent in enumerate(agents):
                agent.store_transition(
                    observations[i],
                    actions[i],
                    rewards[i],
                    next_obs[i],
                    done
                )
                agent.train()
                episode_rewards[i] += rewards[i]
                
            observations = next_obs
        
        # Update target networks
        if episode % config.target_update_freq == 0:
            for agent in agents:
                agent.update_target_network()
        
        # Evaluate
        if episode % config.eval_freq == 0:
            eval_reward = evaluate_agents(env, agents, config)
            print(f"Episode {episode}, Eval reward: {eval_reward:.2f}")
            
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                print(f"New best eval reward: {best_eval_reward:.2f}")
        
        # Print training progress
        avg_reward = sum(episode_rewards.values()) / len(agents)
        print(f"Episode {episode}, Training reward: {avg_reward:.2f}, Epsilon: {agents[0].epsilon:.2f}")

if __name__ == "__main__":
    main()