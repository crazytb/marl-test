from src.configs.grid_world_config import GridWorldConfig
from src.environments.grid_world_env import GridWorldEnv

# main.py에서 환경 테스트
if __name__ == "__main__":
    config = GridWorldConfig()
    env = GridWorldEnv(config)
    
    # 간단한 테스트
    obs, _ = env.reset()
    env.render()
    
    for _ in range(10):
        # 랜덤 행동
        actions = {i: env.action_space.sample() for i in range(env.n_agents)}
        obs, rewards, done, _, _ = env.step(actions)
        print("Actions:", actions)
        print("Rewards:", rewards)
        env.render()
        
        if done:
            break