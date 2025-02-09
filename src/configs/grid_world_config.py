# src/configs/grid_world_config.py
class GridWorldConfig:
    def __init__(self):
        self.max_steps = 100
        self.max_episodes = 1000
        self.gamma = 0.99
        self.learning_rate = 1e-4