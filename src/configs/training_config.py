# src/configs/training_config.py
class TrainingConfig:
    def __init__(self):
        # Environment settings
        self.max_steps = 100
        self.max_episodes = 200
        
        # Training settings
        self.gamma = 0.99
        self.learning_rate = 1e-4
        self.target_update_freq = 10  # 목표 네트워크 업데이트 주기
        
        # Evaluation settings
        self.eval_freq = 20  # 평가 주기
        self.eval_episodes = 5  # 평가 에피소드 수