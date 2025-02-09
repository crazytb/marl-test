# src/environments/grid_world_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
from pathlib import Path
# Get absolute path to marl-framework
framework_path = Path(__file__).parent.parent.parent / 'marl-framework'
sys.path.append(str(framework_path))
from environments.base_env import BaseEnvironment

class GridWorldEnv(BaseEnvironment):
    """
    간단한 그리드 월드 멀티에이전트 환경
    - 여러 에이전트가 격자 위에서 이동
    - 각 에이전트는 자신의 목표 지점에 도달해야 함
    - 에이전트끼리 충돌하면 음의 보상
    - 목표 지점에 도달하면 양의 보상
    """
    def __init__(self, config):
        super().__init__(config)
        self.grid_size = 5  # 5x5 격자
        self.n_agents = 2   # 2개의 에이전트
        
        # 행동 공간: 상하좌우 이동
        self.action_space = spaces.Discrete(4)
        
        # 관찰 공간: 자신의 위치(2), 목표 위치(2), 다른 에이전트 위치(2)
        self.observation_space = spaces.Box(
            low=0,
            high=self.grid_size-1,
            shape=(6,),
            dtype=np.float32
        )
        
        # 에이전트와 목표의 초기 위치 설정
        self.agent_positions = None
        self.goal_positions = None
        self.steps = 0
        
    def reset(self, seed=None):
        # seed 파라미터를 부모 클래스의 reset 메서드에 전달
        super().reset(seed=seed)
        self.steps = 0
        
        # 에이전트 초기 위치 무작위 설정
        self.agent_positions = {
            i: self._get_random_position() 
            for i in range(self.n_agents)
        }
        
        # 목표 위치 무작위 설정
        self.goal_positions = {
            i: self._get_random_position() 
            for i in range(self.n_agents)
        }
        
        # 초기 관찰 반환
        observations = {}
        for i in range(self.n_agents):
            observations[i] = self._get_observation(i)
            
        return observations, {}
    
    def _get_random_position(self):
        """무작위 위치 생성"""
        return np.random.randint(0, self.grid_size, size=2)
    
    def _get_observation(self, agent_id):
        """각 에이전트의 관찰 생성"""
        obs = np.zeros(6)
        # 자신의 위치
        obs[0:2] = self.agent_positions[agent_id]
        # 목표 위치
        obs[2:4] = self.goal_positions[agent_id]
        # 다른 에이전트의 위치
        other_agent = (agent_id + 1) % self.n_agents
        obs[4:6] = self.agent_positions[other_agent]
        return obs
    
    def step(self, actions):
        """환경 진행"""
        self.steps += 1
        
        # 이동 방향 정의
        directions = [
            np.array([0, 1]),   # 상
            np.array([0, -1]),  # 하
            np.array([-1, 0]),  # 좌
            np.array([1, 0])    # 우
        ]
        
        # 새로운 위치 계산
        new_positions = {}
        for agent_id, action in actions.items():
            current_pos = self.agent_positions[agent_id]
            new_pos = current_pos + directions[action]
            
            # 격자 범위 체크
            new_pos = np.clip(new_pos, 0, self.grid_size-1)
            new_positions[agent_id] = new_pos
            
        # 충돌 체크
        collision = False
        for i in range(self.n_agents):
            for j in range(i+1, self.n_agents):
                if np.array_equal(new_positions[i], new_positions[j]):
                    collision = True
                    
        # 위치 업데이트 (충돌이 없을 경우만)
        if not collision:
            self.agent_positions = new_positions
            
        # 보상 계산
        rewards = {}
        observations = {}
        for i in range(self.n_agents):
            # 기본 보상
            rewards[i] = -0.1  # 시간에 대한 페널티
            
            # 충돌 페널티
            if collision:
                rewards[i] -= 1.0
                
            # 목표 도달 보상
            if np.array_equal(self.agent_positions[i], self.goal_positions[i]):
                rewards[i] += 10.0
                
            # 관찰 업데이트
            observations[i] = self._get_observation(i)
            
        # 종료 조건 체크
        done = False
        # 모든 에이전트가 목표에 도달
        all_reached = all(
            np.array_equal(self.agent_positions[i], self.goal_positions[i])
            for i in range(self.n_agents)
        )
        # 최대 스텝 도달
        max_steps_reached = self.steps >= self.config.max_steps
        
        done = all_reached or max_steps_reached
        
        return observations, rewards, done, False, {}
    
    def render(self):
        """환경 시각화"""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
        grid.fill('.')
        
        # 에이전트 표시
        for i in range(self.n_agents):
            pos = tuple(self.agent_positions[i])
            grid[pos] = f'A{i}'
            
        # 목표 표시
        for i in range(self.n_agents):
            pos = tuple(self.goal_positions[i])
            if not np.array_equal(self.agent_positions[i], self.goal_positions[i]):
                grid[pos] = f'G{i}'
                
        # 출력
        for row in grid:
            print(' '.join(row))
        print()