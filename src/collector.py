import torch
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
import wandb

from agent import Agent
from episode import Episode
from envs.base import Env
from utils import EpisodeDirManager
from collections import defaultdict

class Collector:
    def __init__(self, env: Env, dataset, episode_manager: EpisodeDirManager):
        self.env = env
        self.dataset = dataset
        self.episode_manager = episode_manager

    def collect(self, agent: Agent, epoch: int, epsilon: float = 0.01, should_sample: bool = True, 
                temperature: float = 1.0, num_steps: int = None, num_episodes: int = None, 
                burn_in: int = 0, **kwargs) -> List[Dict[str, float]]:
        """收集环境交互数据"""
        assert (num_steps is not None) or (num_episodes is not None), "必须指定收集的步数或 episodes 数"
        
        agent.eval()
        metrics = defaultdict(list)
        episodes_collected = 0
        steps_collected = 0
        
        # 初始化环境
        obs = self.env.reset()
        episodes = [Episode() for _ in range(self.env.num_envs)]
        
        with tqdm(total=num_steps or num_episodes, desc="Collecting data") as pbar:
            while (num_episodes is None or episodes_collected < num_episodes) and \
                  (num_steps is None or steps_collected < num_steps):
                
                # 选择动作（带探索）
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, device=agent.device, dtype=torch.float32).unsqueeze(1)
                    action = agent.act(obs_tensor, should_sample=should_sample, temperature=temperature).cpu().numpy()
                
                # 探索：以 epsilon 概率随机选择动作
                if epsilon > 0:
                    for i in range(self.env.num_envs):
                        if np.random.rand() < epsilon:
                            action[i] = np.random.randint(self.env.num_actions)
                
                # 执行动作
                next_obs, reward, done, info = self.env.step(action)
                
                # 记录数据
                for i in range(self.env.num_envs):
                    episodes[i].add_step(
                        observation=obs[i],
                        action=action[i],
                        reward=reward[i],
                        done=done[i]
                    )
                    
                    steps_collected += 1
                    metrics['step_reward'].append(reward[i])
                    
                    if done[i]:
                        # 保存 episode
                        episodes_collected += 1
                        metrics['episode_length'].append(len(episodes[i]))
                        metrics['episode_reward'].append(sum(episodes[i].rewards))
                        
                        # 保存部分 episode 用于可视化
                        if episodes_collected % (max(1, num_episodes // self.episode_manager.max_num_episodes)) == 0:
                            episode_dir = self.episode_manager.get_new_episode_dir(epoch)
                            episodes[i].save(episode_dir)
                        
                        # 添加到数据集
                        self.dataset.add_episode(episodes[i])
                        
                        # 重置 episode
                        episodes[i] = Episode()
                        
                        pbar.update(1)
                
                obs = next_obs
        
        # 计算统计指标
        stats = {
            'collection/avg_step_reward': np.mean(metrics['step_reward']),
            'collection/avg_episode_length': np.mean(metrics['episode_length']) if metrics['episode_length'] else 0,
            'collection/avg_episode_reward': np.mean(metrics['episode_reward']) if metrics['episode_reward'] else 0,
            'collection/episodes_collected': episodes_collected,
            'collection/steps_collected': steps_collected
        }
        
        return [stats]
