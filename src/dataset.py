import torch
import psutil
import numpy as np
from typing import List, Dict, Optional, Tuple, Iterator
from dataclasses import dataclass
from omegaconf import DictConfig
from .episode import Episode


@dataclass
class DatasetConfig:
    """数据集配置类"""
    max_num_episodes: Optional[int] = None  # 最大存储Episode数（None表示无限制）
    max_ram_usage: Optional[str] = None     # 最大内存占用（如"30G"）
    name: str = "dataset"                   # 数据集名称


class EpisodesDataset:
    """IRIS原版基类：存储Episode并采样序列数据"""
    def __init__(self, config: DictConfig):
        self.config = DatasetConfig(
            max_num_episodes=config.get("max_num_episodes"),
            max_ram_usage=config.get("max_ram_usage"),
            name=config.get("name", "dataset")
        )
        self.episodes: List[Episode] = []  # 存储所有Episode
        self.device = torch.device("cpu")  # 默认CPU（可切换）

    def add_episode(self, episode: Episode):
        """添加Episode到数据集"""
        # 1. 检查最大Episode数限制
        if self.config.max_num_episodes is not None and len(self.episodes) >= self.config.max_num_episodes:
            self.episodes.pop(0)  # 移除最早的Episode
        
        # 2. 添加新Episode
        self.episodes.append(episode)

    def get_total_steps(self) -> int:
        """返回数据集总步数"""
        return sum(len(ep) for ep in self.episodes)

    def sample_episode(self) -> Episode:
        """随机采样一个Episode"""
        if len(self.episodes) == 0:
            raise ValueError("Dataset is empty!")
        return self.episodes[np.random.randint(0, len(self.episodes))]

    def sample_sequence(
        self,
        sequence_length: int,
        sample_from_start: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        核心方法：采样固定长度的序列（供模型训练）
        :param sequence_length: 序列长度（如64）
        :param sample_from_start: 是否从Episode开头采样（测试用）
        :return: 序列数据（obs/actions/rewards等）
        """
        # 1. 采样一个足够长的Episode
        episode = self.sample_episode()
        while len(episode) < sequence_length:
            episode = self.sample_episode()

        # 2. 确定采样起始位置
        if sample_from_start:
            start_idx = 0
        else:
            max_start_idx = len(episode) - sequence_length
            start_idx = np.random.randint(0, max_start_idx + 1)
        end_idx = start_idx + sequence_length

        # 3. 获取序列并转为tensor
        sequence = episode.get_sequence(start_idx, end_idx)
        for k, v in sequence.items():
            sequence[k] = v.to(self.device)
        return sequence

    def sample_batch(
        self,
        batch_size: int,
        sequence_length: int,
        sample_from_start: bool = False
    ) -> Dict[str, torch.Tensor]:
        """采样批次序列（训练核心）"""
        batch = {}
        for _ in range(batch_size):
            try:
                sequence = self.sample_sequence(sequence_length, sample_from_start)
                for k, v in sequence.items():
                    if k not in batch:
                        batch[k] = []
                    batch[k].append(v.unsqueeze(0))  # 添加batch维度
            except Exception as e:
                print(f"采样序列出错: {e}")
                continue
        
        # 合并batch
        for k in batch:
            batch[k] = torch.cat(batch[k], dim=0)
        return batch

    def clear(self):
        """清空数据集"""
        self.episodes = []

    def __len__(self) -> int:
        return len(self.episodes)


class EpisodesDatasetRamMonitoring(EpisodesDataset):
    """训练集子类：监控内存占用，避免爆内存"""
    def __init__(self, config: DictConfig):
        super().__init__(config)
        # 解析最大内存占用（如"30G" → 30*1024*1024*1024字节）
        if self.config.max_ram_usage is not None:
            self.max_ram_bytes = self._parse_ram_limit(self.config.max_ram_usage)
        else:
            self.max_ram_bytes = None

    def _parse_ram_limit(self, ram_str: str) -> int:
        """解析内存限制字符串（如"30G" → 字节数）"""
        ram_str = ram_str.strip().upper()
        if ram_str.endswith("G"):
            return int(ram_str[:-1]) * 1024**3
        elif ram_str.endswith("M"):
            return int(ram_str[:-1]) * 1024**2
        elif ram_str.endswith("K"):
            return int(ram_str[:-1]) * 1024
        else:
            return int(ram_str)

    def _get_ram_usage(self) -> int:
        """获取当前进程的内存占用（字节）"""
        process = psutil.Process()
        return process.memory_info().rss

    def add_episode(self, episode: Episode):
        """重写add_episode：添加前检查内存"""
        # 先添加新Episode
        self.episodes.append(episode)
        
        # 如果超出最大数量限制，移除最早的
        if self.config.max_num_episodes is not None and len(self.episodes) > self.config.max_num_episodes:
            self.episodes.pop(0)
        
        # 若内存超限，继续移除最早的Episode直到内存达标
        if self.max_ram_bytes is not None:
            while self._get_ram_usage() > self.max_ram_bytes and len(self.episodes) > 0:
                removed = self.episodes.pop(0)
                print(f"内存超限，移除最早的episode (剩余 {len(self.episodes)} 个)")
