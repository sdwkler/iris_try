import random
import numpy as np
import torch
import torch.optim as optim
from omegaconf import DictConfig
from pathlib import Path
import shutil
from PIL import Image
from typing import Dict, Union



class SimpleNamespace:
    """简单的命名空间类，用于存储损失等数据"""
    def __init__(self,** kwargs):
        self.__dict__.update(kwargs)


def set_seed(seed: int) -> None:
    """设置随机种子，确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def configure_optimizer(model: torch.nn.Module, learning_rate: float, weight_decay: float = 0.0) -> torch.optim.Optimizer:
    """配置优化器，对不同参数应用不同的权重衰减"""
    # 分离出不需要权重衰减的参数（如偏置和LayerNorm权重）
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    return optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)


class EpisodeDirManager:
    """管理 episode 存储目录，自动清理旧的 episode"""
    def __init__(self, directory: Path, max_num_episodes: int):
        self.directory = directory
        self.max_num_episodes = max_num_episodes
        self.directory.mkdir(parents=True, exist_ok=True)

    def clean_old_episodes(self) -> None:
        """清理旧的 episode，只保留最新的 max_num_episodes 个"""
        episodes = sorted(self.directory.glob('episode_*'), key=lambda x: int(x.stem.split('_')[1]))
        if len(episodes) > self.max_num_episodes:
            for episode in episodes[:-self.max_num_episodes]:
                if episode.is_dir():
                    shutil.rmtree(episode)
                else:
                    episode.unlink()

    def get_new_episode_dir(self, epoch: int) -> Path:
        """获取新的 episode 存储目录"""
        episode_id = len(list(self.directory.glob('episode_*')))
        episode_dir = self.directory / f'episode_{epoch}_{episode_id}'
        episode_dir.mkdir(exist_ok=True)
        self.clean_old_episodes()
        return episode_dir

def make_reconstructions_from_batch(
    batch: Dict[str, torch.Tensor],
    tokenizer: torch.nn.Module,
    save_dir: Union[str, Path],
    num_samples: int = 4,
    prefix: str = "recon"
) -> None:
    """
    从数据批次中生成原始图像和重建图像并保存，用于可视化Tokenizer的重建效果
    
    Args:
        batch: 包含观测数据的字典，必须包含 'observations' 键，值为形状 (B, H, W, C) 的张量/数组
        tokenizer: 训练好的Tokenizer实例，需实现 encode_decode 方法
        save_dir: 保存图像的目录路径
        num_samples: 从批次中抽取的样本数量（默认4个）
        prefix: 保存文件的前缀，用于区分不同阶段的重建结果（如训练/测试）
    """
    # 处理保存目录
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在，不存在则创建
    
    # 从批次中提取观测数据（取前num_samples个样本）
    observations = batch['observations'][:num_samples]  # 形状: (num_samples, H, W, C)
    
    # 确保观测数据是numpy数组（处理可能的torch张量情况）
    if isinstance(observations, torch.Tensor):
        observations = observations.detach().cpu().numpy()
    
    # 获取模型设备（确保输入与模型在同一设备）
    device = next(tokenizer.parameters()).device
    
    # 使用Tokenizer进行编码和解码，生成重建图像
    with torch.no_grad():  # 关闭梯度计算，节省内存
        # 将观测数据转换为张量并移动到模型设备
        obs_tensor = torch.tensor(
            observations,
            device=device,
            dtype=torch.float32
        )
        # 调用Tokenizer的encode_decode方法生成重建结果
        reconstructions = tokenizer.encode_decode(
            obs_tensor,
            should_preprocess=True,   # 启用预处理（归一化+维度转换）
            should_postprocess=True   # 启用后处理（维度转换+转回0-255）
        )
    
    # 确保重建结果是numpy数组（处理可能的torch张量情况）
    if isinstance(reconstructions, torch.Tensor):
        reconstructions = reconstructions.detach().cpu().numpy()
    
    # 保存原始图像和重建图像
    for i in range(num_samples):
        # 保存原始图像
        orig_img = Image.fromarray(observations[i].astype(np.uint8))
        orig_img.save(save_dir / f"{prefix}_orig_{i}.png")
        
        # 保存重建图像
        recon_img = Image.fromarray(reconstructions[i].astype(np.uint8))
        recon_img.save(save_dir / f"{prefix}_recon_{i}.png")

