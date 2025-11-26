import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

class Agent(nn.Module):
    def __init__(self, tokenizer: nn.Module, world_model: nn.Module, actor_critic: nn.Module):
        super().__init__()
        self.tokenizer = tokenizer
        self.world_model = world_model
        self.actor_critic = actor_critic

    def act(self, obs: torch.Tensor, hidden_state: Optional[Dict[str, torch.Tensor]] = None, 
            should_sample: bool = True, temperature: float = 1.0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        生成动作
        :param obs: 观测数据 [B, C, H, W] 或 [B, T, C, H, W]
        :param hidden_state: 隐藏状态
        :param should_sample: 是否采样（否则取argmax）
        :param temperature: 采样温度
        :return: 动作和新的隐藏状态
        """
        # 处理观测
        with torch.no_grad():
            # 编码观测
            obs_tokens = self.tokenizer.encode(obs, should_preprocess=True)
            
            # 初始化隐藏状态（如果需要）
            if hidden_state is None:
                hidden_state = self.world_model.initial_state(obs.shape[0])
            
            # 预热世界模型
            hidden_state = self.world_model.burn_in(obs_tokens, hidden_state)
            
            # 准备AC输入
            ac_input = hidden_state['hidden'][:, -1:]  # 取最后一个状态
            
            # 获取动作分布
            ac_output = self.actor_critic(ac_input)
            logits = ac_output.logits_actions[:, -1]  # [B, act_vocab_size]
            
            # 采样或取最大动作
            if should_sample and temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                action = torch.distributions.Categorical(probs).sample()
            else:
                action = torch.argmax(logits, dim=-1)
                
            return action, hidden_state

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """兼容接口，直接返回动作（用于推理）"""
        action, _ = self.act(obs, should_sample=False)
        return action

    def load(self, path_to_checkpoint: str, load_tokenizer: bool = True, 
             load_world_model: bool = True, load_actor_critic: bool = True, 
             device: torch.device = torch.device('cpu')) -> None:
        """加载模型权重"""
        checkpoint = torch.load(path_to_checkpoint, map_location=device)
        
        if load_tokenizer and 'tokenizer' in checkpoint:
            self.tokenizer.load_state_dict(checkpoint['tokenizer'])
            
        if load_world_model and 'world_model' in checkpoint:
            self.world_model.load_state_dict(checkpoint['world_model'])
            
        if load_actor_critic and 'actor_critic' in checkpoint:
            self.actor_critic.load_state_dict(checkpoint['actor_critic'])
