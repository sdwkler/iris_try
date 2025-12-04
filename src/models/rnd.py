import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class RNDModel(nn.Module):
    def __init__(self, input_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
    
    def forward(self, x):
        return self.net(x)

class RND(nn.Module):  # 确保 RND 继承自 nn.Module（关键！）
    def __init__(self, input_dim, device, predictor_hidden=256, target_hidden=256, lr=1e-4):
        super(RND, self).__init__()  # 必须调用父类构造函数
        self.device = device
        self.input_dim = input_dim

        # 目标网络（不更新）
        self.target_net = nn.Sequential(
            nn.Linear(input_dim, target_hidden),
            nn.ReLU(),
            nn.Linear(target_hidden, target_hidden),
            nn.ReLU(),
            nn.Linear(target_hidden, input_dim)
        ).to(device)
        
        # 预测网络（更新）
        self.predictor_net = nn.Sequential(
            nn.Linear(input_dim, predictor_hidden),
            nn.ReLU(),
            nn.Linear(predictor_hidden, predictor_hidden),
            nn.ReLU(),
            nn.Linear(predictor_hidden, input_dim)
        ).to(device)

        # 冻结目标网络参数
        for param in self.target_net.parameters():
            param.requires_grad = False

        # 优化器
        self.optimizer = optim.Adam(self.predictor_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # 存储内在奖励（用于日志）
        self.intrinsic_rewards = None

    def compute_intrinsic(self, x):
        """计算内在奖励（预测误差）"""
        with torch.no_grad():
            target_output = self.target_net(x)
        predictor_output = self.predictor_net(x)
        # 计算每个样本的MSE误差（作为内在奖励）
        error = torch.mean((predictor_output - target_output) ** 2, dim=1)
        self.intrinsic_rewards = error
        return error

    def update(self, x):
        """更新预测网络"""
        self.optimizer.zero_grad()
        target_output = self.target_net(x)
        predictor_output = self.predictor_net(x)
        loss = self.loss_fn(predictor_output, target_output)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    # ✅ 核心修复1：实现 state_dict() 方法
    def state_dict(self):
        return {
            'predictor_net': self.predictor_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'target_net': self.target_net.state_dict()  # 可选：保存目标网络
        }

    # ✅ 核心修复2：实现 load_state_dict() 方法（配套加载）
    def load_state_dict(self, state_dict):
        self.predictor_net.load_state_dict(state_dict['predictor_net'])
        self.target_net.load_state_dict(state_dict['target_net'])
        self.optimizer.load_state_dict(state_dict['optimizer'])