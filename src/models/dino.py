import torch
import torch.nn as nn
import timm
from PIL import Image
import torchvision.transforms as transforms

class DINOv2FeatureExtractor(nn.Module):
    def __init__(self, model_name="vit_small_patch14_dinov2.lvd142m", input_size=(518, 518)):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.input_size = input_size
        self.transform = transforms.Compose([
            transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # 冻结DINOv2参数
        for param in self.model.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        """
        x: (batch_size, seq_len, 3, H, W) 或 (batch_size, 3, H, W)
        返回: (batch_size, seq_len, embedding_dim) 或 (batch_size, embedding_dim)
        """
        original_shape = x.shape
        batch_size = original_shape[0]
        
        # 处理序列输入
        if len(original_shape) == 5:  # 包含时序维度
            seq_len = original_shape[1]
            x = x.reshape(-1, *original_shape[2:])  # (batch_size*seq_len, 3, H, W)
        
        # 特征提取
        features = self.model(x)
        
        # 恢复序列维度
        if len(original_shape) == 5:
            features = features.reshape(batch_size, seq_len, -1)
            
        return features
    
    def preprocess(self, obs):
        """处理原始观测数据为模型输入格式"""
        if isinstance(obs, list):
            obs = [Image.fromarray(o.astype('uint8')).convert('RGB') for o in obs]
            return torch.stack([self.transform(o) for o in obs])
        else:
            img = Image.fromarray(obs.astype('uint8')).convert('RGB')
            return self.transform(img).unsqueeze(0)
