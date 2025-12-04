import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

class DINOv2FeatureExtractor(nn.Module):
    def __init__(self, model_name="dinov2_vits14", input_size=(224, 224)):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.eval()
        self.input_size = input_size
        self.transform = transforms.Compose([
            transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            feats = self.model.forward_features(x)
            # 如果直接是Tensor
            if isinstance(feats, torch.Tensor):
                return feats
            # 尝试常见key
            for key in ['cls_token_embedding', 'x_norm_clsstoken', 'x_norm_clstoken', 'last_hidden_state']:
                if key in feats:
                    return feats[key]
            # fallback: 返回feat第一个维度
            return list(feats.values())[0]

    def preprocess(self, obs):
        if isinstance(obs, list):
            obs = [Image.fromarray(o.astype('uint8')).convert('RGB') for o in obs]
            return torch.stack([self.transform(o) for o in obs])
        else:
            img = Image.fromarray(obs.astype('uint8')).convert('RGB')
            return self.transform(img)