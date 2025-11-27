# iris_try -> iris-transformer (proposed)

这是 iris_try 的一个重构分支（iris-transformer），目标是把项目改造为 IRIS 风格的视觉+transformer pipeline（简化版），包含：
- 环境（Tetris-like）并渲染为图像帧
- 小型 CNN 视觉 encoder
- Transformer backbone（处理帧序列）
- DQN 风格离散动作策略
- 可配置的 training/trainer/replay

快速运行：
1. 安装依赖：
   pip install -r requirements.txt
2. 运行训练：
   python -m src.main --config config/config.yaml

注意：
- 这是一个起点（简化实现）。后续可以逐步引入更接近 IRIS 的 tokenizers（如 VQ-VAE）、并行采集、优先回放、更多训练技巧等。
