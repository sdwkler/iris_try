```markdown
# iris_MZ (Montezuma's Revenge prototype)

This branch contains a compact IRIS-like pipeline tuned for Montezuma's Revenge:
- VQ-VAE visual tokenizer (pretrainable)
- Transformer backbone policy (PPO)
- RND intrinsic reward for exploration
- Vectorized Atari envs (gymnasium + AutoROM)

Quickstart
1. Create virtual env and install:
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

2. Download ROMs (AutoROM):
   AutoROM --accept-license

3. Edit config/config.yaml if needed (num_envs, image_size, device).

4. Pretrain VQ (optional but recommended):
   python -m src.main --config config/config.yaml
   (set mode: pretrain_vq in config to run pretraining)

5. Train policy (PPO + RND):
   set mode: train and run:
   python -m src.main --config config/config.yaml

6. Watch agent:
   python tests/watch_agent.py --config config/config.yaml --model outputs/model_final.pth

Notes
- Montezuma is hard; use many envs (num_envs>=16) and a GPU for meaningful progress.
- Start with small settings for smoke tests (image_size [64,64], num_envs=4, total_updates small).
- This is a starting implementation; you can add more exploration modules (episodic count, hierarchy, imitation).
```