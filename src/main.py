#!/usr/bin/env python3
import argparse
import yaml
from src.utils import set_seed, ensure_dir, get_device
from src.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    ensure_dir(cfg.get("output_dir","outputs"))
    ensure_dir(cfg.get("logging", {}).get("video_dir", "outputs/videos"))
    set_seed(cfg.get("seed", 0))
    device = get_device(cfg.get("device","auto"))
    print("Using device:", device)
    trainer = Trainer(cfg, device=device)
    mode = cfg.get("mode","train")
    if mode == "pretrain_vq":
        trainer.pretrain_vq()
    elif mode == "train":
        trainer.train()
    elif mode == "eval":
        trainer.evaluate(episodes=cfg.get("logging", {}).get("eval_episodes", 5))
    else:
        raise ValueError("Unknown mode: " + str(mode))

if __name__ == "__main__":
    main()