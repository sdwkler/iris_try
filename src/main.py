#!/usr/bin/env python3
import argparse
import yaml
from src.utils import set_seed, ensure_dir
from src.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    ensure_dir(cfg.get("output_dir", "outputs"))
    set_seed(cfg.get("seed", 0))
    trainer = Trainer(cfg)
    if cfg.get("mode", "train") == "train":
        trainer.train()
    else:
        trainer.evaluate(episodes=cfg.get("eval_episodes", 10))

if __name__ == "__main__":
    main()