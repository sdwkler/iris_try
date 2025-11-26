import hydra
from omegaconf import DictConfig
from trainer import Trainer
from omegaconf import OmegaConf

@hydra.main(config_path="../config", config_name="trainer", version_base="1.1")
def main(cfg: DictConfig):
    # 确保配置可被正确解析
    OmegaConf.resolve(cfg)
    trainer = Trainer(cfg)
    trainer.run()

if __name__ == "__main__":
    main()
