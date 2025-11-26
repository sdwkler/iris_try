from collections import defaultdict
from functools import partial
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, Optional, Tuple

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import gymnasium as gym
from gymnasium.wrappers import FrameStack

from agent import Agent
from collector import Collector
from envs.base import SingleProcessEnv, MultiProcessEnv
from episode import Episode
from utils import make_reconstructions_from_batch
from models.actor_critic import ActorCritic, ActorCriticOutput
from models.world_model import WorldModel, LossOutput
from utils import configure_optimizer, EpisodeDirManager, set_seed

class Trainer:
    def __init__(self, cfg: DictConfig) -> None:
        # 初始化wandb
        self.wandb_run = wandb.init(
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
            resume="allow",** cfg.wandb
        )

        if cfg.common.seed is not None:
            set_seed(cfg.common.seed)

        self.cfg = cfg
        self.start_epoch = 1
        self.device = torch.device(cfg.common.device if torch.cuda.is_available() else "cpu")

        # 初始化目录
        self.ckpt_dir = Path(cfg.common.checkpoint_dir) if "checkpoint_dir" in cfg.common else Path('checkpoints')
        self.media_dir = Path('media')
        self.episode_dir = self.media_dir / 'episodes'
        self.reconstructions_dir = self.media_dir / 'reconstructions'

        # 确保目录存在
        self.ckpt_dir.mkdir(exist_ok=True, parents=True)
        self.media_dir.mkdir(exist_ok=True, parents=True)
        self.episode_dir.mkdir(exist_ok=True, parents=True)
        self.reconstructions_dir.mkdir(exist_ok=True, parents=True)

        #  episode管理器
        episode_manager_train = EpisodeDirManager(
            self.episode_dir / 'train', 
            max_num_episodes=cfg.collection.train.num_episodes_to_save
        )
        episode_manager_test = EpisodeDirManager(
            self.episode_dir / 'test', 
            max_num_episodes=cfg.collection.test.num_episodes_to_save
        )
        self.episode_manager_imagination = EpisodeDirManager(
            self.episode_dir / 'imagination', 
            max_num_episodes=cfg.evaluation.actor_critic.num_episodes_to_save
        )

        # 创建环境函数
        def create_env(cfg_env, num_envs):
            def env_fn():
                env = instantiate(cfg_env)
                # 确保帧堆叠正确应用
                if not any(isinstance(w, FrameStack) for w in env.wrappers):
                    env = FrameStack(env, num_stack=4)
                return env
            
            return MultiProcessEnv(env_fn, num_envs) if num_envs > 1 else SingleProcessEnv(env_fn)

        self.train_env = None
        self.test_env = None
        self.train_dataset = None
        self.test_dataset = None
        self.train_collector = None
        self.test_collector = None
        
        if self.cfg.training.should:
            self.train_env = create_env(cfg.env.train, cfg.collection.train.num_envs)
            self.train_dataset = instantiate(cfg.datasets.train)
            self.train_collector = Collector(
                self.train_env, 
                self.train_dataset, 
                episode_manager_train
            )

        if self.cfg.evaluation.should:
            self.test_env = create_env(cfg.env.test, cfg.collection.test.num_envs)
            self.test_dataset = instantiate(cfg.datasets.test)
            self.test_collector = Collector(
                self.test_env, 
                self.test_dataset, 
                episode_manager_test
            )

        # 确保至少有一个环境可用
        assert self.cfg.training.should or self.cfg.evaluation.should, "至少需要启用训练或评估"
        env = self.train_env if self.cfg.training.should else self.test_env

        # 初始化模型
        self.tokenizer = instantiate(cfg.tokenizer)
        self.world_model = WorldModel(
            obs_vocab_size=self.tokenizer.vocab_size, 
            act_vocab_size=env.num_actions, 
            config=instantiate(cfg.world_model.config)
        )
        self.actor_critic = ActorCritic(
            act_vocab_size=env.num_actions, 
            use_original_obs=cfg.actor_critic.use_original_obs
        )
        self.agent = Agent(
            self.tokenizer, 
            self.world_model, 
            self.actor_critic
        ).to(self.device)

        # 打印模型参数数量
        print(f'{sum(p.numel() for p in self.agent.tokenizer.parameters())} parameters in agent.tokenizer')
        print(f'{sum(p.numel() for p in self.agent.world_model.parameters())} parameters in agent.world_model')
        print(f'{sum(p.numel() for p in self.agent.actor_critic.parameters())} parameters in agent.actor_critic')

        # 初始化优化器
        self.optimizer_tokenizer = torch.optim.Adam(
            self.agent.tokenizer.parameters(), 
            lr=cfg.training.learning_rate
        )
        self.optimizer_world_model = configure_optimizer(
            self.agent.world_model, 
            cfg.training.learning_rate, 
            cfg.training.world_model.weight_decay
        )
        self.optimizer_actor_critic = torch.optim.Adam(
            self.agent.actor_critic.parameters(), 
            lr=cfg.training.learning_rate
        )

        # 加载检查点
        if cfg.initialization.path_to_checkpoint is not None:
            self.load_checkpoint(cfg.initialization.path_to_checkpoint)

        if cfg.common.resume:
            self.load_latest_checkpoint()

    def run(self) -> None:
        for epoch in range(self.start_epoch, 1 + self.cfg.common.epochs):
            print(f"\nEpoch {epoch} / {self.cfg.common.epochs}\n")
            start_time = time.time()
            to_log = []

            # 收集训练数据
            if self.cfg.training.should:
                if epoch <= self.cfg.collection.train.stop_after_epochs:
                    collection_metrics = self.train_collector.collect(
                        self.agent, 
                        epoch, 
                        **self.cfg.collection.train.config
                    )
                    to_log.extend(collection_metrics)
                
                # 训练智能体
                train_metrics = self.train_agent(epoch)
                to_log.extend(train_metrics)

            # 评估
            if self.cfg.evaluation.should and (epoch % self.cfg.evaluation.every == 0):
                if self.test_dataset is not None:
                    self.test_dataset.clear()
                eval_collection_metrics = self.test_collector.collect(
                    self.agent, 
                    epoch,** self.cfg.collection.test.config
                )
                to_log.extend(eval_collection_metrics)
                
                eval_metrics = self.eval_agent(epoch)
                to_log.extend(eval_metrics)

            # 保存检查点
            if self.cfg.training.should and (epoch % self.cfg.common.checkpoint_every == 0):
                self.save_checkpoint(epoch)

            # 记录时间
            to_log.append({
                'duration': (time.time() - start_time) / 3600,
                'epoch': epoch
            })
            
            # 日志记录
            for metrics in to_log:
                wandb.log(metrics)

        self.finish()

    def train_agent(self, epoch: int) -> list:
        self.agent.train()
        metrics_tokenizer, metrics_world_model, metrics_actor_critic = {}, {}, {}

        cfg_tokenizer = self.cfg.training.tokenizer
        cfg_world_model = self.cfg.training.world_model
        cfg_actor_critic = self.cfg.training.actor_critic

        # 训练tokenizer
        if epoch > cfg_tokenizer.start_after_epochs and self.train_dataset.get_total_steps() > 0:
            metrics_tokenizer = self.train_component(
                self.agent.tokenizer, 
                self.optimizer_tokenizer, 
                sequence_length=1, 
                sample_from_start=True, 
                loss_weights=self.cfg.tokenizer.loss_weights,
                **cfg_tokenizer
            )
        self.agent.tokenizer.eval()

        # 训练世界模型
        if epoch > cfg_world_model.start_after_epochs and self.train_dataset.get_total_steps() > 0:
            metrics_world_model = self.train_component(
                self.agent.world_model, 
                self.optimizer_world_model, 
                sequence_length=self.cfg.common.sequence_length, 
                sample_from_start=True, 
                tokenizer=self.agent.tokenizer,** cfg_world_model
            )
        self.agent.world_model.eval()

        # 训练actor-critic
        if epoch > cfg_actor_critic.start_after_epochs and self.train_dataset.get_total_steps() > 0:
            metrics_actor_critic = self.train_component(
                self.agent.actor_critic, 
                self.optimizer_actor_critic, 
                sequence_length=1 + self.cfg.training.actor_critic.burn_in, 
                sample_from_start=False, 
                tokenizer=self.agent.tokenizer, 
                world_model=self.agent.world_model,
                **cfg_actor_critic
            )
        self.agent.actor_critic.eval()

        return [{** metrics_tokenizer, **metrics_world_model,** metrics_actor_critic, 'epoch': epoch}]

    def train_component(self, component: nn.Module, optimizer: torch.optim.Optimizer, 
                       steps_per_epoch: int, batch_num_samples: int, grad_acc_steps: int, 
                       max_grad_norm: Optional[float], sequence_length: int, 
                       sample_from_start: bool,** kwargs_loss: Any) -> Dict[str, float]:
        loss_total_epoch = 0.0
        intermediate_losses = defaultdict(float)

        component.train()
        for _ in tqdm(range(steps_per_epoch), desc=f"Training {component.__class__.__name__}", file=sys.stdout):
            optimizer.zero_grad()
            for _ in range(grad_acc_steps):
                try:
                    batch = self.train_dataset.sample_batch(
                        batch_num_samples, 
                        sequence_length, 
                        sample_from_start
                    )
                    batch = self._to_device(batch)
                    
                    losses = component.compute_loss(batch, **kwargs_loss) / grad_acc_steps
                    loss_total_step = losses.loss_total
                    loss_total_step.backward()
                    
                    loss_total_epoch += loss_total_step.item() / steps_per_epoch
                    
                    for loss_name, loss_value in losses.intermediate_losses.items():
                        intermediate_losses[f"{component.__class__.__name__}/train/{loss_name}"] += loss_value / steps_per_epoch
                except Exception as e:
                    print(f"训练批次出错: {e}")
                    continue

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(component.parameters(), max_grad_norm)

            optimizer.step()

        metrics = {
            f'{component.__class__.__name__}/train/total_loss': loss_total_epoch,
            **intermediate_losses
        }
        return metrics

    @torch.no_grad()
    def eval_agent(self, epoch: int) -> list:
        self.agent.eval()
        metrics = []

        cfg_tokenizer = self.cfg.evaluation.tokenizer
        cfg_world_model = self.cfg.evaluation.world_model
        cfg_actor_critic = self.cfg.evaluation.actor_critic

        # 评估tokenizer
        if epoch > cfg_tokenizer.start_after_epochs and len(self.test_dataset) > 0:
            tokenizer_metrics = self.eval_component(
                self.agent.tokenizer, 
                self.test_dataset,
                cfg_tokenizer.batch_num_samples, 
                sequence_length=1
            )
            metrics.append(tokenizer_metrics)

        # 评估世界模型
        if epoch > cfg_world_model.start_after_epochs and len(self.test_dataset) > 0:
            world_model_metrics = self.eval_component(
                self.agent.world_model, 
                self.test_dataset,
                cfg_world_model.batch_num_samples, 
                sequence_length=self.cfg.common.sequence_length,
                tokenizer=self.agent.tokenizer
            )
            metrics.append(world_model_metrics)

        # 评估actor-critic
        if epoch > cfg_actor_critic.start_after_epochs and len(self.test_dataset) > 0:
            actor_critic_metrics = self.eval_component(
                self.agent.actor_critic, 
                self.test_dataset,
                cfg_actor_critic.batch_num_samples,
                sequence_length=1 + self.cfg.training.actor_critic.burn_in,
                tokenizer=self.agent.tokenizer,
                world_model=self.agent.world_model
            )
            metrics.append(actor_critic_metrics)

        return [{** m, 'epoch': epoch} for m in metrics]

    @torch.no_grad()
    def eval_component(self, component: nn.Module, dataset, batch_num_samples: int, 
                      sequence_length: int,** kwargs) -> Dict[str, float]:
        component.eval()
        loss_total = 0.0
        intermediate_losses = defaultdict(float)
        steps = self.cfg.evaluation.steps_per_component

        for _ in tqdm(range(steps), desc=f"Evaluating {component.__class__.__name__}", file=sys.stdout):
            batch = dataset.sample_batch(batch_num_samples, sequence_length, sample_from_start=True)
            batch = self._to_device(batch)
            
            losses = component.compute_loss(batch, **kwargs)
            loss_total += losses.loss_total.item() / steps
            
            for loss_name, loss_value in losses.intermediate_losses.items():
                intermediate_losses[f"{component.__class__.__name__}/eval/{loss_name}"] += loss_value.item() / steps

        return {
            f'{component.__class__.__name__}/eval/total_loss': loss_total,
            **intermediate_losses
        }

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """将批次数据移动到指定设备"""
        return {k: v.to(self.device) for k, v in batch.items()}

    def save_checkpoint(self, epoch: int):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_tokenizer': self.optimizer_tokenizer.state_dict(),
            'optimizer_world_model': self.optimizer_world_model.state_dict(),
            'optimizer_actor_critic': self.optimizer_actor_critic.state_dict(),
            'wandb_run_id': self.wandb_run.id
        }
        torch.save(checkpoint, self.ckpt_dir / f'checkpoint_epoch_{epoch}.pt')

    def load_checkpoint(self, path: str):
        """加载指定检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(checkpoint['agent_state_dict'])
        self.optimizer_tokenizer.load_state_dict(checkpoint['optimizer_tokenizer'])
        self.optimizer_world_model.load_state_dict(checkpoint['optimizer_world_model'])
        self.optimizer_actor_critic.load_state_dict(checkpoint['optimizer_actor_critic'])
        self.start_epoch = checkpoint['epoch'] + 1

    def load_latest_checkpoint(self):
        """加载最新检查点"""
        checkpoints = list(self.ckpt_dir.glob('checkpoint_epoch_*.pt'))
        if not checkpoints:
            print("没有找到检查点，从头开始训练")
            return
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
        print(f"加载最新检查点: {latest_checkpoint}")
        self.load_checkpoint(str(latest_checkpoint))

    def finish(self):
        """结束训练"""
        if self.train_env is not None:
            self.train_env.close()
        if self.test_env is not None:
            self.test_env.close()
        wandb.finish()
