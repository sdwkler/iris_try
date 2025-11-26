import random
from collections import deque, namedtuple
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from src.models import CNNEncoder, TransformerPolicy

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

class VisualDQNAgent:
    def __init__(self, cfg, obs_shape, action_n, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.action_n = action_n
        self.cfg = cfg
        # obs_shape: (seq_len, C, H, W)
        self.seq_len = obs_shape[0]
        c = obs_shape[1]
        self.encoder = CNNEncoder(in_channels=c, conv_channels=cfg.get("conv_channels", (32,64,128)), fc_dim=cfg.get("d_model", 256)).to(self.device)
        self.policy = TransformerPolicy(d_model=cfg.get("d_model",256),
                                        n_heads=cfg.get("n_heads",4),
                                        num_layers=cfg.get("num_layers",3),
                                        mlp_dim=cfg.get("mlp_dim",512),
                                        seq_len=self.seq_len,
                                        action_n=action_n,
                                        dropout=cfg.get("dropout",0.1)).to(self.device)
        self.target_encoder = CNNEncoder(in_channels=c, conv_channels=cfg.get("conv_channels", (32,64,128)), fc_dim=cfg.get("d_model", 256)).to(self.device)
        self.target_policy = TransformerPolicy(d_model=cfg.get("d_model",256),
                                               n_heads=cfg.get("n_heads",4),
                                               num_layers=cfg.get("num_layers",3),
                                               mlp_dim=cfg.get("mlp_dim",512),
                                               seq_len=self.seq_len,
                                               action_n=action_n,
                                               dropout=cfg.get("dropout",0.1)).to(self.device)
        # sync targets
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.target_policy.load_state_dict(self.policy.state_dict())

        params = list(self.encoder.parameters()) + list(self.policy.parameters())
        self.optimizer = optim.Adam(params, lr=cfg.get("lr", 1e-4))
        self.gamma = cfg.get("gamma", 0.99)
        self.eps_start = cfg.get("eps_start", 1.0)
        self.eps_end = cfg.get("eps_end", 0.05)
        self.eps_decay = cfg.get("eps_decay", 20000)
        self.steps_done = 0
        self.update_target_every = cfg.get("update_target_every", 1000)
        self.replay = ReplayBuffer(cfg.get("replay_size", 50000))
        self.batch_size = cfg.get("batch_size", 64)
        self.min_replay_size = cfg.get("min_replay_size", 1000)

    def select_action(self, state):  # state: numpy array (seq_len, C, H, W)
        eps = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if random.random() < eps:
            return random.randrange(self.action_n)
        else:
            self.encoder.eval(); self.policy.eval()
            with torch.no_grad():
                s = torch.FloatTensor(state).to(self.device)  # (seq_len, C, H, W)
                s = s.unsqueeze(0)  # (1, seq_len, C, H, W)
                # encode each frame
                b, seq, C, H, W = s.shape
                s = s.view(b*seq, C, H, W)
                z = self.encoder(s)  # (b*seq, d)
                z = z.view(b, seq, -1)  # (b, seq, d)
                q = self.policy(z)  # (b, action_n)
                return int(torch.argmax(q, dim=1).item())

    def push(self, state, action, reward, next_state, done):
        # store raw numpy arrays
        self.replay.push(state, action, reward, next_state, done)

    def optimize(self):
        if len(self.replay) < max(self.min_replay_size, self.batch_size):
            return None
        batch = self.replay.sample(self.batch_size)
        state = np.stack(batch.state)          # (B, seq, C, H, W)
        next_state = np.stack(batch.next_state)
        action = np.array(batch.action)
        reward = np.array(batch.reward, dtype=np.float32)
        done = np.array(batch.done, dtype=np.float32)

        # convert to tensors
        B, seq, C, H, W = state.shape
        state_t = torch.FloatTensor(state).to(self.device).view(B*seq, C, H, W)
        next_state_t = torch.FloatTensor(next_state).to(self.device).view(B*seq, C, H, W)

        # encode
        z = self.encoder(state_t)           # (B*seq, d)
        z = z.view(B, seq, -1)              # (B, seq, d)
        q_vals = self.policy(z)             # (B, action_n)

        with torch.no_grad():
            z_next = self.target_encoder(next_state_t)
            z_next = z_next.view(B, seq, -1)
            next_q = self.target_policy(z_next)
            max_next_q = next_q.max(1)[0].cpu().numpy()

        # target
        targets = reward + (1 - done) * self.gamma * max_next_q
        q_pred = q_vals.gather(1, torch.LongTensor(action).unsqueeze(1).to(self.device)).squeeze(1)

        loss = F.mse_loss(q_pred, torch.FloatTensor(targets).to(self.device))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.policy.parameters()), 1.0)
        self.optimizer.step()

        # update target
        if self.steps_done % self.update_target_every == 0:
            self.target_encoder.load_state_dict(self.encoder.state_dict())
            self.target_policy.load_state_dict(self.policy.state_dict())

        return float(loss.item())

    def save(self, path):
        torch.save({
            'encoder': self.encoder.state_dict(),
            'policy': self.policy.state_dict(),
            'target_encoder': self.target_encoder.state_dict(),
            'target_policy': self.target_policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done
        }, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(data['encoder'])
        self.policy.load_state_dict(data['policy'])
        self.target_encoder.load_state_dict(data['target_encoder'])
        self.target_policy.load_state_dict(data['target_policy'])
        self.optimizer.load_state_dict(data['optimizer'])
        self.steps_done = data.get('steps_done', 0)