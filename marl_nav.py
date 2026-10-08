"""
Multi-robot navigation on an 8x8 chessboard: CNN policy + PPO (parameter sharing).

Run:  python marl_nav.py --robots 2 --updates 300
Saves: policy.pt

Cells are (col, row) with col = file a..h (x), row = rank 1..8 (y).
Actions: 0 stay, 1 +y (up), 2 -y (down), 3 +x (right), 4 -x (left)
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

DELTAS = np.array([(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)])
N_ACTIONS = len(DELTAS)
N_CHANNELS = 4


def build_obs(pos, goals, size):
    """Per-robot observation, shape (n, 4, size, size).
    ch0: my position   ch1: other robots   ch2: my goal   ch3: other goals
    Used by BOTH training and MuJoCo deployment so they always match."""
    n = len(pos)
    obs = np.zeros((n, N_CHANNELS, size, size), dtype=np.float32)
    for i in range(n):
        obs[i, 0, pos[i][1], pos[i][0]] = 1
        obs[i, 2, goals[i][1], goals[i][0]] = 1
        for j in range(n):
            if j != i:
                obs[i, 1, pos[j][1], pos[j][0]] = 1
                obs[i, 3, goals[j][1], goals[j][0]] = 1
    return obs


class GridNavEnv:
    def __init__(self, n_robots=2, size=8, max_steps=64, seed=None):
        self.n, self.size, self.max_steps = n_robots, size, max_steps
        self.rng = np.random.default_rng(seed)

    def reset(self):
        cells = self.rng.choice(self.size ** 2, size=2 * self.n, replace=False)
        xy = [(c % self.size, c // self.size) for c in cells]
        self.pos = np.array(xy[: self.n])
        self.goals = np.array(xy[self.n:])
        self.t = 0
        return build_obs(self.pos, self.goals, self.size)

    def _dist(self, pos):
        return np.abs(pos - self.goals).sum(axis=1)

    def step(self, actions):
        n = self.n
        old_pos, old_dist = self.pos.copy(), self._dist(self.pos)
        old_on = old_dist == 0

        prop = np.clip(old_pos + DELTAS[actions], 0, self.size - 1)
        blocked = np.zeros(n, dtype=bool)
        blocked |= (actions != 0) & (prop == old_pos).all(axis=1)  # walked into wall

        # Resolve conflicts until stable: same target cell, moving into a
        # robot that stays, or swapping places -> the mover stays put.
        changed = True
        while changed:
            changed = False
            for i in range(n):
                if (prop[i] == old_pos[i]).all():
                    continue
                for j in range(n):
                    if j == i:
                        continue
                    same_cell = (prop[i] == prop[j]).all()
                    swap = (prop[i] == old_pos[j]).all() and (prop[j] == old_pos[i]).all()
                    if same_cell or swap:
                        prop[i] = old_pos[i]
                        blocked[i] = True
                        changed = True
                        break

        self.pos = prop
        self.t += 1
        new_dist = self._dist(self.pos)
        on = new_dist == 0

        reward = 0.1 * (old_dist - new_dist).astype(np.float32)   # progress shaping
        reward += 1.0 * (on.astype(np.float32) - old_on.astype(np.float32))  # arrive/leave
        reward -= 0.02 * (~on)                                    # time pressure
        reward -= 0.3 * blocked                                   # collision / wall
        all_done = bool(on.all())
        if all_done:
            reward += 2.0                                         # team bonus
        done = all_done or self.t >= self.max_steps
        return (build_obs(self.pos, self.goals, self.size), reward, done,
                {"success": all_done})


class ActorCritic(nn.Module):
    def __init__(self, size=8):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(N_CHANNELS, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * size * size, 256), nn.ReLU(),
        )
        self.actor = nn.Linear(256, N_ACTIONS)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        h = self.cnn(x)
        return self.actor(h), self.critic(h).squeeze(-1)


def train(n_robots, size, updates, rollout=1024, epochs=4, mb=256,
          gamma=0.98, lam=0.95, clip=0.2, lr=3e-4, ent=0.01, out="policy.pt"):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    env = GridNavEnv(n_robots, size)
    net = ActorCritic(size).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    obs = env.reset()
    ep_ok, ep_n = 0, 0

    for u in range(updates):
        O = np.zeros((rollout, n_robots, N_CHANNELS, size, size), np.float32)
        A = np.zeros((rollout, n_robots), np.int64)
        LP = np.zeros((rollout, n_robots), np.float32)
        V = np.zeros((rollout, n_robots), np.float32)
        R = np.zeros((rollout, n_robots), np.float32)
        D = np.zeros(rollout, np.float32)

        for t in range(rollout):
            o = torch.as_tensor(obs, device=dev)
            with torch.no_grad():
                logits, v = net(o)
                dist = Categorical(logits=logits)
                a = dist.sample()
                lp = dist.log_prob(a)
            nobs, r, done, info = env.step(a.cpu().numpy())
            O[t], A[t], LP[t], V[t], R[t], D[t] = obs, a.cpu().numpy(), lp.cpu().numpy(), v.cpu().numpy(), r, done
            if done:
                ep_n += 1
                ep_ok += info["success"]
                obs = env.reset()
            else:
                obs = nobs

        with torch.no_grad():
            _, last_v = net(torch.as_tensor(obs, device=dev))
        last_v = last_v.cpu().numpy()

        adv = np.zeros_like(R)
        gae = np.zeros(n_robots, np.float32)
        for t in reversed(range(rollout)):
            nv = last_v if t == rollout - 1 else V[t + 1]
            nonterm = 1.0 - D[t]
            delta = R[t] + gamma * nv * nonterm - V[t]
            gae = delta + gamma * lam * nonterm * gae
            adv[t] = gae
        ret = adv + V

        flat = lambda x: x.reshape(rollout * n_robots, *x.shape[2:])
        bO = torch.as_tensor(flat(O), device=dev)
        bA = torch.as_tensor(flat(A), device=dev)
        bLP = torch.as_tensor(flat(LP), device=dev)
        bAdv = torch.as_tensor(flat(adv), device=dev)
        bRet = torch.as_tensor(flat(ret), device=dev)
        bAdv = (bAdv - bAdv.mean()) / (bAdv.std() + 1e-8)

        N = bO.shape[0]
        for _ in range(epochs):
            perm = torch.randperm(N, device=dev)
            for s in range(0, N, mb):
                idx = perm[s:s + mb]
                logits, v = net(bO[idx])
                dist = Categorical(logits=logits)
                ratio = torch.exp(dist.log_prob(bA[idx]) - bLP[idx])
                pg = -torch.min(ratio * bAdv[idx],
                                torch.clamp(ratio, 1 - clip, 1 + clip) * bAdv[idx]).mean()
                vf = 0.5 * (v - bRet[idx]).pow(2).mean()
                loss = pg + 0.5 * vf - ent * dist.entropy().mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

        if (u + 1) % 10 == 0:
            rate = ep_ok / max(ep_n, 1)
            print(f"update {u+1}/{updates}  episodes {ep_n}  success rate {rate:.2f}")
            ep_ok, ep_n = 0, 0
            torch.save({"model": net.state_dict(), "size": size, "robots": n_robots}, out)

    torch.save({"model": net.state_dict(), "size": size, "robots": n_robots}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--robots", type=int, default=2)
    p.add_argument("--size", type=int, default=8)
    p.add_argument("--updates", type=int, default=300)
    args = p.parse_args()
    train(args.robots, args.size, args.updates)
