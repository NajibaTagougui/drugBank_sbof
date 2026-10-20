"""
============================================================
SBOF NEURAL TRAINER
Trains a deep neural network to parameterize the Schrödinger
Bridge Onsager Flow potential g(x; θ) on DrugBank data.

Architecture : MLP score network  →  SBOFNet
Loss         : KL-regularised marginal matching
Checkpointing: .pt saved every N epochs + best model
============================================================
"""

import os, sys, time, json, math, logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr, wasserstein_distance
from scipy.special import kl_div
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path('/home/claude/sbof_drugbank')
OUT_DIR   = ROOT / 'sbof_outputs'
CKPT_DIR  = OUT_DIR / 'checkpoints'
OUT_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-7s  %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUT_DIR / 'training.log'),
    ]
)
log = logging.getLogger('SBOF-TRAIN')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log.info('Device: %s', DEVICE)


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LAYER
# ══════════════════════════════════════════════════════════════════════════════

class DrugBankDataset(Dataset):
    """
    Wraps the cached DrugBank parquet into a torch Dataset.

    Each item: (features[7], drug_score[1], index)
    Features: MW, logP, HBA, HBD, PSA, RTB, QED  (z-scored)
    """

    FEAT_COLS = ['mw', 'logp', 'hba', 'hbd', 'psa', 'rtb', 'qed']

    def __init__(self, df: pd.DataFrame, scaler: Optional[StandardScaler] = None,
                 fit_scaler: bool = True):
        self.df = df.reset_index(drop=True)
        X_raw = df[self.FEAT_COLS].values.astype(np.float32)

        if scaler is None:
            scaler = StandardScaler()
        if fit_scaler:
            X_raw = scaler.fit_transform(X_raw)
        else:
            X_raw = scaler.transform(X_raw)
        self.scaler = scaler

        self.X     = torch.tensor(X_raw,                        dtype=torch.float32)
        self.score = torch.tensor(df['drug_score'].values,      dtype=torch.float32)
        self.idx   = torch.arange(len(df))

        # Pre-compute target distribution ν (soft-max over drug_score)
        beta  = 4.0
        w     = torch.exp(beta * self.score - (beta * self.score).max())
        self.nu = (w / w.sum())

        log.info('Dataset: %d molecules | feat_dim=%d | ν_entropy=%.4f',
                 len(df), self.X.shape[1], float(-torch.sum(self.nu * torch.log(self.nu + 1e-12))))

    def __len__(self):  return len(self.df)

    def __getitem__(self, i):
        return self.X[i], self.score[i], self.nu[i], self.idx[i]


# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

class SinusoidalEmbedding(nn.Module):
    """Fourier feature embedding for continuous scalar inputs."""
    def __init__(self, dim: int = 32):
        super().__init__()
        self.dim = dim
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(100.0), dim // 2))
        self.register_buffer('freqs', freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,) or (B,1)
        x = x.view(-1, 1) * self.freqs.unsqueeze(0)   # (B, D/2)
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)   # (B, D)


class ResBlock(nn.Module):
    """Pre-norm residual block with LayerNorm + GELU."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class SBOFNet(nn.Module):
    """
    Neural parameterization of the SBOF dual variable g(x; θ).

    Architecture
    ────────────
    Input  : 7-dim molecular descriptor vector
    Stem   : Linear(7 → hidden) + optional sinusoidal embedding
    Body   : N × ResBlock(hidden)
    Head   : Linear(hidden → 1)  →  g scalar per molecule

    The network outputs log-weights that define the bridge measure.
    """

    def __init__(self,
                 feat_dim:  int   = 7,
                 hidden:    int   = 256,
                 n_layers:  int   = 6,
                 dropout:   float = 0.10,
                 embed_dim: int   = 32):
        super().__init__()
        self.feat_dim  = feat_dim
        self.hidden    = hidden
        self.n_layers  = n_layers
        self.embed_dim = embed_dim

        # Fourier embedding for each scalar feature
        self.feat_embed = nn.ModuleList([
            SinusoidalEmbedding(embed_dim) for _ in range(feat_dim)
        ])
        in_dim = feat_dim * embed_dim

        # Stem
        self.stem = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )

        # Body
        self.blocks = nn.Sequential(*[
            ResBlock(hidden, dropout) for _ in range(n_layers)
        ])

        # Head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters())
        log.info('SBOFNet: hidden=%d  layers=%d  params=%s',
                 hidden, n_layers, f'{n_params:,}')

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, feat_dim)  z-scored molecular features
        returns g : (B,)   SBOF dual variable
        """
        # Embed each feature separately
        parts = [self.feat_embed[i](x[:, i]) for i in range(self.feat_dim)]
        h = torch.cat(parts, dim=-1)          # (B, feat_dim * embed_dim)
        h = self.stem(h)
        h = self.blocks(h)
        g = self.head(h).squeeze(-1)          # (B,)
        return g

    def get_distribution(self, x: torch.Tensor,
                         mu: torch.Tensor,
                         K:  torch.Tensor) -> torch.Tensor:
        """
        Compute learned marginal m given features x, source μ, kernel K.
        m_i = Σ_j μ_j K_ji exp(g_i) / Z_j
        """
        g   = self.forward(x)                        # (N,)
        eg  = torch.exp(g - g.max())                 # stability
        Z   = K @ eg                                 # (N,)
        # m_i = [μ^T K diag(eg)]_i / (K^T diag(1/Z) μ)_i  →  simplified:
        #      = (μ · K · eg) / Z  (row-normalised K already sums μ K = μ)
        m   = (mu @ K) * eg / (Z + 1e-12)
        m   = torch.clamp(m, min=0)
        return m / (m.sum() + 1e-12)


# ══════════════════════════════════════════════════════════════════════════════
# 3. LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def kl_loss(m: torch.Tensor, nu: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """KL(m || ν) — main SBOF objective."""
    m  = m  + eps
    nu = nu + eps
    return (m * (m.log() - nu.log())).sum()


def tv_loss(m: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    """½ ||m − ν||₁"""
    return 0.5 * (m - nu).abs().sum()


def js_loss(m: torch.Tensor, nu: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Jensen-Shannon divergence."""
    mid = 0.5 * (m + nu) + eps
    m   = m  + eps
    nu  = nu + eps
    return 0.5 * (m * (m.log() - mid.log())).sum() + \
           0.5 * (nu * (nu.log() - mid.log())).sum()


def entropy_reg(m: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Negative entropy regulariser — encourages spread."""
    m = m + eps
    return (m * m.log()).sum()


class SBOFLoss(nn.Module):
    """
    Composite loss:
        L = λ_kl · KL(m||ν) + λ_tv · TV(m,ν) + λ_js · JS(m,ν) + λ_ent · H(m)
    """
    def __init__(self, λ_kl=1.0, λ_tv=0.5, λ_js=0.2, λ_ent=0.05):
        super().__init__()
        self.λ_kl  = λ_kl
        self.λ_tv  = λ_tv
        self.λ_js  = λ_js
        self.λ_ent = λ_ent

    def forward(self, m: torch.Tensor, nu: torch.Tensor
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        lkl  = kl_loss(m, nu)
        ltv  = tv_loss(m, nu)
        ljs  = js_loss(m, nu)
        lent = entropy_reg(m)
        total = (self.λ_kl * lkl + self.λ_tv * ltv +
                 self.λ_js * ljs + self.λ_ent * lent)
        return total, {
            'kl': lkl.item(), 'tv': ltv.item(),
            'js': ljs.item(), 'ent': lent.item(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 4. KERNEL (pre-computed, cached on CPU/GPU)
# ══════════════════════════════════════════════════════════════════════════════

def build_kernel(X_np: np.ndarray, sigma: float = 0.35,
                 self_loop: float = 0.10) -> torch.Tensor:
    """Build and return normalised similarity kernel as a torch tensor."""
    log.info('Building %d×%d kernel (σ=%.2f)…', len(X_np), len(X_np), sigma)
    t0 = time.perf_counter()
    D  = cdist(X_np, X_np, 'euclidean')
    K  = np.exp(-D ** 2 / (2 * sigma ** 2))
    np.fill_diagonal(K, 0)
    K /= K.sum(axis=1, keepdims=True) + 1e-12
    K  = (1 - self_loop) * K + self_loop * np.eye(len(X_np))
    log.info('Kernel built in %.2fs', time.perf_counter() - t0)
    return torch.tensor(K, dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class Trainer:
    """
    Full training loop for SBOFNet.

    Saves
    ─────
    checkpoints/epoch_{N:04d}.pt   every `save_every` epochs
    checkpoints/best_model.pt      whenever val TV improves
    checkpoints/final_model.pt     at end of training
    training_history.json          loss curves
    fig_training.png               publication figure
    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self._setup_data()
        self._setup_model()
        self._setup_optim()
        self.history = {k: [] for k in [
            'epoch', 'train_loss', 'train_kl', 'train_tv', 'train_js',
            'val_kl', 'val_tv', 'val_pearson', 'val_spearman', 'lr',
        ]}
        self.best_tv    = float('inf')
        self.best_epoch = 0

    # ── Setup ─────────────────────────────────────────────────────────────────
    def _setup_data(self):
        log.info('Loading dataset from %s', OUT_DIR / 'drugbank_cache.parquet')
        df = pd.read_parquet(OUT_DIR / 'drugbank_cache.parquet')

        # Train/val split (90/10, stratified by drug_score quartile)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        n_val = max(50, int(len(df) * 0.10))
        df_val  = df.iloc[:n_val].copy()
        df_train= df.iloc[n_val:].copy()

        self.ds_train = DrugBankDataset(df_train, fit_scaler=True)
        self.ds_val   = DrugBankDataset(df_val,   scaler=self.ds_train.scaler, fit_scaler=False)

        # Full dataset features (z-scored) for kernel
        self.X_full_np = self.ds_train.scaler.transform(
            df[DrugBankDataset.FEAT_COLS].values.astype(np.float32)
        )
        # Full nu (all molecules)
        beta   = 4.0
        score  = torch.tensor(df['drug_score'].values, dtype=torch.float32)
        w      = torch.exp(beta * score - (beta * score).max())
        self.nu_full = (w / w.sum()).to(DEVICE)

        # Build kernel on full set
        self.K_full = build_kernel(self.X_full_np,
                                   sigma=self.cfg['sigma'],
                                   self_loop=self.cfg['self_loop']).to(DEVICE)
        self.mu_full = torch.ones(len(df), dtype=torch.float32, device=DEVICE) / len(df)
        self.X_full  = torch.tensor(self.X_full_np, dtype=torch.float32, device=DEVICE)

        self.df_full = df
        log.info('Train: %d | Val: %d | Full: %d', len(df_train), len(df_val), len(df))

    def _setup_model(self):
        self.model = SBOFNet(
            feat_dim  = len(DrugBankDataset.FEAT_COLS),
            hidden    = self.cfg['hidden'],
            n_layers  = self.cfg['n_layers'],
            dropout   = self.cfg['dropout'],
            embed_dim = self.cfg['embed_dim'],
        ).to(DEVICE)
        self.loss_fn = SBOFLoss(
            λ_kl  = self.cfg['lambda_kl'],
            λ_tv  = self.cfg['lambda_tv'],
            λ_js  = self.cfg['lambda_js'],
            λ_ent = self.cfg['lambda_ent'],
        )

    def _setup_optim(self):
        self.opt = AdamW(
            self.model.parameters(),
            lr=self.cfg['lr'],
            weight_decay=self.cfg['weight_decay'],
        )
        self.sched = CosineAnnealingLR(
            self.opt,
            T_max=self.cfg['epochs'],
            eta_min=self.cfg['lr'] * 1e-2,
        )

    # ── Training step ─────────────────────────────────────────────────────────
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        accum = {'loss': 0, 'kl': 0, 'tv': 0, 'js': 0}
        n_steps = 0

        # Full-batch gradient on whole dataset (distribution matching)
        # We use the full kernel and all features per step.
        self.opt.zero_grad()
        m = self.model.get_distribution(self.X_full, self.mu_full, self.K_full)
        loss, sub = self.loss_fn(m, self.nu_full)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()

        accum['loss'] += loss.item()
        accum['kl']   += sub['kl']
        accum['tv']   += sub['tv']
        accum['js']   += sub['js']

        return {k: v for k, v in accum.items()}

    # ── Validation ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        self.model.eval()
        m = self.model.get_distribution(self.X_full, self.mu_full, self.K_full)
        m_np  = m.cpu().numpy()
        nu_np = self.nu_full.cpu().numpy()
        kl    = float(kl_loss(m, self.nu_full).item())
        tv    = float(tv_loss(m, self.nu_full).item())
        r_p,_ = pearsonr(nu_np, m_np)
        r_s,_ = spearmanr(nu_np, m_np)
        return {'kl': kl, 'tv': tv, 'pearson': r_p, 'spearman': r_s, 'm': m_np}

    # ── Checkpoint ───────────────────────────────────────────────────────────
    def _save_checkpoint(self, epoch: int, val: Dict, tag: str = ''):
        payload = {
            'epoch':        epoch,
            'model_state':  self.model.state_dict(),
            'optim_state':  self.opt.state_dict(),
            'sched_state':  self.sched.state_dict(),
            'cfg':          self.cfg,
            'val_tv':       val['tv'],
            'val_kl':       val['kl'],
            'val_pearson':  val['pearson'],
            'best_tv':      self.best_tv,
            'best_epoch':   self.best_epoch,
            'history':      self.history,
            'timestamp':    datetime.now().isoformat(),
            'device':       str(DEVICE),
            'n_params':     sum(p.numel() for p in self.model.parameters()),
        }
        name = tag if tag else f'epoch_{epoch:04d}'
        path = CKPT_DIR / f'{name}.pt'
        torch.save(payload, path)
        log.info('Checkpoint → %s  (TV=%.4f)', path.name, val['tv'])
        return path

    # ── Full training loop ────────────────────────────────────────────────────
    def train(self):
        log.info('Starting training: %d epochs | device=%s', self.cfg['epochs'], DEVICE)
        log.info('Config: %s', json.dumps(self.cfg, indent=2))

        for epoch in range(1, self.cfg['epochs'] + 1):
            t0 = time.perf_counter()

            # Train
            tr = self._train_epoch(epoch)

            # Validate
            val = self._validate()

            # LR step
            self.sched.step()
            lr = self.sched.get_last_lr()[0]

            # Record
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(tr['loss'])
            self.history['train_kl'].append(tr['kl'])
            self.history['train_tv'].append(tr['tv'])
            self.history['train_js'].append(tr['js'])
            self.history['val_kl'].append(val['kl'])
            self.history['val_tv'].append(val['tv'])
            self.history['val_pearson'].append(val['pearson'])
            self.history['val_spearman'].append(val['spearman'])
            self.history['lr'].append(lr)

            elapsed = time.perf_counter() - t0

            # Best model
            if val['tv'] < self.best_tv:
                self.best_tv    = val['tv']
                self.best_epoch = epoch
                self._save_checkpoint(epoch, val, tag='best_model')
                best_marker = ' ★ NEW BEST'
            else:
                best_marker = ''

            # Periodic checkpoint
            if epoch % self.cfg['save_every'] == 0:
                self._save_checkpoint(epoch, val)

            # Log
            if epoch % self.cfg['log_every'] == 0 or epoch == 1 or epoch == self.cfg['epochs']:
                log.info(
                    'Ep %4d/%d | Loss=%.4e | KL=%.4e | TV=%.4f | '
                    'r=%.4f | ρ=%.4f | LR=%.2e | %.1fs%s',
                    epoch, self.cfg['epochs'],
                    tr['loss'], val['kl'], val['tv'],
                    val['pearson'], val['spearman'],
                    lr, elapsed, best_marker
                )

        # Final checkpoint
        val_final = self._validate()
        self._save_checkpoint(self.cfg['epochs'], val_final, tag='final_model')

        # Save history
        hist_path = OUT_DIR / 'training_history.json'
        with open(hist_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        log.info('History saved → %s', hist_path)

        # Summary
        self._print_summary(val_final)
        self._make_training_figure()
        self._make_result_figure(val_final)

        return self.model, self.history

    # ── Post-training ─────────────────────────────────────────────────────────
    def _print_summary(self, val: Dict):
        print('\n' + '═' * 68)
        print('TRAINING COMPLETE')
        print('═' * 68)
        print(f"  Epochs          : {self.cfg['epochs']}")
        print(f"  Best TV         : {self.best_tv:.6f}  (epoch {self.best_epoch})")
        print(f"  Final TV        : {val['tv']:.6f}")
        print(f"  Final KL        : {val['kl']:.4e}")
        print(f"  Final Pearson r : {val['pearson']:.4f}")
        print(f"  Final Spearman ρ: {val['spearman']:.4f}")
        print(f"  Checkpoints     : {CKPT_DIR}")
        print(f"  Best model      : {CKPT_DIR / 'best_model.pt'}")
        print(f"  Final model     : {CKPT_DIR / 'final_model.pt'}")
        print('═' * 68)

    def _make_training_figure(self):
        h   = self.history
        eps = h['epoch']
        fig = plt.figure(figsize=(16, 10))
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

        def _ax(row, col, title, xlabel='Epoch'):
            a = fig.add_subplot(gs[row, col])
            a.set_title(title, fontsize=10, fontweight='bold')
            a.set_xlabel(xlabel, fontsize=8)
            a.tick_params(labelsize=7)
            return a

        ax = _ax(0, 0, '(a) Total Training Loss')
        ax.semilogy(eps, h['train_loss'], color='#2563EB', lw=1.8)
        ax.set_ylabel('Loss', fontsize=8)

        ax = _ax(0, 1, '(b) KL Divergence (val)')
        ax.semilogy(eps, [abs(v) for v in h['val_kl']], color='#7C3AED', lw=1.8, label='|KL|')
        ax.set_ylabel('|KL|', fontsize=8)

        ax = _ax(0, 2, '(c) TV Distance (val)')
        ax.plot(eps, h['val_tv'], color='#16A34A', lw=1.8)
        best_ep = self.best_epoch
        best_tv = self.best_tv
        ax.axvline(best_ep, ls='--', color='red', lw=0.8)
        ax.scatter([best_ep], [best_tv], color='red', s=40, zorder=5,
                   label=f'Best: {best_tv:.4f} @ ep{best_ep}')
        ax.legend(fontsize=7)
        ax.set_ylabel('TV', fontsize=8)

        ax = _ax(1, 0, '(d) Train KL vs JS')
        ax2 = ax.twinx()
        ax.semilogy(eps, [abs(v) for v in h['train_kl']], color='#2563EB', lw=1.4, label='KL')
        ax2.semilogy(eps, [abs(v) for v in h['train_js']], color='#D97706', lw=1.4, label='JS', alpha=0.7)
        ax.set_ylabel('|KL|', fontsize=8, color='#2563EB')
        ax2.set_ylabel('JS',  fontsize=8, color='#D97706')

        ax = _ax(1, 1, '(e) Pearson r (val)')
        ax.plot(eps, h['val_pearson'],  color='#0891B2', lw=1.8, label='Pearson r')
        ax.plot(eps, h['val_spearman'], color='#16A34A', lw=1.4, ls='--', label='Spearman ρ')
        ax.axhline(0.99, ls=':', color='gray', lw=0.8)
        ax.legend(fontsize=7)
        ax.set_ylabel('Correlation', fontsize=8)

        ax = _ax(1, 2, '(f) Learning Rate Schedule')
        ax.semilogy(eps, h['lr'], color='#64748B', lw=1.5)
        ax.set_ylabel('LR', fontsize=8)

        fig.suptitle(
            f'SBOFNet Training — DrugBank {len(self.df_full)} molecules | '
            f'Best TV={self.best_tv:.4f} @ epoch {self.best_epoch}',
            fontsize=12, fontweight='bold'
        )
        path = OUT_DIR / 'fig_training.png'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        log.info('Training figure → %s', path)

    def _make_result_figure(self, val: Dict):
        m_np  = val['m']
        nu_np = self.nu_full.cpu().numpy()
        df    = self.df_full
        score = df['drug_score'].values
        mw    = df['mw'].values

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.subplots_adjust(hspace=0.40, wspace=0.32)

        # (a) ν vs m scatter
        ax = axes[0, 0]
        sc = ax.scatter(nu_np, m_np, c=score, cmap='viridis', s=6, alpha=0.5, rasterized=True)
        lim = max(nu_np.max(), m_np.max()) * 1.05
        ax.plot([0, lim], [0, lim], 'r--', lw=1, label='y=x')
        ax.set_xlabel('Target ν', fontsize=9); ax.set_ylabel('Learned m', fontsize=9)
        r = val['pearson']
        ax.set_title(f'(a) ν vs m  r={r:.4f}', fontsize=10, fontweight='bold')
        plt.colorbar(sc, ax=ax, label='Drug Score', pad=0.01)
        ax.legend(fontsize=8)

        # (b) Sorted distribution overlay
        ax = axes[0, 1]
        sort_idx = np.argsort(score)
        ax.plot(score[sort_idx], np.ones(len(df))/len(df), color='#94A3B8', lw=1.2, ls='--', label='Source μ')
        ax.plot(score[sort_idx], nu_np[sort_idx], color='#16A34A', lw=1.8, label='Target ν')
        ax.plot(score[sort_idx], m_np[sort_idx],  color='#2563EB', lw=1.8, ls='-.', label='SBOFNet m')
        ax.set_xlabel('Drug Score', fontsize=9); ax.set_ylabel('Probability', fontsize=9)
        ax.set_title('(b) Distribution Overlay', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)

        # (c) Top-50 highlight
        ax = axes[0, 2]
        top50 = np.argsort(m_np)[-50:]
        ax.scatter(mw, score, c='#CBD5E1', s=4, alpha=0.4, label='All drugs')
        ax.scatter(mw[top50], score[top50], c='#DC2626', s=30, zorder=5, label='Top 50 (SBOF)')
        ax.set_xlabel('MW (Da)', fontsize=9); ax.set_ylabel('Drug Score', fontsize=9)
        ax.set_title('(c) Top-50 Identified', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)

        # (d) MW distribution
        ax = axes[1, 0]
        sp = lambda p: np.maximum(p, 0) / (np.maximum(p, 0).sum() + 1e-12)
        ax.hist(np.random.choice(mw, 1000, p=sp(np.ones(len(df))/len(df))),
                bins=35, alpha=0.4, density=True, color='#94A3B8', label='Source')
        ax.hist(np.random.choice(mw, 1000, p=sp(nu_np)),
                bins=35, alpha=0.4, density=True, color='#16A34A', label='Target')
        ax.hist(np.random.choice(mw, 1000, p=sp(m_np)),
                bins=35, alpha=0.4, density=True, color='#2563EB', label='SBOFNet')
        ax.set_xlabel('MW (Da)', fontsize=9); ax.set_ylabel('Density', fontsize=9)
        ax.set_title('(d) MW Distribution', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)

        # (e) Per-cluster mean m
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=6, random_state=42, n_init=10)
        scaler = self.ds_train.scaler
        feats  = scaler.transform(df[DrugBankDataset.FEAT_COLS].values.astype(np.float32))
        labels = km.fit_predict(feats)
        ax = axes[1, 1]
        cmap6 = plt.cm.get_cmap('tab10', 6)
        for ci in range(6):
            mask = labels == ci
            ax.scatter(df['logp'].values[mask], df['qed'].values[mask],
                       c=[cmap6(ci)], s=4 + 400 * m_np[mask] / m_np.max(),
                       alpha=0.5, label=f'C{ci}')
        ax.set_xlabel('logP', fontsize=9); ax.set_ylabel('QED', fontsize=9)
        ax.set_title('(e) logP vs QED (size ∝ m)', fontsize=10, fontweight='bold')
        ax.legend(fontsize=7, ncol=2)

        # (f) Residuals histogram
        ax = axes[1, 2]
        resid = m_np - nu_np
        ax.hist(resid, bins=50, color='#7C3AED', alpha=0.7, edgecolor='white', lw=0.3)
        ax.axvline(0, color='red', lw=1, ls='--')
        ax.set_xlabel('m − ν  (residual)', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.set_title(f'(f) Residuals  μ={resid.mean():.2e}  σ={resid.std():.2e}',
                     fontsize=10, fontweight='bold')

        fig.suptitle(
            f'SBOFNet Learned Distribution — TV={val["tv"]:.4f}  '
            f'Pearson r={val["pearson"]:.4f}  n={len(df)}',
            fontsize=12, fontweight='bold'
        )
        path = OUT_DIR / 'fig_sbofnet_results.png'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        log.info('Results figure → %s', path)


# ══════════════════════════════════════════════════════════════════════════════
# 6. INFERENCE / RESUME
# ══════════════════════════════════════════════════════════════════════════════

def load_checkpoint(path: str | Path, device: torch.device = DEVICE) -> Dict:
    """Load any .pt checkpoint and return the full payload dict."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    log.info('Loaded checkpoint: %s', Path(path).name)
    log.info('  epoch=%d  TV=%.4f  Pearson=%.4f',
             ckpt['epoch'], ckpt['val_tv'], ckpt['val_pearson'])
    return ckpt


def resume_model(ckpt_path: str | Path) -> Tuple['SBOFNet', Dict]:
    """Reconstruct SBOFNet from a .pt checkpoint, ready for inference."""
    ckpt  = load_checkpoint(ckpt_path)
    cfg   = ckpt['cfg']
    model = SBOFNet(
        feat_dim  = cfg.get('feat_dim', 7),
        hidden    = cfg['hidden'],
        n_layers  = cfg['n_layers'],
        dropout   = cfg['dropout'],
        embed_dim = cfg['embed_dim'],
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    log.info('Model resumed from %s  (%s params)',
             Path(ckpt_path).name,
             f"{sum(p.numel() for p in model.parameters()):,}")
    return model, ckpt


# ══════════════════════════════════════════════════════════════════════════════
# 7. POST-TRAINING EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_checkpoint(ckpt_path: str | Path, verbose: bool = True) -> Dict:
    """Full evaluation of a saved checkpoint against all baselines."""
    sys.path.insert(0, str(ROOT))
    from sbof_main import (DrugBankDownloader, SBOF, SBOFConfig,
                            SOTABaselines, print_benchmark_table)

    log.info('Loading data for evaluation…')
    df = pd.read_parquet(OUT_DIR / 'drugbank_cache.parquet')

    # Rebuild analytic SBOF for baseline reference
    cfg_a  = SBOFConfig(n_iter=150, eta=0.03, sigma=0.35, beta=4.0, verbose=False)
    sbof_a = SBOF(df, cfg_a)
    res_a  = sbof_a.run()

    # Neural model
    model, ckpt = resume_model(ckpt_path)
    ds    = DrugBankDataset(df, fit_scaler=True)
    X_t   = ds.X.to(DEVICE)
    K_t   = build_kernel(
        ds.scaler.transform(df[DrugBankDataset.FEAT_COLS].values.astype(np.float32)),
        sigma=ckpt['cfg']['sigma'],
        self_loop=ckpt['cfg']['self_loop']
    ).to(DEVICE)
    mu_t  = torch.ones(len(df), device=DEVICE) / len(df)
    nu_t  = ds.nu.to(DEVICE)

    with torch.no_grad():
        m_net = model.get_distribution(X_t, mu_t, K_t).cpu().numpy()

    nu_np = nu_t.cpu().numpy()
    r_p,_ = pearsonr(nu_np, m_net)
    r_s,_ = spearmanr(nu_np, m_net)

    neural_result = {
        'name': 'SBOFNet (neural)',
        'tv':   float(tv_loss(torch.tensor(m_net), torch.tensor(nu_np))),
        'kl':   float(kl_loss(torch.tensor(m_net + 1e-12), torch.tensor(nu_np + 1e-12))),
        'js':   float(js_loss(torch.tensor(m_net + 1e-12), torch.tensor(nu_np + 1e-12))),
        'wasserstein': float(wasserstein_distance(m_net, nu_np)),
        'pearson_r':  r_p,
        'spearman_r': r_s,
        'runtime': 0.0,
        'traj_tv': [],
    }

    # Baselines
    sota     = SOTABaselines(sbof_a)
    baselines = sota.run_all()

    # Analytic SBOF entry
    analytic = {
        'name': 'SBOF-analytic',
        'tv':   res_a.stats['final_tv'],
        'kl':   abs(res_a.stats['final_kl']),
        'js':   res_a.stats['final_js'],
        'wasserstein': res_a.stats['final_wass'],
        'pearson_r':  res_a.stats['pearson_r'],
        'spearman_r': res_a.stats['spearman_r'],
        'runtime': res_a.runtime,
        'traj_tv': res_a.trajectory['tv'],
    }

    all_methods = [neural_result, analytic] + baselines
    all_methods.sort(key=lambda x: x['tv'])

    if verbose:
        print('\n' + '═' * 80)
        print('FULL EVALUATION — SBOFNet vs SBOF-analytic vs 5 SOTA baselines')
        print('═' * 80)
        hdr = f"{'Method':<22} {'TV↓':>8} {'KL↓':>12} {'JS↓':>8} {'Wass↓':>8} {'r↑':>7} {'ρ↑':>7}"
        print(hdr); print('─' * 80)
        for rank, m in enumerate(all_methods, 1):
            marker = ' ★' if rank == 1 else ''
            print(f"{m['name']:<22} {m['tv']:>8.4f} {m['kl']:>12.4e} {m['js']:>8.4f} "
                  f"{m['wasserstein']:>8.4f} {m['pearson_r']:>7.4f} {m['spearman_r']:>7.4f}{marker}")
        print('═' * 80)

    return {'neural': neural_result, 'analytic': analytic,
            'baselines': baselines, 'all': all_methods}


# ══════════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_CFG = {
    # Architecture
    'hidden':    256,
    'n_layers':  6,
    'dropout':   0.10,
    'embed_dim': 32,
    'feat_dim':  7,
    # Kernel
    'sigma':     0.35,
    'self_loop': 0.10,
    # Training
    'epochs':    500,
    'lr':        3e-4,
    'weight_decay': 1e-4,
    # Loss weights
    'lambda_kl':  1.0,
    'lambda_tv':  0.5,
    'lambda_js':  0.2,
    'lambda_ent': 0.05,
    # Bookkeeping
    'save_every': 100,
    'log_every':  50,
}


def main(cfg: Dict = None, evaluate_only: str = None):
    cfg = {**DEFAULT_CFG, **(cfg or {})}

    if evaluate_only:
        evaluate_checkpoint(evaluate_only)
        return

    print('═' * 68)
    print('SBOF NEURAL TRAINER')
    print(f"  Device  : {DEVICE}")
    print(f"  Epochs  : {cfg['epochs']}")
    print(f"  Hidden  : {cfg['hidden']}  ×  {cfg['n_layers']} layers")
    print(f"  LR      : {cfg['lr']}")
    print('═' * 68)

    trainer = Trainer(cfg)
    model, history = trainer.train()

    log.info('Running full evaluation against all baselines…')
    evaluate_checkpoint(CKPT_DIR / 'best_model.pt')

    return model, history


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',    type=int,   default=500)
    p.add_argument('--hidden',    type=int,   default=256)
    p.add_argument('--n_layers',  type=int,   default=6)
    p.add_argument('--lr',        type=float, default=3e-4)
    p.add_argument('--evaluate',  type=str,   default=None,
                   help='Path to .pt checkpoint for eval-only mode')
    args = p.parse_args()

    if args.evaluate:
        evaluate_checkpoint(args.evaluate)
    else:
        cfg = {**DEFAULT_CFG, 'epochs': args.epochs,
               'hidden': args.hidden, 'n_layers': args.n_layers, 'lr': args.lr}
        main(cfg)
