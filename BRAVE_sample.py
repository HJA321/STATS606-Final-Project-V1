from __future__ import annotations
import math
import time
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cvxpy as cp
import matplotlib.pyplot as plt

# ------------------------- 1. Data utilities ---------------------------------

class AllHistoryDataset(Dataset):
    """Sliding window dataset producing (Xpast, Yfuture) pairs."""
    def __init__(self, df: pd.DataFrame, tickers: list[str], tau: int = 63, stride: int = 1):
        super().__init__()
        self.df      = df.copy()
        self.tau     = tau
        self.stride  = stride
        self.tickers = tickers
        self.m       = len(tickers)

        # expect columns [tk_open, tk_close, tk_high, tk_low] per ticker
        exp_cols = []
        for tk in tickers:
            exp_cols += [f"{tk}_open", f"{tk}_close", f"{tk}_high", f"{tk}_low"]
        if list(df.columns) != exp_cols:
            raise ValueError("Column order must be open,close,high,low per ticker")

        # compute close-to-close returns
        close = df[[f"{tk}_close" for tk in tickers]].values
        rets  = (np.roll(close, -1, axis=0) - close) / close
        rets[-1, :] = np.nan
        self.returns = rets

        T = len(df)
        self.ends = [t for t in range(tau, T - tau) if (t - tau) % stride == 0]

    def __len__(self):
        return len(self.ends)

    def __getitem__(self, idx):
        end_t = self.ends[idx]
        start = end_t - self.tau
        past  = self.df.iloc[start:end_t].values.reshape(self.tau, self.m, 4).astype(np.float32)
        fut   = self.returns[end_t:end_t+self.tau].astype(np.float32)
        return torch.from_numpy(past), torch.from_numpy(fut)

# ------------------------- 2. Model components -------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        seq = x.size(0)
        return x + self.pe[:seq].unsqueeze(1)

def make_transformer_encoder(d_model, n_heads, n_layers, dropout=0.1):
    layer = nn.TransformerEncoderLayer(d_model, n_heads, 4*d_model, dropout, batch_first=False)
    return nn.TransformerEncoder(layer, n_layers)

class Encoder(nn.Module):
    def __init__(self, m, tau, d_model=128, latent_dim=32, n_heads=4, n_layers=2):
        super().__init__()
        self.m           = m
        self.tau         = tau
        self.d_stock     = d_model // 8
        self.d_quarter   = d_model // 8
        self.stock_embed = nn.Embedding(m, self.d_stock)
        self.quarter_embed = nn.Embedding(1024, self.d_quarter)
        self.input_fc    = nn.Linear(4 + self.d_stock + self.d_quarter, d_model)
        self.pos_enc     = PositionalEncoding(d_model)
        self.transformer = make_transformer_encoder(d_model, n_heads, n_layers)
        self.mu_head     = nn.Linear(d_model, latent_dim)
        self.logvar_head = nn.Linear(d_model, latent_dim)

    def forward(self, past: torch.Tensor):
        τ, m, _ = past.shape
        device  = past.device
        ids   = torch.arange(m, device=device)
        s_emb = self.stock_embed(ids).unsqueeze(0).expand(τ, -1, -1)
        q_ids = torch.zeros(τ, dtype=torch.long, device=device)
        q_emb = self.quarter_embed(q_ids).unsqueeze(1).expand(-1, m, -1)
        x = torch.cat([past, s_emb, q_emb], dim=-1)
        x = self.input_fc(x)
        x = x.reshape(τ*m, -1).unsqueeze(1)
        x = self.pos_enc(x)
        h = self.transformer(x)
        h = h.mean(dim=0)
        mu     = self.mu_head(h).squeeze(0)
        logvar = self.logvar_head(h).squeeze(0)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, m, d_latent, n_bins=21):
        super().__init__()
        self.m      = m
        self.n_bins = n_bins
        self.mlp    = nn.Sequential(
            nn.Linear(d_latent, 2*d_latent), nn.ReLU(),
            nn.Linear(2*d_latent, m * n_bins)
        )

    def forward(self, z: torch.Tensor):
        logits = self.mlp(z)
        return logits.view(-1, self.m, self.n_bins)

class VAE(nn.Module):
    def __init__(self, enc: Encoder, dec: Decoder):
        super().__init__()
        self.encoder = enc
        self.decoder = dec

    def forward(self, past: torch.Tensor):
        mu, logvar = self.encoder(past)
        std        = (0.5 * logvar).exp()
        z          = mu + std * torch.randn_like(std)
        logits     = self.decoder(z.unsqueeze(0))
        return logits, mu, logvar

def elbo_loss(logits, y_future, mu, logvar, n_bins, beta=0.3):
    logits_flat = logits.squeeze(0)
    edges       = torch.linspace(-0.2, 0.2, n_bins + 1, device=logits.device)
    bin_w       = edges[1] - edges[0]
    r0          = y_future[0].clamp(edges[0], edges[-2] + 1e-6)
    idx         = ((r0 - edges[0]) / bin_w).floor().long().clamp(0, n_bins - 1)
    recon       = F.cross_entropy(logits_flat, idx, reduction="mean")
    kl          = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl

# ----------------------- 3. Trainer ------------------------------------------

class BRAVETrainer:
    def __init__(self, tickers, tau=63, model_dim=128, latent_dim=16, device="cpu"):
        self.m       = len(tickers)
        self.tau     = tau
        self.device  = device
        self.encoder = Encoder(self.m, tau, model_dim, latent_dim).to(device)
        self.decoder = Decoder(self.m, latent_dim, n_bins=21).to(device)
        self.vae     = VAE(self.encoder, self.decoder)
        self.student = Encoder(self.m, tau, model_dim, latent_dim).to(device)
        self.mse     = nn.MSELoss()

    def train_teacher(self, loader: DataLoader, epochs=25, lr=1e-4):
        opt = torch.optim.Adam(self.vae.parameters(), lr)
        for ep in range(epochs):
            tot = 0.0
            for past, y in loader:
                past = past.squeeze(0).to(self.device)
                y    = y.squeeze(0).to(self.device)
                logits, mu, logvar = self.vae(past)
                loss = elbo_loss(logits, y, mu, logvar, self.decoder.n_bins)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item()
            print(f"[Teacher] Epoch {ep+1}/{epochs}, avg loss={tot/len(loader):.4f}")

    def train_student(self, loader: DataLoader, epochs=25, lr=1e-4):
        for p in self.decoder.parameters(): p.requires_grad=False
        for p in self.encoder.parameters(): p.requires_grad=False
        opt = torch.optim.Adam(self.student.parameters(), lr)
        for ep in range(epochs):
            tot = 0.0
            for past, _ in loader:
                past = past.squeeze(0).to(self.device)
                with torch.no_grad():
                    mu_t, logvar_t = self.encoder(past)
                mu_s, logvar_s = self.student(past)
                loss = self.mse(mu_s, mu_t) + self.mse(logvar_s, logvar_t)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item()
            print(f"[Student] Epoch {ep+1}/{epochs}, avg loss={tot/len(loader):.6f}")

    @torch.no_grad()
    def sample_scenarios(self, past: torch.Tensor, N=1000):
        mu, logvar = self.student(past.to(self.device))
        std        = (0.5 * logvar).exp()
        edges      = torch.linspace(-0.2, 0.2, self.decoder.n_bins+1, device=self.device)
        arr        = []
        for _ in range(N):
            z     = mu + std * torch.randn_like(std)
            probs = self.decoder(z.unsqueeze(0)).softmax(-1).squeeze(0)
            idx   = torch.multinomial(probs, 1).squeeze(1)
            lo, hi= edges[idx], edges[idx+1]
            arr.append(((hi-lo)*torch.rand_like(lo)+lo).cpu().numpy())
        return np.stack(arr)

# ----------------------- 4. Utils --------------------------------------------

def compute_returns(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    prices = df[[f"{tk}_close" for tk in tickers]]
    ret    = prices.pct_change().iloc[1:]
    ret.columns = tickers
    return ret

def solve_cvar(scenarios: np.ndarray,
               α: float = 0.2,
               λ1: float = 0.1,
               λ2: float = 0.1,
               σ_max: float = 0.3,
               w_prev: np.ndarray | None = None) -> np.ndarray:
    N, m = scenarios.shape
    w = cp.Variable(m, nonneg=True)
    t = cp.Variable()
    z = cp.Variable(N, nonneg=True)
    r_hat = scenarios.mean(axis=0)
    obj   = r_hat @ w
    if w_prev is not None:
        obj = obj - λ1 * cp.norm1(w - w_prev)
    cons = [
        z >= -scenarios @ w - t,
        cp.sum(z)/((1-α)*N) + t <= σ_max,
        cp.sum(w) == 1
    ]
    prob = cp.Problem(cp.Maximize(obj), cons)
    prob.solve(solver=cp.SCS, verbose=False)
    return w.value

# ----------------------- 5. Rolling backtest --------------------------------

def rolling_backtest(
    df: pd.DataFrame,
    tickers: list[str],
    τ: int = 63,
    step: int = 63,
    E_t: int = 5,
    E_s: int = 5,
    N_scen: int = 300,
    α: float = 0.2,
    λ1: float = 0.1,
    λ2: float = 0.1,
    σ_max: float = 0.3,
    max_periods: int = 5,
    gamma: float = 0.4,
) -> pd.DataFrame:
    records = []
    w_prev  = None
    periods = 0
    for start in range(200, len(df) - τ - step + 1, step):
        if periods >= max_periods:
            break
        periods += 1

        ds     = AllHistoryDataset(df.iloc[:start+τ], tickers, tau=τ, stride=1)
        loader = DataLoader(ds, batch_size=1)
        brave  = BRAVETrainer(tickers, tau=τ, device="cpu")
        brave.train_teacher(loader, epochs=E_t)
        brave.train_student(loader, epochs=E_s)

        past, _ = ds[-1]
        scen    = brave.sample_scenarios(past, N=N_scen)
        w_cvar  = solve_cvar(scen, α, λ1, λ2, σ_max, w_prev)

        # mix with equal weight
        m      = len(tickers)
        w_eq   = np.ones(m)/m
        w_opt  = gamma*w_cvar + (1-gamma)*w_eq
        w_opt  = np.clip(w_opt, 0, None); w_opt /= w_opt.sum()
        w_prev = w_opt.copy()

        test_df = df.iloc[start+τ:start+τ+step]
        R_test  = compute_returns(test_df, tickers)
        strat_r = R_test.values @ w_opt

        for date, row_r, sr in zip(R_test.index, R_test.values, strat_r):
            rec = {"date": date, "strategy_return": sr}
            for tk, r in zip(tickers, row_r): rec[f"ret_{tk}"] = r
            for tk, w_ in zip(tickers, w_opt): rec[f"w_{tk}"]   = w_
            records.append(rec)

    df_out = pd.DataFrame(records).set_index("date").sort_index()
    ret_cols                  = [f"ret_{tk}" for tk in tickers]
    df_out["equal_weight"]    = df_out[ret_cols].mean(axis=1)
    df_out["historical_pred"] = (
        df_out[ret_cols].rolling(τ).mean().fillna(method="bfill")
        .dot(np.ones(len(tickers))/len(tickers))
    )
    return df_out

# ----------------------- 6. Main and hyperparameter search ------------------

if __name__ == "__main__":
    # load and preprocess data
    data = pd.read_csv("combined_sp500_data.csv", parse_dates=["date"], index_col="date")
    try:
        end   = data.index.max()
        start = end - pd.DateOffset(years=4)
        data  = data.loc[start:end]
    except:
        data  = data.iloc[-3000:]

    tickers = ["AAPL","MSFT","GOOG","A","ABT","ACGL","AMT","AMZN", "AMD", "BAC", "BLK",
               "BA", "CVS", "MCD", "META", "MS", "NFLX", "NVDA", "NKE", "PEP"]
    cols    = [f"{tk}_{col}" for tk in tickers for col in ["open","close","high","low"]]
    df      = data[cols].dropna()

    # hyperparameter grid: 4 × 3 × 3 = 36 configs
    param_grid = {
        "gamma": [0.3, 0.5, 0.7, 0.9],
        "α":     [0.1, 0.2, 0.3],
        "λ1":    [0.0, 0.1, 0.5],
    }
    results   = []
    start_all = time.time()

    for gamma, α, λ1 in itertools.product(param_grid["gamma"],
                                          param_grid["α"],
                                          param_grid["λ1"]):
        t0 = time.time()
        df_out = rolling_backtest(
            df, tickers,
            τ=63, step=63,
            E_t=5, E_s=5,
            N_scen=300,
            α=α, λ1=λ1, λ2=0.1, σ_max=0.3,
            max_periods=5,
            gamma=gamma
        )
        cum   = (1 + df_out).cumprod() - 1
        final = cum["strategy_return"].iloc[-1]

        # plot & save with dynamic filename
        plt.figure(figsize=(10,4))
        plt.plot(cum.index, cum["strategy_return"], label="BRAVE+CVaR")
        plt.plot(cum.index, cum["equal_weight"],    label="Equal Weight")
        plt.plot(cum.index, cum["historical_pred"], label="Historical Pred")
        plt.legend()
        plt.title(f"γ={gamma}, α={α}, λ1={λ1} → {final:.2%}")
        plt.ylabel("Cumulative Return")
        plt.xlabel("Date")
        fname = f"rb_search_g{gamma}_a{α}_l{λ1}.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=300)
        plt.close()

        results.append({
            "gamma": gamma, "α": α, "λ1": λ1,
            "final_return": final,
            "time_min":    (time.time()-t0)/60
        })
        print(f"Done γ={gamma}, α={α}, λ1={λ1} → ret={final:.2%}, time={(time.time()-t0)/60:.1f} min")

    total_min = (time.time()-start_all)/60
    print(f"Total grid search time: {total_min:.1f} min")

    df_res = pd.DataFrame(results).sort_values("final_return", ascending=False)
    print("Top configurations:\n", df_res.head(5))