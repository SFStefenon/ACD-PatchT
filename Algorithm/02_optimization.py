gpu_use = True
gpu_number = "1"
if gpu_use == True:
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
    device = torch.device("cuda:0")
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    print(f"GPU {gpu_number}")

complete = True
if complete: # Complete experiment
    EPOCHS = 500
    RUNS = 50
    N_TRIALS = 100
    TUNING_EPOCHS = 100
else:
    EPOCHS = 2
    RUNS = 2
    N_TRIALS = 2
    TUNING_EPOCHS = 2

import math, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import optuna
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HORIZON = 7
FINAL_EPOCHS = EPOCHS
TUNING_SEED = 1
FINAL_SEEDS = list(range(1, RUNS+1))

from pathlib import Path
OUTPUT_DIR = Path("Results_02_optimization")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

raw_values = pd.read_csv("tucurui.csv", sep=";", decimal=",")["Natural Flow"].dropna().to_numpy(np.float32)
test_start = int(len(raw_values) * 0.8)
tuning_values, raw_test = raw_values[:test_start], raw_values[test_start:]
val_start = int(len(tuning_values) * 0.8)
raw_train, raw_val = tuning_values[:val_start], tuning_values[val_start:]

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_train_windows(data, lookback):
    x, y = [], []
    for i in range(len(data) - lookback - HORIZON + 1):
        x.append(data[i:i + lookback]); y.append(data[i + lookback:i + lookback + HORIZON])
    return torch.tensor(np.asarray(x)).unsqueeze(-1), torch.tensor(np.asarray(y)).unsqueeze(-1)

def make_future_windows(context, future, lookback):
    data = np.concatenate([context[-lookback:], future]); x, y = [], []
    for i in range(len(future) - HORIZON + 1):
        x.append(data[i:i + lookback]); y.append(data[i + lookback:i + lookback + HORIZON])
    return torch.tensor(np.asarray(x)).unsqueeze(-1), torch.tensor(np.asarray(y)).unsqueeze(-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length=5000):
        super().__init__(); position = torch.arange(max_length).unsqueeze(1)
        scale = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_length, d_model); encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale[:encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0))
    def forward(self, x): return x + self.encoding[:, :x.size(1)]

class TimestepEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__(); self.network = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
    def forward(self, timestep, total_steps): return self.network(timestep.float().unsqueeze(-1) / total_steps)

class InputEncoder(nn.Module):
    def __init__(self, lookback, patch_size, d_model):
        super().__init__()
        if lookback % patch_size: raise ValueError("LOOKBACK must be divisible by PATCH_SIZE.")
        self.patch_size = patch_size; self.projection = nn.Linear(patch_size, d_model)
        self.position = PositionalEncoding(d_model)
    def forward(self, x):
        return self.position(self.projection(x.squeeze(-1).unfold(1, self.patch_size, self.patch_size)))

class Forecaster(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, dropout, ffn_multiplier):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ffn_multiplier*d_model, dropout,
                                           activation="gelu", batch_first=True)
        self.backbone = nn.TransformerEncoder(layer, n_layers)
        self.output_projection = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, HORIZON))
    def forward(self, x):
        representation = self.backbone(x).mean(1)
        return self.output_projection(representation), representation

class DiffusionModel(nn.Module):
    def __init__(self, d_model, hidden):
        super().__init__(); self.timestep_embedding = TimestepEmbedding(d_model)
        self.input_layer = nn.Linear(3*d_model, hidden)
        self.block1 = nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.block2 = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.output_layer = nn.Linear(hidden, d_model)
    def forward(self, noisy_target, condition, timestep, total_steps):
        hidden = self.input_layer(torch.cat([noisy_target, condition, self.timestep_embedding(timestep, total_steps)], -1))
        return self.output_layer(self.block2(self.block1(hidden)) + hidden)

class FullModel(nn.Module):
    def __init__(self, lookback, patch_size, d_model, n_heads, n_layers, diffusion_steps,
                 dropout, ffn_multiplier, diffusion_hidden):
        super().__init__(); self.diffusion_steps = diffusion_steps
        self.input_encoder = InputEncoder(lookback, patch_size, d_model)
        self.forecaster = Forecaster(d_model, n_heads, n_layers, dropout, ffn_multiplier)
        self.target_encoder = nn.Sequential(nn.Linear(HORIZON, d_model), nn.LayerNorm(d_model))
        self.diffusion_model = DiffusionModel(d_model, diffusion_hidden)
        betas = torch.linspace(1e-4, 0.02, diffusion_steps)
        self.register_buffer("alpha_bars", torch.cumprod(1.0 - betas, 0))
    def forward(self, x, y=None):
        forecast, condition = self.forecaster(self.input_encoder(x))
        if y is None: return forecast
        forecast_loss = nn.functional.mse_loss(forecast, y.squeeze(-1))
        target = self.target_encoder(y.squeeze(-1))
        timestep = torch.randint(self.diffusion_steps, (target.size(0),), device=target.device)
        noise = torch.randn_like(target); alpha_bar = self.alpha_bars[timestep].unsqueeze(-1)
        noisy_target = alpha_bar.sqrt() * target + (1 - alpha_bar).sqrt() * noise
        predicted_noise = self.diffusion_model(noisy_target, condition, timestep, self.diffusion_steps)
        return forecast, forecast_loss, nn.functional.mse_loss(predicted_noise, noise)

def loaders(train, future, lookback, batch_size, shuffle_seed):
    x_train, y_train = make_train_windows(train, lookback)
    x_future, y_future = make_future_windows(train, future, lookback)
    generator = torch.Generator().manual_seed(shuffle_seed)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, generator=generator)
    future_loader = DataLoader(TensorDataset(x_future, y_future), batch_size=batch_size, shuffle=False)
    return train_loader, future_loader

def train_model(model, loader, epochs, diffusion_weight, learning_rate, weight_decay, grad_clip, trial=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(epochs):
        model.train(); epoch_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device); optimizer.zero_grad()
            _, forecast_loss, diffusion_loss = model(x, y)
            loss = forecast_loss + diffusion_weight * diffusion_loss
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), grad_clip); optimizer.step()
            epoch_loss += loss.item()
        if trial is not None and (epoch + 1) % 10 == 0:
            trial.report(epoch_loss / len(loader), epoch)
            if trial.should_prune(): raise optuna.TrialPruned()

def evaluate(model, loader, scale_mean, scale_std):
    model.eval(); predictions, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            predictions.append(model(x.to(device)).cpu()); targets.append(y.squeeze(-1))
    predictions = torch.cat(predictions).numpy().ravel() * scale_std + scale_mean
    targets = torch.cat(targets).numpy().ravel() * scale_std + scale_mean
    error = predictions - targets
    return {"RMSE": np.sqrt(np.mean(error**2)), "MAE": np.mean(np.abs(error)),
            "SMAPE": 100 * np.mean(2 * np.abs(error) / (np.abs(targets) + np.abs(predictions) + 1e-8))}

def objective(trial):
    params = {
        "lookback": trial.suggest_categorical("LOOKBACK", [64, 128, 256, 512]),
        "patch_size": trial.suggest_categorical("PATCH_SIZE", [4, 8, 16, 32]),
        "d_model": trial.suggest_categorical("D_MODEL", [32, 64, 128, 256]),
        "n_heads": trial.suggest_categorical("N_HEADS", [2, 4, 8, 16, 32]),
        "n_layers": trial.suggest_int("N_LAYERS", 1, 5),
        "diffusion_steps": trial.suggest_categorical("DIFFUSION_STEPS", [25, 50, 100, 200]),
        "dropout": trial.suggest_float("DROPOUT", 0.0, 0.3),
        "ffn_multiplier": trial.suggest_categorical("FFN_MULTIPLIER", [2, 4, 8, 16]),
        "diffusion_hidden": trial.suggest_categorical("DIFFUSION_HIDDEN", [64, 128, 256, 512])}
    diffusion_weight = trial.suggest_float("DIFFUSION_WEIGHT", 1e-4, 5e-1, log=True)
    batch_size = trial.suggest_categorical("BATCH_SIZE", [4, 16, 32, 64])
    learning_rate = trial.suggest_float("LEARNING_RATE", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("WEIGHT_DECAY", 1e-6, 1e-2, log=True)
    grad_clip = trial.suggest_categorical("GRAD_CLIP", [0.5, 1.0, 2.0, 4.0])
    if params["lookback"] >= len(raw_train) or params["lookback"] % params["patch_size"]:
        raise optuna.TrialPruned()
    set_seed(TUNING_SEED); scale_mean, scale_std = raw_train.mean(), raw_train.std()
    train = (raw_train - scale_mean) / scale_std; val = (raw_val - scale_mean) / scale_std
    train_loader, val_loader = loaders(train, val, params["lookback"], batch_size, TUNING_SEED)
    model = FullModel(**params).to(device)
    train_model(model, train_loader, TUNING_EPOCHS, diffusion_weight, learning_rate, weight_decay, grad_clip, trial)
    rmse = evaluate(model, val_loader, scale_mean, scale_std)["RMSE"]
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return float(rmse)

sampler = optuna.samplers.TPESampler(seed=TUNING_SEED)
pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=30)
study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
study.optimize(objective, n_trials=N_TRIALS)
study.trials_dataframe().to_csv(OUTPUT_DIR / "full_model_tuning_trials.csv", index=False)
print("Best validation RMSE:", study.best_value)
print("Best hyperparameters:", study.best_params)

best = study.best_params
model_params = {"lookback": best["LOOKBACK"], "patch_size": best["PATCH_SIZE"], "d_model": best["D_MODEL"],
                "n_heads": best["N_HEADS"], "n_layers": best["N_LAYERS"], "diffusion_steps": best["DIFFUSION_STEPS"],
                "dropout": best["DROPOUT"], "ffn_multiplier": best["FFN_MULTIPLIER"],
                "diffusion_hidden": best["DIFFUSION_HIDDEN"]}
scale_mean, scale_std = tuning_values.mean(), tuning_values.std()
final_train = (tuning_values - scale_mean) / scale_std; final_test = (raw_test - scale_mean) / scale_std
results = []
for seed in FINAL_SEEDS:
    set_seed(seed)
    train_loader, test_loader = loaders(final_train, final_test, best["LOOKBACK"], best["BATCH_SIZE"], seed)
    model = FullModel(**model_params).to(device)
    train_model(model, train_loader, FINAL_EPOCHS, best["DIFFUSION_WEIGHT"], best["LEARNING_RATE"],
                best["WEIGHT_DECAY"], best["GRAD_CLIP"])
    metrics = evaluate(model, test_loader, scale_mean, scale_std)
    results.append({"Seed": seed, "Parameters": sum(p.numel() for p in model.parameters() if p.requires_grad), **best, **metrics})
    print(f"Seed {seed}: RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}, SMAPE={metrics['SMAPE']:.2f}")
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / "full_model_optimized_runs.csv", index=False)
summary = results_df[["RMSE", "MAE", "SMAPE"]].agg(["mean", "std"])
summary.to_csv(OUTPUT_DIR / "full_model_optimized_summary.csv")
print("\nFinal test summary\n", summary)

# Plots
trials = study.trials_dataframe().query("state == 'COMPLETE'").sort_values("number").copy()
trials["Trial"], trials["Best RMSE"] = trials["number"]+1, trials["value"].cummin()
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
norm, cmap = Normalize(trials["value"].min(), trials["value"].max()), "turbo_r"

ax[0,0].plot(trials["Trial"], trials["value"], "o-", alpha=.7, label="Trial RMSE")
ax[0,0].plot(trials["Trial"], trials["Best RMSE"], color="#E63946", lw=2.5, label="Best RMSE")
ax[0,0].scatter(study.best_trial.number+1, study.best_value, s=220, marker="*", color="#FFD166", edgecolor="black", zorder=5)
ax[0,0].set(xlabel="Trial", ylabel="Validation RMSE (m³/s)"); ax[0, 0].legend(frameon=True)

sc = ax[0,1].scatter(trials["params_LOOKBACK"], trials["params_PATCH_SIZE"], c=trials["value"],
    s=40+trials["params_D_MODEL"], cmap=cmap, norm=norm, edgecolor="black", alpha=.85)
ax[0,1].set(xlabel="Lookback", ylabel="Patch size"); fig.colorbar(sc, ax=ax[0,1], label="Validation RMSE (m³/s)")

sc = ax[1,0].scatter(trials["params_LEARNING_RATE"], trials["params_DIFFUSION_WEIGHT"], c=trials["value"],
    s=40+2*trials["params_BATCH_SIZE"], cmap=cmap, norm=norm, edgecolor="black", alpha=.85)
ax[1,0].set(xscale="log", yscale="log", xlabel="Learning rate", ylabel="Diffusion weight")
fig.colorbar(sc, ax=ax[1,0], label="Validation RMSE (m³/s)")

importance = optuna.importance.get_param_importances(study)
labels = {"LOOKBACK":"Lookback", "PATCH_SIZE":"Patch size", "D_MODEL":"Model dimension", "N_HEADS":"Attention heads",
    "N_LAYERS":"Transformer layers", "DIFFUSION_STEPS":"Diffusion steps", "DIFFUSION_WEIGHT":"Diffusion weight",
    "BATCH_SIZE":"Batch size", "LEARNING_RATE":"Learning rate", "DROPOUT":"Dropout", "WEIGHT_DECAY":"Weight decay",
    "FFN_MULTIPLIER":"FFN multiplier", "DIFFUSION_HIDDEN":"Diffusion hidden", "GRAD_CLIP":"Gradient clipping"}
names, scores = list(importance), np.array(list(importance.values())); order = np.argsort(scores)
bars = ax[1,1].barh([labels.get(names[i], names[i]) for i in order], scores[order],
    color=plt.cm.magma(np.linspace(.3, .85, len(order))))
ax[1,1].set(xlabel="Importance"); ax[1,1].bar_label(bars, fmt="%.2f", padding=3)

for label, a in zip("ABCD", ax.ravel()):
    a.set_title(f"{label})", loc="left", fontweight="bold"); a.grid(True, alpha=.25)
    a.spines[["top", "right"]].set_visible(False)
plt.savefig(OUTPUT_DIR / "full_model_hypertuning.pdf", bbox_inches="tight"); plt.show()
