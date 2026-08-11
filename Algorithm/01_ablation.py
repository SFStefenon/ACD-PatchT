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
else:
    EPOCHS = 2
    RUNS = 2

import random
import numpy as np
import pandas as pd
import torch
from torch import nn
import math
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOOKBACK = 512
HORIZON = 7
PATCH_SIZE = 8
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
DIFFUSION_STEPS = 100
DIFFUSION_WEIGHT = 0.1
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

from pathlib import Path
OUTPUT_DIR = Path("Results_01_ablation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("tucurui.csv", sep=";", decimal=",")
data = df[["Natural Flow", "UPH610010000"]].dropna()
values = data["Natural Flow"].to_numpy(dtype=np.float32)
precipitation = data["UPH610010000"].to_numpy(dtype=np.float32)
samples = np.arange(len(values))
fig, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(samples, values, color="k", linewidth=1, label="Natural flow")
ax1.set_xlabel("Time step")
ax1.set_ylabel(r"Natural flow ($\mathrm{m^3/s}$)", color="k")
ax1.tick_params(axis="y", labelcolor="k")
ax1.set_xlim(0, len(values) - 1)
ax1.grid(True, alpha=0.3)
ax2 = ax1.twinx()
ax2.plot(samples, precipitation, color="blue", linewidth=1, alpha=0.6, label="Precipitation")
ax2.set_ylabel("Precipitation (mm)", color="blue")
ax2.tick_params(axis="y", labelcolor="blue")
lines = ax1.get_lines() + ax2.get_lines()
ax1.legend(lines, [line.get_label() for line in lines], loc="upper right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "tucurui_time_series.pdf", bbox_inches="tight")
plt.show()

split = int(len(values) * 0.8)
mean, std = values[:split].mean(), values[:split].std()
values = (values - mean) / std # Normalization

train_values = values[:split]
test_values = values[split - LOOKBACK:]

ABLATION_SEEDS, ABLATION_EPOCHS = list(range(1, RUNS+1)), EPOCHS

def make_windows(data):
    x, y = [], []
    for i in range(len(data) - LOOKBACK - HORIZON + 1):
        x.append(data[i:i + LOOKBACK])
        y.append(data[i + LOOKBACK:i + LOOKBACK + HORIZON])
    return torch.tensor(np.array(x)).unsqueeze(-1), torch.tensor(np.array(y)).unsqueeze(-1)

x_train, y_train = make_windows(train_values)
x_test, y_test = make_windows(test_values)

train_dataset = TensorDataset(x_train, y_train)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length=5000):
        super().__init__()
        position = torch.arange(max_length).unsqueeze(1)
        scale = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale[:encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]

class TimestepEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(1, d_model),nn.SiLU(),
                                     nn.Linear(d_model, d_model))

    def forward(self, timestep, total_steps):
        timestep = timestep.float().unsqueeze(-1) / total_steps
        return self.network(timestep)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class FlexibleInputEncoder(nn.Module):
    def __init__(self, sequence_length, patch_size, d_model, use_patches=True, use_position=True):
        super().__init__()
        self.use_patches, self.use_position, self.patch_size = use_patches, use_position, patch_size
        if use_patches and sequence_length % patch_size:
            raise ValueError("LOOKBACK must be divisible by PATCH_SIZE.")
        self.projection = nn.Linear(patch_size if use_patches else 1, d_model)
        if use_position: self.position = PositionalEncoding(d_model)

    def forward(self, x):
        if self.use_patches: x = x.squeeze(-1).unfold(1, self.patch_size, self.patch_size)
        x = self.projection(x)
        return self.position(x) if self.use_position else x

class FlexibleForecaster(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, horizon):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, 0.1,
            activation="gelu", batch_first=True)
        self.backbone = nn.TransformerEncoder(layer, n_layers)
        self.output_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, horizon))

    def forward(self, x):
        representation = self.backbone(x).mean(dim=1)
        return self.output_projection(representation), representation

class AblationDiffusionModel(nn.Module):
    def __init__(self, target_dimension, condition_dimension, hidden_dimension=256, use_condition=True, use_timestep=True, use_residual=True):
        super().__init__()
        self.use_condition, self.use_timestep, self.use_residual = use_condition, use_timestep, use_residual
        if use_timestep: self.timestep_embedding = TimestepEmbedding(condition_dimension)
        input_dimension = target_dimension + condition_dimension * (use_condition + use_timestep)
        self.input_layer = nn.Linear(input_dimension, hidden_dimension)
        self.block1 = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dimension, hidden_dimension), nn.SiLU())
        self.block2 = nn.Sequential(nn.Linear(hidden_dimension, hidden_dimension), nn.SiLU(), nn.Linear(hidden_dimension, hidden_dimension))
        self.output_layer = nn.Linear(hidden_dimension, target_dimension)

    def forward(self, noisy_target, condition, timestep, total_steps):
        features = [noisy_target]
        if self.use_condition: features.append(condition)
        if self.use_timestep: features.append(self.timestep_embedding(timestep, total_steps))
        hidden = self.input_layer(torch.cat(features, dim=-1)); residual = hidden
        hidden = self.block2(self.block1(hidden))
        return self.output_layer(hidden + residual if self.use_residual else hidden)

class AblationDiffusionLLM(nn.Module):
    def __init__(self, lookback, horizon, patch_size, d_model, n_heads, n_layers, diffusion_steps,
                 use_diffusion=True, use_condition=True, use_timestep=True, use_diffusion_residual=True,
                 use_position=True, use_patches=True):
        super().__init__()
        self.diffusion_steps, self.use_diffusion = diffusion_steps, use_diffusion
        self.input_encoder = FlexibleInputEncoder(lookback, patch_size, d_model, use_patches, use_position)
        self.forecaster = FlexibleForecaster(d_model, n_heads, n_layers, horizon)
        if use_diffusion:
            self.target_encoder = nn.Sequential(nn.Linear(horizon, d_model), nn.LayerNorm(d_model))
            self.diffusion_model = AblationDiffusionModel(d_model, d_model, 256, use_condition, use_timestep, use_diffusion_residual)
            betas = torch.linspace(1e-4, 0.02, diffusion_steps)
            self.register_buffer("alpha_bars", torch.cumprod(1.0 - betas, dim=0))

    def encode_input(self, x):
        return self.forecaster(self.input_encoder(x))

    def diffusion_loss(self, target_embedding, condition):
        timestep = torch.randint(0, self.diffusion_steps, (target_embedding.size(0),), device=target_embedding.device)
        noise = torch.randn_like(target_embedding); alpha_bar = self.alpha_bars[timestep].unsqueeze(-1)
        noisy_target = alpha_bar.sqrt() * target_embedding + (1.0 - alpha_bar).sqrt() * noise
        predicted_noise = self.diffusion_model(noisy_target, condition, timestep, self.diffusion_steps)
        return nn.functional.mse_loss(predicted_noise, noise)

    def forward(self, x, y=None):
        forecast, condition = self.encode_input(x)
        if y is None: return forecast
        forecast_loss = nn.functional.mse_loss(forecast, y.squeeze(-1))
        diffusion_loss = self.diffusion_loss(self.target_encoder(y.squeeze(-1)), condition) if self.use_diffusion else x.new_zeros(())
        return forecast, forecast_loss, diffusion_loss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model, train_loader, epochs, diffusion_weight):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    history = {"total": [], "forecast": [], "diffusion": []}
    for _ in range(epochs):
        model.train(); losses = np.zeros(3)
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device); optimizer.zero_grad()
            _, forecast_loss, diffusion_loss = model(x_batch, y_batch)
            loss = forecast_loss + diffusion_weight * diffusion_loss
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            losses += [loss.item(), forecast_loss.item(), diffusion_loss.item()]
        for key, value in zip(history, losses / len(train_loader)): history[key].append(value)
    return history

def evaluate_model(model, test_loader):
    model.eval(); predictions, targets = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            predictions.append(model(x_batch.to(device)).cpu())
            targets.append(y_batch.squeeze(-1))
    predictions = torch.cat(predictions).numpy().ravel() * std + mean
    targets = torch.cat(targets).numpy().ravel() * std + mean
    error = predictions - targets
    return {"MSE": np.mean(error**2),"RMSE": np.sqrt(np.mean(error**2)), "MAE": np.mean(np.abs(error)),
        "SMAPE": 100 * np.mean(2 * np.abs(error) / (np.abs(targets) + np.abs(predictions) + 1e-8))}

BASE = dict(use_diffusion=True, use_condition=True, use_timestep=True, use_diffusion_residual=True,
            use_position=True, use_patches=True)
ablation_configurations = {
    "Full model": BASE,
    "Without diffusion": {**BASE, "use_diffusion": False, "use_condition": False, "use_timestep": False, "use_diffusion_residual": False},
    "Unconditional diffusion": {**BASE, "use_condition": False},
    "Without timestep embedding": {**BASE, "use_timestep": False},
    "Without diffusion residual": {**BASE, "use_diffusion_residual": False},
    "Without positional encoding": {**BASE, "use_position": False},
    "Without patching": {**BASE, "use_patches": False},
}

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

all_results, all_histories = [], {}
for architecture_name, configuration in ablation_configurations.items():
    print(f"\nArchitecture: {architecture_name}"); all_histories[architecture_name] = []
    for run_index, seed in enumerate(ABLATION_SEEDS, 1):
        set_seed(seed)
        model = AblationDiffusionLLM(LOOKBACK, HORIZON, PATCH_SIZE, D_MODEL, N_HEADS, N_LAYERS,
                                     DIFFUSION_STEPS, **configuration).to(device)
        history = train_model(model, train_loader, ABLATION_EPOCHS, DIFFUSION_WEIGHT)
        metrics = evaluate_model(model, test_loader)
        all_histories[architecture_name].append(history)
        all_results.append({"Architecture": architecture_name, "Seed": seed,
                            "Parameters": count_parameters(model), **metrics})
        print(f"Run {run_index}/{len(ABLATION_SEEDS)} | Seed: {seed} | RMSE: {metrics['RMSE']:.6f} | MAE: {metrics['MAE']:.6f}")
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

results_df = pd.DataFrame(all_results)
summary_df = (results_df.groupby("Architecture", as_index=False)
              .agg(Parameters=("Parameters", "first"), RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
                   MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
                   SMAPE_mean=("SMAPE", "mean"), SMAPE_std=("SMAPE", "std"))
              .sort_values("RMSE_mean").reset_index(drop=True))
full_rmse = summary_df.loc[summary_df["Architecture"].eq("Full model"), "RMSE_mean"].iloc[0]
summary_df["RMSE_change_percent"] = 100 * (summary_df["RMSE_mean"] - full_rmse) / full_rmse
print("\nAblation summary\n", summary_df.to_string(index=False))
results_df.to_csv(OUTPUT_DIR / "ablation_individual_runs.csv", index=False)
summary_df.to_csv(OUTPUT_DIR / "ablation_summary.csv", index=False)

# Plot 1: RMSE with variability
plot_df = summary_df.sort_values("RMSE_mean", ascending=False)
positions = np.arange(len(plot_df))
plt.figure(figsize=(7, 5))
plt.barh(positions, plot_df["RMSE_mean"], xerr=plot_df["RMSE_std"], color="green", capsize=4, alpha=0.85)
plt.yticks(positions, plot_df["Architecture"])
plt.xlabel("RMSE")
plt.ylabel("Architecture")
#plt.title("Architecture ablation study")
plt.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ablation_rmse.pdf", bbox_inches="tight")
plt.show()

# Plot 2: change relative to the full architecture
# Positive values indicate that removing or changing the component increased the error.
plot_df = summary_df[summary_df["Architecture"] != "Full model"].sort_values(
    "RMSE_change_percent", ascending=False)
positions = np.arange(len(plot_df))
plt.figure(figsize=(7, 5))
plt.barh(positions, plot_df["RMSE_change_percent"], color="blue", alpha=0.8)
plt.axvline(0, color="black", linestyle="--", linewidth=1.5)
plt.yticks(positions, plot_df["Architecture"])
plt.xlabel("RMSE change relative to the full model (%)")
plt.ylabel("Architecture")
#plt.title("Contribution of each architectural component")
plt.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ablation_relative_rmse.pdf", bbox_inches="tight")
plt.show()

# Plot 3: accuracy versus model size
order = summary_df["Architecture"].tolist()
colors = dict(zip(order, plt.cm.turbo(np.linspace(0, 1, len(order)))))
plt.figure(figsize=(6, 5))
for _, row in summary_df.iterrows():
    name = row["Architecture"]
    plt.scatter(row["Parameters"], row["RMSE_mean"], color=colors[name],
                edgecolor="black", s=100, label=name)

plt.xlabel("Trainable parameters")
plt.ylabel("RMSE")
#plt.title("Accuracy and architectural complexity")
plt.grid(alpha=0.25)
plt.legend(loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ablation_complexity.pdf", bbox_inches="tight")
plt.show()

# Plot 4: average convergence of each architecture
plt.figure(figsize=(6, 5))
for name in order:
    curves = np.array([history["forecast"] for history in all_histories[name]])
    plt.plot(np.arange(1, ABLATION_EPOCHS + 1), curves.mean(axis=0),
             color=colors[name], linewidth=2, label=name)

plt.xlabel("Epoch")
plt.ylabel("Forecasting loss")
#plt.title("Convergence of the ablated architectures")
plt.grid(alpha=0.25)
plt.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ablation_convergence.pdf", bbox_inches="tight")
plt.show()

TOLERANCE_PERCENT = 1.0
best_rmse = summary_df["RMSE_mean"].min()
eligible_architectures = summary_df[
    summary_df["RMSE_mean"]
    <= best_rmse * (1 + TOLERANCE_PERCENT / 100)].copy()
selected_architecture = eligible_architectures.sort_values(
    ["Parameters", "RMSE_mean"]).iloc[0]
print("\nSelected architecture")
print(selected_architecture)

from scipy.stats import wilcoxon

full_rmse = results_df[results_df["Architecture"].eq("Full model")].sort_values("Seed")["RMSE"].to_numpy()
statistical_results = []

for architecture in ablation_configurations:
    if architecture != "Full model":
        variant_rmse = results_df[results_df["Architecture"].eq(architecture)].sort_values("Seed")["RMSE"].to_numpy()
        statistic, p_value = wilcoxon(variant_rmse, full_rmse)
        statistical_results.append({"Architecture": architecture, "Wilcoxon_statistic": statistic, "p_value": p_value})

statistical_df = pd.DataFrame(statistical_results)

print("\nWilcoxon tests\n", statistical_df.to_string(index=False))
statistical_df.to_csv(OUTPUT_DIR / "ablation_wilcoxon.csv", index=False)
