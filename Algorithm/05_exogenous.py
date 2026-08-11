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

import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Configuration
# ============================================================

PARAMS = {
    "lookback": 64,
    "patch_size": 16,
    "d_model": 128,
    "n_heads": 8,
    "n_layers": 1,
    "diffusion_steps": 25,
    "dropout": 0.15642808590645504,
    "ffn_multiplier": 4,
    "diffusion_hidden": 512,
    "diffusion_weight": 0.00023119084834595678,
    "batch_size": 64,
    "learning_rate": 8.024951685274957e-05,
    "weight_decay": 0.009468512309958347,
    "grad_clip": 1.0,
}

HORIZON = 7
SEEDS = list(range(1, RUNS+1))
DATA_PATH = "tucurui.csv"
TARGET_COLUMN = "Natural Flow"
EXOG_COLUMN = "UPH610010000"
OUTPUT_DIR = Path("Results_05_exogenous_comparison")

# For a quick code check, use for example EPOCHS = 2 and SEEDS = [1, 2].
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Reproducibility and data preparation
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_windows(features, target, lookback, horizon):
    x, y = [], []
    for i in range(len(features) - lookback - horizon + 1):
        x.append(features[i:i + lookback])
        y.append(target[i + lookback:i + lookback + horizon])
    return (torch.tensor(np.asarray(x), dtype=torch.float32),
            torch.tensor(np.asarray(y), dtype=torch.float32).unsqueeze(-1))


def load_data(path):
    columns = [TARGET_COLUMN, EXOG_COLUMN]
    frame = pd.read_csv(path, sep=";", decimal=",")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing columns in {path}: {missing}")
    frame = frame[columns].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    if len(frame) <= PARAMS["lookback"] + HORIZON:
        raise ValueError("The dataset is too short after removing missing values.")
    return frame


def prepare_datasets(frame, feature_columns):
    split = int(len(frame) * 0.8)
    train = frame.iloc[:split]

    feature_mean = train[feature_columns].mean()
    feature_std = train[feature_columns].std(ddof=0).replace(0, 1)
    target_mean = float(train[TARGET_COLUMN].mean())
    target_std = float(train[TARGET_COLUMN].std(ddof=0))
    if target_std == 0:
        raise ValueError("Natural Flow has zero standard deviation in the training set.")

    features = ((frame[feature_columns] - feature_mean) / feature_std).to_numpy(np.float32)
    target = ((frame[TARGET_COLUMN] - target_mean) / target_std).to_numpy(np.float32)

    # The test data include only the lookback observations preceding the split.
    x_train, y_train = make_windows(features[:split], target[:split], PARAMS["lookback"], HORIZON)
    x_test, y_test = make_windows(features[split - PARAMS["lookback"]:],
                                  target[split - PARAMS["lookback"]:],
                                  PARAMS["lookback"], HORIZON)
    if len(x_train) == 0 or len(x_test) == 0:
        raise ValueError("No train/test windows were created. Check the split and window sizes.")

    return {
        "train": TensorDataset(x_train, y_train),
        "test": TensorDataset(x_test, y_test),
        "target_mean": target_mean,
        "target_std": target_std,
        "split": split,
    }


# ============================================================
# Diffusion Transformer
# ============================================================

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
        self.network = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))

    def forward(self, timestep, total_steps):
        return self.network(timestep.float().unsqueeze(-1) / total_steps)


class InputEncoder(nn.Module):
    def __init__(self, lookback, patch_size, n_features, d_model):
        super().__init__()
        if lookback % patch_size:
            raise ValueError("lookback must be divisible by patch_size.")
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size * n_features, d_model)
        self.position = PositionalEncoding(d_model)

    def forward(self, x):
        # unfold produces [batch, patches, features, patch_size].
        patches = x.unfold(1, self.patch_size, self.patch_size)
        patches = patches.reshape(x.size(0), patches.size(1), -1)
        return self.position(self.projection(patches))


class Forecaster(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, horizon, dropout, ffn_multiplier):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_multiplier * d_model,
            dropout=dropout, activation="gelu", batch_first=True)
        self.backbone = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output_projection = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, horizon))

    def forward(self, x):
        representation = self.backbone(x).mean(dim=1)
        return self.output_projection(representation), representation


class DiffusionModel(nn.Module):
    def __init__(self, target_dimension, condition_dimension, hidden_dimension):
        super().__init__()
        self.timestep_embedding = TimestepEmbedding(condition_dimension)
        self.input_layer = nn.Linear(target_dimension + 2 * condition_dimension, hidden_dimension)
        self.block1 = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dimension, hidden_dimension), nn.SiLU())
        self.block2 = nn.Sequential(nn.Linear(hidden_dimension, hidden_dimension), nn.SiLU(),
                                    nn.Linear(hidden_dimension, hidden_dimension))
        self.output_layer = nn.Linear(hidden_dimension, target_dimension)

    def forward(self, noisy_target, condition, timestep, total_steps):
        timestep_embedding = self.timestep_embedding(timestep, total_steps)
        hidden = self.input_layer(torch.cat([noisy_target, condition, timestep_embedding], dim=-1))
        return self.output_layer(self.block2(self.block1(hidden)) + hidden)


class FullDiffusionTransformer(nn.Module):
    def __init__(self, config, horizon, n_features):
        super().__init__()
        d_model = config["d_model"]
        self.diffusion_steps = config["diffusion_steps"]
        self.input_encoder = InputEncoder(config["lookback"], config["patch_size"],
                                          n_features, d_model)
        self.forecaster = Forecaster(d_model, config["n_heads"], config["n_layers"], horizon,
                                     config["dropout"], config["ffn_multiplier"])
        self.target_encoder = nn.Sequential(nn.Linear(horizon, d_model), nn.LayerNorm(d_model))
        self.diffusion_model = DiffusionModel(d_model, d_model, config["diffusion_hidden"])
        betas = torch.linspace(1e-4, 0.02, self.diffusion_steps)
        self.register_buffer("alpha_bars", torch.cumprod(1.0 - betas, dim=0))

    def encode_input(self, x):
        return self.forecaster(self.input_encoder(x))

    def encode_target(self, y):
        return self.target_encoder(y.squeeze(-1))

    def diffusion_loss(self, target_embedding, condition):
        timestep = torch.randint(0, self.diffusion_steps, (target_embedding.size(0),),
                                 device=target_embedding.device)
        noise = torch.randn_like(target_embedding)
        alpha = self.alpha_bars[timestep].unsqueeze(-1)
        noisy_target = alpha.sqrt() * target_embedding + (1 - alpha).sqrt() * noise
        predicted_noise = self.diffusion_model(noisy_target, condition, timestep,
                                               self.diffusion_steps)
        return nn.functional.mse_loss(predicted_noise, noise)

    def forward(self, x, y=None):
        forecast, condition = self.encode_input(x)
        if y is None:
            return forecast
        forecast_loss = nn.functional.mse_loss(forecast, y.squeeze(-1))
        diffusion_loss = self.diffusion_loss(self.encode_target(y), condition)
        return forecast, forecast_loss, diffusion_loss


# ============================================================
# Training and evaluation
# ============================================================

def train_model(model, loader, config, epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    history = []
    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            _, forecast_loss, diffusion_loss = model(x_batch, y_batch)
            loss = forecast_loss + config["diffusion_weight"] * diffusion_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            epoch_loss += loss.item()
        history.append(epoch_loss / len(loader))
    return np.asarray(history)


def predict(model, loader, target_mean, target_std):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            predictions.append(model(x_batch.to(DEVICE)).cpu())
            targets.append(y_batch.squeeze(-1))
    predictions = torch.cat(predictions).numpy() * target_std + target_mean
    targets = torch.cat(targets).numpy() * target_std + target_mean
    return predictions, targets


def calculate_metrics(predictions, targets):
    error = predictions - targets
    return {
        "MSE": float(np.mean(error ** 2)),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAE": float(np.mean(np.abs(error))),
        "SMAPE": float(100 * np.mean(2 * np.abs(error) /
                                     (np.abs(targets) + np.abs(predictions) + 1e-8))),
    }


def run_configuration(name, feature_columns, frame):
    prepared = prepare_datasets(frame, feature_columns)
    test_loader = DataLoader(prepared["test"], batch_size=PARAMS["batch_size"], shuffle=False)
    rows, outputs, histories = [], [], []

    print(f"\n{name}: {feature_columns}")
    for run, seed in enumerate(SEEDS, 1):
        set_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(prepared["train"], batch_size=PARAMS["batch_size"],
                                  shuffle=True, generator=generator)
        model = FullDiffusionTransformer(PARAMS, HORIZON, len(feature_columns)).to(DEVICE)
        histories.append(train_model(model, train_loader, PARAMS, EPOCHS))
        predictions, targets = predict(model, test_loader, prepared["target_mean"],
                                       prepared["target_std"])
        metrics = calculate_metrics(predictions, targets)
        rows.append({"Configuration": name, "Seed": seed,
                     "Parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                     **metrics})
        outputs.append((predictions, targets))
        print(f"Run {run:02d}/{len(SEEDS)} | seed={seed:02d} | "
              f"RMSE={metrics['RMSE']:.3f} | MAE={metrics['MAE']:.3f} | "
              f"SMAPE={metrics['SMAPE']:.3f}%")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = pd.DataFrame(rows)
    representative = int((results["RMSE"] - results["RMSE"].mean()).abs().idxmin())
    return {
        "results": results,
        "outputs": outputs,
        "histories": np.stack(histories),
        "representative": representative,
    }


def summarize(all_results):
    rows = []
    for name, group in all_results.groupby("Configuration", sort=False):
        row = {"Configuration": name, "Parameters": int(group["Parameters"].iloc[0])}
        for metric in ["RMSE", "MAE", "SMAPE"]:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_tests(all_results):
    without = all_results.query("Configuration == 'Without exogenous'").set_index("Seed")
    with_exog = all_results.query("Configuration == 'With exogenous'").set_index("Seed")
    common = without.index.intersection(with_exog.index)
    rows = []
    for metric in ["RMSE", "MAE", "SMAPE"]:
        baseline = without.loc[common, metric].to_numpy()
        exogenous = with_exog.loc[common, metric].to_numpy()
        difference = exogenous - baseline
        try:
            statistic, p_value = wilcoxon(exogenous, baseline, alternative="two-sided")
        except ValueError:
            statistic, p_value = np.nan, 1.0
        rows.append({
            "Metric": metric,
            "Mean_without": baseline.mean(),
            "Mean_with": exogenous.mean(),
            "Change_percent": 100 * (exogenous.mean() - baseline.mean()) / baseline.mean(),
            "Improved_seeds": int(np.sum(difference < 0)),
            "Wilcoxon_statistic": statistic,
            "p_value": p_value,
        })
    return pd.DataFrame(rows)


# ============================================================
# Plots
# ============================================================

def horizon_metrics(predictions, targets):
    error = predictions - targets
    return {
        "MAE": np.mean(np.abs(error), axis=(0, 1)),
        "RMSE": np.sqrt(np.mean(error ** 2, axis=(0, 1))),
        "SMAPE": 100 * np.mean(2 * np.abs(error) /
                               (np.abs(targets) + np.abs(predictions) + 1e-8), axis=(0, 1)),
    }





OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
frame = load_data(DATA_PATH)
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Rows after joint missing-value removal: {len(frame)}")

configurations = {
    "Without exogenous": [TARGET_COLUMN],
    "With exogenous": [TARGET_COLUMN, EXOG_COLUMN],
}
experiments = {name: run_configuration(name, columns, frame)
               for name, columns in configurations.items()}
all_results = pd.concat([experiment["results"] for experiment in experiments.values()],
                        ignore_index=True)
summary = summarize(all_results)
tests = paired_tests(all_results)

all_results.to_csv(OUTPUT_DIR / "individual_runs.csv", index=False)
summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
tests.to_csv(OUTPUT_DIR / "paired_wilcoxon.csv", index=False)

# Plots
plt.style.use("seaborn-v0_8-whitegrid")
colors = {"Without exogenous": "#4C78A8", "With exogenous": "#E45756"}
names, future = list(experiments), np.arange(1, HORIZON + 1)

def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# Forecast comparison
fig, ax = plt.subplots(figsize=(6, 4))
length = min(300, *(len(exp["outputs"][exp["representative"]][1])
                    for exp in experiments.values()))

for name, exp in experiments.items():
    pred, true = exp["outputs"][exp["representative"]]
    ax.plot(pred[:length, 0], color=colors[name], lw=1.2, label=name)

ax.plot(true[:length, 0], "k", lw=1.4, label="Actual", zorder=3)
ax.set(xlabel="Test origin", ylabel="Flow (m³/s)")
ax.legend(frameon=True, facecolor="white", framealpha=1)
save(fig, "forecast_with_and_without_exogenous")

# Forecast comparison
fig, ax = plt.subplots(figsize=(10, 4))
length = min(300, *(len(exp["outputs"][exp["representative"]][1])
                    for exp in experiments.values()))

for name, exp in experiments.items():
    pred, true = exp["outputs"][exp["representative"]]
    ax.plot(pred[:length, 0], color=colors[name], lw=1.2, label=name)

ax.plot(true[:length, 0], "k", lw=1.4, label="Actual", zorder=3)
ax.set(xlabel="Test origin", ylabel="Flow (m³/s)")
ax.legend(frameon=True, facecolor="white", framealpha=1)
save(fig, "forecast_with_and_without_exogenous2")

# Mean errors
fig, ax = plt.subplots(figsize=(6, 4))
ax2, x, width = ax.twinx(), np.arange(2), .25
ax.set_axisbelow(True)
ax.grid(True, linestyle="--", alpha=.4, zorder=0)
ax2.grid(False)
for i, metric in enumerate(["RMSE", "MAE"]):
    ax.bar(x+(i-1)*width, summary[f"{metric}_mean"], width,
           yerr=summary[f"{metric}_std"], capsize=4, alpha=.85,
           label=metric, zorder=3)
ax2.bar(x+width, summary["SMAPE_mean"], width,
        yerr=summary["SMAPE_std"], capsize=4, alpha=.85,
        color="green", label="SMAPE", zorder=3)
ax.set(xticks=x, xticklabels=["Without\nexogenous", "With\nexogenous"],
       ylabel="Error (m³/s)")
ax2.set_ylabel("SMAPE (%)")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax2.legend(h1+h2, l1+l2, loc="lower right", frameon=True,
           facecolor="white", framealpha=1)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "mean_errors.pdf", bbox_inches="tight")
plt.show()
plt.close(fig)

# Paired RMSE
pivot = all_results.pivot(index="Seed", columns="Configuration", values="RMSE")
fig, ax = plt.subplots(figsize=(6, 4))
for _, row in pivot.iterrows():
    ax.plot([0, 1], row[names].values, color=".75", lw=.8)
for i, name in enumerate(names):
    ax.scatter(np.full(len(pivot), i), pivot[name], s=25,
               color=colors[name], label=name, zorder=2)
ax.set(xticks=[0, 1], xticklabels=["Without\nexogenous", "With\nexogenous"],
       ylabel="RMSE (m³/s)")
ax.legend(ncol=2, frameon=True, facecolor="white", framealpha=1)
save(fig, "paired_rmse")

# Horizon RMSE
fig, ax = plt.subplots(figsize=(6, 4))
for name, exp in experiments.items():
    pred = np.stack([x[0] for x in exp["outputs"]])
    true = np.stack([x[1] for x in exp["outputs"]])
    ax.plot(future, horizon_metrics(pred, true)["RMSE"], "o-",
            color=colors[name], label=name)
ax.set(xlabel="Forecast horizon", ylabel="RMSE (m³/s)", xticks=future)
ax.legend(ncol=2, frameon=True, facecolor="white", framealpha=1)
save(fig, "horizon_rmse")

# Actual versus predicted
fig, ax = plt.subplots(figsize=(6, 4))
values = []
for name, exp in experiments.items():
    pred, true = exp["outputs"][exp["representative"]]
    ax.scatter(true.ravel(), pred.ravel(), s=10, alpha=.18,
               color=colors[name], label=name)
    values.extend([true, pred])
lower, upper = min(x.min() for x in values), max(x.max() for x in values)
margin = .03 * (upper-lower)
ax.plot([lower, upper], [lower, upper], "k--")
ax.set(xlim=(lower-margin, upper+margin), ylim=(lower-margin, upper+margin),
       xlabel="Actual flow (m³/s)", ylabel="Predicted flow (m³/s)",
       aspect="equal")
ax.legend(ncol=2, frameon=True, facecolor="white", framealpha=1)
save(fig, "actual_vs_predicted")

# Representative forecasts
for name, exp in experiments.items():
    i = exp["representative"]
    pred, true = exp["outputs"][i]
    length = min(300, len(true))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(true[:length, 0], "k", lw=1.4, label="Actual")
    ax.plot(pred[:length, 0], color=colors[name], lw=1.2, label="Predicted")
    ax.set(xlabel="Test origin", ylabel="Flow (m³/s)")
    ax.legend(ncol=2)
    save(fig, f"forecast_{name.lower().replace(' ', '_')}")

print("\nSummary\n", summary.to_string(index=False))
print("\nPaired comparison (negative change favors the exogenous model)\n",
      tests.to_string(index=False))
print(f"\nResults saved in: {OUTPUT_DIR.resolve()}")
