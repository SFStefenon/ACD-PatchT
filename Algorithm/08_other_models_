gpu_use = True
complete = True
generalization_analisys = True

gpu_number = "1"
if gpu_use == True:
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
    device = torch.device("cuda:0")
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    print(f"GPU {gpu_number}")

# Sudeste/Centro-Oeste - Fio dagua - DOCE - DCUBAG - BAGUARI
# Nordeste - Reservatório com Usina - SAO FRANCISCO - SFSOBR - SOBRADINHO
# Sul - Reservatório com Usina - URUGUAI - RIUHCN - CAMPOS NOVOS
# Norte	- Fio dagua - AMAZONAS - AMUSBM - BELO MONTE

for HORIZONS in range (13,16,2):
    print(f"Running {HORIZONS}")
    HORIZON = HORIZONS
    
    from pathlib import Path
    if generalization_analisys == True:
        OUTPUT_DIR = Path("Results_03_opt_model_generalization")
        DATA_DIR = Path("dados_hidrologicos")
        RESERVOIR_ID = "DCUBAG"
        FLOW_COLUMN = "val_vazaoafluente"
    else:
        OUTPUT_DIR = Path("Results_03_opt_model")
        DATA_PATH = "tucurui.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if complete: # Complete experiment
        EPOCHS = 500
        RUNS = 50
    else:
        EPOCHS = 2
        RUNS = 2
    
    import time
    import os
    import math
    import random
    import numpy as np
    import pandas as pd
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    
    ###############################################################################
    # Configuration
    
    params = {
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
    
    SEEDS = list(range(1, RUNS+1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ###############################################################################
    # Used Classes
    def set_seed(seed):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def make_windows(data, lookback, horizon):
        x, y = [], []
        for i in range(len(data) - lookback - horizon + 1):
            x.append(data[i:i + lookback]); y.append(data[i + lookback:i + lookback + horizon])
        return (torch.tensor(np.array(x), dtype=torch.float32).unsqueeze(-1),
                torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1))
    
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
        def __init__(self, lookback, patch_size, d_model):
            super().__init__()
            if lookback % patch_size:
                raise ValueError("params['lookback'] must be divisible by params['patch_size'].")
            self.patch_size = patch_size
            self.projection = nn.Linear(patch_size, d_model)
            self.position = PositionalEncoding(d_model)
        def forward(self, x):
            patches = x.squeeze(-1).unfold(1, self.patch_size, self.patch_size)
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
        def __init__(self, config, horizon):
            super().__init__()
            d_model = config["d_model"]
            self.diffusion_steps = config["diffusion_steps"]
            self.input_encoder = InputEncoder(config["lookback"], config["patch_size"], d_model)
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
            alpha_bar = self.alpha_bars[timestep].unsqueeze(-1)
            noisy_target = alpha_bar.sqrt() * target_embedding + (1.0 - alpha_bar).sqrt() * noise
            predicted_noise = self.diffusion_model(noisy_target, condition, timestep, self.diffusion_steps)
            return nn.functional.mse_loss(predicted_noise, noise)
        def forward(self, x, y=None):
            forecast, condition = self.encode_input(x)
            if y is None: return forecast
            target = y.squeeze(-1)
            forecast_loss = nn.functional.mse_loss(forecast, target)
            diffusion_loss = self.diffusion_loss(self.encode_target(y), condition)
            return forecast, forecast_loss, diffusion_loss
    
    def train_model(model, train_loader, config, epochs):
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                      weight_decay=config["weight_decay"])
        history = {"total": [], "forecast": [], "diffusion": []}
        for _ in range(epochs):
            model.train(); losses = np.zeros(3)
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                _, forecast_loss, diffusion_loss = model(x_batch, y_batch)
                loss = forecast_loss + config["diffusion_weight"] * diffusion_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                optimizer.step()
                losses += [loss.item(), forecast_loss.item(), diffusion_loss.item()]
            for key, value in zip(history, losses / len(train_loader)): history[key].append(value)
        return history
    
    def evaluate_model(model, test_loader, data_mean, data_std):
        model.eval(); predictions, targets = [], []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                predictions.append(model(x_batch.to(device)).cpu())
                targets.append(y_batch.squeeze(-1))
        predictions = torch.cat(predictions).numpy() * data_std + data_mean
        targets = torch.cat(targets).numpy() * data_std + data_mean
        error = predictions - targets
        metrics = {"MSE": np.mean(error**2), "RMSE": np.sqrt(np.mean(error**2)),
                "MAE": np.mean(np.abs(error)),
                "SMAPE": 100 * np.mean(2 * np.abs(error) /
                                        (np.abs(targets) + np.abs(predictions) + 1e-8))}
        return metrics, predictions, targets
    
    def gradient_saliency(model, x):
        model.eval(); x = x.clone().detach().to(device).requires_grad_(True)
        model(x).mean().backward()
        saliency = x.grad.abs().squeeze(-1)
        return (saliency / (saliency.amax(1, keepdim=True) + 1e-8)).detach().cpu().numpy()
    
    def occlusion_importance(model, x, patch_size):
        model.eval(); x = x.clone().detach().to(device)
        with torch.no_grad(): baseline = model(x)
        importance = []
        for i in range(x.size(1) // patch_size):
            occluded = x.clone(); start = i * patch_size
            occluded[:, start:start + patch_size] = x.mean(1, keepdim=True)
            with torch.no_grad(): importance.append((baseline - model(occluded)).abs().mean(1).cpu().numpy())
        importance = np.stack(importance, axis=1)
        return importance / (importance.max(1, keepdims=True) + 1e-8)
    
    def horizon_saliency(model, x):
        model.eval(); x = x.clone().detach().to(device).requires_grad_(True); forecast = model(x); maps = []
        for h in range(forecast.size(1)):
            model.zero_grad(); x.grad = None; forecast[:, h].sum().backward(retain_graph=True)
            maps.append(x.grad.abs().squeeze(-1).mean(0).cpu().numpy())
        maps = np.stack(maps)
        return maps / (maps.max(1, keepdims=True) + 1e-8)
    
    def diffusion_loss_by_timestep(model, x, y, samples_per_step=5):
        model.eval(); x, y = x.to(device), y.to(device)
        with torch.no_grad(): _, condition = model.encode_input(x); target = model.encode_target(y)
        steps = np.unique(np.linspace(0, model.diffusion_steps - 1, 20, dtype=int)); losses = []
        for step in steps:
            step_losses = []
            for _ in range(samples_per_step):
                t = torch.full((x.size(0),), step, device=device, dtype=torch.long)
                noise = torch.randn_like(target); alpha = model.alpha_bars[t].unsqueeze(-1)
                noisy = alpha.sqrt() * target + (1 - alpha).sqrt() * noise
                with torch.no_grad(): predicted = model.diffusion_model(noisy, condition, t, model.diffusion_steps)
                step_losses.append(nn.functional.mse_loss(predicted, noise).item())
            losses.append(np.mean(step_losses))
        return steps, np.array(losses)
    
    ###############################################################################
    # Load Data
    if generalization_analisys == True:
        files = sorted(DATA_DIR.glob("DADOS_HIDROLOGICOS_HO_*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found in {DATA_DIR.resolve()}")
        
        columns = ["id_reservatorio", "nom_reservatorio", "din_instante", FLOW_COLUMN]
        data = pd.concat(
            [pd.read_csv(f, sep=";", decimal=",", usecols=columns) for f in files],
            ignore_index=True,)
        
        data["din_instante"] = pd.to_datetime(data["din_instante"], errors="coerce")
        data[FLOW_COLUMN] = pd.to_numeric(data[FLOW_COLUMN], errors="coerce")
        data = (data[data["id_reservatorio"].eq(RESERVOIR_ID)]
                .dropna(subset=["din_instante", FLOW_COLUMN])
                .sort_values("din_instante")
                .drop_duplicates("din_instante", keep="last")
                .set_index("din_instante"))
        
        # For daily natural/affluent flow, using the daily mean
        data = data.resample("D").agg({FLOW_COLUMN: "mean","nom_reservatorio": 
                                       "first",}).dropna(subset=[FLOW_COLUMN])
        
        print(f"{data['nom_reservatorio'].iloc[0]}: {data.index.min()} to "
              f"{data.index.max()} ({len(data)} observations)")
        
        values = data[FLOW_COLUMN].to_numpy(np.float32)
        split = int(0.8 * len(values))
        data_mean, data_std = values[:split].mean(), values[:split].std()
        values = (values - data_mean) / data_std
        
        train_values = values[:split]
        test_values = values[split - params["lookback"]:]
        
        x_train, y_train = make_windows(train_values, params["lookback"], HORIZON)
        x_test, y_test = make_windows(test_values, params["lookback"], HORIZON)
        
        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False)
    
    else:
        # Load Data
        values = pd.read_csv(DATA_PATH, sep=";", decimal=",")["Natural Flow"].dropna().to_numpy(np.float32)
        split = int(len(values) * 0.8)
        data_mean, data_std = values[:split].mean(), values[:split].std()
        values = (values - data_mean) / data_std
        train_values, test_values = values[:split], values[split - params["lookback"]:]
        x_train, y_train = make_windows(train_values, params["lookback"], HORIZON)
        x_test, y_test = make_windows(test_values, params["lookback"], HORIZON)
        train_dataset, test_dataset = TensorDataset(x_train, y_train), TensorDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False)
    ###############################################################################
    # Run the Experiment
    
    results, histories, run_outputs, states = [], [], [], []
    for run, seed in enumerate(SEEDS, 1):
        set_seed(seed)
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], 
                                  shuffle=True, generator=generator)
        model = FullDiffusionTransformer(params, HORIZON).to(device)
        
        start = time.time()
        histories.append(train_model(model, train_loader, params, EPOCHS))
        train_time = time.time() - start
        
        start = time.time()
        metrics, predictions, targets = evaluate_model(
            model, test_loader, data_mean, data_std)
        test_time = time.time() - start
    
        metrics, predictions, targets = evaluate_model(model, test_loader, data_mean, data_std)
        results.append({"Seed": seed,
        "Parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        **metrics, "Train": train_time, "Test": test_time})
        run_outputs.append((predictions, targets)); states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        #print(f"Run {run}/{len(SEEDS)} | Seed: {seed} | RMSE: {metrics['RMSE']:.6f} | "
              #f"MAE: {metrics['MAE']:.6f} | SMAPE: {metrics['SMAPE']:.6f}")
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    results_df = pd.DataFrame(results)
    summary_df = pd.DataFrame([{
        "Parameters": results_df["Parameters"].iloc[0],
        "RMSE_mean": results_df["RMSE"].mean(), "RMSE_std": results_df["RMSE"].std(),
        "MAE_mean": results_df["MAE"].mean(), "MAE_std": results_df["MAE"].std(),
        "SMAPE_mean": results_df["SMAPE"].mean(), "SMAPE_std": results_df["SMAPE"].std()}])
    #print("\nFull-model summary\n", summary_df.to_string(index=False))
    results_df.to_csv(OUTPUT_DIR / "full_model_individual_runs.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "full_model_summary.csv", index=False)
    
    representative = (results_df["RMSE"] - results_df["RMSE"].mean()).abs().idxmin()
    model = FullDiffusionTransformer(params, HORIZON).to(device)
    model.load_state_dict(states[representative])
    predictions, targets = run_outputs[representative]
    #print(f"Plots use seed {results_df.loc[representative, 'Seed']}, whose RMSE is closest to the multi-seed mean.")
    
    ###############################################################################
    if generalization_analisys == False:
        # Plot of Results
        plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
                             "legend.fontsize": 10, "figure.dpi": 120})
        steps, future = np.arange(1, EPOCHS + 1), np.arange(1, HORIZON + 1)
        mean_history = {key: np.mean([h[key] for h in histories], axis=0) for key in histories[0]}
        errors = predictions - targets
        mae = np.abs(errors).mean(0)
        rmse = np.sqrt((errors**2).mean(0))
        smape = 100 * np.mean(2 * np.abs(errors) / (np.abs(targets) + np.abs(predictions) + 1e-8), axis=0)
        
        plt.figure(figsize=(6, 4))
        for key, label in [("total", "Total loss"), ("forecast", "Forecast loss"), ("diffusion", "Diffusion loss")]:
            plt.plot(steps, mean_history[key], linewidth=2.2, label=label)
        plt.xlabel("Epoch"); plt.ylabel("Loss"); #plt.title("Mean training convergence across seeds")
        plt.grid(alpha=.25); plt.legend(); plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True));
        name = "training_losses.pdf"
        plt.tight_layout(); plt.savefig(OUTPUT_DIR / name, bbox_inches="tight"); plt.show()
        
        fig, ax = plt.subplots(figsize=(6, 4))
        l1 = ax.plot(future, mae, "o-", label="MAE")
        l2 = ax.plot(future, rmse, "s-", label="RMSE")
        ax.set(xlabel="Forecast horizon", ylabel="Error (m³/s)"); ax.grid(alpha=.25)
        ax2 = ax.twinx()
        l3 = ax2.plot(future, smape, "^-", color="green", label="SMAPE")
        ax2.set_ylabel("SMAPE (%)")
        lines = l1 + l2 + l3
        ax.legend(lines, [line.get_label() for line in lines], loc="upper left")
        name = "horizon_errors.pdf"
        plt.tight_layout(); plt.savefig(OUTPUT_DIR / name, bbox_inches="tight"); plt.show()
        
        plt.figure(figsize=(6, 4))
        image = plt.imshow(np.abs(errors[:min(100, len(errors))]), aspect="auto", origin="lower")
        plt.colorbar(image, label="Absolute error (m³/s)"); plt.xlabel("Forecast horizon"); plt.ylabel("Test window")
        plt.xticks(np.arange(HORIZON), future); #plt.title("Forecast-error heatmap"); 
        name = "error_heatmap.pdf"
        plt.tight_layout(); plt.savefig(OUTPUT_DIR / name, bbox_inches="tight"); plt.show()
        
        plt.figure(figsize=(6, 4))
        actual, predicted = targets.ravel(), predictions.ravel()
        end = max(actual.max(), predicted.max())
        plt.scatter(actual, predicted, alpha=.25, color="g")
        plt.plot([0, end], [0, end], "k--")
        plt.xlim(-500, end+500); plt.ylim(-500, end+500)
        plt.xlabel("Actual flow (m³/s)"); plt.ylabel("Predicted flow (m³/s)"); plt.grid(); 
        name = "actual_vs_predicted.pdf"
        plt.tight_layout(); plt.savefig(OUTPUT_DIR / name, bbox_inches="tight"); plt.show()
        
        plt.figure(figsize=(6, 4)); plt.hist(errors.ravel(), bins=50, density=True, alpha=.75, edgecolor="black", linewidth=.5)
        plt.axvline(errors.ravel().mean(), ls="--", lw=2, label=f"Mean = {errors.ravel().mean():.2f}", color="k")
        plt.xlabel("Residual (m³/s)"); plt.ylabel("Density"); #plt.title("Residual distribution"); 
        plt.grid(axis="y", alpha=.25); plt.legend(); 
        name = "residual_distribution.pdf"
        plt.tight_layout(); plt.savefig(OUTPUT_DIR / name, bbox_inches="tight"); plt.show()
          
        example = min(10, len(predictions) - 1)
        plt.figure(figsize=(6, 4)); plt.plot(future, targets[example], "o-", lw=2.3, label="Actual", color="b")
        plt.plot(future, predictions[example], "s--", lw=2.4, label="Prediction", color="r")
        plt.fill_between(future, targets[example], predictions[example], alpha=.15)
        plt.xlabel("Forecast horizon"); plt.ylabel("Natural flow (m³/s)"); #plt.title(f"Forecast example {example}")
        plt.grid(alpha=.25); plt.legend(); 
        name = "forecast_example.pdf"
        plt.tight_layout(); plt.savefig(OUTPUT_DIR / name, bbox_inches="tight"); plt.show()
        
        window_rmse = np.sqrt((errors**2).mean(1)); indices = {"best": window_rmse.argmin(),
            "median": np.argsort(window_rmse)[len(window_rmse)//2], "worst": window_rmse.argmax()}
        for name, index in indices.items():
            plt.figure(figsize=(6, 4)); plt.plot(future, targets[index], "o-", lw=2.3, label="Actual", color="b")
            plt.plot(future, predictions[index], "s--", lw=2.4, label="Prediction", color="r")
            plt.fill_between(future, targets[index], predictions[index], alpha=.15); plt.xlabel("Forecast horizon")
            plt.ylabel("Natural flow (m³/s)"); 
            #plt.title(f"{name.title()} forecast — RMSE = {window_rmse[index]:.2f} m³/s")
            plt.grid(alpha=.25); plt.legend(); 
            name = str(f"{name}_forecast.pdf")
            plt.tight_layout(); plt.savefig(OUTPUT_DIR / name, bbox_inches="tight"); plt.show()
        
        x, y = test_dataset[example]; batch = x.unsqueeze(0); 
        saliency = gradient_saliency(model, batch)[0]
        input_values = x.squeeze(-1).numpy() * data_std + data_mean; history_axis = np.arange(-params["lookback"], 0)
        importance = horizon_saliency(model, batch); plt.figure(figsize=(6, 4)); 
        image = plt.imshow(importance, aspect="auto", origin="lower")
        plt.colorbar(image, label="Normalized gradient importance"); 
        plt.xlabel("Historical input time step"); plt.ylabel("Forecast horizon")
        positions = np.linspace(0, params["lookback"] - 1, 9, dtype=int); 
        plt.xticks(positions, history_axis[positions]); plt.yticks(np.arange(HORIZON), future)
        #plt.title("Input importance by forecast horizon"); 
        name = "horizon_saliency_heatmap.pdf"
        plt.tight_layout(); plt.savefig(OUTPUT_DIR / name, bbox_inches="tight"); plt.show()
        
        patch_importance = occlusion_importance(model, batch, params["patch_size"])[0]; 
        patch_axis = np.arange(len(patch_importance))
        plt.figure(figsize=(6, 4)); plt.bar(patch_axis, patch_importance, color="b"); 
        plt.xlabel("Historical input patch"); plt.ylabel("Normalized importance"); 
        #plt.title("Patch occlusion importance"); 
        plt.grid(axis="y", alpha=.25); 
        name = "patch_occlusion_importance.pdf"
        plt.tight_layout(); plt.savefig(OUTPUT_DIR / name, bbox_inches="tight"); plt.show()
        
        fig, ax1 = plt.subplots(figsize=(6, 4)); 
        ax1.plot(history_axis, input_values, lw=2, label="Input series", color="k")
        ax1.set(xlabel="Time step relative to forecast", ylabel="Natural flow (m³/s)"); ax1.grid(alpha=.25)
        ax2 = ax1.twinx(); ax2.fill_between(history_axis, 0, saliency, alpha=.25, color="tab:orange", label="Gradient importance")
        ax2.set_ylabel("Normalized importance"); ax2.set_ylim(0, 1.05)
        lines1, labels1 = ax1.get_legend_handles_labels(); 
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center"); 
        ax1.set_xlim(-params["lookback"], 0)
        #plt.title("Input-time importance"); 
        name = "input_saliency.pdf"
        plt.tight_layout(); plt.savefig(OUTPUT_DIR / name, bbox_inches="tight"); plt.show()
        
        number = min(200, len(test_dataset)); 
        global_saliency = np.mean([gradient_saliency(model, test_dataset[i][0].unsqueeze(0))[0] for i in range(number)], axis=0)
        global_saliency /= global_saliency.max() + 1e-8; 
        plt.figure(figsize=(6, 4)); plt.plot(history_axis, global_saliency, lw=2.3, color="b")
        plt.fill_between(history_axis, 0, global_saliency, alpha=.2, color="b"); 
        plt.xlabel("Time step relative to forecast")
        plt.ylabel("Mean normalized importance"); plt.grid(alpha=.25); 
        #plt.title(f"Global temporal importance across {number} test windows"); 
        name = "global_temporal_importance.pdf"
        plt.tight_layout(); plt.savefig(OUTPUT_DIR / name, bbox_inches="tight"); plt.show()
        
        size = min(64, len(test_dataset)); x_diff = torch.stack([test_dataset[i][0] for i in range(size)]); 
        y_diff = torch.stack([test_dataset[i][1] for i in range(size)])
        diffusion_steps, losses = diffusion_loss_by_timestep(model, x_diff, y_diff)
        plt.figure(figsize=(6, 4)); plt.plot(diffusion_steps, losses, "o-", lw=2.2, color="k"); 
        plt.xlabel("Diffusion timestep")
        plt.ylabel("Noise-prediction MSE"); plt.grid(alpha=.25); 
        # plt.title("Diffusion loss by timestep"); 
        name = "diffusion_timestep_loss.pdf"
        plt.tight_layout(); plt.savefig(OUTPUT_DIR / name, bbox_inches="tight"); plt.show()
        
        n = min(300, len(predictions)); axis = np.arange(n)
        plt.figure(figsize=(6, 4)); plt.plot(axis, targets[:n, 0], "-", lw=2, label="Actual", color="b")
        plt.plot(axis, predictions[:n, 0], "--", lw=2, label="Prediction", color="r")
        plt.xlabel("Test window"); plt.ylabel("Natural flow (m³/s)"); #plt.title("One-step-ahead test predictions")
        plt.grid(alpha=.25); plt.legend(); plt.show()
        name = "continuous_predictions.pdf"
        plt.tight_layout(); plt.savefig(OUTPUT_DIR / name, bbox_inches="tight"); plt.show()
        
        results.append({"Seed": seed,
            "Parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            **metrics, "Train": train_time, "Test": test_time})
        
    def sci(x):
        exponent = int(np.floor(np.log10(abs(x)))) if x != 0 else 0
        coefficient = x / 10**exponent
        return f"{coefficient:.2f}$\\times 10^{{{exponent}}}$"
        
    print(f"Ours & {HORIZON}"
        f" & {sci(results_df.RMSE.mean())} $\\pm$ {sci(results_df.RMSE.std())}"
        f" & {sci(results_df.MAE.mean())} $\\pm$ {sci(results_df.MAE.std())}"
        f" & {sci(results_df.SMAPE.mean())} $\\pm$ {sci(results_df.SMAPE.std())}"
        f" & {sci(results_df.Train.mean())}"
        f" & {sci(results_df.Test.mean())} \\\\")
