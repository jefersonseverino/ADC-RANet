import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha 
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  
        focal_weight = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class FocalLossFLSD(nn.Module):
    def __init__(self, threshold_low=0.2, threshold_high=0.8, reduction='mean'):
        super().__init__()
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        gamma = torch.where(
            pt < 0.2,
            torch.full_like(pt, 5.0),
            torch.full_like(pt, 3.0),
        )

        focal_weight = (1.0 - pt) ** gamma
        loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def mixup_data(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, outputs, y_a, y_b, lam):
    if not isinstance(outputs, list):
        outputs = [outputs]

    loss = 0.0
    for output in outputs:
        loss += lam * criterion(output, y_a) + (1 - lam) * criterion(output, y_b)

    return loss

def cutmix_data(x, y, alpha=1.0):
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    _, _, H, W = x.shape
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    cy = np.random.randint(H)
    cx = np.random.randint(W)

    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    lam = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)

    return mixed_x, y, y[index], lam

def cutmix_criterion(criterion, outputs, y_a, y_b, lam):
    if not isinstance(outputs, list):
        outputs = [outputs]

    loss = 0.0
    for output in outputs:
        loss += lam * criterion(output, y_a) + (1 - lam) * criterion(output, y_b)

    return loss

class PerExitTemperatureScaling(nn.Module):
    def __init__(self, n_exits):
        super().__init__()
        # Inicializa T=1.0 (log(1)=0) pra cada exit
        self.log_temperatures = nn.Parameter(torch.zeros(n_exits))

    @property
    def temperatures(self):
        # T sempre positivo via exp
        return self.log_temperatures.exp()

    def forward(self, logits_list):
        """Calibra logits de todos os exits."""
        calibrated = []
        for k, logits in enumerate(logits_list):
            T = self.temperatures[k]
            calibrated.append(logits / T)
        return calibrated

    def calibrate_exit(self, logits, exit_idx):
        """Calibra logits de um exit específico."""
        T = self.temperatures[exit_idx]
        return logits / T


def fit_temperature_scaling(model, val_loader, n_exits, max_iter=100, lr=0.01):
    model.eval()
    device = next(model.parameters()).device

    all_logits = [[] for _ in range(n_exits)]
    all_targets = []

    with torch.no_grad():
        for input, target in val_loader:
            target = target.to(device)
            if isinstance(input, list):
                input = [inp.to(device) for inp in input]
            else:
                input = input.to(device)

            output = model(input)
            if not isinstance(output, list):
                output = [output]

            for k in range(n_exits):
                all_logits[k].append(output[k])
            all_targets.append(target)

    all_targets = torch.cat(all_targets)
    for k in range(n_exits):
        all_logits[k] = torch.cat(all_logits[k])

    temp_scaler = PerExitTemperatureScaling(n_exits).to(device)

    optimizer = torch.optim.LBFGS(
        temp_scaler.parameters(), lr=lr, max_iter=max_iter
    )

    def closure():
        optimizer.zero_grad()
        loss = 0.0
        for k in range(n_exits):
            scaled = temp_scaler.calibrate_exit(all_logits[k], k)
            loss += F.cross_entropy(scaled, all_targets)
        loss.backward()
        return loss

    optimizer.step(closure)

    print("\n[Temperature Scaling] Per-exit temperatures:")
    for k in range(n_exits):
        T_k = temp_scaler.temperatures[k].item()
        print(f"  Exit {k+1}: T = {T_k:.4f}")

    return temp_scaler

@torch.no_grad()
def fit_temperature_scaling_grid(model, val_loader, n_exits, 
                                  t_min=0.1, t_max=10.0, t_step=0.1, n_bins=15):
    model.eval()
    device = next(model.parameters()).device

    all_logits = [[] for _ in range(n_exits)]
    all_targets = []

    for input, target in val_loader:
        target = target.to(device)
        if isinstance(input, list):
            input = [inp.to(device) for inp in input]
        else:
            input = input.to(device)
        output = model(input)
        if not isinstance(output, list):
            output = [output]
        for k in range(n_exits):
            all_logits[k].append(output[k])
        all_targets.append(target)

    all_targets = torch.cat(all_targets)
    for k in range(n_exits):
        all_logits[k] = torch.cat(all_logits[k])

    temp_scaler = PerExitTemperatureScaling(n_exits).to(device)
    temperatures = torch.arange(t_min, t_max + t_step, t_step)

    for k in range(n_exits):
        best_ece, best_t = float('inf'), 1.0
        for t in temperatures:
            scaled = all_logits[k] / t.item()
            ece, _, _, _ = compute_ece(scaled, all_targets, n_bins=n_bins)
            if ece < best_ece:
                best_ece = ece
                best_t = t.item()
        
        temp_scaler.log_temperatures.data[k] = math.log(best_t)
        print(f"  Exit {k+1}: T = {best_t:.2f}  (val ECE = {best_ece:.4f})")

    return temp_scaler

class PerExitVectorScaling(nn.Module):
    def __init__(self, n_exits, n_classes):
        super().__init__()
        self.W = nn.ParameterList([
            nn.Parameter(torch.ones(n_classes)) for _ in range(n_exits)
        ])
        self.b = nn.ParameterList([
            nn.Parameter(torch.zeros(n_classes)) for _ in range(n_exits)
        ])

    def forward(self, logits_list):
        calibrated = []
        for k, logits in enumerate(logits_list):
            calibrated.append(logits * self.W[k] + self.b[k])
        return calibrated

    def calibrate_exit(self, logits, exit_idx):
        return logits * self.W[exit_idx] + self.b[exit_idx]


def fit_vector_scaling(model, val_loader, n_exits, n_classes,
                       max_iter=100, lr=0.01):
    model.eval()
    device = next(model.parameters()).device

    all_logits = [[] for _ in range(n_exits)]
    all_targets = []

    with torch.no_grad():
        for input, target in val_loader:
            target = target.to(device)
            if isinstance(input, list):
                input = [inp.to(device) for inp in input]
            else:
                input = input.to(device)

            output = model(input)
            if not isinstance(output, list):
                output = [output]

            for k in range(n_exits):
                all_logits[k].append(output[k])
            all_targets.append(target)

    all_targets = torch.cat(all_targets)
    for k in range(n_exits):
        all_logits[k] = torch.cat(all_logits[k])

    vec_scaler = PerExitVectorScaling(n_exits, n_classes).to(device)

    optimizer = torch.optim.LBFGS(
        vec_scaler.parameters(), lr=lr, max_iter=max_iter
    )

    def closure():
        optimizer.zero_grad()
        loss = 0.0
        for k in range(n_exits):
            scaled = vec_scaler.calibrate_exit(all_logits[k], k)
            loss += F.cross_entropy(scaled, all_targets)
        loss.backward()
        return loss

    optimizer.step(closure)

    print("\n[Vector Scaling] Per-exit parameters:")
    for k in range(n_exits):
        w_mean = vec_scaler.W[k].data.mean().item()
        b_mean = vec_scaler.b[k].data.mean().item()
        print(f"  Exit {k+1}: W_mean={w_mean:.4f}, b_mean={b_mean:.4f}")

    return vec_scaler

@torch.no_grad()
def compute_ece(logits, targets, n_bins=15):
    probs = F.softmax(logits, dim=1)
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(targets)

    ece = torch.zeros(1, device=logits.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & \
                 (confidences <= bin_boundaries[i + 1])
        count = in_bin.float().sum().item()
        bin_counts.append(count)

        if count > 0:
            acc = accuracies[in_bin].float().mean().item()
            conf = confidences[in_bin].mean().item()
            ece += (count / len(targets)) * abs(acc - conf)
            bin_accs.append(acc)
            bin_confs.append(conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)

    return ece.item(), bin_accs, bin_confs, bin_counts


@torch.no_grad()
def evaluate_calibration(model, loader, n_exits, calibrator=None):
    model.eval()
    if calibrator is not None:
        calibrator.eval()

    device = next(model.parameters()).device
    all_logits = [[] for _ in range(n_exits)]
    all_targets = []

    for input, target in loader:
        target = target.to(device)
        if isinstance(input, list):
            input = [inp.to(device) for inp in input]
        else:
            input = input.to(device)

        output = model(input)
        if not isinstance(output, list):
            output = [output]

        for k in range(n_exits):
            all_logits[k].append(output[k])
        all_targets.append(target)

    all_targets = torch.cat(all_targets)
    for k in range(n_exits):
        all_logits[k] = torch.cat(all_logits[k])

    results = {}
    for k in range(n_exits):
        ece_raw, _, _, _ = compute_ece(all_logits[k], all_targets)

        if calibrator is not None:
            calibrated = calibrator.calibrate_exit(all_logits[k], k)
            ece_cal, _, _, _ = compute_ece(calibrated, all_targets)
        else:
            ece_cal = None

        preds = all_logits[k].argmax(dim=1)
        acc = preds.eq(all_targets).float().mean().item() * 100

        results[k] = {
            'ece_raw': ece_raw,
            'ece_calibrated': ece_cal,
            'accuracy': acc,
        }

        if ece_cal is not None:
            print(f"  Exit {k+1}: acc={acc:.2f}%  "
                  f"ECE_raw={ece_raw:.4f}  ECE_cal={ece_cal:.4f}  "
                  f"(Δ={ece_raw - ece_cal:+.4f})")
        else:
            print(f"  Exit {k+1}: acc={acc:.2f}%  ECE={ece_raw:.4f}")

    return results

def build_criterion(args):
    label_smoothing = getattr(args, 'label_smoothing', 0.0)
    if getattr(args, 'use_focal_loss', False):
        gamma = getattr(args, 'focal_gamma', 2.0)
        if args.focal_flsd:
            print(f"[Criterion] FocalLoss FLSD (gamma_mid={gamma})")
            return FocalLossFLSD().cuda()
        else:
            print(f"[Criterion] FocalLoss (gamma={gamma})")
            return FocalLoss(gamma=gamma).cuda()

    return nn.CrossEntropyLoss(label_smoothing=label_smoothing).cuda()


def run_post_hoc_calibration(model, val_loader, test_loader, args):
    calibration = getattr(args, 'calibration', None)
    if calibration is None:
        return None

    n_exits = args.num_exits

    print(f"\n{'='*60}")
    print(f"Post-hoc Calibration: {calibration}")
    print(f"{'='*60}")

    print("\n--- Before calibration ---")
    evaluate_calibration(model, test_loader, n_exits)

    if calibration == 'temp_scaling':
        calibrator = fit_temperature_scaling(model, val_loader, n_exits)
    elif calibration == 'temp_scaling_grid':
        calibrator = fit_temperature_scaling_grid(model, val_loader, n_exits)
    elif calibration == 'vector_scaling':
        calibrator = fit_vector_scaling(
            model, val_loader, n_exits, args.num_classes
        )
    else:
        print(f"Unknown calibration method: {calibration}")
        return None

    print("\n--- After calibration ---")
    evaluate_calibration(model, test_loader, n_exits, calibrator)

    return calibrator

def post_training_threshold_optimization(
    val_loader, model, router, flops_per_exit, calibration_size=0.2
):
    import optuna

    thresholds = router.get_hard_thresholds().cpu().tolist()
    n_exits = router.n_exits
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)

    # Use calibration reduced version of validation set for faster threshold search
    n_samples = len(val_loader.dataset)
    n_calibration = int(n_samples * calibration_size)
    calibration_subset = torch.utils.data.Subset(
        val_loader.dataset, range(n_calibration)
    )
    calibration_loader = torch.utils.data.DataLoader(
        calibration_subset,
        batch_size=val_loader.batch_size,
        shuffle=False,
        num_workers=val_loader.num_workers,
        pin_memory=False,
        collate_fn=val_loader.collate_fn,
    )

    def objective(trial):
        new_thresholds = []
        for k in range(n_exits - 1):
            min_th = max(0.0, thresholds[k] - 0.5)
            max_th = min(1.0, thresholds[k] + 0.5)
            t_k = trial.suggest_float(f"T_{k}", min_th, max_th)
            new_thresholds.append(t_k)
        new_thresholds.append(-1e8)  # Last exit always accepts
        router.set_hard_thresholds(new_thresholds)
        routed_acc, avg_flops, _ = routed_evaluate(
            calibration_loader, model, router, flops_per_exit
        )
        print(routed_acc, avg_flops)
        return routed_acc, avg_flops

    study = optuna.create_study(
        directions=["maximize", "minimize"],
        sampler=optuna.samplers.GPSampler(),
        pruner=pruner,
    )

    study.optimize(objective, n_trials=200)
    optuna.visualization.plot_pareto_front(
        study, target_names=["Routed Accuracy", "Avg FLOPs"]
    )

    print(f"Number of trials on the Pareto front: {len(study.best_trials)}")

    trial_with_highest_accuracy = max(study.best_trials, key=lambda t: t.values[0])
    print("Trial with highest accuracy: ")
    print(f"\tnumber: {trial_with_highest_accuracy.number}")
    print(f"\tparams: {trial_with_highest_accuracy.params}")
    print(f"\tvalues: {trial_with_highest_accuracy.values}")

    trial_with_lowest_flops = min(study.best_trials, key=lambda t: t.values[1])
    print("Trial with lowest FLOPs: ")
    print(f"\tnumber: {trial_with_lowest_flops.number}")
    print(f"\tparams: {trial_with_lowest_flops.params}")
    print(f"\tvalues: {trial_with_lowest_flops.values}")

    # define the router with the best accruracy point on the Pareto front
    best_accuracy_trial = max(study.best_trials, key=lambda t: t.values[0])
    best_thresholds = [best_accuracy_trial.params[f"T_{k}"] for k in range(n_exits - 1)]
    print(best_thresholds)
    best_thresholds.append(-1e8)
    router.set_hard_thresholds(best_thresholds)
