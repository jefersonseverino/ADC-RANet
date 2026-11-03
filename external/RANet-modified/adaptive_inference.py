from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import os
import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# from useful.plot_flop_acc import plot_flop_acc_curve

def log_exit_distribution(results, args):
    import csv
    """Log per-exit sample distribution to W&B, print, and save CSV."""
    n_stage = args.num_exits
    flops = torch.load(os.path.join(args.save, "flops.pth"))
    
    print("\n" + "=" * 70)
    print("EXIT DISTRIBUTION ANALYSIS")
    print("=" * 70)
    
    points = []
    for r in results:
        f_val = r["flops"].item() if torch.is_tensor(r["flops"]) else r["flops"]
        points.append((f_val, r["acc"], r))
    points.sort(key=lambda x: x[0])
    
    pareto = []
    best_acc = -1
    for f, a, r in points:
        if a > best_acc:
            pareto.append((f, a, r))
            best_acc = a
    
    pareto_select = getattr(args, "pareto_select", "knee")
    
    if pareto_select == "first":
        selected = pareto[0][2]
        select_label = "FIRST (lowest FLOPs)"
    elif pareto_select == "last":
        selected = pareto[-1][2]
        select_label = "LAST (highest accuracy)"
    else:  # knee
        if len(pareto) >= 2:
            f_min, a_min = pareto[0][0], pareto[0][1]
            f_max, a_max = pareto[-1][0], pareto[-1][1]
            f_range = f_max - f_min if f_max != f_min else 1.0
            a_range = a_max - a_min if a_max != a_min else 1.0
            best_dist = -1
            selected = pareto[0][2]
            for f, a, r in pareto:
                fn = (f - f_min) / f_range
                an = (a - a_min) / a_range
                dist = an - fn
                if dist > best_dist:
                    best_dist = dist
                    selected = r
        else:
            selected = pareto[0][2]
        select_label = "KNEE (best cost-benefit)"
    
    for idx, (f, a, r) in enumerate(pareto):
        marker = f" <-- {pareto_select.upper()}" if r is selected else ""
        print(f"  Pareto #{idx+1}: p={r['p']}, acc={a:.2f}%, flops={f/1e6:.2f}M{marker}")
    
    sel_idx = [r for _, _, r in pareto].index(selected) + 1
    print(f"\nSelected: point #{sel_idx} of {len(pareto)} ({select_label})")
    
    f_val = selected["flops"].item() if torch.is_tensor(selected["flops"]) else selected["flops"]
    print(f"  acc={selected['acc']:.2f}%, flops={f_val/1e6:.2f}M")
    print(f"  Thresholds: {[f'{t:.4f}' for t in selected['T'].tolist()]}")

    has_ece = "exit_ece" in selected

    print(f"  {'Exit':<8} {'Samples':<10} {'%':<8} {'Acc':<8} {'FLOPs (M)':<10}", end="")
    if has_ece:
        print(f" {'ECE_routed':<12} {'ECE_all':<10} {'Avg Conf':<10} {'SignedBias':<12}", end="")
    print()
    print("  " + "-" * (50 + (48 if has_ece else 0)))
    
    for k in range(n_stage):
        count = int(selected["exit_counts"][k].item())
        pct = count * 100.0 / selected["n_samples"]
        acc_k = (selected["exit_correct"][k] * 100.0 / max(count, 1)).item()
        print(f"    @{k+1:<5} {count:<10} {pct:<8.1f} {acc_k:<8.1f} "
              f"{flops[k]/1e6:<10.2f}", end="")
        if has_ece and count > 0:
            # signed bias = avg_conf - avg_acc (negative = underconfident)
            avg_conf = selected['exit_avg_conf'][k]
            # try to compute avg_acc from provided fields
            if 'exit_acc' in selected:
                avg_acc = selected['exit_acc'][k]
            else:
                avg_acc = (selected['exit_correct'][k].item() * 1.0 / max(count, 1))
            signed_bias = avg_conf - avg_acc
            ece_all = selected.get('exit_ece_all', [None]*n_stage)[k]
            print(f" {selected['exit_ece'][k]:<12.4f} {ece_all:<10.4f} {avg_conf:<10.4f} {signed_bias:<12.4f}", end="")
        elif has_ece:
            print(f" {'---':<10} {'---':<10} {'---':<12}", end="")
        print()
    
    if wandb.run is not None:
        for k in range(n_stage):
            count = int(selected["exit_counts"][k].item())
            pct = count * 100.0 / selected["n_samples"]
            acc_k = (selected["exit_correct"][k] * 100.0 / max(count, 1)).item()
            log_dict = {
                f"exit_{k+1}_samples": count,
                f"exit_{k+1}_pct": pct,
                f"exit_{k+1}_acc_routed": acc_k,
            }
            if has_ece and count > 0:
                log_dict[f"exit_{k+1}_ece_routed"] = selected["exit_ece"][k]
                log_dict[f"exit_{k+1}_ece_all"] = selected.get("exit_ece_all", [None]*n_stage)[k]
                log_dict[f"exit_{k+1}_avg_conf_routed"] = selected["exit_avg_conf"][k]
                signed_bias = selected['exit_avg_conf'][k] - (selected['exit_correct'][k].item() * 1.0 / max(count, 1))
                log_dict[f"exit_{k+1}_signed_bias_routed"] = signed_bias
            wandb.log(log_dict)
        
        pareto_table = wandb.Table(columns=["acc", "flops_M"])
        for f, a, r in pareto:
            pareto_table.add_data(a, f / 1e6)
        wandb.log({"pareto_front": pareto_table})
        
        columns = ["p", "acc", "flops_M"] + \
                  [f"exit_{k+1}_count" for k in range(n_stage)] + \
                  [f"exit_{k+1}_pct" for k in range(n_stage)]
        table = wandb.Table(columns=columns)
        for r in results:
            f_r = r["flops"].item() if torch.is_tensor(r["flops"]) else r["flops"]
            row = [r["p"], r["acc"], f_r / 1e6]
            for k in range(n_stage):
                row.append(int(r["exit_counts"][k].item()))
            for k in range(n_stage):
                row.append(r["exit_counts"][k].item() * 100.0 / r["n_samples"])
            table.add_data(*row)
        wandb.log({"exit_distribution_table": table})
        
        data = [[f"Exit {k+1}", int(selected["exit_counts"][k].item())]
                for k in range(n_stage)]
        table_bar = wandb.Table(data=data, columns=["Exit", "Samples"])
        wandb.log({
            "exit_distribution_bar": wandb.plot.bar(
                table_bar, "Exit", "Samples",
                title=f"Exit Distribution ({select_label})"
            )
        })
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    saves_dir = os.path.join(script_dir, "saves")
    os.makedirs(saves_dir, exist_ok=True)
    
    save_id = os.path.basename(args.save.rstrip("/"))
    save_name = f"{args.wandb_group}_{save_id}"
    
    csv_all = os.path.join(saves_dir, f"{save_name}_sweep_{args.pareto_select}.csv")
    with open(csv_all, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["p", "acc", "flops", "is_pareto", "is_selected"] + \
                 [f"exit_{k+1}_count" for k in range(n_stage)] + \
                 [f"exit_{k+1}_pct" for k in range(n_stage)] + \
                 [f"exit_{k+1}_acc" for k in range(n_stage)]
        if has_ece:
            header += [f"exit_{k+1}_ece_routed" for k in range(n_stage)]
            header += [f"exit_{k+1}_ece_all" for k in range(n_stage)]
            header += [f"exit_{k+1}_avg_conf" for k in range(n_stage)]
            header += [f"exit_{k+1}_signed_bias_routed" for k in range(n_stage)]
        writer.writerow(header)
        
        pareto_ps = {r["p"] for _, _, r in pareto}
        for r in results:
            f_r = r["flops"].item() if torch.is_tensor(r["flops"]) else r["flops"]
            row = [
                r["p"],
                round(r["acc"], 4),
                round(f_r, 2),
                r["p"] in pareto_ps,
                r is selected,
            ]
            for k in range(n_stage):
                row.append(int(r["exit_counts"][k].item()))
            for k in range(n_stage):
                row.append(round(r["exit_counts"][k].item() * 100.0 / r["n_samples"], 2))
            for k in range(n_stage):
                count = max(int(r["exit_counts"][k].item()), 1)
                row.append(round((r["exit_correct"][k] * 100.0 / count).item(), 2))
            if has_ece:
                for k in range(n_stage):
                    row.append(round(r["exit_ece"][k], 4))
                for k in range(n_stage):
                    row.append(round(r.get("exit_ece_all", [0.0]*n_stage)[k], 4))
                for k in range(n_stage):
                    row.append(round(r["exit_avg_conf"][k], 4))
                for k in range(n_stage):
                    row.append(round(r["exit_signed_bias"][k], 4))
            writer.writerow(row)
    
    csv_selected = os.path.join(saves_dir, f"{save_name}_selected_{args.pareto_select}.csv")
    with open(csv_selected, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["exit", "samples", "pct", "acc_routed", "flops_M", "threshold"]
        if has_ece:
            header += ["ece_routed", "ece_all", "avg_conf", "signed_bias"]
        writer.writerow(header)
        for k in range(n_stage):
            count = int(selected["exit_counts"][k].item())
            pct = round(count * 100.0 / selected["n_samples"], 2)
            acc_k = round((selected["exit_correct"][k] * 100.0 / max(count, 1)).item(), 2)
            flops_k = round(flops[k] / 1e6, 2)
            t_k = round(selected["T"][k].item(), 6)
            row = [k + 1, count, pct, acc_k, flops_k, t_k]
            if has_ece:
                row += [round(selected["exit_ece"][k], 4),
                    round(selected.get("exit_ece_all", [0.0]*n_stage)[k], 4),
                    round(selected["exit_avg_conf"][k], 4),
                    round(selected.get("exit_signed_bias", [0.0]*n_stage)[k], 4)]
            writer.writerow(row)

    print(f"\nSaved to:")
    print(f"  {csv_all}")
    print(f"  {csv_selected}")

def compute_ece_from_probs(probs, targets, n_bins=15):
    if probs.numel() == 0:
        return 0.0, 0.0, 0.0
    
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(targets.long())
    
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        count = mask.sum().item()
        if count > 0:
            avg_conf = confidences[mask].mean().item()
            avg_acc = accuracies[mask].float().mean().item()
            ece += (count / len(confidences)) * abs(avg_acc - avg_conf)
    
    return ece, confidences.mean().item(), accuracies.float().mean().item()


def calibrate_temperature(softmax_logits, targets, lr=0.01, max_iter=50):
    """
    Learn one temperature scalar per exit that minimises NLL on the
    validation set.

    Args:
        softmax_logits: (n_exits, n_samples, n_classes)
        targets:        (n_samples,)
        lr:             learning rate for L-BFGS.
        max_iter:       max L-BFGS iterations per exit.

    Returns:
        temperatures: list[float] of length n_exits.
    """
    n_stage = softmax_logits.size(0)
    targets_long = targets.long()
    temperatures = []

    for k in range(n_stage):
        raw = torch.log(softmax_logits[k].clamp(min=1e-8))
        temp = nn.Parameter(torch.ones(1, device=raw.device))
        optimizer = torch.optim.LBFGS([temp], lr=lr, max_iter=max_iter)
        nll = nn.CrossEntropyLoss()

        def _closure(raw=raw, temp=temp):
            optimizer.zero_grad()
            loss = nll(raw / temp.clamp(min=0.01), targets_long)
            loss.backward()
            return loss

        optimizer.step(_closure)
        temperatures.append(temp.item())
        print(f"  Exit {k}: temperature = {temp.item():.4f}")

    return temperatures


def apply_temperature(softmax_logits, temperatures):
    """
    Re-scale softmax probs with learned temperatures.

    Args:
        softmax_logits: (n_exits, n_samples, n_classes)
        temperatures:   list[float] of length n_exits

    Returns:
        calibrated: (n_exits, n_samples, n_classes)
    """
    n_stage = softmax_logits.size(0)
    calibrated = torch.empty_like(softmax_logits)
    for k in range(n_stage):
        raw = torch.log(softmax_logits[k].clamp(min=1e-8))
        calibrated[k] = F.softmax(raw / max(temperatures[k], 0.01), dim=1)
    return calibrated


# ---------------------------------------------------------------------------
# Confidence metrics
# ---------------------------------------------------------------------------
def compute_confidence(softmax_logits, metric="max_softmax"):
    """
    Compute per-exit confidence from softmax probabilities.

    Args:
        softmax_logits: (n_exits, n_samples, n_classes)
        metric: 'max_softmax', 'margin', or 'hybrid'
                hybrid = max_softmax on first half of exits, margin on second half

    Returns:
        confidence: (n_exits, n_samples)
        argmax_preds: (n_exits, n_samples)
    """
    n_exits = softmax_logits.size(0)
    argmax_preds = softmax_logits.argmax(dim=2)

    if metric == "margin":
        top2 = softmax_logits.topk(2, dim=2).values
        confidence = top2[:, :, 0] - top2[:, :, 1]
    elif metric == "hybrid":
        cutoff = n_exits // 2  # first half uses max_softmax
        max_conf = softmax_logits.max(dim=2).values
        top2 = softmax_logits.topk(2, dim=2).values
        margin_conf = top2[:, :, 0] - top2[:, :, 1]
        confidence = torch.empty_like(max_conf)
        confidence[:cutoff] = max_conf[:cutoff]
        confidence[cutoff:] = margin_conf[cutoff:]
    else: 
        confidence = softmax_logits.max(dim=2).values

    return confidence, argmax_preds


def _extract_pareto_front(flops_list, acc_list, configs):
    points = list(zip(flops_list, acc_list, configs))
    points.sort(key=lambda x: x[0])

    pareto = []
    best_acc = -1
    for flops, acc, cfg in points:
        if acc > best_acc:
            pareto.append((flops, acc, cfg))
            best_acc = acc

    p_flops = [x[0] for x in pareto]
    p_acc = [x[1] for x in pareto]
    p_configs = [x[2] for x in pareto]
    return p_flops, p_acc, p_configs


def plot_flop_acc_curve(all_flops, all_acc, pareto_flops, pareto_acc, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.plot(
        all_flops,
        all_acc,
        label="All Configurations",
        marker="o",
        linestyle="-",
        color="blue",
    )
    plt.plot(
        pareto_flops,
        pareto_acc,
        label="Pareto Front",
        marker="x",
        linestyle="--",
        color="red",
    )
    plt.xlabel("FLOPs (Millions)")
    plt.ylabel("Accuracy (%)")
    plt.title("FLOPs vs Accuracy Curve")
    plt.legend()
    plt.grid(True)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()


def dynamic_evaluate(model, test_loader, val_loader, args):
    tester = Tester(model, args)
    if os.path.exists(os.path.join(args.save, "logits_single.pth")):
        val_pred, val_target, test_pred, test_target = torch.load(
            os.path.join(args.save, "logits_single.pth")
        )
    else:
        val_pred, val_target = tester.calc_logit(val_loader)
        test_pred, test_target = tester.calc_logit(test_loader)
        torch.save(
            (val_pred, val_target, test_pred, test_target),
            os.path.join(args.save, "logits_single.pth"),
        )

    flops = torch.load(os.path.join(args.save, "flops.pth"))

    if getattr(args, "calibrate", False):
        print("\n[Temperature Scaling] Calibrating per-exit temperatures on val set...")
        temperatures = calibrate_temperature(val_pred, val_target)
        print(f"[Temperature Scaling] Applying to val and test logits...")
        val_pred = apply_temperature(val_pred, temperatures)
        test_pred = apply_temperature(test_pred, temperatures)
        torch.save(temperatures, os.path.join(args.save, "temperatures.pth"))

    confidence_metric = getattr(args, "confidence_metric", "max_softmax")
    print(f"[Confidence] Using metric: {confidence_metric}")

    method = getattr(args, "threshold_method", "sweep")

    all_acc, all_flops, all_T = [], [], []
    all_samples = {}

    if method == "bayesian":
        if not HAS_OPTUNA:
            raise ImportError(
                "Optuna is required for bayesian method. Install with: pip install optuna"
            )

        val_conf = val_pred.max(dim=2)[0]
        test_conf = test_pred.max(dim=2)[0]

        n_trials = getattr(args, "bayesian_trials", 200)
        print(f"Running Bayesian Optimization with {n_trials} trials...")

        results = optimize_thresholds_bayesian(
            val_conf,
            val_pred,
            val_target,
            test_conf,
            test_pred,
            test_target,
            flops,
            n_trials=n_trials,
        )

        for i, r in enumerate(results):
            acc_test, exp_flops, exit_buckets, _ = tester.dynamic_eval_with_threshold(
                test_pred, test_target, flops, r["T"]
            )
            all_acc.append(acc_test)
            all_flops.append(exp_flops)
            all_T.append(r["T"])
            all_samples[i + 1] = exit_buckets
            print(
                f"  Trial {i+1}: test_acc={acc_test:.2f}%, flops={exp_flops/1e6:.1f}M"
            )

        configs = list(range(1, len(results) + 1))
    else:
        for p in range(1, 40):
            _p = torch.FloatTensor(1).fill_(p * 1.0 / 20)
            probs = torch.exp(torch.log(_p) * torch.range(1, args.num_exits))
            probs /= probs.sum()

            acc_val, _, T = tester.dynamic_eval_find_threshold(
                val_pred,
                val_target,
                probs,
                flops,
                confidence_metric=confidence_metric,
            )

            acc_test, exp_flops, exit_buckets, _ = tester.dynamic_eval_with_threshold(
                test_pred,
                test_target,
                flops,
                T,
                confidence_metric=confidence_metric,
            )

            all_acc.append(acc_test)
            all_flops.append(exp_flops)
            all_T.append(T)
            all_samples[p] = exit_buckets

        configs = list(range(1, 40))

    flops_budget_M = getattr(args, "flops_budget", None)
    if flops_budget_M is not None and HAS_OPTUNA:
        budget = flops_budget_M * 1e6
        n_trials = 500  # getattr(args, 'budget_trials', 500)
        n_exits = args.num_exits
        n_folds = getattr(args, "optuna_folds", 5)

        val_conf = val_pred.max(dim=2).values
        conf_lo = [val_conf[k].min().item() for k in range(n_exits)]
        conf_hi = [val_conf[k].max().item() for k in range(n_exits)]

        n_val = val_pred.size(1)
        perm = torch.randperm(n_val)
        fold_size = n_val // n_folds
        fold_idx = []
        for f_i in range(n_folds):
            s = f_i * fold_size
            e = s + fold_size if f_i < n_folds - 1 else n_val
            fold_idx.append(perm[s:e])

        print(f"\n[Budget] Maximize accuracy with FLOPs <= {flops_budget_M:.1f}M")
        print(f"  {n_trials} trials, {n_folds}-fold CV, budget={budget/1e6:.1f}M")

        def budget_objective(trial):
            T_vals = []
            for k in range(n_exits - 1):
                T_vals.append(trial.suggest_float(f"T_{k}", conf_lo[k], conf_hi[k]))
            T_vals.append(-1e8)
            T_tensor = torch.tensor(T_vals)

            fold_accs = []
            fold_flops = []
            for ho in range(n_folds):
                ho_logits = val_pred[:, fold_idx[ho], :]
                ho_targets = val_target[fold_idx[ho]]
                acc_f, flops_f, _, _ = tester.dynamic_eval_with_threshold(
                    ho_logits,
                    ho_targets,
                    flops,
                    T_tensor,
                )
                fold_accs.append(acc_f)
                fold_flops.append(
                    flops_f.item() if torch.is_tensor(flops_f) else flops_f
                )

            mean_acc = sum(fold_accs) / n_folds
            mean_flops = sum(fold_flops) / n_folds

            if mean_flops > budget:
                return -1.0  # infeasible
            return mean_acc

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=min(20, n_trials // 5),
                multivariate=True,
                seed=111,
            ),
        )

        for i, (a, f, t) in enumerate(zip(all_acc, all_flops, all_T)):
            f_v = f.item() if torch.is_tensor(f) else f
            if f_v <= budget:
                params = {}
                valid = True
                for k in range(n_exits - 1):
                    v = t[k].item() if torch.is_tensor(t[k]) else t[k]
                    v = max(conf_lo[k], min(conf_hi[k], v))
                    params[f"T_{k}"] = v
                study.enqueue_trial(params)

        study.optimize(budget_objective, n_trials=n_trials, show_progress_bar=True)

        n_added = 0
        seen = set()
        for trial in sorted(
            study.trials,
            key=lambda t: t.value if t.value is not None else -999,
            reverse=True,
        )[:20]:
            if trial.value is None or trial.value < 0:
                continue
            T_vals = [trial.params[f"T_{k}"] for k in range(n_exits - 1)] + [-1e8]
            T_key = tuple(round(v, 6) for v in T_vals)
            if T_key in seen:
                continue
            seen.add(T_key)
            T_tensor = torch.tensor(T_vals)
            acc_test, exp_flops, exit_buckets, _ = tester.dynamic_eval_with_threshold(
                test_pred,
                test_target,
                flops,
                T_tensor,
            )
            all_acc.append(acc_test)
            all_flops.append(exp_flops)
            all_T.append(T_tensor)
            all_samples[len(all_acc)] = exit_buckets
            n_added += 1
            ef = exp_flops.item() if torch.is_tensor(exp_flops) else exp_flops
            mark = " OK" if ef <= budget else " (over)"
            print(f"  acc={acc_test:.2f}%, flops={ef/1e6:.2f}M{mark}")

        print(f"  Added {n_added} budget-constrained points")
        configs = list(range(1, len(all_acc) + 1))

    pareto_flops, pareto_acc, pareto_configs = _extract_pareto_front(
        [f.item() if torch.is_tensor(f) else f for f in all_flops],
        all_acc,
        configs,
    )

    with open(os.path.join(args.save, "dynamic.txt"), "w") as fout:
        fout.write("# All operating points\n")
        for acc, ef in zip(all_acc, all_flops):
            ef_val = ef.item() if torch.is_tensor(ef) else ef
            fout.write("{}\t{}\n".format(acc, ef_val))

    for acc, ef in zip(all_acc, all_flops):
        ef_val = ef.item() if torch.is_tensor(ef) else ef
        print("Config: acc={:.3f}, flops={:.2f}M".format(acc, ef_val / 1e6))

    filepath = args.evaluate_from.split("/")

    filename = ""
    for paths in filepath:
        if "default" in paths:
            filename = paths
            break

    plot_flop_acc_curve(
        all_flops,
        all_acc,
        pareto_flops,
        pareto_acc,
        save_path=os.path.join(
            "imgs",
            f"flop_acc_curve_RANet_{args.calibrate}_{confidence_metric}_{filename}.png",
        ),
    )

    pareto_thresholds = [all_T[c - 1] for c in pareto_configs]
    torch.save(
        {
            "pareto_flops": pareto_flops,
            "pareto_acc": pareto_acc,
            "pareto_configs": pareto_configs,
            "pareto_thresholds": pareto_thresholds,
            "all_flops": all_flops,
            "all_acc": all_acc,
        },
        os.path.join(args.save, "pareto.pth"),
    )

    torch.save([all_flops, all_acc], os.path.join(args.save, "dynamic.pth"))
    torch.save(all_samples, os.path.join(args.save, "exit_samples_by_p.pth"))

    return all_acc, all_flops, pareto_flops, pareto_acc, pareto_thresholds

def evaluate_fixed_threshold(model, test_loader, val_loader, args, T, calibration_mode=None):
    """
    Same routing logic as evaluate_exit_distribution, but T is a fixed, absolute
    confidence threshold vector you choose yourself (not searched by quota).
    Use the SAME T for every model/calibration you want to compare, so any
    difference in exit-acceptance comes purely from the confidence values
    themselves (i.e. from calibration), not from quota-matching.
 
    T: list/tensor of length n_stage, e.g. [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, -1e9]
       (last exit is usually set very low / -inf so every remaining sample exits there)
    """
    tester = Tester(model, args)
 
    if os.path.exists(os.path.join(args.save, "logits_single.pth")):
        val_pred, val_target, test_pred, test_target = torch.load(
            os.path.join(args.save, "logits_single.pth")
        )
    else:
        val_pred, val_target = tester.calc_logit(val_loader)
        test_pred, test_target = tester.calc_logit(test_loader)
 
    if calibration_mode == "temp_scaling":
        temperatures = calibrate_temperature(val_pred, val_target)
        test_pred = apply_temperature(test_pred, temperatures)
 
    n_stage = test_pred.size(0)
    n_sample = test_target.size(0)
    confidence_metric = getattr(args, "confidence_metric", "max_softmax")
 
    max_preds, argmax_preds = compute_confidence(test_pred, metric=confidence_metric)
 
    exit_counts = torch.zeros(n_stage)
    exit_correct = torch.zeros(n_stage)
 
    for i in range(n_sample):
        gold = int(test_target[i].item())
        for k in range(n_stage):
            if max_preds[k][i].item() >= T[k]:
                exit_counts[k] += 1
                if gold == int(argmax_preds[k][i].item()):
                    exit_correct[k] += 1
                break
 
    exit_pct = [100.0 * c.item() / n_sample for c in exit_counts]
    exit_acc = [100.0 * exit_correct[k].item() / exit_counts[k].item() if exit_counts[k] > 0 else 0.0
                for k in range(n_stage)]
 
    return {"exit_pct": exit_pct, "exit_acc": exit_acc, "exit_counts": exit_counts}

def evaluate_exit_distribution(model, test_loader, val_loader, args, calibration_mode=None):
    tester = Tester(model, args)
 
    if os.path.exists(os.path.join(args.save, "logits_single.pth")):
        val_pred, val_target, test_pred, test_target = torch.load(
            os.path.join(args.save, "logits_single.pth")
        )
    else:
        val_pred, val_target = tester.calc_logit(val_loader)
        test_pred, test_target = tester.calc_logit(test_loader)
 
    temperatures = None
    if calibration_mode == "temp_scaling":
        print("[TempScaling] Fitting per-exit temperature on val set...")
        temperatures = calibrate_temperature(val_pred, val_target)
        val_pred = apply_temperature(val_pred, temperatures)
        test_pred = apply_temperature(test_pred, temperatures)
    
    flops = torch.load(os.path.join(args.save, "flops.pth"))
    n_stage = val_pred.size(0)
    n_sample = test_target.size(0)
    confidence_metric = getattr(args, "confidence_metric", "max_softmax")
 
    results = []
 
    for p in range(1, 5):
        _p = torch.FloatTensor(1).fill_(p * 1.0 / 2.5)
        probs = torch.exp(torch.log(_p) * torch.range(1, args.num_exits))
        probs /= probs.sum()
 
        acc_val, _, T = tester.dynamic_eval_find_threshold(
            val_pred,
            val_target,
            probs,
            flops,
            confidence_metric=confidence_metric,
        )
        acc_test, exp_flops, exit_buckets, _ = tester.dynamic_eval_with_threshold(
            test_pred,
            test_target,
            flops,
            T,
            confidence_metric=confidence_metric,
        )
 
        max_preds, argmax_preds = compute_confidence(
            test_pred, metric=confidence_metric
        )

        # --- compute per-exit ECE on the full test set (not just routed samples)
        exit_ece_all = []
        exit_avg_conf_all = []
        exit_acc_all = []
        for k in range(n_stage):
            ece_all_k, conf_all_k, acc_all_k = compute_ece_from_probs(test_pred[k], test_target)
            exit_ece_all.append(ece_all_k)
            exit_avg_conf_all.append(conf_all_k)
            exit_acc_all.append(acc_all_k)
 
        exit_counts = torch.zeros(n_stage)
        exit_correct = torch.zeros(n_stage)
        exit_sample_indices = [[] for _ in range(n_stage)]
 
        for i in range(n_sample):
            gold = int(test_target[i].item())
            for k in range(n_stage):
                if max_preds[k][i].item() >= T[k]:
                    exit_counts[k] += 1
                    exit_sample_indices[k].append(i)
                    if gold == int(argmax_preds[k][i].item()):
                        exit_correct[k] += 1
                    break
 
        exit_ece = []
        exit_avg_conf = []
        exit_signed_bias = []
        for k in range(n_stage):
            indices = exit_sample_indices[k]
            if len(indices) > 0:
                idx_tensor = torch.LongTensor(indices)
                routed_probs = test_pred[k][idx_tensor]
                routed_targets = test_target[idx_tensor]
                ece_k, conf_k, acc_k = compute_ece_from_probs(routed_probs, routed_targets)
                exit_ece.append(ece_k)
                exit_avg_conf.append(conf_k)
            else:
                exit_ece.append(0.0)
                exit_avg_conf.append(0.0)

        # compute signed bias globally (avg_conf_all - acc_all)
        for k in range(n_stage):
            signed = exit_avg_conf_all[k] - exit_acc_all[k]
            exit_signed_bias.append(signed)
 
        results.append(
            {
                "p": p,
                "acc": acc_test,
                "flops": exp_flops,
                "T": T,
                "exit_counts": exit_counts.clone(),
                "exit_correct": exit_correct.clone(),
                "n_samples": n_sample,
                "exit_ece": exit_ece,          # list[float], len = n_stage (routed)
                "exit_avg_conf": exit_avg_conf, # list[float], len = n_stage (routed)
                "exit_signed_bias": exit_signed_bias, # list[float], len = n_stage (routed)
                "exit_ece_all": exit_ece_all,  # list[float], len = n_stage (all samples)
                "exit_avg_conf_all": exit_avg_conf_all, # list[float], len = n_stage (all samples)
                "exit_acc_all": exit_acc_all,
            }
        )
 
    return results


def optimize_thresholds_bayesian(
    val_conf,  # (n_exits, n_samples)
    val_logits,  # (n_exits, n_samples, n_classes)
    val_targets,  # (n_samples,)
    test_conf,
    test_logits,
    test_targets,
    flops,
    n_trials=200,
):
    """
    Use Optuna's multi-objective Bayesian Optimization to find
    Pareto-optimal thresholds.
    """
    n_stage = val_conf.size(0)
    _, argmax_preds = val_logits.max(dim=2)

    def evaluate_thresholds(T, conf, logits, targets):
        """Standard sequential threshold evaluation."""
        n_stage, n_sample = conf.size()
        max_preds, argmax_p = logits.max(dim=2)

        acc, exp = 0, torch.zeros(n_stage)
        for i in range(n_sample):
            for k in range(n_stage):
                if conf[k][i].item() >= T[k]:
                    if int(targets[i]) == int(argmax_p[k][i]):
                        acc += 1
                    exp[k] += 1
                    break

        expected_flops = sum((exp[k] / n_sample) * flops[k] for k in range(n_stage))
        return acc * 100.0 / n_sample, expected_flops

    def objective(trial):
        T = []
        for k in range(n_stage - 1):
            # Search range: from min to max confidence at this exit
            lo = val_conf[k].min().item()
            hi = val_conf[k].max().item()
            t_k = trial.suggest_float(f"T_{k}", lo, hi)
            T.append(t_k)
        T.append(-1e8)  # last exit accepts all

        T_tensor = torch.tensor(T)

        acc, exp_flops = evaluate_thresholds(
            T_tensor, val_conf, val_logits, val_targets
        )

        return acc, exp_flops

    study = optuna.create_study(
        directions=["maximize", "minimize"],
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=20,  # Random exploration first
            multivariate=True,  # Model correlations between T_k's
        ),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Extract Pareto-optimal trials
    pareto_trials = study.best_trials

    results = []
    for trial in pareto_trials:
        T = [trial.params[f"T_{k}"] for k in range(n_stage - 1)] + [-1e8]
        T_tensor = torch.tensor(T)

        # Evaluate on TEST set (not val)
        acc_test, flops_test = evaluate_thresholds(
            T_tensor, test_conf, test_logits, test_targets
        )
        results.append(
            {
                "T": T_tensor,
                "val_acc": trial.values[0],
                "val_flops": trial.values[1],
                "test_acc": acc_test,
                "test_flops": flops_test,
            }
        )

    return results


class Tester(object):
    def __init__(self, model, args=None):
        self.args = args
        self.model = model
        self.softmax = nn.Softmax(dim=1).cuda()

    def calc_logit(self, dataloader):
        self.model.eval()
        n_stage = self.args.num_exits
        logits = [[] for _ in range(n_stage)]
        targets = []
        for i, (input, target) in enumerate(dataloader):
            targets.append(target)
            input = input.cuda()
            target = target.cuda()
            with torch.no_grad():
                input_var = torch.autograd.Variable(input)
                output = self.model(input_var)
                if not isinstance(output, list):
                    output = [output]
                for b in range(n_stage):
                    # Gnerate a vector of probabilities for each exit using softmax
                    _t = self.softmax(output[b])

                    # Store the logits for each exit in a list
                    logits[b].append(_t)

            if i % self.args.print_freq == 0:
                print("Generate Logit: [{0}/{1}]".format(i, len(dataloader)))

        for b in range(n_stage):
            # Concatenate the tensors for each exit along the batch dimension
            logits[b] = torch.cat(logits[b], dim=0)

        size = (n_stage, logits[0].size(0), logits[0].size(1))
        ts_logits = torch.Tensor().resize_(size).zero_()
        for b in range(n_stage):
            ts_logits[b].copy_(logits[b])

        targets = torch.cat(targets, dim=0)
        ts_targets = torch.Tensor().resize_(size[1]).copy_(targets)

        return ts_logits, ts_targets

    def dynamic_eval_find_threshold(
        self, logits, targets, p, flops, confidence_metric="max_softmax"
    ):
        """
        logits: m * n * c
        m: Stages
        n: Samples
        c: Classes
        """
        # logits = val_pred
        # targets = val_target
        # p = probs
        # flops = flops

        n_stage, n_sample, c = logits.size()

        max_preds, argmax_preds = compute_confidence(logits, metric=confidence_metric)

        _, sorted_idx = max_preds.sort(dim=1, descending=True)

        filtered = torch.zeros(n_sample)
        T = torch.Tensor(n_stage).fill_(1e8)

        for k in range(n_stage - 1):
            acc, count = 0.0, 0
            # Number of samples that should exit at stage k based on the probability p[k]
            out_n = math.floor(n_sample * p[k])
            for i in range(n_sample):
                ori_idx = sorted_idx[k][i]
                if filtered[ori_idx] == 0:
                    count += 1
                    if count == out_n:
                        T[k] = max_preds[k][ori_idx]
                        break
            filtered.add_(max_preds[k].ge(T[k]).type_as(filtered))

        T[n_stage - 1] = -1e8  # accept all of the samples at the last stage

        acc_rec, exp = torch.zeros(n_stage), torch.zeros(n_stage)
        acc, expected_flops = 0, 0
        for i in range(n_sample):
            gold_label = targets[i]
            for k in range(n_stage):
                if max_preds[k][i].item() >= T[k]:  # force the sample to exit at k
                    if int(gold_label.item()) == int(argmax_preds[k][i].item()):
                        acc += 1
                        acc_rec[k] += 1
                    exp[k] += 1
                    break
        acc_all = 0
        for k in range(n_stage):
            _t = 1.0 * exp[k] / n_sample
            expected_flops += _t * flops[k]
            acc_all += acc_rec[k]

        return acc * 100.0 / n_sample, expected_flops, T

    def dynamic_eval_with_threshold(
        self, logits, targets, flops, T, confidence_metric="max_softmax"
    ):
        n_stage, n_sample, _ = logits.size()
        max_preds, argmax_preds = compute_confidence(logits, metric=confidence_metric)

        exit_buckets = {
            i: {j: [] for j in range(n_stage)} for i in range(1000)
        }  # for each exit use a bucket to keep track of samples outputing from it

        acc_rec, exp = torch.zeros(n_stage), torch.zeros(n_stage)
        acc, expected_flops = 0, 0
        for i in range(n_sample):
            gold_label = targets[i]
            for k in range(n_stage):
                if max_preds[k][i].item() >= T[k]:  # force to exit at k
                    _g = int(gold_label.item())
                    _pred = int(argmax_preds[k][i].item())
                    if _g == _pred:
                        acc += 1
                        acc_rec[k] += 1
                    exp[k] += 1
                    exit_buckets[int(gold_label)][k].append(i)
                    break

        acc_all, sample_all = 0, 0
        for k in range(n_stage):
            _t = exp[k] * 1.0 / n_sample
            sample_all += exp[k]
            expected_flops += _t * flops[k]
            acc_all += acc_rec[k]

        return (
            acc * 100.0 / n_sample,
            expected_flops,
            exit_buckets,
            {
                "exit_counts": exp,
                "exit_correct": acc_rec,
                "n_samples": n_sample,
            },
        )
