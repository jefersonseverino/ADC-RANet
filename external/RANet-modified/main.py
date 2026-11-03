import os
import math
import time
import shutil
import models

from dataloader import (
    get_dataloaders,
    get_dataloader_distech,
    get_dataloader_cifar10,
    get_dataloader_cifar100,
    get_dataloader_imagenet100,
    get_dataloader_tinyimagenet
)
from args import args
from adaptive_inference import dynamic_evaluate, log_exit_distribution
from op_counter import measure_model

import torch
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from collections import OrderedDict
try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call

import wandb
import numpy as np
import random

from calibration import (
    build_criterion,
    cutmix_data,
    fit_temperature_scaling_grid,
    mixup_data,
    mixup_criterion,
    run_post_hoc_calibration,
    evaluate_calibration,
)

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from timm.utils import ModelEmaV2

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
#torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

#torch.use_deterministic_algorithms(True, warn_only=True)

#wandb.init(mode="offline")


class TemperatureAnnealer:
    def __init__(self, mu=0.5, tau_star=1.05, tau_init=1.0):
        self.mu = mu
        self.tau_star = tau_star
        self.tau = tau_init

    def step(self, teacher_outputs):
        with torch.no_grad():
            max_conf = 0.0
            count = 0
            for out in teacher_outputs:
                probs = F.softmax(out / self.tau, dim=1)
                max_conf += probs.max(dim=1)[0].mean().item()
                count += 1
            avg_conf = max_conf / count
        if avg_conf > self.mu:
            self.tau *= self.tau_star
        return self.tau

def setup_distributed():
    if not dist.is_available():
        return 0, False
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, False
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank, True


def compute_exit_gap(test_loader, model):
    model.eval()
    if args.num_exits < 2:
        raise ValueError("Need at least two exits to compute an exit gap profile")

    gap_meters = [AverageMeter() for _ in range(args.num_exits - 1)]

    end = time.time()
    with torch.no_grad():
        for i, (input, _) in enumerate(test_loader):
            if torch.cuda.is_available():
                input = input.cuda()

            if isinstance(input, list):
                input_var = [torch.autograd.Variable(inp) for inp in input]
                batch_size = input[0].size(0)
            else:
                input_var = torch.autograd.Variable(input)
                batch_size = input.size(0)

            output, _ = model(input_var)
            if not isinstance(output, list):
                output = [output]

            for start_idx in range(len(output) - 1):
                log_p = F.log_softmax(output[start_idx], dim=1)
                for end_idx in range(start_idx + 1, len(output)):
                    log_q = F.log_softmax(output[end_idx], dim=1)
                    kl_pq = F.kl_div(
                        log_p, log_q, log_target=True, reduction="batchmean"
                    )
                    kl_qp = F.kl_div(
                        log_q, log_p, log_target=True, reduction="batchmean"
                    )
                    sym_kl = 0.5 * (kl_pq + kl_qp)
                    distance = end_idx - start_idx
                    gap_meters[distance - 1].update(sym_kl.item(), batch_size)
            if i % args.print_freq == 0:
                print(
                    "Gap eval: [{0}/{1}]\t"
                    "Gap(d=1) {gap:.6f}".format(
                        i + 1,
                        len(test_loader),
                        gap=gap_meters[0].avg if gap_meters else 0.0,
                    )
                )

    result_file = os.path.join(args.save, "ExitGapResults.txt")
    with open(result_file, "w+") as fd:
        fd.write("ExitGapResults\n")
        for j in range(len(gap_meters)):
            gap_str = " distance={d} symKL {gap:.6f}".format(
                d=j + 1,
                gap=gap_meters[j].avg,
            )
            print(gap_str)
            fd.write(gap_str + "\n")

        monotonic = all(
            gap_meters[d].avg <= gap_meters[d + 1].avg
            for d in range(len(gap_meters) - 1)
        )
        monotonic_str = f"monotonic_non_decreasing={monotonic}"
        print(monotonic_str)
        fd.write(monotonic_str + "\n")

    torch.save([g.avg for g in gap_meters], os.path.join(args.save, "exit_gap.pth"))
    return [g.avg for g in gap_meters]

def compute_gradient_conflict(loader, model, max_batches=50):
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    K = args.num_exits

    cos_sum = torch.zeros(K, K)
    neg_cnt = torch.zeros(K, K)
    pair_cnt = torch.zeros(K, K)
    norm_sum = torch.zeros(K, K) 

    for i, (input, target) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda(non_blocking=True)

        output, _ = model(input)
        if not isinstance(output, list):
            output = [output]

        grads = []
        for k in range(K):
            loss_k = F.cross_entropy(output[k], target)
            g = torch.autograd.grad(
                loss_k, params, retain_graph=(k < K - 1), allow_unused=True
            )
            grads.append(g)

        for a in range(K):
            for b in range(a + 1, K):
                shared = [
                    idx for idx in range(len(params))
                    if grads[a][idx] is not None and grads[b][idx] is not None
                ]
                if not shared:
                    continue
                ga = torch.cat([grads[a][idx].flatten() for idx in shared])
                gb = torch.cat([grads[b][idx].flatten() for idx in shared])

                cos = F.cosine_similarity(ga, gb, dim=0).item()
                cos_sum[a, b] += cos
                cos_sum[b, a] += cos
                neg_cnt[a, b] += float(cos < 0)
                neg_cnt[b, a] += float(cos < 0)
                pair_cnt[a, b] += 1
                pair_cnt[b, a] += 1
                norm_sum[a, b] += ga.norm().item()
                norm_sum[b, a] += gb.norm().item()

        if i % args.print_freq == 0:
            print("Grad-conflict eval batch [{}/{}]".format(i + 1, len(loader)))

    valid = pair_cnt > 0
    cos_mean = torch.zeros(K, K)
    neg_frac = torch.zeros(K, K)
    norm_mean = torch.zeros(K, K)
    cos_mean[valid] = cos_sum[valid] / pair_cnt[valid]
    neg_frac[valid] = neg_cnt[valid] / pair_cnt[valid]
    norm_mean[valid] = norm_sum[valid] / pair_cnt[valid]

    result_file = os.path.join(args.save, "GradConflictResults.txt")
    with open(result_file, "w+") as fd:
        fd.write("GradConflictResults (max_batches={})\n".format(max_batches))
        for a in range(K):
            for b in range(a + 1, K):
                if not valid[a, b]:
                    continue
                line = (
                    " exits=({a},{b}) cos={c:.4f} neg_frac={n:.3f} "
                    "norm_{a}={na:.4f} norm_{b}={nb:.4f} ratio={r:.3f}".format(
                        a=a + 1, b=b + 1,
                        c=cos_mean[a, b].item(),
                        n=neg_frac[a, b].item(),
                        na=norm_mean[a, b].item(),
                        nb=norm_mean[b, a].item(),
                        r=(norm_mean[b, a] / (norm_mean[a, b] + 1e-12)).item(),
                    )
                )
                print(line)
                fd.write(line + "\n")

    torch.save(
        {"cos_mean": cos_mean, "neg_frac": neg_frac, "norm_mean": norm_mean},
        os.path.join(args.save, "grad_conflict.pth"),
    )
    return cos_mean, neg_frac, norm_mean

def compute_dead_neuron_stats(test_loader, model, threshold=1e-3, max_batches=None,
                               layer_filter=None, track_permanently_dead=True,
                               sample_fraction=1.0, seed=0):
    model.eval()
    torch.manual_seed(seed)

    class _LayerStats:
        __slots__ = (
            "total_elements", "dead_instant", "sum_abs",
            "channel_max_abs", "num_channels", "_dims_to_reduce",
        )

        def __init__(self):
            self.total_elements = 0
            self.dead_instant = None
            self.sum_abs = None
            self.channel_max_abs = None
            self.num_channels = None
            self._dims_to_reduce = None

        def update(self, x):
            if sample_fraction < 1.0:
                flat = x.reshape(x.shape[0], x.shape[1], -1) if x.dim() >= 2 else x.reshape(x.shape[0], -1)
                n_spatial = flat.shape[-1] if x.dim() >= 2 else flat.shape[-1]
                k = max(1, int(n_spatial * sample_fraction))
                if x.dim() >= 2 and n_spatial > k:
                    idx = torch.randperm(n_spatial, device=x.device)[:k]
                    flat = flat.index_select(-1, idx)
                x = flat

            abs_x = x.abs()
            self.total_elements += x.numel()

            batch_dead = (abs_x < threshold).sum()
            batch_sum_abs = abs_x.sum()

            if self.dead_instant is None:
                self.dead_instant = batch_dead
                self.sum_abs = batch_sum_abs
            else:
                self.dead_instant += batch_dead
                self.sum_abs += batch_sum_abs

            if track_permanently_dead and x.dim() >= 2:
                if self._dims_to_reduce is None:
                    self.num_channels = x.shape[1]
                    self._dims_to_reduce = [d for d in range(x.dim()) if d != 1]

                batch_channel_max = abs_x.amax(dim=self._dims_to_reduce)
                if self.channel_max_abs is None:
                    self.channel_max_abs = batch_channel_max
                else:
                    torch.maximum(
                        self.channel_max_abs, batch_channel_max,
                        out=self.channel_max_abs,
                    )
            elif self.num_channels is None and x.dim() >= 2:
                self.num_channels = x.shape[1]

        def summary(self):
            mean_abs = (self.sum_abs / max(self.total_elements, 1)).item()
            instant_dead_frac = (self.dead_instant / max(self.total_elements, 1)).item()
            permanently_dead_frac = None
            if self.channel_max_abs is not None:
                permanently_dead_frac = (
                    (self.channel_max_abs < threshold).float().mean()
                ).item()
            return {
                "mean_abs_activation": mean_abs,
                "instant_dead_fraction": instant_dead_frac,
                "permanently_dead_fraction": permanently_dead_frac,
                "num_channels": self.num_channels,
            }

    stats = {}
    handles = []

    for name, module in model.named_modules():
        if not isinstance(module, (nn.ReLU, nn.GELU)):
            continue
        if layer_filter is not None and not layer_filter(name, module):
            continue

        layer_stat = _LayerStats()
        stats[name] = layer_stat

        def hook(module, inp, out, _stat=layer_stat):
            _stat.update(out.detach() if out.requires_grad else out)

        handles.append(module.register_forward_hook(hook))

    if not stats:
        for h in handles:
            h.remove()
        raise ValueError(
            "Nenhuma camada nn.ReLU/nn.GELU (apos layer_filter, se usado) "
            "foi encontrada no modelo."
        )

    print(
        f"Dead-neuron eval: {len(stats)} camadas monitoradas, "
        f"sample_fraction={sample_fraction}"
    )

    end = time.time()
    with torch.no_grad():
        for i, (input, _target) in enumerate(test_loader):
            if max_batches is not None and i >= max_batches:
                break

            if isinstance(input, list):
                input_var = [inp.cuda() for inp in input] if torch.cuda.is_available() else input
            else:
                input_var = input.cuda() if torch.cuda.is_available() else input
            print(i)
            model(input_var)

            if i % args.print_freq == 0:
                elapsed = time.time() - end
                print(
                    "Dead-neuron eval: [{0}/{1}]\telapsed {2:.2f}s".format(
                        i + 1, len(test_loader), elapsed
                    )
                )

    for h in handles:
        h.remove()

    results = {name: s.summary() for name, s in stats.items()}

    instant_vals = [v["instant_dead_fraction"] for v in results.values()]
    perm_vals = [
        v["permanently_dead_fraction"]
        for v in results.values()
        if v["permanently_dead_fraction"] is not None
    ]
    mean_instant = sum(instant_vals) / len(instant_vals) if instant_vals else float("nan")
    mean_perm = sum(perm_vals) / len(perm_vals) if perm_vals else float("nan")

    result_file = os.path.join(args.save, "DeadNeuronResults.txt")
    with open(result_file, "w+") as fd:
        fd.write(
            "DeadNeuronResults (threshold={}, sample_fraction={})\n".format(
                threshold, sample_fraction
            )
        )
        for name, s in results.items():
            perm_str = (
                "n/a" if s["permanently_dead_fraction"] is None
                else "{:.6f}".format(s["permanently_dead_fraction"])
            )
            line = (
                " layer={name} instant_dead={inst:.6f} "
                "permanently_dead={perm} mean_abs_act={mabs:.6f} "
                "num_channels={ch}".format(
                    name=name,
                    inst=s["instant_dead_fraction"],
                    perm=perm_str,
                    mabs=s["mean_abs_activation"],
                    ch=s["num_channels"],
                )
            )
            print(line)
            fd.write(line + "\n")

        summary_str = (
            "mean_instant_dead={:.6f} mean_permanently_dead={:.6f} "
            "num_layers={}".format(mean_instant, mean_perm, len(results))
        )
        print(summary_str)
        fd.write(summary_str + "\n")

    torch.save(
        {"per_layer": results, "threshold": threshold,
         "sample_fraction": sample_fraction,
         "mean_instant_dead": mean_instant, "mean_permanently_dead": mean_perm},
        os.path.join(args.save, "dead_neuron_stats.pth"),
    )

    return results

import torch
import torch.nn as nn
import torch.nn.functional as F

class MATEWeightNetwork(nn.Module):
    def __init__(self, feat_dims, proj_dim=None):
        super().__init__()
        self.num_exits = len(feat_dims)
        proj_dim = proj_dim or min(feat_dims)
        self.query = nn.ModuleList([nn.Linear(d, proj_dim) for d in feat_dims])
        self.key = nn.ModuleList([nn.Linear(d, proj_dim) for d in feat_dims])

    def forward(self, feats):
        q = torch.stack(
            [self.query[i](feats[i]) for i in range(self.num_exits)], dim=1
        )  
        k = torch.stack(
            [self.key[i](feats[i]) for i in range(self.num_exits)], dim=1
        ) 
        scores = torch.bmm(q, k.transpose(1, 2))  
        weights = torch.softmax(scores, dim=-1)
        return weights


def build_mate_teachers(student_logits_detached, weights):
    stacked = torch.stack(student_logits_detached, dim=1)   
    teacher_logits = torch.bmm(weights, stacked)            
    return [teacher_logits[:, j, :] for j in range(teacher_logits.size(1))]


def mate_loss(student_logits, weight_net, features,
              temperature=3.0, alpha=3.0, alpha_teacher=None):
    if alpha_teacher is None:
        alpha_teacher = alpha

    if features is None:
        raise ValueError(
            "mate_loss requer 'features' (Secao 3.2.1 do MATE usa features "
            "pre-classificador, nao logits). Extraia F^i do seu modelo antes "
            "de chamar esta funcao."
        )

    M = len(student_logits)
    z_detached = [z.detach() for z in student_logits]
    feats = [f.detach() for f in features]  # Secao 3.2.1: features detached

    weights = weight_net(feats)                                    # grad -> weight_net

    teachers_live = build_mate_teachers(z_detached, weights)         # grad -> weight_net (z detached)
    teachers_frozen = [t.detach() for t in teachers_live]             # grad -> ninguem

    log_student_live    = [F.log_softmax(z / temperature, dim=1) for z in student_logits]
    log_student_frozen  = [F.log_softmax(z / temperature, dim=1) for z in z_detached]
    log_teacher_frozen  = [F.log_softmax(t / temperature, dim=1) for t in teachers_frozen]
    log_teacher_live    = [F.log_softmax(t / temperature, dim=1) for t in teachers_live]

    student_loss = student_logits[0].new_zeros(())
    teacher_loss = student_logits[0].new_zeros(())

    for i in range(M):
        for j in range(M):
            student_loss = student_loss + F.kl_div(
                log_student_live[i], log_teacher_frozen[j],
                log_target=True, reduction="batchmean",
            )
        student_loss = student_loss + alpha * F.kl_div(
            log_student_live[i], log_teacher_frozen[i],
            log_target=True, reduction="batchmean",
        )

        for j in range(M):
            teacher_loss = teacher_loss + F.kl_div(
                log_teacher_live[i], log_student_frozen[j],
                log_target=True, reduction="batchmean",
            )
        teacher_loss = teacher_loss + alpha_teacher * F.kl_div(
            log_teacher_live[i], log_student_frozen[i],
            log_target=True, reduction="batchmean",
        )

    student_loss = (temperature ** 2) * student_loss
    teacher_loss = (temperature ** 2) * teacher_loss
    return student_loss, teacher_loss

class HDKDGamma(nn.Module):
    def __init__(self, num_exits):
        super().__init__()
        self.gamma_ce = nn.Parameter(torch.ones(num_exits))
        self.gamma_kd = nn.Parameter(torch.zeros(num_exits, num_exits))


def hdkd_train_loss(outputs, target, criterion, gamma_ce, gamma_kd, temperature):
    K = len(outputs)
    ce_losses = []
    total = outputs[0].new_zeros(())
    for i in range(K):
        ce_i = criterion(outputs[i], target)
        ce_losses.append(ce_i)
        total = total + gamma_ce[i] * ce_i

    for i in range(1, K):  # exit i = professor (mais tardio)
        log_teacher = F.log_softmax(outputs[i].detach() / temperature, dim=1)
        for j in range(i):  # exit j = aluno (mais cedo)
            log_student = F.log_softmax(outputs[j] / temperature, dim=1)
            kd_ij = (temperature ** 2) * F.kl_div(
                log_student, log_teacher, log_target=True, reduction="batchmean"
            )
            total = total + gamma_kd[i, j] * kd_ij

    return total, ce_losses

def hdkd_meta_step(model, is_distributed, input_train, target_train,
                    input_val, target_val, criterion, hdkd_gamma, ema_ce,
                    temperature, xi=0.01, meta_lr=1e-3, alpha=0.5, momentum=0.9):
    base_model = model.module if is_distributed else model
    theta_params = [p for p in base_model.parameters() if p.requires_grad]

    out_train, _ = model(input_train)
    if not isinstance(out_train, list):
        out_train = [out_train]
    ltrain, ce_losses = hdkd_train_loss(
        out_train, target_train, criterion, hdkd_gamma.gamma_ce, hdkd_gamma.gamma_kd, temperature
    )

    grads_theta = torch.autograd.grad(ltrain, theta_params, create_graph=True, retain_graph=True)
    named_params = dict(base_model.named_parameters())
    named_buffers = dict(base_model.named_buffers())
    virtual_params = {
        name: p - xi * g
        for (name, p), g in zip(named_params.items(), grads_theta)
    }
    virtual_params.update(named_buffers)

    out_val, _ = functional_call(base_model, virtual_params, (input_val,))
    if not isinstance(out_val, list):
        out_val = [out_val]

    lval = out_val[0].new_zeros(())
    for k in range(len(out_val)):
        ce_k = criterion(out_val[k], target_val)
        ratio_term = F.softplus(ce_k / max(ema_ce[k], 1e-8) - 1.0)
        lval = lval + alpha * ce_k + (1 - alpha) * ratio_term

    grads_gamma = torch.autograd.grad(lval, [hdkd_gamma.gamma_ce, hdkd_gamma.gamma_kd])
    with torch.no_grad():
        hdkd_gamma.gamma_ce -= meta_lr * grads_gamma[0]
        hdkd_gamma.gamma_kd -= meta_lr * grads_gamma[1]

    with torch.no_grad():
        for k, ce_k in enumerate(ce_losses):
            ema_ce[k] = momentum * ema_ce[k] + (1 - momentum) * ce_k.item()



def compute_exit_cka(test_loader, model, max_batches=None):
    model.eval()
    if args.num_exits < 2:
        raise ValueError("Need at least two exits to compute CKA")

    all_feats = [[] for _ in range(args.num_exits)]

    with torch.no_grad():
        for i, (input, _) in enumerate(test_loader):
            if max_batches is not None and i >= max_batches:
                break
            if torch.cuda.is_available():
                input = input.cuda()
            _, feats = model(input)
            for k, f in enumerate(feats):
                all_feats[k].append(f.cpu())
            if i % args.print_freq == 0:
                print("CKA eval batch [{}/{}]".format(i + 1, len(test_loader)))

    all_feats = [torch.cat(f, dim=0).float() for f in all_feats]  # (N, D_k) por exit

    def linear_cka(X, Y):
        X = X - X.mean(0, keepdim=True)
        Y = Y - Y.mean(0, keepdim=True)
        hsic = (X.T @ Y).norm(p='fro') ** 2
        norm_x = (X.T @ X).norm(p='fro')
        norm_y = (Y.T @ Y).norm(p='fro')
        return (hsic / (norm_x * norm_y + 1e-12)).item()

    n = args.num_exits
    sim_matrix = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = linear_cka(all_feats[i], all_feats[j])

    result_file = os.path.join(args.save, "ExitCKAResults.txt")
    with open(result_file, "w+") as fd:
        fd.write("ExitCKAResults\n")
        for i in range(n):
            row = " ".join(f"{sim_matrix[i,j].item():.4f}" for j in range(n))
            print(f"exit {i+1}: {row}")
            fd.write(f"exit {i+1}: {row}\n")

        for d in range(1, n):
            vals = [sim_matrix[i, i + d].item() for i in range(n - d)]
            avg = sum(vals) / len(vals)
            fd.write(f" distance={d} avg_CKA={avg:.4f}\n")
            print(f" distance={d} avg_CKA={avg:.4f}")

    torch.save(sim_matrix, os.path.join(args.save, "exit_cka.pth"))
    return sim_matrix

def compute_cross_model_cka(test_loader, model_a, model_b, max_batches=None,
                             label_a="A", label_b="B"):
    model_a.eval()
    model_b.eval()

    if args.num_exits < 1:
        raise ValueError("Model has no exits")

    feats_a = [[] for _ in range(args.num_exits)]
    feats_b = [[] for _ in range(args.num_exits)]

    with torch.no_grad():
        for i, (input, _) in enumerate(test_loader):
            if max_batches is not None and i >= max_batches:
                break
            if torch.cuda.is_available():
                input = input.cuda()

            _, fa = model_a(input)
            _, fb = model_b(input)

            for k in range(args.num_exits):
                feats_a[k].append(fa[k].cpu())
                feats_b[k].append(fb[k].cpu())

            if i % args.print_freq == 0:
                print("Cross-model CKA eval batch [{}/{}]".format(
                    i + 1, len(test_loader)
                ))

    feats_a = [torch.cat(f, dim=0).float() for f in feats_a]
    feats_b = [torch.cat(f, dim=0).float() for f in feats_b]

    def linear_cka(X, Y):
        X = X - X.mean(0, keepdim=True)
        Y = Y - Y.mean(0, keepdim=True)
        hsic = (X.T @ Y).norm(p='fro') ** 2
        norm_x = (X.T @ X).norm(p='fro')
        norm_y = (Y.T @ Y).norm(p='fro')
        return (hsic / (norm_x * norm_y + 1e-12)).item()

    per_exit_cka = []
    result_file = os.path.join(args.save, "CrossModelCKAResults.txt")
    with open(result_file, "w+") as fd:
        fd.write(f"CrossModelCKAResults ({label_a} vs {label_b})\n")
        for k in range(args.num_exits):
            cka_k = linear_cka(feats_a[k], feats_b[k])
            per_exit_cka.append(cka_k)
            line = " exit={ext} CKA({a}_vs_{b})={cka:.4f}".format(
                ext=k + 1, a=label_a, b=label_b, cka=cka_k
            )
            print(line)
            fd.write(line + "\n")

    torch.save(
        {"per_exit_cka": per_exit_cka, "label_a": label_a, "label_b": label_b},
        os.path.join(args.save, "cross_model_cka.pth"),
    )
    return per_exit_cka

class MLCSDWeightNetwork(nn.Module):
    def __init__(self, feat_dims):
        super().__init__()
        self.num_exits = len(feat_dims)
        self.cwem = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, self.num_exits),
                nn.BatchNorm1d(self.num_exits),
            )
            for d in feat_dims
        ])

    def forward(self, pooled_feats):
        scores = [self.cwem[i](pooled_feats[i]) for i in range(self.num_exits)]
        weights = [torch.softmax(s, dim=-1) for s in scores]
        return torch.stack(weights, dim=1) 


def build_mlcsd_teachers(student_vals_detached, weights):
    stacked = torch.stack(student_vals_detached, dim=1)  
    teacher_vals = torch.bmm(weights, stacked)            
    return [teacher_vals[:, i, :] for i in range(teacher_vals.size(1))]


def mlcsd_loss(student_logits, weight_net, pooled_feats, features=None,
               temperature=4.0, theta=1.0, sigma=1.0):
    K = len(student_logits)
    z_detached = [z.detach() for z in student_logits]

    weights_live = weight_net(pooled_feats)
    weights_frozen = weights_live.detach()

    logit_teachers_frozen = build_mlcsd_teachers(z_detached, weights_frozen)
    logit_teachers_live = build_mlcsd_teachers(z_detached, weights_live)

    log_student_live = [F.log_softmax(z / temperature, dim=1) for z in student_logits]
    log_student_frozen = [F.log_softmax(z / temperature, dim=1) for z in z_detached]

    kd_loss = student_logits[0].new_zeros(())
    weight_train_loss = student_logits[0].new_zeros(())
    for i in range(K):
        log_pt_frozen = F.log_softmax(logit_teachers_frozen[i] / temperature, dim=1)
        kd_loss = kd_loss + theta * F.kl_div(
            log_student_live[i], log_pt_frozen, log_target=True, reduction="batchmean"
        )
        log_pt_live = F.log_softmax(logit_teachers_live[i] / temperature, dim=1)
        weight_train_loss = weight_train_loss + theta * F.kl_div(
            log_pt_live, log_student_frozen[i], log_target=True, reduction="batchmean"
        )
    kd_loss = (temperature ** 2) * kd_loss
    weight_train_loss = (temperature ** 2) * weight_train_loss

    ht_loss = student_logits[0].new_zeros(())
    if features is not None:
        f_detached = [f.detach() for f in features]
        feat_teachers_frozen = build_mlcsd_teachers(f_detached, weights_frozen)
        for i in range(K):
            ht_loss = ht_loss + sigma * F.mse_loss(features[i], feat_teachers_frozen[i])

    return kd_loss, ht_loss, weight_train_loss

def compute_reliability_diagram(test_loader, model, n_bins=15, exits_to_plot=None,
                                  tag=""):
    import matplotlib
    import matplotlib.pyplot as plt
    model.eval()

    if exits_to_plot is None:
        exits_to_plot = list(range(args.num_exits))

    bin_edges = torch.linspace(0, 1, n_bins + 1)

    conf_sum = {k: torch.zeros(n_bins) for k in exits_to_plot}
    correct_sum = {k: torch.zeros(n_bins) for k in exits_to_plot}
    count = {k: torch.zeros(n_bins) for k in exits_to_plot}

    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda(non_blocking=True)

            output, _ = model(input)
            if not isinstance(output, list):
                output = [output]

            for k in exits_to_plot:
                probs = F.softmax(output[k], dim=1)
                conf, pred = probs.max(dim=1)
                correct = pred.eq(target).float()

                bin_idx = torch.bucketize(conf.cpu(), bin_edges[1:-1])

                for b in range(n_bins):
                    mask = bin_idx == b
                    if mask.any():
                        conf_sum[k][b] += conf.cpu()[mask].sum()
                        correct_sum[k][b] += correct.cpu()[mask].sum()
                        count[k][b] += mask.sum()

            if i % args.print_freq == 0:
                print("Reliability eval: [{0}/{1}]".format(i + 1, len(test_loader)))

    suffix = f"_{tag}" if tag else ""
    results = {}
    result_file = os.path.join(args.save, f"ReliabilityResults{suffix}.txt")

    with open(result_file, "w+") as fd:
        fd.write("ReliabilityResults (n_bins={}, tag={})\n".format(n_bins, tag))

        for k in exits_to_plot:
            valid = count[k] > 0
            avg_conf = torch.zeros(n_bins)
            avg_acc = torch.zeros(n_bins)
            avg_conf[valid] = conf_sum[k][valid] / count[k][valid]
            avg_acc[valid] = correct_sum[k][valid] / count[k][valid]

            total = count[k].sum()
            ece = (
                (count[k][valid] / total) * (avg_conf[valid] - avg_acc[valid]).abs()
            ).sum().item()

            results[k] = {
                "bin_confidence": avg_conf,
                "bin_accuracy": avg_acc,
                "bin_count": count[k],
                "bin_edges": bin_edges,
                "ece": ece,
            }

            line = " exit={ext} ECE={ece:.4f}".format(ext=k + 1, ece=ece)
            print(line)
            fd.write(line + "\n")
            for b in range(n_bins):
                fd.write(
                    "   bin={b} conf={c:.4f} acc={a:.4f} count={n}\n".format(
                        b=b, c=avg_conf[b].item(), a=avg_acc[b].item(), n=int(count[k][b].item())
                    )
                )

            fig, ax = plt.subplots(figsize=(4, 4))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_width = 1.0 / n_bins

            ax.bar(
                bin_centers.numpy(), avg_acc.numpy(), width=bin_width,
                edgecolor="black", color="steelblue", alpha=0.8, label="Accuracy",
            )
            gap = (avg_conf - avg_acc).numpy()
            ax.bar(
                bin_centers.numpy(), gap, bottom=avg_acc.numpy(), width=bin_width,
                edgecolor="black", color="salmon", alpha=0.5, label="Gap (conf-acc)",
            )
            ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Perfect calibration")

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Accuracy")
            title_tag = f" [{tag}]" if tag else ""
            ax.set_title(f"Exit {k+1}{title_tag} — ECE={ece:.3f}")
            ax.legend(loc="upper left", fontsize=8)
            fig.tight_layout()

            img_path = os.path.join(args.save, f"reliability_exit{k+1}{suffix}.pdf")
            fig.savefig(img_path)
            fig.savefig(img_path.replace(".pdf", ".png"), dpi=200)
            plt.close(fig)
            print(f"=> saved reliability diagram '{img_path}'")

    torch.save(results, os.path.join(args.save, f"reliability_diagram{suffix}.pth"))
    return results

def main():
    global args
    best_prec1, best_epoch = 0.0, 0
    best_prec1_ema, best_epoch_ema = 0.0, 0

    # DEFAULT PATH AS DEFAULT-TIMESTAMP
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # NOW SETTING INPUT IMAGE SIZE AS ARGUMENT
    IMG_SIZE = args.img_size

    # CREATE RANET MODEL TO MEASURE FLOPS (RANET IS THE ONLY MODEL INSIDE MODELS)
    model = getattr(models, args.arch)(args)
    args.num_exits = len(model.classifier)

    # Give different weights to different classifiers.
    if args.exit_weights is None:
        args.exit_weights = [1.0] * args.num_exits
    elif len(args.exit_weights) != args.num_exits:
        raise ValueError(
            "exit-weights must match num_exits. "
            f"Got {len(args.exit_weights)} weights for {args.num_exits} exits."
        )
    global n_flops

    # MAYBE USE THIS FUNCTION FOR RESNET TOO?
    # Usa input_sizes se disponível (para LazyRANet)
    widht = IMG_SIZE
    height = IMG_SIZE

    if args.resize:
        # 4:3 aspect for distech data (example)
        height = int(IMG_SIZE * 0.75)

    #n_flops, _ = measure_model(model, widht, height, input_sizes=args.input_sizes)

    # SAVE NUMBER OF FLOPS TO FILE

    #torch.save(n_flops, os.path.join(args.save, "flops.pth"))
    del model

    print(args)
    with open("{}/args.txt".format(args.save), "w") as f:
        print(args, file=f)

    local_rank, is_distributed = setup_distributed()
    is_main = local_rank == 0

    model = getattr(models, args.arch)(args)
    model = model.cuda() if torch.cuda.is_available() else model
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    # model = torch.nn.DataParallel(model.cuda())

    # criterion = nn.CrossEntropyLoss().cuda()
    criterion = build_criterion(args)

    global exit_router

    exit_router = None
    all_params = model.parameters()
    
    mate_net = None
    if args.distilation_type == "mate":
        base_model_for_feats = model.module if is_distributed else model
        feat_dims = [c.linear.in_features for c in base_model_for_feats.classifier]
        mate_net = MATEWeightNetwork(
            feat_dims=feat_dims,
            proj_dim=getattr(args, "mate_proj_dim", None),
        )
        mate_net = mate_net.cuda() if torch.cuda.is_available() else mate_net
        all_params = list(mate_net.parameters()) + list(all_params)

    mlcsd_net = None
    if args.distilation_type == "mlcsd":
        base_model_for_feats = model.module if is_distributed else model
        pooled_dims = [c.linear.in_features for c in base_model_for_feats.classifier]
        mlcsd_net = MLCSDWeightNetwork(feat_dims=pooled_dims)
        mlcsd_net = mlcsd_net.cuda() if torch.cuda.is_available() else mlcsd_net
        all_params = list(mlcsd_net.parameters()) + list(all_params)
    
    hdkd_gamma = None
    hdkd_ema_ce = None
    hdkd_val_iter = None
    if args.distilation_type == "hdkd":
        hdkd_gamma = HDKDGamma(num_exits=args.num_exits)
        hdkd_gamma = hdkd_gamma.cuda() if torch.cuda.is_available() else hdkd_gamma
        hdkd_ema_ce = [1.0] * args.num_exits  # gamma NAO entra no optimizer principal: e' meta-otimizado a parte
        
    optimizer = torch.optim.SGD(
        all_params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    use_ema = getattr(args, "use_ema", False)
    model_ema = None
    if use_ema:
        ema_base_model = model.module if is_distributed else model
        ema_device = "cpu" if getattr(args, "ema_force_cpu", False) else None
        model_ema = ModelEmaV2(
            ema_base_model,
            decay=getattr(args, "ema_decay", 0.9999),
            device=ema_device,
        )
        print(
            "=> using EMA (decay={}, device={})".format(
                getattr(args, "ema_decay", 0.9999),
                ema_device if ema_device is not None else "same as model",
            )
        )

    if args.resume:
        checkpoint = load_checkpoint(args)
        if checkpoint is not None:
            args.start_epoch = checkpoint["epoch"] + 1
            best_prec1 = checkpoint["best_prec1"]
            best_prec1_ema = checkpoint.get("best_prec1_ema", 0.0)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            if exit_router is not None and "exit_router" in checkpoint:
                exit_router.load_state_dict(checkpoint["exit_router"])
                print("Loaded exit_router state from checkpoint")
            if mate_net is not None and "mate_net" in checkpoint:
                mate_net.load_state_dict(checkpoint["mate_net"])
                print("Loaded MATE weight network state from checkpoint")
            if mlcsd_net is not None and "mlcsd_net" in checkpoint:
                mlcsd_net.load_state_dict(checkpoint["mlcsd_net"])
                print("Loaded MLCSD weight network state from checkpoint")
            if mlcsd_net is not None:
                checkpoint_dict["mlcsd_net"] = mlcsd_net.state_dict()
            if model_ema is not None and "state_dict_ema" in checkpoint:
                model_ema.module.load_state_dict(checkpoint["state_dict_ema"])
                print("Loaded EMA model state from checkpoint")

    if args.data == "ImageNet":
        train_loader, val_loader, test_loader = get_dataloaders(args)
    elif args.data == "cifar10":
        train_loader, val_loader, test_loader = get_dataloader_cifar10(args)
    elif args.data == "cifar100":
        train_loader, val_loader, test_loader = get_dataloader_cifar100(args)
    elif args.data  == "imagenet100":
        train_loader, val_loader, test_loader = get_dataloader_imagenet100(args)
    elif args.data == "tinyimagenet":
        train_loader, val_loader, test_loader = get_dataloader_tinyimagenet(args)
    else:
        train_loader, val_loader, test_loader = get_dataloader_distech(
            args, transform=True, resize=args.resize
        )

    if args.evalmode is not None:
        ckpt = torch.load(args.evaluate_from, map_location="cpu")
        state_dict = ckpt["state_dict"]

        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=False)

        if exit_router is not None and "exit_router" in ckpt:
            exit_router.load_state_dict(ckpt["exit_router"])
            print("Loaded exit_router state for evaluation")

        eval_model = model
        if use_ema and getattr(args, "ema_eval", False) and "state_dict_ema" in ckpt:
            ema_state_dict = ckpt["state_dict_ema"]
            if any(k.startswith("module.") for k in ema_state_dict.keys()):
                ema_state_dict = {
                    k.replace("module.", ""): v for k, v in ema_state_dict.items()
                }
            model.load_state_dict(ema_state_dict, strict=False)
            print("Loaded EMA weights for evaluation")
            eval_model = model

        if args.evalmode == "anytime":
            validate(test_loader, eval_model, criterion)
        elif args.evalmode == "dynamic":
            dynamic_evaluate(eval_model, test_loader, val_loader, args)
        elif args.evalmode == "grad_conflict":
            compute_gradient_conflict(
                train_loader, eval_model,
                max_batches=getattr(args, "grad_conflict_max_batches", 50),
            )
            return
        elif args.evalmode == "exit_stats":
            from adaptive_inference import evaluate_exit_distribution
            calib_mode = "temp_scaling" if getattr(args, "calibrate", False) else "none"
            from adaptive_inference import evaluate_fixed_threshold
            T = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, -1e9]
            r_normal = evaluate_fixed_threshold(eval_model, test_loader, val_loader, args, T)
            for k in range(len(T)):
                print(f"@{k+1}   {r_normal['exit_pct'][k]:.1f}       {r_normal['exit_acc'][k]:.1f} ")
        elif args.evalmode == "dead_neurons":
            compute_dead_neuron_stats(
                test_loader,
                eval_model,
                threshold=args.dead_neuron_threshold,
                max_batches=args.dead_neuron_max_batches
            )
        elif args.evalmode == "cka":
            compute_exit_cka(test_loader, eval_model, max_batches=getattr(args, "cka_max_batches", None))
            return
        elif args.evalmode == "reliability":
            compute_reliability_diagram(
                test_loader, eval_model,
                n_bins=getattr(args, "reliability_n_bins", 15),
                exits_to_plot=[0, args.num_exits - 1],  # exit inicial e tardio, como o BkhH pediu
                tag=getattr(args, "reliability_tag", ""),
            )
            return
        elif args.evalmode == "cross_model_cka":
            model_b = getattr(models, args.arch)(args)
            model_b = model_b.cuda() if torch.cuda.is_available() else model_b

            ckpt_b = torch.load(args.evaluate_from_b, map_location="cpu")
            state_dict_b = ckpt_b["state_dict"]
            if any(k.startswith("module.") for k in state_dict_b.keys()):
                state_dict_b = {k.replace("module.", ""): v for k, v in state_dict_b.items()}
            model_b.load_state_dict(state_dict_b, strict=False)

            compute_cross_model_cka(
                test_loader,
                eval_model, 
                model_b,
                max_batches=getattr(args, "cka_max_batches", None),
                label_a=getattr(args, "cka_label_a", "modelA"),
                label_b=getattr(args, "cka_label_b", "modelB"),
            )
            return
        elif args.evalmode == "temperature_scaling":
            pass
        elif args.evalmode == "gap":
            compute_exit_gap(test_loader, eval_model)
            return
        else:
            validate(test_loader, eval_model, criterion)
            dynamic_evaluate(eval_model, test_loader, val_loader, args)

        return

    print("scores")
    scores = [
        "epoch\tlr\ttrain_loss\tval_loss\ttrain_prec1"
        "\tval_prec1\ttrain_prec5\tval_prec5"
    ]

    if is_main:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.save}-{widht}-{height}-{args.data}-{args.seed}-{args.use_se}-{args.epochs}-cmsa-{args.use_cmsa}-nearest",
            group=args.wandb_group,
            config=vars(args),
            #mode="offline"
        )

    temp_annealer = TemperatureAnnealer(
        mu=0.5,
        tau_star=1.05,
        tau_init=1.0,
    )

    val_loss, val_prec1, val_prec5 = 0, 0, 0
    val_loss_ema, val_prec1_ema, val_prec5_ema = 0, 0, 0
    print("Start training\n\n")
    for epoch in range(args.start_epoch, args.epochs):
        print("\nEpoch {}/{}:".format(epoch, args.epochs))
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        if exit_router is not None:
            exit_router.anneal_tau(
                epoch, args.epochs, warmup_epochs=getattr(args, "router_warmup", 10)
            )

        train_loss, train_prec1, train_prec5, lr = train(
            train_loader, model, criterion, optimizer, epoch, temp_annealer,
            model_ema, val_loader=val_loader, mate_net=mate_net, mlcsd_net=mlcsd_net, hdkd_gamma=hdkd_gamma, hdkd_ema_ce=hdkd_ema_ce, is_distributed=is_distributed
        )

        if is_main:
            val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion)

            log_dict = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_prec1": train_prec1,
                "val_prec1": val_prec1,
                "train_prec5": train_prec5,
                "val_prec5": val_prec5,
                "lr": lr,
                "distill_tau": temp_annealer.tau,
            }

            if model_ema is not None:
                val_loss_ema, val_prec1_ema, val_prec5_ema = validate(
                    val_loader, model_ema.module, criterion
                )
                log_dict.update(
                    {
                        "val_loss_ema": val_loss_ema,
                        "val_prec1_ema": val_prec1_ema,
                        "val_prec5_ema": val_prec5_ema,
                    }
                )

            wandb.log(log_dict)

            scores.append(
                ("{}\t{:.3f}" + "\t{:.4f}" * 6).format(
                    epoch,
                    lr,
                    train_loss,
                    val_loss,
                    train_prec1,
                    val_prec1,
                    train_prec5,
                    val_prec5,
                )
            )

        is_best = val_prec1 > best_prec1
        if is_best:
            best_prec1 = val_prec1
            best_epoch = epoch
            print("Best var_prec1 {}".format(best_prec1))

        is_best_ema = model_ema is not None and val_prec1_ema > best_prec1_ema
        if model_ema is not None and val_prec1_ema > best_prec1_ema:
            best_prec1_ema = val_prec1_ema
            best_epoch_ema = epoch
            print("Best EMA var_prec1 {}".format(best_prec1_ema))

        model_filename = "checkpoint_%03d.pth.tar" % epoch
        checkpoint_dict = {
            "epoch": epoch,
            "arch": args.arch,
            "state_dict": model.state_dict(),
            "best_prec1": best_prec1,
            "optimizer": optimizer.state_dict(),
        }
        if exit_router is not None:
            checkpoint_dict["exit_router"] = exit_router.state_dict()
        if model_ema is not None:
            checkpoint_dict["state_dict_ema"] = model_ema.module.state_dict()
            checkpoint_dict["best_prec1_ema"] = best_prec1_ema

        if is_main:
            save_checkpoint(
                checkpoint_dict,
                args,
                is_best,
                model_filename,
                scores,
            )

            if is_best_ema:
                model_dir = os.path.join(args.save, "save_models")
                os.makedirs(model_dir, exist_ok=True)
                best_ema_filename = os.path.join(model_dir, "model_best_ema.pth.tar")
                torch.save(checkpoint_dict, best_ema_filename)
                print("=> saved best EMA checkpoint '{}'".format(best_ema_filename))

        model_path = "%s/save_models/checkpoint_%03d.pth.tar" % (args.save, epoch - 1)
        if os.path.exists(model_path):
            os.remove(model_path)

    best_model_path = os.path.join(args.save, "save_models", "model_best.pth.tar")
    if os.path.exists(best_model_path):
        best_ckpt = torch.load(best_model_path, map_location="cpu")
        model.load_state_dict(best_ckpt["state_dict"])
        if exit_router is not None and "exit_router" in best_ckpt:
            exit_router.load_state_dict(best_ckpt["exit_router"])
        if model_ema is not None and "state_dict_ema" in best_ckpt:
            model_ema.module.load_state_dict(best_ckpt["state_dict_ema"])
        print(
            "=> Loaded best model from epoch {} (val_prec1={:.4f})".format(
                best_ckpt["epoch"], best_ckpt["best_prec1"]
            )
        )
    else:
        print("WARNING: best model checkpoint not found, using last epoch weights")

    final_model = model
    if model_ema is not None and getattr(args, "ema_eval", False):
        final_model = model_ema.module
        print("=> Using EMA weights for final calibration/evaluation")

    calibrator = run_post_hoc_calibration(final_model, val_loader, test_loader, args)

    if calibrator is not None:
        torch.save(calibrator.state_dict(), os.path.join(args.save, "calibrator.pth"))

    print("Best val_prec1: {:.4f} at epoch {}".format(best_prec1, best_epoch))
    if model_ema is not None:
        print(
            "Best val_prec1 (EMA): {:.4f} at epoch {}".format(
                best_prec1_ema, best_epoch_ema
            )
        )
    print("********** Final prediction results **********")

    if is_main:
        loss, prec1, prec5 = validate(test_loader, final_model, criterion)

        test_log = {
            "test_loss": loss,
            "test_prec1": prec1,
            "test_prec5": prec5,
        }

        if model_ema is not None and final_model is not model_ema.module:
            loss_ema, prec1_ema, prec5_ema = validate(test_loader, model_ema.module, criterion)
            test_log.update(
                {
                    "test_loss_ema": loss_ema,
                    "test_prec1_ema": prec1_ema,
                    "test_prec5_ema": prec5_ema,
                }
            )

        wandb.log(test_log)

        # calibration_eval = evaluate_calibration(
        #     final_model, test_loader, n_exits=args.num_exits, calibrator=calibrator
        # )

        # for k, res in calibration_eval.items():
        #     wandb.log(
        #         {
        #             f"exit_{k+1}_ece": res["ece_raw"],
        #             f"exit_{k+1}_accuracy": res["accuracy"],
        #         }
        #     )
        #     if res["ece_calibrated"] is not None:
        #         wandb.log({f"exit_{k+1}_ece_calibrated": res["ece_calibrated"]})

        # if args.data == "cifar100":
        #     args.use_valid = True
        #     train_loader, val_loader, test_loader = get_dataloader_cifar100(args)

        # all_acc, all_flops, pareto_flops, pareto_acc, pareto_thresholds = dynamic_evaluate(
        #     final_model, test_loader, val_loader, args
        # )

    if is_distributed:
        dist.destroy_process_group()

def last_exit_distillation_loss(outputs, temperature=4.0):
    teacher_logits = outputs[-1].detach()
    teacher_probs = F.log_softmax(teacher_logits / temperature, dim=1)

    loss = 0.0
    n_students = len(outputs) - 1
    if n_students == 0:
        return torch.tensor(0.0, device=outputs[0].device)

    for j in range(n_students):
        student_log_probs = F.log_softmax(outputs[j] / temperature, dim=1)
        kl = F.kl_div(
            student_log_probs, teacher_probs, log_target=True, reduction="batchmean"
        )
        loss += kl

    return (temperature**2) * loss / n_students


def cascaded_distillation_loss(outputs, temperature=3.0):
    loss = 0.0
    n_pairs = len(outputs) - 1
    if n_pairs == 0:
        return torch.tensor(0.0, device=outputs[0].device)

    for j in range(n_pairs):
        teacher_logits = outputs[j + 1].detach()
        teacher_probs = F.log_softmax(teacher_logits / temperature, dim=1)
        student_log_probs = F.log_softmax(outputs[j] / temperature, dim=1)
        kl = F.kl_div(
            student_log_probs, teacher_probs, log_target=True, reduction="batchmean"
        )
        loss += kl

    return (temperature**2) * loss / n_pairs


def all_subsequent_exits_distillation_loss(outputs, temperature=3.0):
    loss = 0.0
    n_students = len(outputs) - 1
    if n_students <= 0:
        return torch.tensor(0.0, device=outputs[0].device)

    for j in range(n_students):
        student_log_probs = F.log_softmax(outputs[j] / temperature, dim=1)
        teacher_outputs = outputs[j + 1 :]
        teacher_loss = 0.0
        for teacher_output in teacher_outputs:
            teacher_logits = teacher_output.detach()
            teacher_probs = F.log_softmax(teacher_logits / temperature, dim=1)
            teacher_loss += F.kl_div(
                student_log_probs,
                teacher_probs,
                log_target=True,
                reduction="batchmean",
            )
        loss += teacher_loss / len(teacher_outputs)

    return (temperature**2) * loss / n_students

def naive_all_subsequent_exits_distillation_loss(outputs, temperature=3.0):
    loss = 0.0
    K = len(outputs)
    if K <= 1:
        return torch.tensor(0.0, device=outputs[0].device)

    for j in range(K - 1):
        student_log_probs = F.log_softmax(outputs[j] / temperature, dim=1)
        for i in range(j + 1, K):
            teacher_logits = outputs[i].detach()
            teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=1)

            loss += F.kl_div(
                student_log_probs,
                teacher_log_probs,
                log_target=True,
                reduction="batchmean",
            )

    return (temperature**2) * loss

def train(train_loader, model, criterion, optimizer, epoch, temp_annealer, model_ema=None,
          val_loader=None, mate_net=None, mlcsd_net=None, hdkd_gamma=None, hdkd_ema_ce=None, is_distributed=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1, top5 = [], []
    for i in range(args.num_exits):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    model.train()
    end = time.time()

    hdkd_val_iter = iter(val_loader) if args.distilation_type == "hdkd" else None

    running_lr = None
    for i, (input, target) in enumerate(train_loader):
        lr = adjust_learning_rate(
            optimizer,
            epoch,
            args,
            batch=i,
            nBatch=len(train_loader),
            method=args.lr_type,
        )

        if running_lr is None:
            running_lr = lr

        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True) if torch.cuda.is_available() else target
        if isinstance(input, list):
            input_var = [inp.cuda() if torch.cuda.is_available() else inp for inp in input]
            batch_size = input[0].size(0)
        else:
            input_var = input.cuda() if torch.cuda.is_available() else input
            batch_size = input.size(0)
        target_var = torch.autograd.Variable(target)

        # MIXUP DURING VALIDATION (OPTIONAL, FOR CALIBRATION EVALUATION)
        use_mixup = getattr(args, "use_mixup", False)
        use_cutmix = getattr(args, "use_cutmix", False)

        exits_before_mix = None

        # HDKD (Eq. 5/10 do paper) exige um alvo unico (train e val) para a loss
        # balanceada; mixup/cutmix nao se combinam com o esquema de pesos harmonizados
        # como especificado no paper, entao desligamos a mistura nesse caso.
        if args.distilation_type == "hdkd":
            use_mixup = False
            use_cutmix = False

        if use_mixup and not isinstance(input_var, list):
            input_var, targets_a, targets_b, lam = mixup_data(
                input_var, target_var, alpha=args.mixup_alpha
            )

            output, feats = model(input_var)

        elif use_cutmix and not isinstance(input_var, list):
            input_var, targets_a, targets_b, lam = cutmix_data(
                input_var, target_var, alpha=args.cutmix_alpha
            )

            output, feats = model(input_var)

        else:
            targets_a, targets_b, lam = target_var, target_var, 1.0

            output, feats = model(input_var)

        if not isinstance(output, list):
            output = [output]

        alpha = args.distill_alpha
        exit_weights = args.exit_weights

        warmup_epochs = 20
        T = (
            temp_annealer.step(
                [output[-1]] if args.distilation_type == "last_exit" else output[1:]
            )
            if args.distill_annealing
            else args.distill_temp
        )

        alpha_t = (
            (alpha * (epoch / warmup_epochs) if epoch < warmup_epochs else alpha)
            if args.distill_annealing
            else alpha
        )

        if args.distilation_type == "hdkd":
            try:
                val_input, val_target = next(hdkd_val_iter)
            except StopIteration:
                hdkd_val_iter = iter(val_loader)
                val_input, val_target = next(hdkd_val_iter)
            if torch.cuda.is_available():
                val_input = val_input.cuda(non_blocking=True)
                val_target = val_target.cuda(non_blocking=True)

            hdkd_meta_step(
                model, is_distributed, input_var, target_var, val_input, val_target,
                criterion, hdkd_gamma, hdkd_ema_ce, temperature=T,
            )

            # forward "real" com o gamma ja atualizado, p/ o passo normal de theta
            output, _ = model(input_var)
            if not isinstance(output, list):
                output = [output]

            loss, _ = hdkd_train_loss(
                output, target_var, criterion, hdkd_gamma.gamma_ce, hdkd_gamma.gamma_kd, T
            )
        else:
            loss = 0.0
            for j in range(len(output)):
                weight = exit_weights[j]
                if exits_before_mix is not None and j < exits_before_mix:
                    loss += weight * criterion(output[j], target_var)
                else:
                    loss += weight * (
                        lam * criterion(output[j], targets_a)
                        + (1 - lam) * criterion(output[j], targets_b)
                    )

            if args.distilation_type == "last_exit":
                sd_loss = last_exit_distillation_loss(output, temperature=T)
            elif args.distilation_type == "cascaded":
                sd_loss = cascaded_distillation_loss(output, temperature=T)
            elif args.distilation_type == "all_subsequent_exits":
                sd_loss = all_subsequent_exits_distillation_loss(output, temperature=T)
            elif args.distilation_type == "naive":
                sd_loss = naive_all_subsequent_exits_distillation_loss(output, temperature=T)
            elif args.distilation_type == "mate":
                s_loss, t_loss = mate_loss(
                    output,
                    mate_net,
                    features=feats,
                    temperature=T,
                    alpha=getattr(args, "mate_alpha", 3.0),
                    alpha_teacher=getattr(args, "mate_alpha_teacher", None),
                )
                sd_loss = s_loss + t_loss
            elif args.distilation_type == "mlcsd":
                use_mcfsm = getattr(args, "mlcsd_use_feature", False)
                dims_match = len({f.shape[-1] for f in feats}) == 1
                kd_loss, ht_loss, weight_train_loss = mlcsd_loss(
                    output,
                    mlcsd_net,
                    pooled_feats=feats,
                    features=feats if (use_mcfsm and dims_match) else None,
                    temperature=T,
                    theta=getattr(args, "mlcsd_theta", 1.0),
                    sigma=getattr(args, "mlcsd_sigma", 1.0),
                )
                sd_loss = kd_loss + getattr(args, "mlcsd_beta", 1.0) * ht_loss + weight_train_loss
            else:
                raise ValueError(f"Unsupported distillation type: {args.distilation_type}")

            if args.normalize_loss:
                loss = loss / sum(exit_weights)

            if args.use_distillation:
                loss += alpha_t * sd_loss

        losses.update(loss.item(), batch_size)

        for j in range(len(output)):
            prec1, prec5 = accuracy(
                output[j].data, target, topk=(1, min(5, args.num_classes))
            )
            top1[j].update(prec1.item(), batch_size)
            top5[j].update(prec5.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.avg:.3f}\t"
                "Data {data_time.avg:.3f}\t"
                "Loss {loss.val:.4f}\t"
                "Acc@1 {top1.val:.4f}\t"
                "Acc@5 {top5.val:.4f}".format(
                    epoch,
                    i + 1,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1[-1],
                    top5=top5[-1],
                )
            )

    return losses.avg, top1[-1].avg, top5[-1].avg, running_lr


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    top1, top5 = [], []
    for i in range(args.num_exits):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True) if torch.cuda.is_available() else target
            input = input.cuda() if torch.cuda.is_available() else input

            if isinstance(input, list):
                input_var = [torch.autograd.Variable(inp) for inp in input]
                batch_size = input[0].size(0)
            else:
                input_var = torch.autograd.Variable(input)
                batch_size = input.size(0)
            target_var = torch.autograd.Variable(target)

            data_time.update(time.time() - end)

            output, feats = model(input_var)
            if not isinstance(output, list):
                output = [output]

            loss = 0.0
            for j in range(len(output)):
                loss += args.exit_weights[j] * criterion(output[j], target_var)

            losses.update(loss.item(), batch_size)

            for j in range(len(output)):
                prec1, prec5 = accuracy(
                    output[j].data, target, topk=(1, min(5, args.num_classes))
                )
                top1[j].update(prec1.item(), batch_size)
                top5[j].update(prec5.item(), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print(
                    "Epoch: [{0}/{1}]\t"
                    "Time {batch_time.avg:.3f}\t"
                    "Data {data_time.avg:.3f}\t"
                    "Loss {loss.val:.4f}\t"
                    "Acc@1 {top1.val:.4f}\t"
                    "Acc@5 {top5.val:.4f}".format(
                        i + 1,
                        len(val_loader),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        top1=top1[-1],
                        top5=top5[-1],
                    )
                )

    result_file = os.path.join(args.save, "AnytimeResults.txt")

    fd = open(result_file, "w+")
    fd.write("AnytimeResults" + "\n")
    for j in range(args.num_exits):
        test_str = " @{ext}** flops {flops:.2f}M prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}".format(
            ext=j + 1, flops=n_flops[j] / 1e6, top1=top1[j], top5=top5[j]
        )
        print(test_str)
        fd = open(result_file, "a+")
        fd.write(test_str + "\n")
    fd.close()

    torch.save([e.avg for e in top1], os.path.join(args.save, "acc.pth"))
    return losses.avg, top1[-1].avg, top5[-1].avg


def save_checkpoint(state, args, is_best, filename, result):
    print(args)
    result_filename = os.path.join(args.save, "scores.tsv")
    model_dir = os.path.join(args.save, "save_models")
    latest_filename = os.path.join(model_dir, "latest.txt")
    model_filename = os.path.join(model_dir, filename)
    best_filename = os.path.join(model_dir, "model_best.pth.tar")
    os.makedirs(args.save, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    print("=> saving checkpoint '{}'".format(model_filename))

    torch.save(state, model_filename)

    with open(result_filename, "w") as f:
        print("\n".join(result), file=f)

    with open(latest_filename, "w") as fout:
        fout.write(model_filename)

    if is_best:
        shutil.copyfile(model_filename, best_filename)
        best_filename_epoch = os.path.join(model_dir, "best_model_epoch.txt")
        with open(best_filename_epoch, "w") as fout:
            fout.write(model_filename)

    print("=> saved checkpoint '{}'".format(model_filename))
    return


def load_checkpoint(args):
    model_dir = os.path.join(args.save, "save_models")
    latest_filename = os.path.join(model_dir, "latest.txt")
    if os.path.exists(latest_filename):
        with open(latest_filename, "r") as fin:
            model_filename = fin.readlines()[0].strip()
    else:
        return None
    print("=> loading checkpoint '{}'".format(model_filename))
    state = torch.load(model_filename)
    print("=> loaded checkpoint '{}'".format(model_filename))
    return state


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precor@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # correct_k = correct[:k].view(-1).float().sum(0)
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def adjust_learning_rate(
    optimizer, epoch, args, batch=None, nBatch=None, method="multistep"
):
    if method == "cosine":
        T_total = args.epochs * nBatch
        T_cur = (epoch % args.epochs) * nBatch + batch
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == "multistep":
        if args.data.startswith("cifar"):
            lr, decay_rate = args.lr, 0.1
            if epoch >= args.epochs * 0.75:
                lr *= decay_rate**2
            elif epoch >= args.epochs * 0.5:
                lr *= decay_rate
        else:
            lr = args.lr * (0.1 ** (epoch // 30))
    elif method == "mlcsd":
        decay_start = 150
        decay_every = 30
        decay_rate = 0.1

        if epoch < decay_start:
            lr = args.lr
        else:
            num_decays = (epoch - decay_start) // decay_every + 1
            lr = args.lr * (decay_rate ** num_decays)
    else:
        raise ValueError(f"Unsupported lr schedule method: {method}")

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()