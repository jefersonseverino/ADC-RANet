
import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableExitRouter(nn.Module):
    def __init__(
        self,
        n_exits,
        flops_per_exit,
        init_tau=1.0,
        tau_min=0.1,
        hard=False,
    ):
        super().__init__()
        self.n_exits = n_exits
        self.tau_min = tau_min
        self.hard = hard

        self.thresholds_raw = nn.Parameter(torch.zeros(n_exits - 1))

        self.log_tau = nn.Parameter(torch.tensor(float(init_tau)).log())

        flops = torch.tensor(flops_per_exit, dtype=torch.float32)
        self.register_buffer("flops", flops)
        self.register_buffer("flops_normalized", flops / flops.max())

    @property
    def thresholds(self):
        return torch.sigmoid(self.thresholds_raw)

    @property
    def tau(self):
        return torch.exp(self.log_tau).clamp(min=self.tau_min, max=5.0)

    def forward(self, exit_outputs):
        batch_size = exit_outputs[0].size(0)
        device = exit_outputs[0].device

        exit_softmax = []
        confidences = []
        for k in range(self.n_exits):
            prob = F.softmax(exit_outputs[k], dim=1)
            exit_softmax.append(prob)
            confidences.append(prob.max(dim=1)[0])

        conf_stack = torch.stack(confidences, dim=1)
        tau = self.tau

        T = self.thresholds  # [0, 1] via sigmoid
        exit_logits = conf_stack[:, :-1] - T.unsqueeze(0)  # (B, n-1)
        stay_logits = torch.zeros_like(exit_logits)                      # (B, n-1)
        gate_logits = torch.stack([exit_logits, stay_logits], dim=-1)    # (B, n-1, 2)

        if self.training:
            gate_samples = F.gumbel_softmax(
                gate_logits, tau=tau, hard=self.hard, dim=-1
            )  # (B, n-1, 2)
        else:
            # Eval: deterministic softmax with current temperature (no noise)
            gate_samples = F.softmax(gate_logits / tau, dim=-1)  # (B, n-1, 2)

        gate_probs = gate_samples[:, :, 0]  # (B, n_exits-1)
        gate_probs = gate_probs.clamp(min=1e-6, max=1.0 - 1e-6)

        log_stay = torch.log(1.0 - gate_probs)  # (B, n_exits-1)
        cumulative_log_stay = torch.cumsum(log_stay, dim=1)

        survive_before = torch.cat(
            [torch.zeros(batch_size, 1, device=device), cumulative_log_stay[:, :-1]],
            dim=1,
        )
        survive_prob = torch.exp(survive_before)

        exit_prob_early = gate_probs * survive_prob  # (B, n_exits-1)

        prob_reach_last = torch.exp(cumulative_log_stay[:, -1:])
        exit_probs = torch.cat([exit_prob_early, prob_reach_last], dim=1)

        stacked = torch.stack(exit_softmax, dim=1)   # (B, n_exits, n_classes)
        weights = exit_probs.unsqueeze(2)            # (B, n_exits, 1)
        final_probs = (stacked * weights).sum(dim=1) # (B, n_classes)

        final_logits = torch.log(final_probs + 1e-8)

        expected_flops_norm = (exit_probs * self.flops_normalized.unsqueeze(0)).sum(
            dim=1
        )  # (B,)

        info = {
            "exit_probs": exit_probs,                                    # (B, n_exits)
            "expected_flops_normalized": expected_flops_norm.mean(),      # scalar
            "expected_flops": (exit_probs * self.flops.unsqueeze(0))
            .sum(dim=1)
            .mean(),
            "avg_exit_dist": exit_probs.mean(dim=0).detach(),            # (n_exits,)
            "thresholds": self.thresholds.detach(),  # already in [0,1]
            "tau": tau.item(),
            "confidences": conf_stack.detach(),
        }

        return final_logits, info

    def get_hard_thresholds(self):
        """
        Get thresholds for hard early-exit inference.
        Returns tensor of shape (n_exits,) with T[-1] = -inf
        (last exit always accepts).
        """
        with torch.no_grad():
            T = torch.zeros(self.n_exits)
            T[:-1] = self.thresholds.cpu()  # sigmoid already applied
            T[-1] = 0.0  # last exit always accepts (threshold=0)
        return T

    def set_hard_thresholds(self, T):
        with torch.no_grad():
            if isinstance(T, list):
                T = torch.tensor(T, dtype=self.thresholds_raw.dtype, device=self.thresholds_raw.device)
            #self.thresholds_raw.copy_(T[:self.n_exits - 1])
            T = T[:self.n_exits - 1]
            self.thresholds_raw.copy_(torch.log(T / (1.0 - T)))
            self.log_tau.copy_(torch.tensor(self.tau_min).log())

    def anneal_tau(self, epoch, total_epochs, warmup_epochs=10):
        """
        Anneal temperature from init_tau (soft) down to tau_min (hard).
        Call at the start of each epoch.

        During warmup_epochs, tau stays at 1.0 (soft, exploratory).
        After warmup, tau linearly decreases toward tau_min.
        """
        if epoch < warmup_epochs:
            target_log_tau = torch.tensor(1.0).log()
        else:
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            target_tau = 1.0 - (1.0 - self.tau_min) * progress
            target_tau = max(target_tau, self.tau_min)
            target_log_tau = torch.tensor(target_tau).log()

        with torch.no_grad():
            # Soft EMA update toward target (avoids abrupt jumps)
            self.log_tau.copy_(
                0.9 * self.log_tau + 0.1 * target_log_tau.to(self.log_tau.device)
            )
