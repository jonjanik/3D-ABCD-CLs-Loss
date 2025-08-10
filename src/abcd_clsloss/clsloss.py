# Copyright (c) 2025
# MIT License
#
# Differentiable CLs loss for ABCD with implicit differentiation through
# profile-likelihood fits. This module is intentionally standalone so it can
# be used inside external/private training frameworks.

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions.normal import Normal

# iminuit is used as a very fast/robust local optimizer for the small
# per-bin profile-likelihood fits. It is treated as a black box in the
# forward pass; gradients are recovered via implicit differentiation.
from iminuit import Minuit


class CLsLoss(nn.Module):
    """
    End-to-end CLs loss for an automated ABCD method (3D ABCDisCoTEC style).

    The module:
      1) Builds differentiable ABCD region masks from two classifier scores.
      2) Aggregates signal and background histograms of a final discriminant (mt).
      3) Defines a likelihood with a per-bin non-closure nuisance.
      4) Runs internal profile-likelihood fits and backpropagates with IFT.
      5) Returns a CLs-inspired loss proportional to p_{s+b}(sqrt(q_mu)).

    Important: this is a research prototype and can be numerically unstable.
    """
    def __init__(self, mt_bin_edges: torch.Tensor, mt_min: float, mt_max: float,
                 int_lumi: float = 117100, epsilon: float = 1e-6,
                 steepness: float = 20.0, num_retries: int = 5):
        super().__init__()
        self.register_buffer('mt_bin_edges', mt_bin_edges)
        self.mt_min = float(mt_min)
        self.mt_max = float(mt_max)
        self.epsilon = float(epsilon)
        self.int_lumi = float(int_lumi)
        self.steepness = float(steepness)
        self.num_retries = int(num_retries)

    # ------------ Region assignment (differentiable) -------------------------
    class RegionAssignment(torch.autograd.Function):
        """
        Custom autograd node for soft ABCD region gates.
        The forward computes smooth sigmoid gates; backward passes gradients
        through the sigmoids (no STE tricks here to keep it honest).
        """
        @staticmethod
        def forward(ctx, f1, f2, cut1, cut2, steepness: float):
            s = steepness
            # Save tensors for backward
            ctx.save_for_backward(f1, f2, cut1, cut2)
            ctx.s = s

            in_A = torch.sigmoid(s * (f1 - cut1)) * torch.sigmoid(s * (cut2 - f2))
            in_B = torch.sigmoid(s * (f1 - cut1)) * torch.sigmoid(s * (f2 - cut2))
            in_C = torch.sigmoid(s * (cut1 - f1)) * torch.sigmoid(s * (cut2 - f2))
            in_D = torch.sigmoid(s * (cut1 - f1)) * torch.sigmoid(s * (f2 - cut2))
            masks = torch.stack([in_A, in_B, in_C, in_D], dim=-1)
            return masks

        @staticmethod
        def backward(ctx, grad_out):
            f1, f2, cut1, cut2 = ctx.saved_tensors
            s = ctx.s

            sig = torch.sigmoid
            # Partial derivatives for each mask wrt inputs
            # d/dx sig(s*x) = s*sig(s*x)*(1-sig(s*x))
            def d_sig(x): return s * sig(s * x) * (1.0 - sig(s * x))

            # reuse gates
            g_f1 = sig(s * (f1 - cut1))
            g_f1c = sig(s * (cut1 - f1))
            g_f2a = sig(s * (cut2 - f2))
            g_f2b = sig(s * (f2 - cut2))

            # derivatives
            dg_f1 = d_sig(f1 - cut1)
            dg_f1c = d_sig(cut1 - f1) * (-1.0)
            dg_f2a = d_sig(cut2 - f2) * (-1.0)
            dg_f2b = d_sig(f2 - cut2)

            # masks for clarity
            in_A = g_f1 * g_f2a
            in_B = g_f1 * g_f2b
            in_C = g_f1c * g_f2a
            in_D = g_f1c * g_f2b

            # Chain rule: mask = u*v
            # dmask/df1 = du/df1 * v
            dA_df1 = dg_f1 * g_f2a
            dB_df1 = dg_f1 * g_f2b
            dC_df1 = dg_f1c * g_f2a
            dD_df1 = dg_f1c * g_f2b

            dA_df2 = g_f1 * dg_f2a
            dB_df2 = g_f1 * dg_f2b
            dC_df2 = g_f1c * dg_f2a
            dD_df2 = g_f1c * dg_f2b

            # grads coming from stacked [A,B,C,D]
            gA, gB, gC, gD = grad_out.unbind(dim=-1)

            grad_f1 = gA * dA_df1 + gB * dB_df1 + gC * dC_df1 + gD * dD_df1
            grad_f2 = gA * dA_df2 + gB * dB_df2 + gC * dC_df2 + gD * dD_df2

            # cuts are treated as constants in training; return zeros
            grad_cut1 = torch.zeros_like(cut1)
            grad_cut2 = torch.zeros_like(cut2)

            # steepness is a float, not a tensor
            return grad_f1, grad_f2, grad_cut1, grad_cut2, None

    # ----------------------- Histogramming ----------------------------------
    def compute_mt_hist(self, mt_values: torch.Tensor, weights: torch.Tensor, region_mask: torch.Tensor):
        """
        Weighted histogram of mt within a region. Aggregates over the *full*
        batch to avoid mini-batch biases. Returns counts per bin scaled by
        integrated luminosity.
        """
        nbin = len(self.mt_bin_edges) - 1
        hist = torch.zeros(nbin, device=mt_values.device)
        for i in range(nbin):
            low, high = self.mt_bin_edges[i], self.mt_bin_edges[i+1]
            in_bin = (mt_values >= low) & (mt_values < high)
            hist[i] = (weights * region_mask * in_bin.float() * self.int_lumi).sum()
        return hist

    # ----------------------- Likelihood pieces -------------------------------
    def pois_llh(self, obs: torch.Tensor, exp: torch.Tensor):
        """Poisson log-likelihood, ignoring constants independent of parameters."""
        return obs * torch.log(exp + self.epsilon) - exp - torch.lgamma(obs + 1)

    def gauss_llh(self, theta: torch.Tensor):
        """Standard normal prior for the non-closure nuisance per bin."""
        return -0.5 * theta.pow(2)

    def compute_delta(self, obs_A, obs_B, obs_C, obs_D):
        """Live non-closure estimate from ABCD in data/MC Asimov."""
        bkg_pred = obs_A * obs_D / (obs_C + self.epsilon)
        return (obs_B - bkg_pred) / (bkg_pred + self.epsilon)

    def likelihood(self,
                   obs_A, obs_B, obs_C, obs_D,
                   S_A,   S_B,   S_C,   S_D,
                   mu, theta, nA, nC, nD):
        """
        Full bin-by-bin log-likelihood for regions A,B,C,D.
        nA,nC,nD are background expectations in control regions;
        SR (B) background is TF * (1+delta)^{theta}.
        """
        exp_A = nA + mu * S_A
        exp_C = nC + mu * S_C
        exp_D = nD + mu * S_D

        delta = self.compute_delta(obs_A, obs_B, obs_C, obs_D)
        bkg_SR = (nA * nD / (nC + self.epsilon)) * (1.0 + delta).pow(theta)
        exp_B = bkg_SR + mu * S_B

        llh = (
            self.pois_llh(obs_A, exp_A) +
            self.pois_llh(obs_B, exp_B) +
            self.pois_llh(obs_C, exp_C) +
            self.pois_llh(obs_D, exp_D) +
            self.gauss_llh(theta)
        )
        return llh.sum()

    # ----------------------- Implicit differentiation helpers ----------------
    class _ProfileParams(torch.autograd.Function):
        """
        Fit theta, nA, nC, nD for a fixed mu and return the concatenated
        optimum p* = [theta, nA, nC, nD]. Backward computes gradients wrt
        upstream tensors by inverting the Jacobian of the stationarity
        conditions (implicit function theorem).
        """
        @staticmethod
        def forward(ctx,
                    obs_A, obs_B, obs_C, obs_D,
                    S_A,   S_B,   S_C,   S_D,
                    mu,
                    module: "CLsLoss"):

            nbin = obs_A.numel()

            # define Python NLL for Minuit (non-differentiable wrapper)
            def nll_fn(*params):
                p = torch.tensor(params, device=obs_A.device, dtype=obs_A.dtype)
                θ  = p[:nbin]
                nA = p[nbin:2*nbin]
                nC = p[2*nbin:3*nbin]
                nD = p[3*nbin:4*nbin]
                ll = module.likelihood(obs_A, obs_B, obs_C, obs_D,
                                       S_A, S_B, S_C, S_D,
                                       mu, θ, nA, nC, nD)
                return float(-ll.detach().cpu().item())

            init = np.concatenate([
                np.zeros(obs_A.numel(), dtype=np.float64),
                obs_A.detach().cpu().numpy().astype(np.float64),
                obs_C.detach().cpu().numpy().astype(np.float64),
                obs_D.detach().cpu().numpy().astype(np.float64),
            ])

            names = [f"p{i}" for i in range(len(init))]
            m = Minuit(nll_fn, *init.tolist(), name=names)
            m.limits = [(None, None)] * (4 * nbin)
            m.errordef = Minuit.LIKELIHOOD
            m.migrad()

            best = np.array([m.values[n] for n in m.parameters], dtype=np.float32)
            p_star = torch.tensor(best, device=obs_A.device)

            # Save for backward
            ctx.save_for_backward(obs_A, obs_B, obs_C, obs_D,
                                  S_A, S_B, S_C, S_D, mu, p_star)
            ctx.module = module
            return p_star

        @staticmethod
        def backward(ctx, grad_p):
            obs_A, obs_B, obs_C, obs_D, S_A, S_B, S_C, S_D, mu, p_star = ctx.saved_tensors
            module: CLsLoss = ctx.module
            nbin = obs_A.numel()

            with torch.enable_grad():
                p = p_star.clone().requires_grad_(True)
                θ, nA, nC, nD = p[:nbin], p[nbin:2*nbin], p[2*nbin:3*nbin], p[3*nbin:4*nbin]

                def g_func(_p, A, B, C, D, SA, SB, SC, SD, m):
                    θ_, nA_, nC_, nD_ = _p[:nbin], _p[nbin:2*nbin], _p[2*nbin:3*nbin], _p[3*nbin:4*nbin]
                    llh = module.likelihood(A, B, C, D, SA, SB, SC, SD, m, θ_, nA_, nC_, nD_)
                    return torch.autograd.grad(llh, _p, create_graph=True)[0]

                inputs = (p, obs_A, obs_B, obs_C, obs_D, S_A, S_B, S_C, S_D, mu)
                J = torch.autograd.functional.jacobian(g_func, inputs, vectorize=True)
                J_pp = J[0]

                lambd = 1e-9
                I = torch.eye(J_pp.size(0), device=J_pp.device, dtype=J_pp.dtype)
                invH = torch.inverse(J_pp + lambd * I)

                grads = []
                for i in range(1, 9):
                    J_px = J[i]
                    grad_input = - (J_px.T @ (invH @ grad_p))
                    grads.append(grad_input)

                # Return gradients in the same order as inputs; None for module/mu slot already consumed
                return (None, *grads, None, None)

    def fit_params(self,
                   obs_A, obs_B, obs_C, obs_D,
                   S_A,   S_B,   S_C,   S_D,
                   mu):
        return self._ProfileParams.apply(obs_A, obs_B, obs_C, obs_D,
                                         S_A, S_B, S_C, S_D, mu, self)

    class _ProfileParamsMu(torch.autograd.Function):
        """
        As above but includes mu as a free parameter in the fit.
        """
        @staticmethod
        def forward(ctx,
                    obs_A, obs_B, obs_C, obs_D,
                    S_A,   S_B,   S_C,   S_D,
                    module: "CLsLoss"):

            nbin = obs_A.numel()

            def nll_fn(*params):
                p = torch.tensor(params, device=obs_A.device, dtype=obs_A.dtype)
                θ  = p[:nbin]
                nA = p[nbin:2*nbin]
                nC = p[2*nbin:3*nbin]
                nD = p[3*nbin:4*nbin]
                mu = p[4*nbin]
                ll = module.likelihood(obs_A, obs_B, obs_C, obs_D,
                                       S_A, S_B, S_C, S_D,
                                       mu, θ, nA, nC, nD)
                return float(-ll.detach().cpu().item())

            init = np.concatenate([
                np.zeros(nbin, dtype=np.float64),
                obs_A.detach().cpu().numpy().astype(np.float64),
                obs_C.detach().cpu().numpy().astype(np.float64),
                obs_D.detach().cpu().numpy().astype(np.float64),
                np.array([1.0], dtype=np.float64)
            ])

            names = [f"p{i}" for i in range(len(init))]
            m = Minuit(nll_fn, *init.tolist(), name=names)
            m.limits = [(None, None)] * (4 * nbin) + [(0.0, None)]
            m.errordef = Minuit.LIKELIHOOD
            m.migrad()

            best = np.array([m.values[n] for n in m.parameters], dtype=np.float32)
            p_star = torch.tensor(best, device=obs_A.device)

            ctx.save_for_backward(obs_A, obs_B, obs_C, obs_D, S_A, S_B, S_C, S_D, p_star)
            ctx.module = module
            return p_star

        @staticmethod
        def backward(ctx, grad_p):
            obs_A, obs_B, obs_C, obs_D, S_A, S_B, S_C, S_D, p_star = ctx.saved_tensors
            module: CLsLoss = ctx.module
            nbin = obs_A.numel()

            with torch.enable_grad():
                p = p_star.clone().requires_grad_(True)
                def g_func(_p, A, B, C, D, SA, SB, SC, SD):
                    θ_, nA_, nC_, nD_, mu_ = _p[:nbin], _p[nbin:2*nbin], _p[2*nbin:3*nbin], _p[3*nbin:4*nbin], _p[4*nbin:5*nbin]
                    llh = module.likelihood(A, B, C, D, SA, SB, SC, SD, mu_, θ_, nA_, nC_, nD_)
                    return torch.autograd.grad(llh, _p, create_graph=True)[0]

                inputs = (p, obs_A, obs_B, obs_C, obs_D, S_A, S_B, S_C, S_D)
                J = torch.autograd.functional.jacobian(g_func, inputs, vectorize=True)
                J_pp = J[0]

                lambd = 1e-9
                I = torch.eye(J_pp.size(0), device=J_pp.device, dtype=J_pp.dtype)
                invH = torch.inverse(J_pp + lambd * I)

                grads = []
                for i in range(1, 8):
                    J_px = J[i]
                    grad_input = - (J_px.T @ (invH @ grad_p))
                    grads.append(grad_input)

                return (None, *grads, None)

    def fit_paramsMu(self,
                     obs_A, obs_B, obs_C, obs_D,
                     S_A,   S_B,   S_C,   S_D):
        return self._ProfileParamsMu.apply(obs_A, obs_B, obs_C, obs_D,
                                           S_A, S_B, S_C, S_D, self)

    # ----------------------------- Forward -----------------------------------
    def forward(self, models, features, cuts,
                weights_xs, weights_train, target, data_dict):
        """
        Compute CLs-like loss.

        Parameters
        ----------
        models : tuple(nn.Module, nn.Module)
            Two models producing scalar scores used for ABCD region defs.
        features : Tensor
            Input batch for both models.
        cuts : (float, float)
            Thresholds for ABCD on (f1, f2) respectively.
        weights_xs : Tensor
            Per-event weights used to scale histograms (interpreted as xs).
        weights_train : Tensor
            Currently unused (kept for compatibility / future use).
        target : Tensor[0 or 1]
            Event labels: 0=background, 1=signal.
        data_dict : dict
            Must contain key 'constraint_MT01FatJetMET' with the final
            discriminant scaled to [0,1]. It will be mapped to [mt_min, mt_max].
        """
        f1 = models[0](features)[:, 0]
        f2 = models[1](features)[:, 0]

        masks = self.RegionAssignment.apply(
            f1, f2,
            torch.tensor(float(cuts[0]), device=f1.device, dtype=f1.dtype),
            torch.tensor(float(cuts[1]), device=f2.device, dtype=f2.dtype),
            self.steepness
        )
        in_A, in_B, in_C, in_D = masks.unbind(dim=-1)

        mt = data_dict['constraint_MT01FatJetMET']
        mt = mt * (self.mt_max - self.mt_min) + self.mt_min

        bg  = (target == 0)
        sig = (target == 1)

        B_A = self.compute_mt_hist(mt[bg],  weights_xs[bg],  in_A[bg])
        B_B = self.compute_mt_hist(mt[bg],  weights_xs[bg],  in_B[bg])
        B_C = self.compute_mt_hist(mt[bg],  weights_xs[bg],  in_C[bg])
        B_D = self.compute_mt_hist(mt[bg],  weights_xs[bg],  in_D[bg])

        # Scale down signal templates if they correspond to multiple mass points mirrored, etc.
        S_A = self.compute_mt_hist(mt[sig], weights_xs[sig], in_A[sig]) / 33.0
        S_B = self.compute_mt_hist(mt[sig], weights_xs[sig], in_B[sig]) / 33.0
        S_C = self.compute_mt_hist(mt[sig], weights_xs[sig], in_C[sig]) / 33.0
        S_D = self.compute_mt_hist(mt[sig], weights_xs[sig], in_D[sig]) / 33.0

        # 1) Background-only fit to initialize nuisances using current templates
        mu_bkg = torch.tensor(0.0, device=B_A.device, dtype=B_A.dtype)
        p_bkg = self.fit_params(B_A, B_B, B_C, B_D, S_A, S_B, S_C, S_D, mu_bkg)

        nbin = B_A.numel()
        θ_bkg  = p_bkg[:nbin]
        nA_bkg = p_bkg[nbin:2*nbin]
        nC_bkg = p_bkg[2*nbin:3*nbin]
        nD_bkg = p_bkg[3*nbin:4*nbin]

        # 2) Asimov counts for μ=0 with nuisance θ_bkg (posterior Asimov)
        bkg_SR_asim = (nA_bkg * nD_bkg / (nC_bkg + self.epsilon)) \
                      * (1.0 + self.compute_delta(B_A, B_B, B_C, B_D)).pow(θ_bkg)

        A_asim = nA_bkg
        B_asim = bkg_SR_asim
        C_asim = nC_bkg
        D_asim = nD_bkg

        # 3) Profile-likelihood ratio: numerator at fixed μ=1, denominator with free μ
        mu_fixed = torch.tensor(1.0, device=B_A.device, dtype=B_A.dtype)
        p_hat0 = self.fit_params(A_asim, B_asim, C_asim, D_asim, S_A, S_B, S_C, S_D, mu_fixed)

        θ_hat0, nA_hat0, nC_hat0, nD_hat0 = p_hat0[:nbin], p_hat0[nbin:2*nbin], p_hat0[2*nbin:3*nbin], p_hat0[3*nbin:4*nbin]
        ll0 = self.likelihood(A_asim, B_asim, C_asim, D_asim, S_A, S_B, S_C, S_D, mu_fixed, θ_hat0, nA_hat0, nC_hat0, nD_hat0)

        p_hat1 = self.fit_paramsMu(A_asim, B_asim, C_asim, D_asim, S_A, S_B, S_C, S_D)
        θ_hat1, nA_hat1, nC_hat1, nD_hat1, mu_hat1 = \
            p_hat1[:nbin], p_hat1[nbin:2*nbin], p_hat1[2*nbin:3*nbin], p_hat1[3*nbin:4*nbin], p_hat1[4*nbin:5*nbin]

        ll1 = self.likelihood(A_asim, B_asim, C_asim, D_asim, S_A, S_B, S_C, S_D, mu_hat1, θ_hat1, nA_hat1, nC_hat1, nD_hat1)

        qmu = -2.0 * (ll0 - ll1)
        qmu = torch.clamp(qmu, min=0.0)

        normal = Normal(0.0, 1.0)
        sqrt_q = torch.sqrt(torch.clamp(qmu, min=0.0))
        p_sb = 1.0 - normal.cdf(sqrt_q)

        # Scaled for convenience; users may combine with other losses via M-DMM
        cls_loss = 2.0 * p_sb
        return cls_loss
