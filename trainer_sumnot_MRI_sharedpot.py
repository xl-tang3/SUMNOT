
import argparse
import math
import os
import random
import sys

import torch
from torch import nn                                                                                                                                        
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import save_image
from tqdm import tqdm
from scipy import io as scio
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
import timm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model_sumnot import *
from util import unfreeze, freeze
from utils.universal_dataset_random import MFMRI_TrainDataset, MFMRI_TestDataset


# =========================================================================
# Configuration
# =========================================================================
parser = argparse.ArgumentParser(description="PyTorch MFMRI Unpaired SUMNOT")

parser.add_argument("--batchSize", type=int, default=4, help="Training batch size per GPU")
parser.add_argument("--nEpochs", type=int, default=200, help="Number of epochs")
parser.add_argument("--lr", type=float, default=2e-4, help="Learning Rate")
parser.add_argument("--step", type=int, default=50, help="Learning rate decay step")
parser.add_argument("--cuda", default=True, help="Use cuda?")
parser.add_argument("--resume_base", default=None, type=str, help="Path to resume base model")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number")
parser.add_argument("--threads", type=int, default=8, help="Number of threads")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids")

# Retained for compatibility with original launch commands. The cyclic sampler
# below creates batchSize source slots, so this legacy option does not set slots.
parser.add_argument("--num_sources", type=int, default=3,
                    help="Legacy option; cyclic source-slot count is batchSize")
parser.add_argument("--tau", type=float, default=0.5,
                    help="Finite positive HQ KL coefficient")
parser.add_argument("--data_dir", type=str, default="data/MFMRI", help="Root path of MFMRI dataset")
parser.add_argument("--Sigma", default=10.0, type=float, help="Weight for L1 Loss")
parser.add_argument("--Gamma", default=0.5, type=float, help="Legacy parameter (unused)")
parser.add_argument("--Omega", default=0.1, type=float, help="Weight for SUMNOT ADMM Overall Penalty")
parser.add_argument("--de_type", nargs="+", default=["all"], help='Use "all" or list like "LF_mild" "LF_Severe"')
parser.add_argument("--optimizer", default="RMSprop", type=str)
parser.add_argument("--backbone", default="Restormer", type=str)
parser.add_argument("--type", default="sumnot_mri_unpair", type=str)
parser.add_argument("--patch_size", type=int, default=96, help="Patch size for training crop")
parser.add_argument("--resume_training", type=lambda x: x.lower() in ("true", "1", "yes", "y"), default=False,help="Whether to resume training")
parser.add_argument("--grad_accum_steps", type=int, default=1, help="Number of mini-batches to accumulate before optimizer step")

# Understanding model config
parser.add_argument("--understanding_model_name", default="medmae", type=str)
parser.add_argument("--medmae_ckpt", default="./weights/pre_trained_medmae.pth", type=str, help="Path to MedMAE weights")
parser.add_argument("--projector_dim", type=int, default=-1, help="Auto-set by code")
parser.add_argument("--local_rank", type=int, default=-1)


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def psnr(img1, img2, data_range=1):
    return psnr_loss(img1, img2, data_range=data_range)


def get_understanding_model_and_config(model_name, device, opt):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    model_name = model_name.lower()

    if "medmae" in model_name:
        if is_main_process():
            print(f"[Auto-Config] Loading MedMAE (ViT-Base) from {opt.medmae_ckpt}")
        model = timm.create_model("vit_base_patch16_224", pretrained=False)

        if os.path.exists(opt.medmae_ckpt):
            checkpoint = torch.load(opt.medmae_ckpt, map_location="cpu")
            state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
            model.load_state_dict(state_dict, strict=False)
            if is_main_process():
                print(f"[Success] Loaded MedMAE weights from {opt.medmae_ckpt}")
        else:
            if is_main_process():
                print(f"[Warning] {opt.medmae_ckpt} NOT FOUND! Using random weights.")

        patch_size, embed_dim, target_res = 16, 768, 224

        def forward_fn(model, x):
            return model.forward_features(x)[:, 1:, :]

    elif "dinov2" in model_name:
        if is_main_process():
            print(f"[Auto-Config] Loading DINOv2 ({model_name}) for Unpaired Alignment.")
        model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True)
        patch_size, embed_dim, target_res = 14, 384, 518

        def forward_fn(model, x):
            return model.forward_features(x)[:, 1:, :]

    elif "mae" in model_name or "vit_base" in model_name:
        if is_main_process():
            print(f"[Auto-Config] Loading Natural MAE/ViT ({model_name}) for Unpaired Alignment.")
        model = timm.create_model("vit_base_patch16_224.mae", pretrained=True)
        patch_size, embed_dim, target_res = 16, 768, 224

        def forward_fn(model, x):
            return model.forward_features(x)[:, 1:, :]
    else:
        raise ValueError("Model not supported.")

    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, patch_size, embed_dim, target_res, forward_fn



@torch.no_grad()
def project_inbatch_multiplier_(lambdas, residuals, rho):
    """Exact componentwise projection on the current batch's tuple slots.

    This performs lambda_b <- max(0, lambda_b + rho * r_b) without
    fitting a neural network.  A slot index is reused across random
    batches; it is *not* a persistent identity for the same data tuple.
    """
    if lambdas.shape != residuals.shape:
        raise ValueError("Multiplier and residual must have identical in-batch shape")
    lambdas.add_(rho * residuals.detach()).clamp_(min=0.0)
    return lambdas


def inbatch_source_tuple(features):
    """N cyclic permutations of N IID observations of the mixed LF marginal.

    All N entries appear in each tuple, yielding an N-source stochastic surrogate.
    This is not an exact K=7 joint marginal sample unless sampling is stratified.
    """
    return [torch.roll(features, shifts=k, dims=0) for k in range(features.shape[0])]


def normalize_ot_feature(x):
    """L2-normalize each sample over all feature dimensions; preserve feature shape."""
    return F.normalize(x.flatten(1), p=2, dim=1, eps=1e-8).reshape_as(x)


def quadratic_multimarginal_cost(sources, hq):
    """Squared Euclidean multimarginal cost on unit-L2 features; per tuple."""
    def sqdist(x, y):
        return (x - y).flatten(1).square().sum(dim=1)
    cost = sum((sqdist(x, hq) for x in sources))
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            cost = cost + sqdist(sources[i], sources[j])
    return cost


class Potentials(nn.Module):
    def __init__(self, num_potentials, channels=384, size=128, *, num_sources):
        super().__init__()
        if num_potentials != 2:
            raise ValueError("Expected two heads: conditioned source and HQ.")
        if not isinstance(num_sources, int) or num_sources < 1:
            raise ValueError("num_sources must be a positive integer.")
        if channels < 4 or channels % 4:
            raise ValueError("channels must be positive and divisible by 4.")
        self.num_potentials = num_potentials
        self.num_sources = num_sources
        self.input_channels = channels
        # Preserve the original size convention; this is not the input H/W.
        self.num_features = int(channels * (size / 32) * (size / 32))
        if self.num_features < 4:
            raise ValueError("size is too small for the original MLP.")
        self.potentials = nn.ModuleList([
            self._build_potentials(channels, self.num_features, num_sources),
            self._build_potentials(channels, self.num_features, 0),
        ])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def _build_potentials(self, input_channels, num_fea, condition_dim):
        return nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=4,
                      stride=4, groups=4, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(num_fea + condition_dim, num_fea // 4),
            nn.Linear(num_fea // 4, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
        )

    def _source(self, features, condition):
        if condition is None:
            raise ValueError("Source potential requires a source-slot condition.")
        if isinstance(condition, int):
            if not 0 <= condition < self.num_sources:
                raise ValueError("Source-slot condition is outside [0, num_sources).")
            labels = torch.full((features.shape[0],), condition,
                                device=features.device, dtype=torch.long)
        elif torch.is_tensor(condition):
            if condition.dtype not in (torch.int32, torch.int64):
                raise TypeError("Source-slot conditions must be integer IDs.")
            labels = condition.to(device=features.device, dtype=torch.long)
            if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
                raise ValueError("Expected one source-slot ID per feature sample.")
        else:
            raise TypeError("condition must be an integer or integer tensor.")

        # one_hot rejects negative IDs and IDs >= num_sources.
        code = F.one_hot(labels, num_classes=self.num_sources)
        branch = self.potentials[0]
        hidden = features
        for layer in list(branch.children())[:3]:
            hidden = layer(hidden)
        if hidden.shape[1] != self.num_features:
            raise ValueError(
                f"Flattened convolution has {hidden.shape[1]} features; "
                f"expected {self.num_features}. Check the size argument."
            )
        hidden = torch.cat((hidden, code.to(dtype=hidden.dtype)), dim=1)
        for layer in list(branch.children())[3:]:
            hidden = layer(hidden)
        return hidden

    def forward(self, input, idx=None, condition=None):
        if input.ndim != 4 or input.shape[1] != self.input_channels:
            raise ValueError("Expected features shaped [B, channels, H, W].")
        if idx == 0:
            return self._source(input, condition)
        if idx == 1:
            return self.potentials[1](input)
        if idx is not None:
            raise ValueError("idx must be 0 (source), 1 (HQ), or None.")
        return [self._source(input, condition), self.potentials[1](input)]


def shared_potential_values(pots, sources, hq):
    """Condition source slot k on k; do not interpret k as a degradation ID."""
    n = hq.shape[0]
    k_slots = len(sources)
    module = pots.module if hasattr(pots, "module") else pots
    if k_slots != module.num_sources:
        raise ValueError("Source-slot count must match the configured batchSize.")
    if any(x.shape != hq.shape for x in sources):
        raise ValueError("Source and HQ batches must have identical feature shapes.")
    z_all = torch.cat([*sources, hq], dim=0)
    source_conditions = torch.arange(
        k_slots, device=hq.device, dtype=torch.long
    ).repeat_interleave(n)
    # The source-head output on HQ rows is unused; its condition is a dummy 0.
    conditions = torch.cat([
        source_conditions,
        torch.zeros(n, device=hq.device, dtype=torch.long),
    ])
    p = pots(z_all, condition=conditions)
    src = [p[0][k*n:(k+1)*n].reshape(-1) for k in range(k_slots)]
    f_hq = p[1][k_slots*n:].reshape(-1)
    return src, f_hq


def load_slot_potential_state(pots, checkpoint, resume_training):
    if resume_training:
        if checkpoint.get("potential_condition") != "cyclic_source_slot_v1":
            raise ValueError("Resume needs a slot-conditioned checkpoint. To initialize "
                             "from the original model use --resume_training false.")
        args = checkpoint.get("args", {})
        if args.get("batchSize") != opt.batchSize:
            raise ValueError("Cannot resume with a different source-slot count/batchSize.")
        if args.get("tau") != opt.tau:
            raise ValueError("Checkpoint tau differs from the requested tau.")
    if "Pots" not in checkpoint:
        if resume_training:
            raise ValueError("Resume checkpoint has no potential weights.")
        return
    state = {k[7:] if k.startswith("module.") else k: v
             for k, v in checkpoint["Pots"].items()}
    key = "potentials.0.3.weight"
    expected = pots.state_dict()[key]
    if key in state and state[key].shape != expected.shape:
        old = state[key]
        if old.ndim != 2 or old.shape != (expected.shape[0], pots.num_features):
            raise ValueError("Potential checkpoint architecture is incompatible.")
        padded = old.new_zeros(expected.shape)
        padded[:, :pots.num_features] = old
        state[key] = padded
        if is_main_process():
            print("[Init] Original potential loaded; added condition weights start at zero.")
    pots.load_state_dict(state, strict=True)


def save_checkpoint(restoration_net, Pots, epoch, inbatch_lambdas_by_rank=None):
    if isinstance(opt.de_type, list) and len(opt.de_type) > 0:
        first_item = opt.de_type[0]
        prefix = first_item.split("_")[0] + "_" if "_" in first_item else ""
        if prefix and all(d.startswith(prefix) for d in opt.de_type):
            suffixes = [d.replace(prefix, "") for d in opt.de_type]
            de_type_str = prefix + "-".join(suffixes)
        else:
            de_type_str = "-".join(opt.de_type)
        if len(de_type_str) > 50:
            de_type_str = f"{len(opt.de_type)}domains"
    else:
        de_type_str = str(opt.de_type)

    model_out_path = f"checkpoint/model_{opt.run_name}_{opt.patch_size}_bs{opt.batchSize}.pth"
    state = {
        "epoch": epoch,
        "potential_condition": "cyclic_source_slot_v1",
        "args": vars(opt).copy(),
        "restoration_net": restoration_net.module.state_dict() if hasattr(restoration_net, "module") else restoration_net.state_dict(),
        "Pots": Pots.module.state_dict() if hasattr(Pots, "module") else Pots.state_dict(),
    }

    if inbatch_lambdas_by_rank is not None:
        # Shape [world_size, batchSize], including each DDP worker's slot state.
        state["inbatch_lambdas_by_rank"] = inbatch_lambdas_by_rank.detach().cpu()

    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")

    torch.save(state, model_out_path)
    if is_main_process():
        print(f"Checkpoint saved to {model_out_path}")


def main():
    global opt, restoration_net
    opt = parser.parse_args()
    if opt.batchSize < 1:
        parser.error("--batchSize must be positive")
    if not math.isfinite(opt.tau) or opt.tau <= 0:
        parser.error("--tau must be finite and positive")
    opt.run_name = f"{opt.type}_slotcond_tau{opt.tau}"

    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = 0

    if opt.cuda:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    cudnn.benchmark = True

    if is_main_process():
        print(opt)
        print("------ Training MFMRI with Semi-Unbalanced SUMNOT ------")

    opt.seed = random.randint(1, 10000)
    seed_tensor = torch.tensor(opt.seed, device=device)
    if dist.is_initialized():
        dist.broadcast(seed_tensor, 0)
    opt.seed = seed_tensor.item()
    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)

    u_net, u_patch, u_dim, u_res, u_forward = get_understanding_model_and_config(opt.understanding_model_name, device, opt)
    opt.projector_dim = u_dim

    if opt.backbone == "Restormer":
        restoration_net = Restormer(decoder=True, use_projector=True, projector_out_dim=u_dim)
    else:
        raise ValueError(f"Backbone {opt.backbone} not supported.")

    feat_h = u_res // u_patch
    h_out = (feat_h + 2 * 1 - 4) // 4 + 1
    dynamic_size = h_out * 32
    # The unchanged cyclic sampler constructs batchSize source slots.
    Pots = Potentials(num_potentials=2, channels=u_dim, size=dynamic_size,
                      num_sources=opt.batchSize)

    if opt.cuda:
        restoration_net = restoration_net.to(device)
        Pots = Pots.to(device)

    if opt.resume_base and os.path.isfile(opt.resume_base):
        if is_main_process():
            print(f"=> loading checkpoint '{opt.resume_base}'")
        checkpoint = torch.load(opt.resume_base, map_location=device)
        if opt.resume_training == False:
            opt.start_epoch =   1
        else:
            opt.start_epoch = checkpoint["epoch"] + 1
        new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint["restoration_net"].items()}
        restoration_net.load_state_dict(new_state_dict)
        load_slot_potential_state(Pots, checkpoint, opt.resume_training)

    # One nonnegative exact-projection multiplier per in-batch tuple slot.
    # It is persistent across iterations on each DDP worker, but slot b in
    # different shuffled batches does not refer to an identical data tuple.
    inbatch_lambdas = torch.zeros(opt.batchSize, device=device)
    if opt.resume_base and os.path.isfile(opt.resume_base) and opt.resume_training:
        saved_lambdas = checkpoint.get("inbatch_lambdas_by_rank")
        if saved_lambdas is not None:
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank >= saved_lambdas.size(0):
                raise RuntimeError("Checkpoint has fewer rank multiplier states than the current world size")
            if saved_lambdas.size(1) != opt.batchSize:
                raise RuntimeError("Checkpoint multiplier batch size differs from --batchSize")
            inbatch_lambdas.copy_(saved_lambdas[rank].to(device))
        elif is_main_process():
            print("[Info] Exact in-batch multipliers start at zero (old checkpoints used a multiplier network).")

    restoration_net = DDP(restoration_net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    Pots = DDP(Pots, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    if opt.optimizer == "AdamW":
        net_optimizer = torch.optim.AdamW(restoration_net.parameters(), lr=opt.lr / 2)
        Pots_optimizer = torch.optim.AdamW(Pots.parameters(), lr=opt.lr / 4)
    else:
        net_optimizer = torch.optim.RMSprop(restoration_net.parameters(), lr=opt.lr / 2)
        Pots_optimizer = torch.optim.RMSprop(Pots.parameters(), lr=opt.lr / 4)

    train_set = MFMRI_TrainDataset(opt)
    train_sampler = DistributedSampler(train_set, shuffle=True)
    training_data_loader = DataLoader(
        dataset=train_set,
        num_workers=opt.threads,
        batch_size=opt.batchSize,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True,
    )

    unpaired_sampler = DistributedSampler(train_set, shuffle=True)
    unpaired_loader = DataLoader(
        dataset=train_set,
        num_workers=opt.threads,
        batch_size=opt.batchSize,
        sampler=unpaired_sampler,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if is_main_process():
        val_level = "LF_0.075T"
        val_set = MFMRI_TestDataset(opt, level=val_level)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    u_pkg = {"model": u_net, "patch_size": u_patch, "forward_fn": u_forward, "dim": u_dim, "res": u_res}

    L1LOSS, AlignLOSS, TLOSS, FLOSS, VOL_list, PSNR_list = [], [], [], [], [], []

    if is_main_process() and not os.path.exists(f"./checksample/{opt.run_name}/"):
        os.makedirs(f"./checksample/{opt.run_name}/")

    dist.barrier()

    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train_sampler.set_epoch(epoch)
        unpaired_sampler.set_epoch(epoch + 2026)

        epoch_l1, epoch_f, epoch_align, epoch_t, epoch_vol = train_epoch(
            training_data_loader,
            unpaired_loader,
            net_optimizer,
            Pots_optimizer,
            restoration_net,
            Pots,
            u_pkg,
            epoch,
            local_rank,
            device,
            inbatch_lambdas,
        )

        # Gather tiny rank-local multiplier vectors for correct DDP resume.
        # Every rank must participate, independently of the main-process I/O.
        if dist.is_initialized():
            gathered_lambdas = [torch.empty_like(inbatch_lambdas)
                                for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_lambdas, inbatch_lambdas)
            multipliers_to_save = torch.stack(gathered_lambdas).cpu()
        else:
            multipliers_to_save = inbatch_lambdas.detach().unsqueeze(0).cpu()

        if is_main_process():
            p_curr = evaluate(restoration_net, val_loader, opt) if val_loader else 0.0
            L1LOSS.append(epoch_l1)
            AlignLOSS.append(epoch_align)
            TLOSS.append(epoch_t)
            FLOSS.append(epoch_f)
            VOL_list.append(epoch_vol)
            PSNR_list.append(p_curr)

            scio.savemat(f"L1LOSS_{opt.run_name}.mat", {"L1LOSS": L1LOSS})
            scio.savemat(f"AlignLOSS_{opt.run_name}.mat", {"AlignLOSS": AlignLOSS})
            scio.savemat(f"TLOSS_{opt.run_name}.mat", {"TLOSS": TLOSS})
            scio.savemat(f"FLOSS_{opt.run_name}.mat", {"FLOSS": FLOSS})
            scio.savemat(f"VOL_{opt.run_name}.mat", {"VOL": VOL_list})
            scio.savemat(f"PSNR_{opt.run_name}.mat", {"PSNR": PSNR_list})

            with open(f"./checksample/{opt.run_name}/log.txt", "a") as f:
                f.write(
                    f"Epoch {epoch}, L1: {epoch_l1:.4f}, Align: {epoch_align:.4f}, T_Total: {epoch_t:.4f}, F: {epoch_f:.4f}, Vol: {epoch_vol:.4f}, PSNR: {p_curr:.4f}\n"
                )

            save_checkpoint(restoration_net, Pots, epoch, multipliers_to_save)

        dist.barrier()


def adjust_learning_rate(optimizer, epoch):
    return opt.lr * (0.1 ** (epoch // opt.step))


def projected_al_penalty(R_val, lambd, rho):
    return (torch.relu(lambd + rho * R_val) ** 2 - lambd ** 2) / (2.0 * rho)


def semi_unbalanced_hq_term(f_z, tau):
    v = torch.clamp(-f_z / tau, min=-30.0, max=30.0)
    return -tau * torch.sum(torch.expm1(v))


def train_epoch(training_data_loader, unpaired_loader, net_optimizer, Pots_optimizer,
                restoration_net, Pots, understanding_pkg, epoch, local_rank, device,
                inbatch_lambdas):
    lr = adjust_learning_rate(Pots_optimizer, epoch - 1)
    for param_group in net_optimizer.param_groups:
        param_group["lr"] = lr / 2
    for param_group in Pots_optimizer.param_groups:
        param_group["lr"] = lr / 4

    restoration_net.train()
    Pots.train()
    unpaired_iter = iter(unpaired_loader)
    epoch_l1_loss = epoch_align_loss = epoch_t_loss = epoch_f_loss = epoch_vol = 0.0
    count = 0

    u_model = understanding_pkg["model"]
    u_patch_size = understanding_pkg["patch_size"]
    u_forward = understanding_pkg["forward_fn"]
    u_dim = understanding_pkg["dim"]
    u_res = understanding_pkg["res"]
    feat_size = u_res // u_patch_size
    rho, tau = 0.05, opt.tau
    accum_steps = max(1, int(opt.grad_accum_steps))
    Pots_optimizer.zero_grad(set_to_none=True)
    net_optimizer.zero_grad(set_to_none=True)

    for iteration, batch_A in enumerate(training_data_loader):
        _, degraded_A, target_A = batch_A
        degraded_A = degraded_A.to(device, non_blocking=True)
        target_A = target_A.to(device, non_blocking=True)
        N = degraded_A.shape[0]
        if N != opt.batchSize:
            raise RuntimeError("In-batch multiplier requires full batches (drop_last=True).")
        try:
            batch_B = next(unpaired_iter)
        except StopIteration:
            unpaired_iter = iter(unpaired_loader)
            batch_B = next(unpaired_iter)
        _, degraded_B, target_B = batch_B
        degraded_B = degraded_B.to(device, non_blocking=True)
        target_B = target_B.to(device, non_blocking=True)

        perm_A = torch.randperm(N, device=device)
        degraded_A = degraded_A[perm_A]
        target_A = target_A[perm_A]

        # 保持原版：从 batch B 提取独立的 HQ 特征
        perm_B = torch.randperm(target_B.size(0), device=device)
        target_B = target_B[perm_B]

        with torch.no_grad():
            u_input_B = F.interpolate(
                target_B,
                size=(u_res, u_res),
                mode="bicubic",
                align_corners=False
            )
            u_tokens_B = u_forward(u_model, u_input_B)
            u_feat_B = u_tokens_B.permute(0, 2, 1).reshape(
                -1, u_dim, feat_size, feat_size
            )
            hq_ot = normalize_ot_feature(u_feat_B)

        # F-step. Original two-loader strategy and one Restormer no-grad forward.
        unfreeze(Pots)
        freeze(restoration_net)
        with torch.no_grad():
            _, feat_F = restoration_net(degraded_B)
            if feat_F.shape[-2:] != (feat_size, feat_size):
                feat_F = F.interpolate(feat_F, size=(feat_size, feat_size), mode="bilinear", align_corners=False)
            sources_F = inbatch_source_tuple(normalize_ot_feature(feat_F))
            cost_F = quadratic_multimarginal_cost(sources_F, hq_ot)
        Pots_optimizer.zero_grad(set_to_none=True)
        f_source_F, f_hq_F = shared_potential_values(Pots, sources_F, hq_ot)
        r_F = sum(f_source_F) + f_hq_F - cost_F
        dual_F = sum(p.mean() for p in f_source_F) + semi_unbalanced_hq_term(f_hq_F, tau) / N
        lambda_F = inbatch_lambdas.detach()  # [N] rank-local slot multipliers
        f_loss = -dual_F + projected_al_penalty(r_F, lambda_F, rho).mean()
        f_loss.backward()
        torch.nn.utils.clip_grad_norm_(Pots.parameters(), max_norm=1.0)
        Pots_optimizer.step()
        epoch_f_loss += f_loss.item()

        # T-step: unchanged two batch-sized Restormer forwards, not 7 large passes.
        freeze(Pots)
        unfreeze(restoration_net)
        net_optimizer.zero_grad(set_to_none=True)
        restored_A, _ = restoration_net(degraded_A)
        l1_loss = (restored_A - target_A).abs().mean()
        epoch_l1_loss += l1_loss.item()

        _, feat_T = restoration_net(degraded_B)
        if feat_T.shape[-2:] != (feat_size, feat_size):
            feat_T = F.interpolate(feat_T, size=(feat_size, feat_size), mode="bilinear", align_corners=False)
        sources_T = inbatch_source_tuple(normalize_ot_feature(feat_T))
        cost_T = quadratic_multimarginal_cost(sources_T, hq_ot)
        f_source_T, f_hq_T = shared_potential_values(Pots, sources_T, hq_ot)
        residual_T = sum(f_source_T) + f_hq_T - cost_T  # [N], no batch sum
        lambda_T = inbatch_lambdas.detach()  # same pre-update multipliers as F-step
        penalty_T = projected_al_penalty(residual_T, lambda_T, rho).mean()
        align_T = sum(p.mean() for p in f_source_T) - penalty_T
        net_loss = opt.Sigma * l1_loss + opt.Omega * align_T
        epoch_align_loss += align_T.item()
        epoch_t_loss += net_loss.item()
        epoch_vol += cost_T.detach().sum().item()
        (net_loss / accum_steps).backward()

       
        # Exact *in-batch* componentwise projected multiplier update.
        # This uses residuals evaluated before net_optimizer.step(), just
        # like the original trainer; no additional Restormer pass is needed.
        project_inbatch_multiplier_(inbatch_lambdas, residual_T, rho)

        count += 1
        if ((iteration + 1) % accum_steps == 0) or ((iteration + 1) == len(training_data_loader)):
            torch.nn.utils.clip_grad_norm_(restoration_net.parameters(), max_norm=1.0)
            net_optimizer.step()
            net_optimizer.zero_grad(set_to_none=True)

        if iteration % 10 == 0 and is_main_process():
            with torch.no_grad():
                violation = residual_T.detach().clamp_min(0)    
                violation_mean = violation.mean().item()
                violation_ratio = (residual_T.detach() > 0).float().mean().item()
                violation_max = violation.max().item()
            print("Ep {}({}/{}): L1: {:.4f}, Align: {:.4f}, T_Total: {:.4f}, F: {:.4f}, Vol: {:.4f}, "
                  "R+mean: {:.4f}, R+max: {:.4f}, lambda: {:.4f}".format(
                epoch, iteration, len(training_data_loader), l1_loss.item(), align_T.item(),
                net_loss.item(), f_loss.item(), cost_T.detach().sum().item(),
                violation_mean, violation_max, inbatch_lambdas.mean().item()))
            outdir = "./checksample/{}".format(opt.run_name)
            os.makedirs(outdir, exist_ok=True)
            save_image(restored_A.detach(), "{}/output_A.png".format(outdir))
            save_image(degraded_A.detach(), "{}/degraded_A.png".format(outdir))
            save_image(target_A.detach(), "{}/target_A.png".format(outdir))

    return (epoch_l1_loss / max(count, 1), epoch_f_loss / max(count, 1),
            epoch_align_loss / max(count, 1), epoch_t_loss / max(count, 1),
            epoch_vol / max(count, 1))


def evaluate(model, val_loader, opt):
    if not is_main_process():
        return 0.0

    model.eval()
    print(f"\n---------- Validating on {len(val_loader)} images -----------")

    total_psnr = 0.0
    count = 0

    with torch.no_grad():
        for _, img_in, img_gt in tqdm(val_loader, desc="Epoch Validation", ncols=100):
            if opt.cuda:
                img_in, img_gt = img_in.cuda(), img_gt.cuda()

            _, _, h, w = img_in.shape
            h_new, w_new = h - (h % 8), w - (w % 8)
            img_in, img_gt = img_in[:, :, :h_new, :w_new], img_gt[:, :, :h_new, :w_new]

            restored_img, _ = model(img_in)
            restored_img = restored_img.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
            target_img = img_gt.squeeze(0).cpu().numpy().transpose(1, 2, 0)

            total_psnr += psnr(restored_img, target_img, data_range=1)
            count += 1

    avg_psnr = total_psnr / (count if count > 0 else 1)
    print(f"\n---------- Validation Result: PSNR = {avg_psnr:.4f} dB ----------")
    model.train()
    return avg_psnr


if __name__ == "__main__":
    main()