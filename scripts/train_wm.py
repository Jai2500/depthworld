# from diffusers import StableVideoDiffusionPipeline
import os
import sys

# insert(0), not append: this env ships an unrelated top-level `scripts`
# package in site-packages that would otherwise shadow ours.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datetime
import glob
import json
import math
import signal
import subprocess

# Per-(job, rank) MIOpen cache dir, pre-warmed from a shared find-db.
# Only relevant on AMD/ROCm clusters (avoids MIOpen's find-db rename race
# across ranks); opt-in via CTRLWORLD_USE_MIOPEN_CACHE=1 and
# CTRLWORLD_MIOPEN_CACHE_DIR. CUDA runs should leave it off.
if os.environ.get("CTRLWORLD_USE_MIOPEN_CACHE", "0") == "1":
    import shutil
    _rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    _job = os.environ.get("SLURM_JOB_ID", "local")
    _miopen_base = os.environ.get("CTRLWORLD_MIOPEN_CACHE_DIR", "miopen_cache")
    _user_db = f"{_miopen_base}/user_db.{_job}.{_rank}"
    _cust = f"{_miopen_base}/custom_cache.{_job}.{_rank}"
    _warm = f"{_miopen_base}/warm"
    if os.path.isdir(f"{_warm}/user_db") and not os.path.exists(_user_db):
        shutil.copytree(f"{_warm}/user_db", _user_db)
    else:
        os.makedirs(_user_db, exist_ok=True)
    if os.path.isdir(f"{_warm}/custom_cache") and not os.path.exists(_cust):
        shutil.copytree(f"{_warm}/custom_cache", _cust)
    else:
        os.makedirs(_cust, exist_ok=True)
    os.environ["MIOPEN_USER_DB_PATH"] = _user_db
    os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = _cust

import einops
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import InitProcessGroupKwargs, set_seed
from tqdm.auto import tqdm

from config import wm_args
from models.ctrl_world import CrtlWorld
from models.ema import EMAModuleWrapper
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline


# Module-level emergency-save flag. SIGUSR1 from SLURM (--signal=B:USR1@600)
# sets it; the training loop checkpoints + scontrol-requeues + exits cleanly
# at the next sync_gradients boundary.
_emergency_save_requested = [False]


def _on_sigusr1(signum, frame):
    _emergency_save_requested[0] = True


signal.signal(signal.SIGUSR1, _on_sigusr1)


def _canonical_action_space(action_space: str | None) -> str:
    action_space = str(action_space or "cartesian").lower()
    aliases = {
        "ee": "cartesian",
        "eef": "cartesian",
        "end_effector": "cartesian",
        "cartesian_position": "cartesian",
        "joint": "joint_position",
        "joints": "joint_position",
        "joint_pos": "joint_position",
        "qpos": "joint_position",
    }
    return aliases.get(action_space, action_space)


def _apply_action_space_config(args) -> None:
    args.action_space = _canonical_action_space(getattr(args, "action_space", "cartesian"))
    if args.action_space == "joint_position":
        args.action_dim = 8
    elif args.action_space == "cartesian":
        args.action_dim = 7
    else:
        raise ValueError(
            f"Unknown action_space={args.action_space!r}. "
            "Expected 'cartesian' or 'joint_position'."
        )


def _adapt_state_dict_for_wrapped_modules(state_dict, model):
    """Map checkpoint keys into wrapper modules and drop shape mismatches."""
    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    if any(k.startswith("action_encoder.base.") for k in model_keys):
        state_dict = dict(state_dict)
        remap = {}
        for key, value in list(state_dict.items()):
            if key.startswith("action_encoder.") and not key.startswith("action_encoder.base."):
                new_key = "action_encoder.base." + key[len("action_encoder."):]
                if new_key in model_keys:
                    remap[key] = new_key
        for old_key, new_key in remap.items():
            state_dict[new_key] = state_dict.pop(old_key)
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        target = model_state.get(key)
        if target is not None and tuple(target.shape) != tuple(value.shape):
            skipped.append((key, tuple(value.shape), tuple(target.shape)))
            continue
        filtered[key] = value
    if skipped:
        print(
            f"[ckpt-load] skipped {len(skipped)} shape-mismatched keys; "
            f"first={skipped[:3]}",
            flush=True,
        )
    return filtered

def _build_optimizer(model, args):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=float(args.learning_rate))


def _needs_full_fsdp_state_dict(model, accelerator, args=None):
    unwrapped = accelerator.unwrap_model(model)
    return (
        getattr(unwrapped, "pointmap_decoder", None) is not None
        or getattr(args, "action_space", "cartesian") == "joint_position"
    )


class _FullStateDictForFsdpPlugin:
    """Temporarily flip `fsdp_plugin.state_dict_type` to FULL_STATE_DICT for
    the duration of save_state / load_state. Needed because accelerate's
    save_fsdp_model / load_fsdp_model dispatch on the plugin's setting and
    rebuild their own FSDP.state_dict_type context — an outer FSDP context
    manager gets overridden. SHARDED_STATE_DICT trips a
    `Only torch.contiguous_format memory_format is currently supported`
    ValueError under use_orig_params=True when a small/oddly-shaped camera
    param (gate `(1,)`, missing_pose_token, etc.) has its gathered view
    land non-contiguous; FULL gather to rank 0 sidesteps the broken path.
    """

    def __init__(self, accelerator):
        self.accelerator = accelerator
        self._restore = None

    def __enter__(self):
        from torch.distributed.fsdp import (
            FullOptimStateDictConfig,
            FullStateDictConfig,
            StateDictType,
        )
        plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
        if plugin is None:
            return self
        self._restore = (
            plugin.state_dict_type,
            plugin.state_dict_config,
            plugin.optim_state_dict_config,
        )
        plugin.state_dict_type = StateDictType.FULL_STATE_DICT
        plugin.state_dict_config = FullStateDictConfig(
            offload_to_cpu=True, rank0_only=True
        )
        plugin.optim_state_dict_config = FullOptimStateDictConfig(
            offload_to_cpu=True, rank0_only=True
        )
        return self

    def __exit__(self, *exc):
        plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
        if plugin is None or self._restore is None:
            return False
        (
            plugin.state_dict_type,
            plugin.state_dict_config,
            plugin.optim_state_dict_config,
        ) = self._restore
        return False


def main(args):
    logger = get_logger(__name__, log_level="INFO")

    # Optional model-only warm start from a merged .pt checkpoint. This is
    # intentionally separate from Accelerate resume: auto-resume from an
    # output checkpoint still wins later and restores optimizer/RNG/EMA.
    _warm_start_env = os.environ.get("CTRLWORLD_WARM_START_UNET_CKPT", "").strip()
    _warm_start_step_env = os.environ.get("CTRLWORLD_WARM_START_STEP", "").strip()
    args._warm_start_path = None
    args._warm_start_step = None
    if _warm_start_env and not getattr(args, "ckpt_path", None):
        if not os.path.isfile(_warm_start_env):
            print(
                f"[warm-start] WARNING: CTRLWORLD_WARM_START_UNET_CKPT="
                f"{_warm_start_env!r} does not exist; ignoring.",
                flush=True,
            )
        else:
            args._warm_start_path = _warm_start_env
            print(
                f"[warm-start] CTRLWORLD_WARM_START_UNET_CKPT={_warm_start_env!r} "
                f"will load with strict=False after model construction.",
                flush=True,
            )
            if _warm_start_step_env:
                try:
                    args._warm_start_step = int(_warm_start_step_env)
                    print(
                        f"[warm-start] CTRLWORLD_WARM_START_STEP="
                        f"{args._warm_start_step}",
                        flush=True,
                    )
                except ValueError:
                    print(
                        f"[warm-start] WARNING: CTRLWORLD_WARM_START_STEP="
                        f"{_warm_start_step_env!r} is not an int; ignoring.",
                        flush=True,
                    )

    if getattr(args, "use_swanlab", False):
        import swanlab  # optional; only required when --use_swanlab is set
        swanlab.sync_wandb()
    # Per-collective NCCL op timeout (separate from TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC).
    # Default is 600s but our step-2500 validation can spend ~600-660s on rank 0's
    # video gen alone (especially on the 640-wide Horiz variant whose MIOpen kernels
    # cold-compile for the full-res inference shapes). Non-main ranks block at
    # wait_for_everyone() during that window and trip the default. Bumping to 30 min
    # matches the heartbeat timeout we already raised. Job 18690077 crashed on this
    # exact timeout 2026-05-18; see project memory `validation-nccl-timeout`.
    _pg_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(minutes=30))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb",
        project_dir=args.output_dir,
        kwargs_handlers=[_pg_kwargs],
    )

    # Reproducibility: seed torch / numpy / python random / CUDA. Must come
    # AFTER Accelerator() because device_specific=True queries
    # AcceleratorState().process_index to offset the seed per rank.
    if hasattr(args, "seed") and args.seed is not None:
        set_seed(int(args.seed), device_specific=True)
        if accelerator.is_main_process:
            print(f"[seed] set_seed({int(args.seed)}, device_specific=True)")

    # model and optimizer
    # Per-node serialization: each rank's SVD pipeline load reads ~9 GB from
    # shared storage, and N concurrent reads can stall a networked
    # filesystem. We serialize WITHIN each node (one rank at a time per
    # node) but allow nodes to load in PARALLEL — independent page caches +
    # filesystem clients. Halves the model-build phase vs global N-way
    # serialization.
    _local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    _per_node = min(
        int(os.environ.get("SLURM_GPUS_ON_NODE", "8")),
        accelerator.num_processes,
    )
    for _local_idx in range(_per_node):
        if _local_rank == _local_idx:
            model = CrtlWorld(args)
        accelerator.wait_for_everyone()
    # Only rank 0 reads the checkpoint from disk. With FSDP and
    # sync_module_states=True, accelerator.prepare() broadcasts rank-0 params
    # to the other ranks, so we avoid 8× memory pressure during init.
    # Under plain DDP the broadcast happens in DistributedDataParallel's
    # constructor, so this gating is also safe (and significantly cheaper).
    if args.ckpt_path is not None and accelerator.is_main_process:
        print(f"Loading checkpoint from {args.ckpt_path}!")
        state_dict = torch.load(
            args.ckpt_path, map_location="cpu", weights_only=False
        )
        state_dict = _adapt_state_dict_for_wrapped_modules(state_dict, model)
        _missing, _unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[ckpt-load] missing keys: {len(_missing)}; unexpected keys: {len(_unexpected)}")
        if _missing:
            print(f"[ckpt-load] first missing: {_missing[:5]}")
        if _unexpected:
            print(f"[ckpt-load] first unexpected: {_unexpected[:5]}")
        del state_dict
    elif getattr(args, "_warm_start_path", None) and accelerator.is_main_process:
        print(f"[warm-start] Loading from {args._warm_start_path}", flush=True)
        state_dict = torch.load(
            args._warm_start_path, map_location="cpu", weights_only=False
        )
        state_dict = _adapt_state_dict_for_wrapped_modules(state_dict, model)
        _missing, _unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[warm-start] missing keys: {len(_missing)}; unexpected keys: {len(_unexpected)}", flush=True)
        if _missing:
            print(f"[warm-start] first missing: {_missing[:5]}", flush=True)
        if _unexpected:
            print(f"[warm-start] first unexpected: {_unexpected[:5]}", flush=True)
        del state_dict
    accelerator.wait_for_everyone()
    if os.environ.get("CTRLWORLD_FORCE_FSDP_FLOAT32", "0") == "1":
        model.to(dtype=torch.float32)
        if accelerator.is_main_process:
            print("[fsdp] cast model floating params/buffers to fp32 before FSDP flatten", flush=True)
    # Under FSDP, don't .to(device) before prepare(); FSDP places shards itself.
    if accelerator.distributed_type.value != "FSDP":
        model.to(accelerator.device)
    model.train()
    optimizer = _build_optimizer(model, args)

    # ---- Pre-detect resume dir + wandb_run_id for continuous wandb runs ----
    # Read metadata.json BEFORE init_trackers
    # so we can pass the saved wandb run id back to wandb.init via resume="allow".
    if getattr(args, "resume_from_dir", None) is None:
        _early_glob = os.path.join(args.output_dir, "checkpoint-*")
        _early_dirs = [d for d in glob.glob(_early_glob) if os.path.isdir(d)]

        def _step_of_early(p):
            stem = os.path.basename(p).replace("checkpoint-", "").split("-")[0]
            try:
                return int(stem)
            except ValueError:
                return -1

        _early_dirs = sorted(_early_dirs, key=_step_of_early)
        if _early_dirs and _step_of_early(_early_dirs[-1]) > 0:
            args.resume_from_dir = _early_dirs[-1]
    _resumed_wandb_run_id = None
    if getattr(args, "resume_from_dir", None) and os.path.isdir(args.resume_from_dir):
        _meta_path = os.path.join(args.resume_from_dir, "metadata.json")
        if os.path.isfile(_meta_path):
            with open(_meta_path) as _mf:
                _meta = json.load(_mf)
            _resumed_wandb_run_id = _meta.get("wandb_run_id")

    # logs
    if accelerator.is_main_process:
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        tag = args.tag
        run_name = f"train_{now}_{tag}"
        _init_wandb = {"name": run_name}
        if _resumed_wandb_run_id:
            _init_wandb["id"] = _resumed_wandb_run_id
            _init_wandb["resume"] = "allow"
            print(
                f"[wandb-resume] continuing run id={_resumed_wandb_run_id}",
                flush=True,
            )
        accelerator.init_trackers(
            args.wandb_project_name,
            config={},
            init_kwargs={"wandb": _init_wandb},
        )
        try:
            import wandb as _wandb
            _wandb_run_id_for_save = _wandb.run.id if _wandb.run is not None else None
        except Exception:
            _wandb_run_id_for_save = None
    else:
        _wandb_run_id_for_save = None
        os.makedirs(args.output_dir, exist_ok=True)
        # count parameters num in each part
        num_params = sum(p.numel() for p in model.unet.parameters())
        print(f"Number of parameters in the unet: {num_params / 1000000:.2f}M")
        num_params = sum(p.numel() for p in model.vae.parameters())
        print(f"Number of parameters in the vae: {num_params / 1000000:.2f}M")
        num_params = sum(p.numel() for p in model.image_encoder.parameters())
        print(f"Number of parameters in the image_encoder: {num_params / 1000000:.2f}M")
        num_params = sum(p.numel() for p in model.text_encoder.parameters())
        print(f"Number of parameters in the text_encoder: {num_params / 1000000:.2f}M")
        num_params = sum(p.numel() for p in model.action_encoder.parameters())
        print(
            f"Number of parameters in the action_encoder: {num_params / 1000000:.2f}M"
        )

    # train and val datasets
    # Choose dataset class. `rgb_only` (default) = the standard Dataset_mix
    # that returns RGB latents only. `rgb_depth_horiz` packs RGB + depth
    # latents horizontally per view into a single image-like tensor.
    # The horiz path doubles the spatial width — caller must pass
    # `--width 640` (or equivalent) so per-view shape assertions stay valid.
    _dataset_class = getattr(args, "dataset_class", "rgb_only")
    if _dataset_class == "rgb_depth_horiz":
        from dataset.dataset_droid import Dataset_mix_horiz as _DatasetCls
    else:
        from dataset.dataset_droid import Dataset_mix as _DatasetCls

    train_dataset = _DatasetCls(args, mode="train")
    # val_dataset = _DatasetCls(args, mode="val")
    val_dataset = _DatasetCls(args, mode="train")
    # DataLoader tuning for the real droid_raw_ctrl dataset.
    def _seed_worker(worker_id):
        import numpy as _np, random as _rd
        seed = torch.initial_seed() % 2**32
        _np.random.seed(seed)
        _rd.seed(seed)

    _dl_kwargs = dict(
        batch_size=args.train_batch_size, shuffle=args.shuffle,
        num_workers=args.num_workers, pin_memory=True,
    )
    if args.num_workers > 0:
        _dl_kwargs["persistent_workers"] = True
        _dl_kwargs["prefetch_factor"] = 8
        _dl_kwargs["worker_init_fn"] = _seed_worker

    train_dataloader = torch.utils.data.DataLoader(train_dataset, **_dl_kwargs)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, **_dl_kwargs)

    # Prepare everything with our accelerator
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    # ---- EMA setup (post-prepare). ----
    ema = None
    if getattr(args, "use_ema", False):
        _trainable = [p for p in model.parameters() if p.requires_grad]
        ema = EMAModuleWrapper(
            _trainable,
            decay=getattr(args, "ema_decay", 0.9999),
            device=accelerator.device,
        )
        if accelerator.is_main_process:
            print(
                f"[ema] tracking {sum(p.numel() for p in _trainable):,} trainable "
                f"params (decay={ema.decay})",
                flush=True,
            )

    # ---- Resume from checkpoint dir (explicit or auto-detected) ------------
    # If --resume_from_dir is not given, scan the output dir for the latest
    # `checkpoint-*` and resume from it. Covers manual resubmit AND SLURM
    # --requeue (preemption / NODE_FAIL / time-limit emergency save).
    global_step = 0
    if getattr(args, "_warm_start_step", None) is not None:
        global_step = int(args._warm_start_step)
    resume_from = getattr(args, "resume_from_dir", None)
    if resume_from is None:
        _ckpt_glob = os.path.join(args.output_dir, "checkpoint-*")
        _ckpt_dirs = [
            d
            for d in glob.glob(_ckpt_glob)
            if os.path.isdir(d) and os.path.isfile(os.path.join(d, "metadata.json"))
        ]

        def _step_of(p):
            stem = os.path.basename(p).replace("checkpoint-", "").split("-")[0]
            try:
                return int(stem)
            except ValueError:
                return -1

        _ckpt_dirs = sorted(_ckpt_dirs, key=_step_of)
        if _ckpt_dirs and _step_of(_ckpt_dirs[-1]) > 0:
            resume_from = _ckpt_dirs[-1]
            if accelerator.is_main_process:
                print(
                    f"[auto-resume] no --resume_from_dir given; found existing "
                    f"checkpoint {resume_from} — will resume",
                    flush=True,
                )
    if resume_from is not None and os.path.isdir(resume_from):
        # When camera/pointmap modules are attached, load via FULL_STATE_DICT for the same
        # reason as save below (SHARDED's contiguous-format check trips
        # on small camera params). The on-disk format must match — saves
        # made under this branch wrote a single .bin, not DCP shards.
        if _needs_full_fsdp_state_dict(model, accelerator, args):
            with _FullStateDictForFsdpPlugin(accelerator):
                accelerator.load_state(resume_from)
        else:
            accelerator.load_state(resume_from)
        meta_path = os.path.join(resume_from, "metadata.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as _mf:
                _meta = json.load(_mf)
            global_step = int(_meta.get("global_step", 0))
        if ema is not None:
            _ema_rank_path = os.path.join(
                resume_from, f"ema_rank{accelerator.process_index}.pt"
            )
            if os.path.isfile(_ema_rank_path):
                ema.load_state_dict(
                    torch.load(_ema_rank_path, map_location=accelerator.device)
                )
        if accelerator.is_main_process:
            print(f"[resume] loaded state from {resume_from} at step {global_step}", flush=True)

    ############################ training ##############################
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(
        args.max_train_steps
        * args.gradient_accumulation_steps
        * total_batch_size
        / len(train_dataloader)
    )
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  checkpointing_steps = {args.checkpointing_steps}")
    logger.info(f"  validation_steps = {args.validation_steps}")
    # `global_step` was either initialized to 0 above or restored from
    # `--resume_from_dir`. Don't reset here.
    forward_step = 0
    train_loss = 0.0
    train_loss_extras = {}
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    loss_gen, _ = model(batch)
                avg_loss = accelerator.gather(
                    loss_gen.repeat(args.train_batch_size)
                ).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                unwrapped_model = accelerator.unwrap_model(model)
                aux_log = getattr(unwrapped_model, "_last_aux_log", {})
                for _k, _v in aux_log.items():
                    if not isinstance(_v, torch.Tensor):
                        continue
                    _gv = accelerator.gather(
                        _v.detach().reshape(1).repeat(args.train_batch_size)
                    ).mean()
                    train_loss_extras[_k] = (
                        train_loss_extras.get(_k, 0.0)
                        + _gv.item() / args.gradient_accumulation_steps
                    )
                accelerator.backward(loss_gen)
                params_to_clip = model.parameters()
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                if ema is not None and accelerator.sync_gradients:
                    ema.step(
                        [p for p in model.parameters() if p.requires_grad],
                        global_step,
                    )
                forward_step += 1

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                # Emergency save (SIGUSR1 from SLURM — fires ~600s before
                # walltime). Save a checkpoint, request requeue, exit cleanly.
                if _emergency_save_requested[0]:
                    _emergency_save_requested[0] = False
                    emer_dir = os.path.join(
                        args.output_dir,
                        f"checkpoint-{global_step}-emergency",
                    )
                    if accelerator.is_main_process:
                        print(
                            f"[emergency] SIGUSR1 received — saving {emer_dir}",
                            flush=True,
                        )
                    if _needs_full_fsdp_state_dict(model, accelerator, args):
                        with _FullStateDictForFsdpPlugin(accelerator):
                            accelerator.save_state(emer_dir)
                    else:
                        accelerator.save_state(emer_dir)
                    if accelerator.is_main_process:
                        with open(os.path.join(emer_dir, "metadata.json"), "w") as _mf:
                            json.dump(
                                {
                                    "global_step": global_step,
                                    "tag": args.tag,
                                    "wandb_project_name": args.wandb_project_name,
                                    "wandb_run_name": getattr(args, "wandb_run_name", args.tag),
                                    "wandb_run_id": _wandb_run_id_for_save,
                                    "train_batch_size": args.train_batch_size,
                                    "mixed_precision": args.mixed_precision,
                                    "emergency": True,
                                },
                                _mf,
                                indent=2,
                            )
                    if ema is not None:
                        torch.save(
                            ema.state_dict(),
                            os.path.join(
                                emer_dir,
                                f"ema_rank{accelerator.process_index}.pt",
                            ),
                        )
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        print(
                            "[emergency] checkpoint flushed; idling until SLURM "
                            "kills us at walltime (--requeue then resumes from "
                            "this checkpoint)",
                            flush=True,
                        )
                    # Idle until SLURM sends SIGTERM at walltime. With #SBATCH
                    # --requeue, the walltime kill triggers an auto-requeue
                    # under the same job id, and the next firing's auto-resume
                    # picks up `emer_dir`. scontrol from inside the container
                    # is unreliable (not bind-mounted), so this is the safer
                    # path.
                    import time as _idle_t
                    while True:
                        _idle_t.sleep(60)
                # log loss every 100 steps
                if global_step % 100 == 0:
                    progress_bar.set_postfix({"loss": train_loss})
                    log_payload = {"train_loss": train_loss / 100}
                    for _k, _v in train_loss_extras.items():
                        log_payload[_k] = _v / 100
                    accelerator.log(log_payload, step=global_step)
                    train_loss = 0.0
                    train_loss_extras = {}
                # Full-state checkpoint: model + optimizer + scheduler + RNG +
                # grad scaler (via accelerate) plus per-rank EMA shards and a
                # small JSON of training metadata. Resume via
                # `--resume_from_dir <ckpt_dir>` on the next submission.
                if global_step % args.checkpointing_steps == 0 and global_step > 0:
                    ckpt_dir = os.path.join(
                        args.output_dir, f"checkpoint-{global_step}"
                    )
                    # When camera/pointmap modules are attached the SHARDED_STATE_DICT save
                    # path trips a "Only torch.contiguous_format memory_format
                    # is currently supported" ValueError on some small/oddly
                    # shaped camera params (gate, missing_pose_token, etc.)
                    # under use_orig_params=True. The post-hook also leaves
                    # FSDP's _unshard_params_ctx orphaned on failure, so a
                    # try/except retry asserts inside the next pre-hook.
                    # Workaround: pre-emptively switch to FULL_STATE_DICT for
                    # the save when camera/pointmap is on. Rank-0 CPU gather is a bit
                    # slower but reliable, and the resulting on-disk format
                    # is a single .pt rather than DCP shards (resume from this
                    # ckpt needs the FULL load path, not DCP merge).
                    if _needs_full_fsdp_state_dict(model, accelerator, args):
                        with _FullStateDictForFsdpPlugin(accelerator):
                            accelerator.save_state(ckpt_dir)
                    else:
                        accelerator.save_state(ckpt_dir)
                    if accelerator.is_main_process:
                        with open(os.path.join(ckpt_dir, "metadata.json"), "w") as _mf:
                            json.dump(
                                {
                                    "global_step": global_step,
                                    "tag": args.tag,
                                    "wandb_project_name": args.wandb_project_name,
                                    "wandb_run_name": getattr(args, "wandb_run_name", args.tag),
                                    "wandb_run_id": _wandb_run_id_for_save,
                                    "train_batch_size": args.train_batch_size,
                                    "mixed_precision": args.mixed_precision,
                                },
                                _mf,
                                indent=2,
                            )
                    if ema is not None:
                        torch.save(
                            ema.state_dict(),
                            os.path.join(
                                ckpt_dir,
                                f"ema_rank{accelerator.process_index}.pt",
                            ),
                        )
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        logger.info(f"Saved checkpoint to {ckpt_dir}")
                # generate video every validation_steps. First fire at
                # global_step == validation_steps (NOT at step 5, which would
                # hit a 10+ min validation block at near-init weights and
                # exceed NCCL's 600 s watchdog while other ranks barrier-wait).
                #
                # ALL ranks must enter the gate (not just main) because under
                # FSDP we need to summon_full_params collectively and then
                # wait_for_everyone before resuming training. If only rank 0
                # entered validation, the other ranks would keep stepping and
                # rank 0 would silently desync (TCPStore broken-pipe).
                if global_step > 0 and global_step % args.validation_steps == 0:
                    _run_rgb_validation(
                        model, val_dataset, args, global_step, accelerator, ema=ema,
                    )

                if global_step >= args.max_train_steps:
                    return


def main_val(args):
    accelerator = Accelerator()
    model = CrtlWorld(args)
    # load form val_model_path
    print("load from val_model_path", args.val_model_path)
    model.load_state_dict(torch.load(args.val_model_path))
    model.to(accelerator.device)
    model.eval()
    validate_video_generation(
        model, None, args, 0, "output", 0, accelerator, load_from_dataset=False
    )


def _run_rgb_validation(model, val_dataset, args, global_step, accelerator, ema=None):
    """Collective wrapper around single-modality video sampling.

    Under FSDP we summon full params (collective across ranks); rank 0 then
    runs the actual sampling + mp4/wandb upload. Non-main ranks wait inside
    the summon context so the post-block `wait_for_everyone()` barrier doesn't
    desync against rank 0's long-running video gen (root cause of the
    18679492 TCPStore broken-pipe crash).

    If `ema` is provided, swap EMA weights into the live params for sampling
    (standard diffusion-model practice — cleaner outputs than the latest
    optimizer step). Restore via the temp-buffer pattern after sampling.

    Wall time must stay well under the NCCL watchdog (600s default). The
    config.py default `video_num=2` keeps a step-2500 validation around
    ~400-500s in run 18678919 — same headroom applies here.
    """
    import contextlib
    import time as _t
    t0 = _t.time()
    if accelerator.is_main_process:
        print(
            f"[val] step={global_step} starting RGB validation "
            f"(video_num={args.video_num}, num_inference_steps={args.num_inference_steps}, "
            f"ema={'on' if ema is not None else 'off'})",
            flush=True,
        )
    # EMA swap on ALL ranks (EMA shards live per-rank); subsequent FSDP gather
    # then yields the full EMA params. Restore live shards via temp buffer.
    _trainable = (
        [p for p in model.parameters() if p.requires_grad] if ema is not None else None
    )
    if ema is not None:
        ema.copy_ema_to(_trainable, store_temp=True, grad=False)

    need_summon = accelerator.distributed_type.value == "FSDP"
    if need_summon:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        ctx = FSDP.summon_full_params(model, writeback=False, recurse=True)
    else:
        ctx = contextlib.nullcontext()
    try:
        with ctx:
            if accelerator.is_main_process:
                print(
                    f"[val] step={global_step} summoned full params (t={_t.time()-t0:.1f}s)",
                    flush=True,
                )
                inner = accelerator.unwrap_model(model)
                inner.eval()
                try:
                    with accelerator.autocast():
                        for vid_id in range(args.video_num):
                            tv = _t.time()
                            validate_video_generation(
                                inner, val_dataset, args, global_step,
                                args.output_dir, vid_id, accelerator,
                            )
                            print(
                                f"[val] step={global_step} video {vid_id + 1}/{args.video_num} "
                                f"done in {_t.time()-tv:.1f}s (total {_t.time()-t0:.1f}s)",
                                flush=True,
                            )
                finally:
                    inner.train()
            accelerator.wait_for_everyone()
    finally:
        if ema is not None:
            ema.copy_temp_to(_trainable)
    if accelerator.is_main_process:
        print(
            f"[val] step={global_step} validation done in {_t.time()-t0:.1f}s",
            flush=True,
        )


def validate_video_generation(
    model,
    val_dataset,
    args,
    train_steps,
    videos_dir,
    id,
    accelerator,
    load_from_dataset=True,
):
    device = accelerator.device
    # Callers pass the unwrapped CrtlWorld via `_run_rgb_validation` (under
    # FSDP we summon full params first, then unwrap via accelerator.unwrap_model).
    pipeline = model.pipeline
    videos_row = args.video_num if not args.debug else 1
    videos_col = 2

    # sample from val dataset
    batch_id = list(
        range(0, len(val_dataset), int(len(val_dataset) / videos_row / videos_col))
    )
    batch_id = batch_id[int(id * (videos_col)) : int((id + 1) * (videos_col))]
    batch_list = [val_dataset.__getitem__(id) for id in batch_id]
    video_gt = torch.cat(
        [t["latent"].unsqueeze(0) for i, t in enumerate(batch_list)], dim=0
    ).to(device, non_blocking=True)
    text = [t["text"] for i, t in enumerate(batch_list)]
    actions = torch.cat(
        [t["action"].unsqueeze(0) for i, t in enumerate(batch_list)], dim=0
    ).to(device, non_blocking=True)
    his_latent_gt, future_latent_ft = (
        video_gt[:, : args.num_history],
        video_gt[:, args.num_history :],
    )
    current_latent = future_latent_ft[:, 0]
    print("image", current_latent.shape, "action", actions.shape)
    # Derive num_views from the actual latent shape (3 for DROID, 1 for bridge).
    # args.height/args.width are per-view image dims; SVD VAE is 8x spatial.
    per_view_latent_h = args.height // 8
    per_view_latent_w = args.width // 8
    assert current_latent.shape[1] == 4
    assert current_latent.shape[2] % per_view_latent_h == 0, (
        f"latent height {current_latent.shape[2]} not divisible by per-view "
        f"height {per_view_latent_h} (args.height={args.height})"
    )
    assert current_latent.shape[3] == per_view_latent_w, (
        f"latent width {current_latent.shape[3]} != {per_view_latent_w} "
        f"(args.width={args.width})"
    )
    num_views = current_latent.shape[2] // per_view_latent_h
    stacked_image_height = current_latent.shape[2] * 8
    assert actions.shape[1:] == (
        int(args.num_frames + args.num_history),
        args.action_dim,
    )

    # start generate
    with torch.no_grad():
        bsz = actions.shape[0]
        action_latent = model.action_encoder(
            actions,
            text,
            model.tokenizer,
            model.text_encoder,
            args.frame_level_cond,
        )  # (8, 1, 1024)
        print("action_latent", action_latent.shape)

        _, pred_latents = CtrlWorldDiffusionPipeline.__call__(
            pipeline,
            image=current_latent,
            text=action_latent,
            width=args.width,
            height=stacked_image_height,
            num_frames=args.num_frames,
            history=his_latent_gt,
            num_inference_steps=args.num_inference_steps,
            decode_chunk_size=args.decode_chunk_size,
            max_guidance_scale=args.guidance_scale,
            fps=args.fps,
            motion_bucket_id=args.motion_bucket_id,
            mask=None,
            output_type="latent",
            return_dict=False,
            frame_level_cond=args.frame_level_cond,
            his_cond_zero=args.his_cond_zero,
        )

    pred_latents = einops.rearrange(
        pred_latents, "b f c (m h) (n w) -> (b m n) f c h w", m=num_views, n=1
    )  # (B*num_views, T, 4, per_view_latent_h, per_view_latent_w)
    video_gt = torch.cat([his_latent_gt, future_latent_ft], dim=1)
    video_gt = einops.rearrange(
        video_gt, "b f c (m h) (n w) -> (b m n) f c h w", m=num_views, n=1
    )

    # decode latent
    if video_gt.shape[2] != 3:
        decoded_video = []
        bsz, frame_num = video_gt.shape[:2]
        video_gt = video_gt.flatten(0, 1)
        decode_kwargs = {}
        for i in range(0, video_gt.shape[0], args.decode_chunk_size):
            chunk = (
                video_gt[i : i + args.decode_chunk_size]
                / pipeline.vae.config.scaling_factor
            )
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        video_gt = torch.cat(decoded_video, dim=0)
        video_gt = video_gt.reshape(bsz, frame_num, *video_gt.shape[1:])

        decoded_video = []
        bsz, frame_num = pred_latents.shape[:2]
        pred_latents = pred_latents.flatten(0, 1)
        decode_kwargs = {}
        for i in range(0, pred_latents.shape[0], args.decode_chunk_size):
            chunk = (
                pred_latents[i : i + args.decode_chunk_size]
                / pipeline.vae.config.scaling_factor
            )
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        videos = torch.cat(decoded_video, dim=0)
        videos = videos.reshape(bsz, frame_num, *videos.shape[1:])

    video_gt = (video_gt / 2.0 + 0.5).clamp(0, 1) * 255
    video_gt = (
        video_gt.to(pipeline.unet.dtype)
        .detach()
        .cpu()
        .numpy()
        .transpose(0, 1, 3, 4, 2)
        .astype(np.uint8)
    )
    videos = (videos / 2.0 + 0.5).clamp(0, 1) * 255
    videos = (
        videos.to(pipeline.unet.dtype)
        .detach()
        .cpu()
        .numpy()
        .transpose(0, 1, 3, 4, 2)
        .astype(np.uint8)
    )  # (2,16,256,256,3)
    videos = np.concatenate(
        [video_gt[:, : args.num_history], videos], axis=1
    )  # (2,16,512,256,3)
    videos = np.concatenate([video_gt, videos], axis=-3)  # (2,16,512,256,3)
    videos = np.concatenate([video for video in videos], axis=-2).astype(
        np.uint8
    )  # (16,512,256*batch,3)

    os.makedirs(f"{videos_dir}/samples", exist_ok=True)
    filename = f"{videos_dir}/samples/train_steps_{train_steps}_{id}.mp4"
    try:
        import mediapy
        mediapy.write_video(filename, videos, fps=2)
    except ImportError:
        # mediapy not in container — skip the disk artifact, video panel still
        # exists as a numpy array in memory for callers that want it.
        print(
            f"[train_wm] mediapy not available; skipped writing {filename}",
            flush=True,
        )

    # Upload to wandb under val/video_rgb_<id>. wandb.Video can take either a
    # path to an existing mp4 or a numpy array; we use the path when mediapy
    # wrote one, else the array.
    try:
        import wandb
        if os.path.isfile(filename):
            wandb_video = wandb.Video(filename, fps=2, format="mp4")
        else:
            wandb_video = wandb.Video(videos, fps=2, format="mp4")
        accelerator.log(
            {f"val/video_rgb_{id}": wandb_video},
            step=train_steps,
        )
    except Exception as _e:
        print(f"[train_wm] wandb video upload failed for vid {id}: {_e}", flush=True)
    return


if __name__ == "__main__":
    # reset parameters with command line
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--svd_model_path", type=str, default=None)
    parser.add_argument("--clip_model_path", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument(
        "--no_ckpt",
        action="store_true",
        help="Start from raw SVD UNet (overrides config.py ckpt_path default).",
    )
    parser.add_argument("--dataset_root_path", type=str, default=None)
    parser.add_argument("--dataset_meta_info_path", type=str, default=None)
    parser.add_argument(
        "--dataset_class",
        type=str,
        default="rgb_only",
        choices=["rgb_only", "rgb_depth_horiz"],
        help=(
            "Dataset variant. `rgb_only` (default) = Dataset_mix returning RGB "
            "latents only. `rgb_depth_horiz` = Dataset_mix_horiz that packs "
            "RGB + depth latents side-by-side per view (depth ablation: "
            "single UNet over a 2× wider spatial input). When using "
            "rgb_depth_horiz, also pass --width 640 (was 320 for RGB-only)."
        ),
    )
    parser.add_argument("--dataset_names", type=str, default=None)
    parser.add_argument(
        "--action_space",
        type=str,
        default=None,
        choices=["cartesian", "cartesian_position", "ee", "eef", "end_effector", "joint_position", "joint", "joints", "joint_pos", "qpos"],
        help=(
            "Action conditioning source. Default cartesian = normalized "
            "6D EE pose + gripper. joint_position = normalized Franka "
            "7D joint positions + gripper using robot joint limits."
        ),
    )
    parser.add_argument(
        "--dataset_cfgs",
        type=str,
        default=None,
        help=(
            "Meta-config subdir(s) under --dataset_meta_info_path. Defaults "
            "to --dataset_names. Set explicitly when the data tree is mounted "
            "at filesystem root (dataset_names='') but the {train,val}_sample.json "
            "and stat.json live under a different subdir name."
        ),
    )
    # bridge-correctness knobs (default in config.py is DROID-shaped)
    parser.add_argument("--down_sample", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help=(
            "Per-view image width in pixels (latent W = width / 8). Default "
            "from config.py is 320 (RGB-only). Set to 640 for the "
            "rgb_depth_horiz ablation (RGB+depth packed side-by-side)."
        ),
    )
    # smoke-control knobs
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--validation_steps", type=int, default=None)
    parser.add_argument("--checkpointing_steps", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Accelerator mixed_precision (overrides config.py default fp16).",
    )
    # output naming (config.py freezes output_dir = f"model_ckpt/{tag}" at class
    # definition time, so we recompute output_dir/wandb_run_name when --tag is set)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default=None,
        help="Override the wandb project name (config.py default = droid_example).",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help=(
            "Track an exponential moving average of trainable params (standard "
            "diffusion-model practice). Validation sampling uses EMA weights."
        ),
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.9999,
        help="EMA decay factor (SVD/SD recipe is 0.9999).",
    )
    parser.add_argument(
        "--resume_from_dir",
        type=str,
        default=None,
        help=(
            "Resume training from a previous checkpoint directory produced by "
            "`accelerator.save_state`. Restores model, optimizer, RNG, grad "
            "scaler, EMA shards, and global_step. World size must match."
        ),
    )
    parser.add_argument(
        "--use_swanlab",
        action="store_true",
        help="Route wandb logging through swanlab (requires swanlab API key).",
    )
    args_new = parser.parse_args()
    # If --dataset_cfgs wasn't explicitly given, mirror --dataset_names (the
    # legacy convention where one name covers both the data tree subdir and
    # the meta subdir).
    if args_new.dataset_cfgs is None:
        args_new.dataset_cfgs = args_new.dataset_names
    args = wm_args()

    def merge_args(args, new_args):
        for k, v in new_args.__dict__.items():
            if v is not None and v is not False:
                args.__dict__[k] = v
        return args

    args = merge_args(args, args_new)
    _apply_action_space_config(args)
    if args_new.svd_model_path is not None:
        args.pretrained_model_path = args_new.svd_model_path
    if args_new.clip_model_path is not None:
        args.clip_model_path = args_new.clip_model_path
    args.use_swanlab = args_new.use_swanlab
    if args_new.no_ckpt:
        args.ckpt_path = None
    if args_new.tag is not None:
        args.output_dir = f"model_ckpt/{args.tag}"
        args.wandb_run_name = args.tag

    main(args)
