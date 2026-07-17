import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


# Franka Emika Panda joint position limits, radians. Source: Franka FCI
# "Limits for Panda" q_min/q_max table. These are robot limits, not
# dataset percentiles, so joint-position conditioning has stable scaling.
FRANKA_PANDA_JOINT_MIN = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
    dtype=np.float32,
)
FRANKA_PANDA_JOINT_MAX = np.array(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
    dtype=np.float32,
)


class Dataset_mix(Dataset):
    def __init__(
        self,
        args,
        mode="val",
        his_dropout=True,
        sparse_history=True,
    ):
        """Constructor."""
        super().__init__()
        self.args = args
        self.mode = mode
        self.his_dropout = his_dropout
        self.sparse_history = sparse_history

        # dataset stucture
        # dataset_root_path/dataset_name/annotation_name/mode/traj
        # dataset_root_path/dataset_name/video/mode/traj
        # dataset_root_path/dataset_name/latent_video/mode/traj

        # samples:{'ann_file':xxx, 'frame_idx':xxx, 'dataset_name':xxx}

        # prepare all datasets path
        self.dataset_path_all = []
        self.samples_all = []
        self.samples_len = []
        self.norm_all = []

        dataset_root_path = args.dataset_root_path
        dataset_names = args.dataset_names.split("+")
        dataset_meta_info_path = args.dataset_meta_info_path
        dataset_cfgs = args.dataset_cfgs.split("+")
        self.prob = args.prob
        for dataset_name, dataset_cfg in zip(dataset_names, dataset_cfgs):
            data_json_path = (
                f"{dataset_meta_info_path}/{dataset_cfg}/{mode}_sample.json"
            )

            with open(data_json_path, "r") as f:
                samples = json.load(f)
            dataset_path = [
                os.path.join(dataset_root_path, dataset_name) for sample in samples
            ]
            print(f"ALL dataset, {len(samples)} samples in total")
            self.dataset_path_all.append(dataset_path)
            self.samples_all.append(samples)
            self.samples_len.append(len(samples))

            # prepare normalization. When dataset_name is empty (sqfs-mounted
            # data tree at the filesystem root → dataset_path is "/"), the meta
            # subdir for stat.json comes from dataset_cfg instead.
            _stat_dir = dataset_name if dataset_name else dataset_cfg
            with open(f"{dataset_meta_info_path}/{_stat_dir}/stat.json", "r") as f:
                data_stat = json.load(f)
                state_p01 = np.array(data_stat["state_01"])[None, :]
                state_p99 = np.array(data_stat["state_99"])[None, :]
                self.norm_all.append((state_p01, state_p99))

        self.max_id = max(self.samples_len)
        print("samples_len:", self.samples_len, "max_id:", self.max_id)

        self.gripper_filter_mode = str(
            getattr(self.args, "gripper_filter_mode", "none")
        ).lower()
        self.gripper_delta_thresh = float(
            getattr(self.args, "gripper_delta_thresh", 0.01)
        )
        self.gripper_min_events = int(getattr(self.args, "gripper_min_events", 1))
        self.gripper_filter_train_only = bool(
            getattr(self.args, "gripper_filter_train_only", False)
        )
        self.gripper_filter_sample_prob = float(
            getattr(self.args, "gripper_filter_sample_prob", 1.0)
        )
        self._use_gripper_filter = self._should_enable_gripper_filter()
        self._valid_sample_indices_by_skip = None
        self._dataset_choices_by_skip = None
        self._samples_len_unfiltered = list(self.samples_len)
        self._max_id_unfiltered = int(self.max_id)
        if self._use_gripper_filter:
            self._build_gripper_filter_index()
            self._build_sampling_distribution_for_filter()
            if self.gripper_filter_sample_prob >= 1.0:
                self._update_lengths_after_filter()
            self._print_gripper_filter_summary()

    def __len__(self):
        return self.max_id

    def _should_enable_gripper_filter(self):
        valid_modes = {"none", "open", "close", "both"}
        if self.gripper_filter_mode not in valid_modes:
            raise ValueError(
                f"Unknown gripper_filter_mode={self.gripper_filter_mode}. "
                "Expected one of: none, open, close, both."
            )
        if self.gripper_filter_mode == "none":
            return False
        if self.gripper_filter_train_only and self.mode != "train":
            return False
        if not (0.0 <= self.gripper_filter_sample_prob <= 1.0):
            raise ValueError("gripper_filter_sample_prob must be in [0, 1].")
        if self.gripper_filter_sample_prob == 0.0:
            return False
        if self.gripper_delta_thresh < 0:
            raise ValueError("gripper_delta_thresh must be non-negative.")
        if self.gripper_min_events < 0:
            raise ValueError("gripper_min_events must be non-negative.")
        return True

    def _skip_candidates(self):
        return [1] if self.mode == "val" else [1, 2]

    def _count_gripper_events(self, gripper_values):
        deltas = np.diff(gripper_values)
        if self.gripper_filter_mode == "open":
            return int(np.sum(deltas > self.gripper_delta_thresh))
        if self.gripper_filter_mode == "close":
            return int(np.sum(deltas < -self.gripper_delta_thresh))
        return int(np.sum(np.abs(deltas) > self.gripper_delta_thresh))

    def _clip_has_gripper_event(self, sample, gripper_series, frame_len, skip):
        frame_now = int(sample["frame_ids"][0])
        frame_horizon = int(self.args.num_frames)
        rgb_id = np.arange(
            frame_now, frame_now + frame_horizon * skip, skip, dtype=np.int64
        )
        rgb_id = np.clip(rgb_id, 0, int(frame_len))
        state_id = rgb_id * int(self.args.down_sample)
        state_id = np.clip(state_id, 0, int(gripper_series.shape[0]) - 1)
        gripper_values = gripper_series[state_id]
        event_count = self._count_gripper_events(gripper_values)
        return event_count >= self.gripper_min_events

    def _build_gripper_filter_index(self):
        skip_candidates = self._skip_candidates()
        self._valid_sample_indices_by_skip = [
            {skip: [] for skip in skip_candidates} for _ in range(len(self.samples_all))
        ]

        for dataset_id, (samples, dataset_paths) in enumerate(
            zip(self.samples_all, self.dataset_path_all)
        ):
            episode_cache = {}
            for sample_idx, sample in enumerate(samples):
                episode_id = sample["episode_id"]
                if episode_id not in episode_cache:
                    dataset_dir = dataset_paths[sample_idx]
                    ann_file = (
                        f"{dataset_dir}/{self.args.annotation_name}/"
                        f"{self.mode}/{episode_id}.json"
                    )
                    with open(ann_file, "r") as f:
                        label = json.load(f)
                    gripper_series = np.asarray(
                        label["observation.state.gripper_position"], dtype=np.float32
                    ).reshape(-1)
                    joint_len = len(label["observation.state.joint_position"]) - 1
                    frame_len = int(np.floor(joint_len / int(self.args.down_sample)))
                    episode_cache[episode_id] = (gripper_series, frame_len)

                gripper_series, frame_len = episode_cache[episode_id]
                for skip in skip_candidates:
                    if self._clip_has_gripper_event(
                        sample, gripper_series, frame_len, skip
                    ):
                        self._valid_sample_indices_by_skip[dataset_id][skip].append(
                            sample_idx
                        )

    def _build_sampling_distribution_for_filter(self):
        base_prob = np.asarray(self.prob, dtype=np.float64)
        if base_prob.shape[0] != len(self.samples_all):
            raise ValueError(
                f"Probability length ({base_prob.shape[0]}) does not match number of datasets "
                f"({len(self.samples_all)})."
            )
        if np.any(base_prob < 0):
            raise ValueError("Dataset sampling probabilities must be non-negative.")

        self._dataset_choices_by_skip = {}
        for skip in self._skip_candidates():
            dataset_ids = [
                dataset_id
                for dataset_id in range(len(self.samples_all))
                if len(self._valid_sample_indices_by_skip[dataset_id][skip]) > 0
            ]
            if len(dataset_ids) == 0:
                raise ValueError(
                    f"Gripper filter removed all samples for mode={self.mode}, skip={skip}. "
                    "Relax gripper filtering thresholds or disable filtering."
                )
            weights = base_prob[dataset_ids]
            if float(weights.sum()) <= 0.0:
                raise ValueError(
                    f"Sum of sampling probabilities is zero for datasets available at skip={skip}."
                )
            weights = weights / weights.sum()
            self._dataset_choices_by_skip[skip] = (dataset_ids, weights)

    def _update_lengths_after_filter(self):
        filtered_lengths = []
        for dataset_id in range(len(self.samples_all)):
            skip_lengths = [
                len(self._valid_sample_indices_by_skip[dataset_id][skip])
                for skip in self._skip_candidates()
            ]
            filtered_lengths.append(max(skip_lengths))

        self.samples_len = filtered_lengths
        self.max_id = max(self.samples_len)

    def _print_gripper_filter_summary(self):
        print(
            "gripper_filter:",
            f"mode={self.gripper_filter_mode}",
            f"delta_thresh={self.gripper_delta_thresh}",
            f"min_events={self.gripper_min_events}",
            f"train_only={self.gripper_filter_train_only}",
            f"sample_prob={self.gripper_filter_sample_prob}",
        )
        for skip in self._skip_candidates():
            kept = [
                len(self._valid_sample_indices_by_skip[dataset_id][skip])
                for dataset_id in range(len(self.samples_all))
            ]
            print(f"gripper_filter skip={skip}: kept_per_dataset={kept}")
        print("samples_len:", self.samples_len, "max_id:", self.max_id)

    def _load_latent_video(self, video_path, frame_ids):
        with open(video_path, "rb") as file:
            video_tensor = torch.load(file)
            video_tensor.requires_grad = False
        max_frames = video_tensor.size()[0]
        frame_ids = [
            int(frame_id) if frame_id < max_frames else max_frames - 1
            for frame_id in frame_ids
        ]
        frame_data = video_tensor[frame_ids]
        return frame_data

    def _get_frames(
        self, label, frame_ids, cam_id, pre_encode, video_dir, use_img_cond=False
    ):
        # directly load videos latent after svd-vae encoder
        assert cam_id is not None
        assert pre_encode == True
        if pre_encode:
            video_path = label["latent_videos"][cam_id]["latent_video_path"]
            video_path = os.path.join(video_dir, video_path)
            try:
                frames = self._load_latent_video(video_path, frame_ids)
            except:
                video_path = video_path.replace("latent_videos", "latent_videos_svd")
                frames = self._load_latent_video(video_path, frame_ids)
        return frames

    def _get_obs(self, label, frame_ids, cam_id, pre_encode, video_dir):
        if cam_id is None:
            temp_cam_id = random.choice(self.cam_ids)
        else:
            temp_cam_id = cam_id
        frames = self._get_frames(
            label,
            frame_ids,
            cam_id=temp_cam_id,
            pre_encode=pre_encode,
            video_dir=video_dir,
        )
        return frames, temp_cam_id

    def _action_space(self) -> str:
        action_space = str(getattr(self.args, "action_space", "cartesian")).lower()
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

    def build_action(self, label, state_id, state_p01, state_p99):
        action_space = self._action_space()
        if action_space == "cartesian":
            cartesian_pose = np.array(label["observation.state.cartesian_position"])[
                state_id
            ]
            gripper_pose = np.array(label["observation.state.gripper_position"])[state_id]
            if gripper_pose.ndim == 1:
                gripper_pose = gripper_pose[:, np.newaxis]
            action = np.concatenate((cartesian_pose, gripper_pose), axis=-1)
            return self.normalize_bound(action, state_p01, state_p99)

        if action_space == "joint_position":
            joint_pose = np.array(
                label["observation.state.joint_position"], dtype=np.float32
            )[state_id]
            joint_pose = self.normalize_bound(
                joint_pose,
                FRANKA_PANDA_JOINT_MIN[None, :],
                FRANKA_PANDA_JOINT_MAX[None, :],
            )
            gripper_pose = np.array(
                label["observation.state.gripper_position"], dtype=np.float32
            )[state_id]
            if gripper_pose.ndim == 1:
                gripper_pose = gripper_pose[:, np.newaxis]
            gripper_pose = self.normalize_bound(
                gripper_pose,
                np.array([[0.0]], dtype=np.float32),
                np.array([[1.0]], dtype=np.float32),
            )
            return np.concatenate((joint_pose, gripper_pose), axis=-1)

        raise ValueError(
            f"Unknown action_space={action_space!r}. "
            "Expected 'cartesian' or 'joint_position'."
        )

    def normalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
        eps: float = 1e-8,
    ) -> np.ndarray:
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
        return np.clip(ndata, clip_min, clip_max)

    def denormalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
        eps=1e-8,
    ) -> np.ndarray:
        clip_range = clip_max - clip_min
        rdata = (data - clip_min) / clip_range * (data_max - data_min) + data_min
        return rdata

    def __getitem__(self, index):
        if self._use_gripper_filter:
            skip = random.randint(1, 2) if self.mode != "val" else 1
            use_filtered = random.random() < self.gripper_filter_sample_prob
            if use_filtered:
                dataset_ids, weights = self._dataset_choices_by_skip[skip]
                dataset_id = int(np.random.choice(dataset_ids, p=weights))
            else:
                dataset_id = np.random.choice(len(self.samples_all), p=self.prob)
        else:
            dataset_id = np.random.choice(len(self.samples_all), p=self.prob)
            skip = random.randint(1, 2) if self.mode != "val" else 1

        samples = self.samples_all[dataset_id]
        dataset_path = self.dataset_path_all[dataset_id]
        state_p01, state_p99 = self.norm_all[dataset_id]

        if self._use_gripper_filter:
            if use_filtered:
                valid_indices = self._valid_sample_indices_by_skip[dataset_id][skip]
                sample_idx = valid_indices[index % len(valid_indices)]
            else:
                sample_idx = index % len(samples)
        else:
            sample_idx = index % len(samples)

        sample = samples[sample_idx]
        dataset_dir = dataset_path[sample_idx]

        # get annotation
        frame_ids = sample["frame_ids"]
        ann_file = f"{dataset_dir}/{self.args.annotation_name}/{self.mode}/{sample['episode_id']}.json"
        with open(ann_file, "r") as f:
            label = json.load(f)

        # since we downsample the video from 15hz to 5 hz to save the storage space, the frame id is 1/3 of the state id
        joint_len = len(label["observation.state.joint_position"]) - 1
        frame_len = np.floor(joint_len / int(self.args.down_sample))
        skip_his = int(skip * 4) if self.sparse_history else skip
        p = random.random()
        if self.his_dropout and (p < 0.15):
            skip_his = 0

        # rgb_id and state_id
        frame_now = frame_ids[0]
        rgb_id = []
        for i in range(self.args.num_history, 0, -1):
            rgb_id.append(int(frame_now - i * skip_his))
        rgb_id.append(frame_now)
        for i in range(1, self.args.num_frames):
            rgb_id.append(int(frame_now + i * skip))
        rgb_id = np.array(rgb_id)
        rgb_id = np.clip(rgb_id, 0, frame_len).tolist()
        rgb_id = [int(frame_id) for frame_id in rgb_id]
        state_id = np.array(rgb_id) * self.args.down_sample

        # prepare data
        data = dict()

        # instructions
        data["text"] = label["texts"][0] if label.get("texts") else ""

        # stack tokens of multi-view; num_views inferred from the annotation.
        # DROID has 3 views @ 24 latent-height -> 72; bridge has 1 view @ 32 -> 32.
        num_views = len(label["latent_videos"])
        latnt_conds = []
        for cam_id in range(num_views):
            latnt, _ = self._get_obs(
                label, rgb_id, cam_id, pre_encode=True, video_dir=dataset_dir
            )
            latnt_conds.append(latnt)
        latent = torch.cat(latnt_conds, dim=2).float()
        data["latent"] = latent

        # prepare action cond data
        action = self.build_action(label, state_id, state_p01, state_p99)
        data["action"] = torch.tensor(action).float()

        return data


if __name__ == "__main__":
    from config import wm_args

    args = wm_args()
    train_dataset = Dataset_mix(args, mode="val")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True
    )
    for data in tqdm(train_loader, total=len(train_loader)):
        print(data["ann_file"])


# ===========================================================================
# Depth-aware variants.
#
# `Dataset_mix_dual`  — RGB + depth latents as separate batch fields.
# `Dataset_mix_horiz` — RGB|depth packed side-by-side along width per view,
#                       views stacked along height, as ONE `latent` field
#                       shaped (T, 4, H*num_views, 2*W). `args.width` must be
#                       2x the per-view RGB width (e.g. 640 for 320).
#
# Depth latents live at the same path as the RGB ones with `latent_videos`
# replaced by `latent_videos_depth` (override via the `depth_root` /
# `depth_folder` constructor kwargs when depth lives in a separate tree).
# ===========================================================================

_DEPTH_FOLDER = "latent_videos_depth"


class Dataset_mix_dual(Dataset_mix):
    """RGB + depth latents as separate `latent` / `latent_depth` fields."""

    def __init__(self, *args, depth_root=None, depth_folder=None, **kwargs):
        super().__init__(*args, **kwargs)
        # depth_root=None -> use the same dataset_dir as the RGB stream
        # (everything in one tree). depth_root="/some/mount" -> read depth
        # latents from that absolute root instead.
        self._depth_root = depth_root
        # depth_folder=None -> "latent_videos_depth" (default).
        self._depth_folder = depth_folder if depth_folder is not None else _DEPTH_FOLDER

    # ------------------------------------------------------------------
    # Shared clip sampling. Extracted verbatim from the previous per-class
    # __getitem__ replicas — the RNG draw order must stay identical so that
    # seeded sampling (e.g. eval's RNG-replay metadata wrapper) reproduces
    # the same clips.
    # ------------------------------------------------------------------
    def _sample_clip(self, index):
        if self._use_gripper_filter:
            skip = random.randint(1, 2) if self.mode != "val" else 1
            use_filtered = random.random() < self.gripper_filter_sample_prob
            if use_filtered:
                dataset_ids, weights = self._dataset_choices_by_skip[skip]
                dataset_id = int(np.random.choice(dataset_ids, p=weights))
            else:
                dataset_id = np.random.choice(len(self.samples_all), p=self.prob)
        else:
            dataset_id = np.random.choice(len(self.samples_all), p=self.prob)
            skip = random.randint(1, 2) if self.mode != "val" else 1

        samples = self.samples_all[dataset_id]
        dataset_path = self.dataset_path_all[dataset_id]
        state_p01, state_p99 = self.norm_all[dataset_id]

        if self._use_gripper_filter:
            if use_filtered:
                valid_indices = self._valid_sample_indices_by_skip[dataset_id][skip]
                sample_idx = valid_indices[index % len(valid_indices)]
            else:
                sample_idx = index % len(samples)
        else:
            sample_idx = index % len(samples)

        sample = samples[sample_idx]
        dataset_dir = dataset_path[sample_idx]

        frame_ids = sample["frame_ids"]
        ann_file = f"{dataset_dir}/{self.args.annotation_name}/{self.mode}/{sample['episode_id']}.json"
        with open(ann_file, "r") as f:
            label = json.load(f)

        joint_len = len(label["observation.state.joint_position"]) - 1
        frame_len = np.floor(joint_len / int(self.args.down_sample))
        skip_his = int(skip * 4) if self.sparse_history else skip
        p = random.random()
        if self.his_dropout and (p < 0.15):
            skip_his = 0

        frame_now = frame_ids[0]
        rgb_id = []
        for i in range(self.args.num_history, 0, -1):
            rgb_id.append(int(frame_now - i * skip_his))
        rgb_id.append(frame_now)
        for i in range(1, self.args.num_frames):
            rgb_id.append(int(frame_now + i * skip))
        rgb_id = np.array(rgb_id)
        rgb_id = np.clip(rgb_id, 0, frame_len).tolist()
        rgb_id = [int(frame_id) for frame_id in rgb_id]
        state_id = np.array(rgb_id) * self.args.down_sample

        episode_id = int(sample["episode_id"])
        return label, dataset_dir, rgb_id, state_id, state_p01, state_p99, episode_id

    def _load_depth_latent(self, label, cam_id, dataset_dir, rgb_id):
        """Load the depth latent matching one view's RGB latent clip."""
        depth_subpath = label["latent_videos"][cam_id]["latent_video_path"]
        depth_subpath = depth_subpath.replace("latent_videos", self._depth_folder)
        depth_root = self._depth_root if self._depth_root is not None else dataset_dir
        depth_path = os.path.join(depth_root, depth_subpath)
        return self._load_latent_video(depth_path, rgb_id)

    def _maybe_stash_sample(self, episode_id, rgb_id, num_views, dataset_dir):
        """Subclasses (e.g. the pointmap dataset) opt in to post-hoc
        inspection of the sampled clip by defining `_stash_sample`."""
        stash = getattr(self, "_stash_sample", None)
        if stash is not None:
            stash(episode_id, rgb_id, num_views, self.mode, dataset_dir)

    def __getitem__(self, index):
        (label, dataset_dir, rgb_id, state_id, state_p01, state_p99,
         episode_id) = self._sample_clip(index)

        data = dict()
        data["text"] = label["texts"][0] if label.get("texts") else ""

        num_views = len(label["latent_videos"])
        rgb_conds = []
        depth_conds = []
        for cam_id in range(num_views):
            rgb_latent, _ = self._get_obs(
                label, rgb_id, cam_id, pre_encode=True, video_dir=dataset_dir
            )
            rgb_conds.append(rgb_latent)
            depth_conds.append(
                self._load_depth_latent(label, cam_id, dataset_dir, rgb_id)
            )
        data["latent"] = torch.cat(rgb_conds, dim=2).float()
        data["latent_depth"] = torch.cat(depth_conds, dim=2).float()

        action = self.build_action(label, state_id, state_p01, state_p99)
        data["action"] = torch.tensor(action).float()

        self._maybe_stash_sample(episode_id, rgb_id, num_views, dataset_dir)
        return data


class Dataset_mix_horiz(Dataset_mix_dual):
    """RGB + depth latents concatenated horizontally per view (single-stream)."""

    def __getitem__(self, index):
        (label, dataset_dir, rgb_id, state_id, state_p01, state_p99,
         episode_id) = self._sample_clip(index)

        data = dict()
        data["text"] = label["texts"][0] if label.get("texts") else ""

        num_views = len(label["latent_videos"])
        per_view_combined = []
        for cam_id in range(num_views):
            rgb_latent, _ = self._get_obs(
                label, rgb_id, cam_id, pre_encode=True, video_dir=dataset_dir
            )
            depth_latent = self._load_depth_latent(label, cam_id, dataset_dir, rgb_id)
            # Horizontal concat per view: (T, C, H, W) + (T, C, H, W) -> (T, C, H, 2W).
            per_view_combined.append(
                torch.cat([rgb_latent, depth_latent], dim=3)
            )
        # Vertical stack of views along H: (T, C, H*num_views, 2W).
        data["latent"] = torch.cat(per_view_combined, dim=2).float()

        action = self.build_action(label, state_id, state_p01, state_p99)
        data["action"] = torch.tensor(action).float()

        self._maybe_stash_sample(episode_id, rgb_id, num_views, dataset_dir)
        return data
