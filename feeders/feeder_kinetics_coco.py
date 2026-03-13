import pickle
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from feeders import tools

# COCO-17 keypoint order (V=17):
# 0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
# 5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
# 9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
# 13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle
COCO17_PARTS = {
    "head": [0, 1, 2, 3, 4],
    "l_arm": [5, 7, 9],
    "r_arm": [6, 8, 10],
    "torso": [5, 6, 11, 12],
    "l_leg": [11, 13, 15],
    "r_leg": [12, 14, 16],
}


def drop_part(
    data_numpy: np.ndarray, parts=("l_arm", "r_arm", "l_leg", "r_leg"), p=0.5
) -> np.ndarray:
    """Randomly zero-out one body part across all frames (simple part-drop augmentation).
    data_numpy: (C,T,V,M)
    """
    if np.random.rand() >= p:
        return data_numpy
    part = parts[np.random.randint(0, len(parts))]
    joints = COCO17_PARTS[part]
    data_numpy[:, :, joints, :] = 0
    return data_numpy


class Feeder(Dataset):
    """Kinetics skeleton feeder (COCO-17 layout).

    Output per sample: (C, T, V, M) with C=3 (x, y, score), V=17, M=2.

    Supported input formats:
      1. Single .npz file with x_train/y_train and x_test/y_test arrays
      2. Directory with multiple .npz shards (kinetics_coco17_*.npz)
      3. MMAction2/PYSKL HRNet annotations (.pkl) plus kpfiles/*.pkl keypoint blobs
    """

    def __init__(
        self,
        data_path,
        label_path=None,
        p_interval=1,
        split="train",
        aug_method="z",
        intra_p=0.5,
        inter_p=0.0,
        window_size=-1,
        debug=False,
        thres=64,
        uniform=False,
        partition=False,
        num_people=2,
        num_points=17,
        max_shards=-1,
        lazy_load=False,
        keypoint_path=None,
        num_person_in=5,
    ):
        self.debug = debug
        self.data_path = data_path
        self.split = split
        self.aug_method = aug_method
        self.intra_p = intra_p
        self.inter_p = inter_p
        self.window_size = window_size
        self.p_interval = p_interval
        self.thres = thres
        self.uniform = uniform
        self.partition = partition
        self.num_people = num_people
        self.num_points = num_points
        self.max_shards = max_shards
        self.lazy_load = lazy_load
        self.max_cache_shards = 8
        self.num_person_in = max(num_person_in, num_people)
        self.keypoint_path = Path(keypoint_path or label_path) if (keypoint_path or label_path) else None

        self.data_format = None
        self.data = None
        self.annotations = []
        self.label = np.array([], dtype=np.int64)
        self.sample_name = []

        self.shard_files = []
        self.shard_indices = []
        self.shard_cache = {}
        self.shard_access_order = []
        self.x_key = None
        self.y_key = None

        self.load_data()

    def _to_nctvm(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 3:
            N, T, D = x.shape
            expected = self.num_people * self.num_points * 3
            if expected != D:
                raise ValueError(f"Expected last dim {expected}, got {D}")
            x = x.reshape(N, T, self.num_people, self.num_points, 3)
            return x.transpose(0, 4, 1, 3, 2)

        if x.ndim == 5:
            if x.shape[-1] != 3:
                raise ValueError("Expected last dim C=3 (x,y,score)")
            if x.shape[2] == self.num_people and x.shape[3] == self.num_points:
                return x.transpose(0, 4, 1, 3, 2)
            if x.shape[2] == self.num_points and x.shape[3] == self.num_people:
                return x.transpose(0, 4, 1, 2, 3)
            raise ValueError(f"Unrecognized 5D layout: {x.shape}")

        raise ValueError(f"Unsupported input ndim: {x.ndim}")

    def _build_lazy_index(self, data_dir: Path, x_key: str, y_key: str) -> np.ndarray:
        print(f"[LAZY] Building index from {data_dir}...")
        npz_files = sorted(data_dir.glob("kinetics_coco17_*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No kinetics_coco17_*.npz files found in {data_dir}")

        if self.max_shards > 0:
            npz_files = npz_files[: self.max_shards]

        self.shard_files = npz_files
        all_labels = []

        for shard_idx, npz_path in enumerate(npz_files):
            print(f"[LAZY] Reading shard {shard_idx + 1}/{len(npz_files)}: {npz_path.name}")
            npz = np.load(npz_path, allow_pickle=True)
            y_shard = npz[y_key]
            labels = np.where(y_shard > 0)[1]
            all_labels.append(labels)
            for local_idx in range(len(labels)):
                self.shard_indices.append((shard_idx, local_idx))

        print(f"[LAZY] Index built: {len(self.shard_indices)} samples from {len(npz_files)} shards")
        return np.concatenate(all_labels, axis=0)

    def _load_sharded_data(self, data_dir: Path, x_key: str, y_key: str) -> tuple[np.ndarray, np.ndarray]:
        npz_files = sorted(data_dir.glob("kinetics_coco17_*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No kinetics_coco17_*.npz files found in {data_dir}")

        if self.max_shards > 0:
            npz_files = npz_files[: self.max_shards]

        all_x, all_y = [], []
        for npz_path in npz_files:
            npz = np.load(npz_path, allow_pickle=True)
            all_x.append(npz[x_key])
            all_y.append(npz[y_key])

        x = np.concatenate(all_x, axis=0)
        y = np.concatenate(all_y, axis=0)
        return x, y

    def _infer_keypoint_root(self, annotation_path: Path) -> Path:
        candidates = [
            annotation_path.parent / "k400_kpfiles_2d" / "kpfiles",
            annotation_path.parent / "kpfiles",
            annotation_path.parent / "k400_kpfiles_2d",
        ]
        for candidate in candidates:
            if candidate.exists():
                if candidate.is_dir() and (candidate / "kpfiles").is_dir():
                    return candidate / "kpfiles"
                return candidate
        raise FileNotFoundError(
            "Could not infer keypoint_path from annotation file. "
            "Pass keypoint_path explicitly, e.g. data/k400_kpfiles_2d/kpfiles."
        )

    def _normalize_keypoint_root(self, annotation_path: Path) -> Path:
        keypoint_root = self.keypoint_path or self._infer_keypoint_root(annotation_path)
        if keypoint_root.is_dir() and (keypoint_root / "kpfiles").is_dir():
            return keypoint_root / "kpfiles"
        return keypoint_root

    def _load_mmaction_annotations(self, annotation_path: Path) -> list[dict]:
        with annotation_path.open("rb") as handle:
            ann_file = pickle.load(handle)

        if not isinstance(ann_file, dict) or "annotations" not in ann_file:
            raise ValueError(
                f"Expected MMAction2/PYSKL annotation dict with 'annotations' key, got {type(ann_file)}"
            )

        split_key = {"train": "train", "test": "val", "val": "val"}.get(self.split)
        if split_key is None:
            raise NotImplementedError("split only supports train/test/val for MMAction2 annotations")

        split_names = ann_file.get("split", {}).get(split_key)
        if split_names is None:
            raise KeyError(
                f"Split '{split_key}' not found in annotation file. Available: {list(ann_file.get('split', {}).keys())}"
            )

        split_name_set = set(split_names)
        annotations = [
            ann
            for ann in ann_file["annotations"]
            if ann.get("frame_dir") in split_name_set
        ]
        if not annotations:
            raise ValueError(f"No annotations found for split '{split_key}' in {annotation_path}")
        return annotations

    def _resolve_mmaction_kpfile(self, ann: dict) -> Path:
        candidates = []
        frame_dir = ann.get("frame_dir")
        if frame_dir:
            candidates.append(self.keypoint_path / f"{frame_dir}.pkl")

        raw_file = ann.get("raw_file")
        if raw_file:
            candidates.append(self.keypoint_path / Path(str(raw_file)).name)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"Could not find kpfile for sample '{frame_dir}'. Tried: {[str(path) for path in candidates]}"
        )

    def _select_person_candidates(
        self, keypoint: np.ndarray, frame_inds: np.ndarray, det_scores: np.ndarray, total_frames: int
    ) -> np.ndarray:
        frame_data = np.zeros((total_frames, self.num_person_in, self.num_points, 3), dtype=np.float32)

        for frame_idx in np.unique(frame_inds):
            detection_indices = np.where(frame_inds == frame_idx)[0]
            if detection_indices.size == 0:
                continue

            ranked = detection_indices[np.argsort(-det_scores[detection_indices])]
            ranked = ranked[: self.num_person_in]
            frame_data[frame_idx, : len(ranked)] = keypoint[ranked]

        return frame_data

    def _normalize_frame_data(self, frame_data: np.ndarray, ann: dict) -> np.ndarray:
        img_shape = ann.get("img_shape") or ann.get("original_shape")
        if not img_shape or len(img_shape) != 2:
            return frame_data

        height, width = img_shape
        if height <= 0 or width <= 0:
            return frame_data

        frame_data = frame_data.copy()
        frame_data[..., 0] = np.clip(frame_data[..., 0] / float(width), 0.0, 1.0) - 0.5
        frame_data[..., 1] = np.clip(frame_data[..., 1] / float(height), 0.0, 1.0) - 0.5

        zero_mask = frame_data[..., 2] <= 0
        frame_data[..., 0][zero_mask] = 0
        frame_data[..., 1][zero_mask] = 0
        return frame_data

    def _load_mmaction_sample(self, index: int) -> np.ndarray:
        ann = self.annotations[index]
        kpfile = self._resolve_mmaction_kpfile(ann)
        with kpfile.open("rb") as handle:
            kp_blob = pickle.load(handle)

        keypoint = np.asarray(kp_blob["keypoint"], dtype=np.float32)
        if keypoint.ndim != 3 or keypoint.shape[1] != self.num_points or keypoint.shape[2] != 3:
            raise ValueError(f"Unexpected keypoint shape {keypoint.shape} in {kpfile}")

        frame_inds = np.asarray(ann["frame_inds"], dtype=np.int64)
        if frame_inds.ndim != 1 or frame_inds.shape[0] != keypoint.shape[0]:
            raise ValueError(
                f"frame_inds mismatch for {kpfile}: {frame_inds.shape} vs {keypoint.shape[0]}"
            )

        total_frames = int(ann.get("total_frames") or 0)
        if frame_inds.size == 0:
            total_frames = max(total_frames, 1)
            return np.zeros((3, total_frames, self.num_points, self.num_people), dtype=np.float32)

        if frame_inds.min() >= 1:
            frame_inds = frame_inds - 1
        total_frames = max(total_frames, int(frame_inds.max()) + 1)

        box_score = ann.get("box_score")
        if box_score is not None:
            det_scores = np.asarray(box_score, dtype=np.float32)
            if det_scores.shape[0] != keypoint.shape[0]:
                det_scores = keypoint[..., 2].mean(axis=1)
        else:
            det_scores = keypoint[..., 2].mean(axis=1)

        frame_data = self._select_person_candidates(keypoint, frame_inds, det_scores, total_frames)
        frame_data = self._normalize_frame_data(frame_data, ann)

        person_scores = frame_data[..., 2].sum(axis=(0, 2))
        person_order = np.argsort(-person_scores)[: self.num_people]
        frame_data = frame_data[:, person_order]
        return frame_data.transpose(3, 0, 2, 1)

    def load_data(self):
        data_path = Path(self.data_path)

        if data_path.is_file() and data_path.suffix == ".npz":
            self.data_format = "npz"
            npz = np.load(data_path, allow_pickle=True)
            if self.split == "train":
                self.data = npz["x_train"]
                self.label = np.where(npz["y_train"] > 0)[1]
            elif self.split == "test":
                self.data = npz["x_test"]
                self.label = np.where(npz["y_test"] > 0)[1]
            else:
                raise NotImplementedError("split only supports train/test")
            self.data = self._to_nctvm(self.data)

        elif data_path.is_dir():
            self.data_format = "npz_shards"
            if self.split == "train":
                shard_dir = data_path / "train"
                self.x_key, self.y_key = "x_train", "y_train"
            elif self.split == "test":
                shard_dir = data_path / "val"
                self.x_key, self.y_key = "x_test", "y_test"
            else:
                raise NotImplementedError("split only supports train/test")

            if not shard_dir.exists():
                raise FileNotFoundError(f"Shard directory not found: {shard_dir}")

            if self.lazy_load:
                self.label = self._build_lazy_index(shard_dir, self.x_key, self.y_key)
                self.data = None
            else:
                x, y = self._load_sharded_data(shard_dir, self.x_key, self.y_key)
                self.data = self._to_nctvm(x)
                self.label = np.where(y > 0)[1]

        elif data_path.is_file() and data_path.suffix in {".pkl", ".pickle"}:
            self.data_format = "mmaction_pkl"
            self.keypoint_path = self._normalize_keypoint_root(data_path)
            self.annotations = self._load_mmaction_annotations(data_path)
            self.label = np.asarray([ann["label"] for ann in self.annotations], dtype=np.int64)
            self.data = None
        else:
            raise FileNotFoundError(
                f"data_path must be a .npz file, shard directory, or MMAction2/PYSKL .pkl file: {data_path}"
            )

        if self.debug:
            if self.data is not None:
                self.data = self.data[:200]
            if self.annotations:
                self.annotations = self.annotations[:200]
            self.label = self.label[:200]
            if self.lazy_load:
                self.shard_indices = self.shard_indices[:200]

        if self.data_format == "mmaction_pkl":
            self.sample_name = [ann.get("frame_dir", f"sample_{i}") for i, ann in enumerate(self.annotations)]
        elif self.split == "train":
            self.sample_name = ["train_" + str(i) for i in range(len(self.label))]
        elif self.split == "test":
            self.sample_name = ["test_" + str(i) for i in range(len(self.label))]
        else:
            self.sample_name = ["sample_" + str(i) for i in range(len(self.label))]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        if self.data_format == "mmaction_pkl":
            data_numpy = self._load_mmaction_sample(index)
        elif self.lazy_load and self.data is None:
            shard_idx, local_idx = self.shard_indices[index]

            if shard_idx not in self.shard_cache:
                npz = np.load(self.shard_files[shard_idx], allow_pickle=True)
                x_shard = self._to_nctvm(npz[self.x_key])
                self.shard_cache[shard_idx] = x_shard
                self.shard_access_order.append(shard_idx)

                while len(self.shard_cache) > self.max_cache_shards:
                    oldest_key = self.shard_access_order.pop(0)
                    if oldest_key in self.shard_cache:
                        del self.shard_cache[oldest_key]
            else:
                if shard_idx in self.shard_access_order:
                    self.shard_access_order.remove(shard_idx)
                    self.shard_access_order.append(shard_idx)

            data_numpy = np.array(self.shard_cache[shard_idx][local_idx])
        else:
            data_numpy = np.array(self.data[index])

        label = int(self.label[index])

        valid_mask = data_numpy.sum(0).sum(-1).sum(-1) != 0
        valid_indices = np.where(valid_mask)[0]
        valid_frame_num = int(valid_indices[-1] + 1) if valid_indices.size > 0 else data_numpy.shape[1]
        num_people_present = np.sum(data_numpy.sum(0).sum(0).sum(0) != 0)

        if self.uniform:
            data_numpy, index_t = tools.valid_crop_uniform(
                data_numpy, valid_frame_num, self.p_interval, self.window_size, self.thres
            )
        else:
            data_numpy, index_t = tools.valid_crop_resize(
                data_numpy, valid_frame_num, self.p_interval, self.window_size, self.thres
            )

        if self.split == "train":
            p = np.random.rand(1)
            if p < self.intra_p:
                if self.partition and ("p" in self.aug_method):
                    data_numpy = drop_part(data_numpy, p=0.5)

                if "a" in self.aug_method and np.random.rand(1) < 0.5:
                    data_numpy = data_numpy[:, :, :, np.array([1, 0])]
                if "b" in self.aug_method and num_people_present == 2 and np.random.rand(1) < 0.5:
                    axis_next = np.random.randint(0, 1)
                    temp = data_numpy.copy()
                    C, T, V, _ = data_numpy.shape
                    temp[:, :, :, axis_next] = np.zeros((C, T, V))
                    data_numpy = temp

                if "1" in self.aug_method:
                    data_numpy = tools.shear(data_numpy, p=0.5)
                if "2" in self.aug_method:
                    data_numpy = tools.rotate(data_numpy, p=0.5)
                if "3" in self.aug_method:
                    data_numpy = tools.scale(data_numpy, p=0.5)
                if "4" in self.aug_method:
                    data_numpy = tools.spatial_flip(data_numpy, p=0.5, num_points=self.num_points)
                if "5" in self.aug_method:
                    data_numpy, index_t = tools.temporal_flip(data_numpy, index_t, p=0.5)
                if "6" in self.aug_method:
                    data_numpy = tools.gaussian_noise(data_numpy, p=0.5)
                if "7" in self.aug_method:
                    data_numpy = tools.gaussian_filter(data_numpy, p=0.5)
                if "8" in self.aug_method:
                    data_numpy = tools.drop_axis(data_numpy, p=0.5)
                if "9" in self.aug_method:
                    data_numpy = tools.drop_joint(data_numpy, p=0.5)

        return data_numpy, index_t, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
