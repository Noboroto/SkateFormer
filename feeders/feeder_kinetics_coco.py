import numpy as np
from torch.utils.data import Dataset
from feeders import tools

# COCO-17 keypoint order (V=17):
# 0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
# 5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
# 9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
# 13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle
COCO17_PARTS = {
    "head":      [0, 1, 2, 3, 4],
    "l_arm":     [5, 7, 9],
    "r_arm":     [6, 8, 10],
    "torso":     [5, 6, 11, 12],
    "l_leg":     [11, 13, 15],
    "r_leg":     [12, 14, 16],
}

def drop_part(data_numpy: np.ndarray, parts=("l_arm","r_arm","l_leg","r_leg"), p=0.5) -> np.ndarray:
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
    """Kinetics skeleton feeder (COCO-17 layout) for SkateFormer.

    Output per sample: (C, T, V, M) with C=3 (x, y, score), V=17, M=2.

    Recommended .npz structure:
      - x_train: (N, T, M, V, C) OR (N, T, V, M, C) OR (N, T, M*V*C)
      - y_train: one-hot (N, num_classes)
      - x_test, y_test similarly
    """

    def __init__(self, data_path, label_path=None, p_interval=1, split='train',
                 aug_method='z', intra_p=0.5, inter_p=0.0, window_size=-1,
                 debug=False, thres=64, uniform=False, partition=False,
                 num_people=2, num_points=17):
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
        self.partition = partition  # kept for API symmetry; not used by default for COCO
        self.num_people = num_people
        self.num_points = num_points
        self.load_data()

    def _to_nctvm(self, x: np.ndarray) -> np.ndarray:
        # Convert to (N, C, T, V, M)
        if x.ndim == 3:
            # (N, T, M*V*C)
            N, T, D = x.shape
            expected = self.num_people * self.num_points * 3
            if D != expected:
                raise ValueError(f"Expected last dim {expected}, got {D}")
            x = x.reshape(N, T, self.num_people, self.num_points, 3)
            return x.transpose(0, 4, 1, 3, 2)

        if x.ndim == 5:
            # (N,T,M,V,C) or (N,T,V,M,C)
            if x.shape[-1] != 3:
                raise ValueError("Expected last dim C=3 (x,y,score)")
            if x.shape[2] == self.num_people and x.shape[3] == self.num_points:
                return x.transpose(0, 4, 1, 3, 2)
            if x.shape[2] == self.num_points and x.shape[3] == self.num_people:
                return x.transpose(0, 4, 1, 2, 3)
            raise ValueError(f"Unrecognized 5D layout: {x.shape}")

        raise ValueError(f"Unsupported input ndim: {x.ndim}")

    def load_data(self):
        npz = np.load(self.data_path, allow_pickle=True)
        if self.split == 'train':
            self.data = npz['x_train']
            self.label = np.where(npz['y_train'] > 0)[1]
        elif self.split == 'test':
            self.data = npz['x_test']
            self.label = np.where(npz['y_test'] > 0)[1]
        else:
            raise NotImplementedError("split only supports train/test")

        self.data = self._to_nctvm(self.data)  # N,C,T,V,M
        if self.debug:
            self.data = self.data[:200]
            self.label = self.label[:200]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy = np.array(self.data[index])  # C,T,V,M
        label = int(self.label[index])

        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        num_people_present = np.sum(data_numpy.sum(0).sum(0).sum(0) != 0)

        if self.uniform:
            data_numpy, index_t = tools.valid_crop_uniform(
                data_numpy, valid_frame_num, self.p_interval, self.window_size, self.thres
            )
        else:
            data_numpy, index_t = tools.valid_crop_resize(
                data_numpy, valid_frame_num, self.p_interval, self.window_size, self.thres
            )

        if self.split == 'train':
            p = np.random.rand(1)
            if p < self.intra_p:
                # Optional COCO part-based augmentation (enable by setting partition: True)
                if self.partition and ('p' in self.aug_method):
                    data_numpy = drop_part(data_numpy, p=0.5)

                if 'a' in self.aug_method and np.random.rand(1) < 0.5:
                    data_numpy = data_numpy[:, :, :, np.array([1, 0])]
                if 'b' in self.aug_method and num_people_present == 2 and np.random.rand(1) < 0.5:
                    axis_next = np.random.randint(0, 1)
                    temp = data_numpy.copy()
                    C, T, V, M = data_numpy.shape
                    temp[:, :, :, axis_next] = np.zeros((C, T, V))
                    data_numpy = temp

                if '1' in self.aug_method:
                    data_numpy = tools.shear(data_numpy, p=0.5)
                if '2' in self.aug_method:
                    data_numpy = tools.rotate(data_numpy, p=0.5)
                if '3' in self.aug_method:
                    data_numpy = tools.scale(data_numpy, p=0.5)
                if '4' in self.aug_method:
                    data_numpy = tools.spatial_flip(data_numpy, p=0.5)
                if '5' in self.aug_method:
                    data_numpy, index_t = tools.temporal_flip(data_numpy, index_t, p=0.5)
                if '6' in self.aug_method:
                    data_numpy = tools.gaussian_noise(data_numpy, p=0.5)
                if '7' in self.aug_method:
                    data_numpy = tools.gaussian_filter(data_numpy, p=0.5)
                if '8' in self.aug_method:
                    data_numpy = tools.drop_axis(data_numpy, p=0.5)
                if '9' in self.aug_method:
                    data_numpy = tools.drop_joint(data_numpy, p=0.5)

        return data_numpy, index_t, label, index
