import numpy as np
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
import random
import os
import torch
import scipy.spatial.distance
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pathlib import Path
import pytorch3d
import trimesh
import open3d as o3d

from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from prefetch_generator import BackgroundGenerator
from tqdm.notebook import tqdm


class PointCloudData(Dataset):
    def __init__(self, root_dir, valid=False, folder="train"):
        self.root_dir = root_dir
        folders = [directory for directory in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir / directory)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.valid = valid
        self.files = []
        for category in self.classes.keys():
            new_dir = root_dir / Path(category) / folder
            for file in os.listdir(new_dir):
                if file.endswith('.obj'):
                    sample = {}
                    sample['mesh_path'] = new_dir / file
                    sample['category'] = category
                    self.files.append(sample)

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
    
    def __len__(self):
        return len(self.files)

    def __preproc__(self, file):
        mesh = trimesh.load(file)
        v, f = torch.from_numpy(mesh.vertices).float(), torch.from_numpy(mesh.faces).long()

        # normalize
        center = v.mean(0)
        verts = v - center
        scale = max(verts.abs().max(0)[0])
        v = verts / scale

        mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(v),
            triangles=o3d.utility.Vector3iVector(f))
        mesh = mesh.subdivide_loop(number_of_iterations=1)
        v = torch.from_numpy(np.array(mesh.vertices)).float()
        f = torch.from_numpy(np.array(mesh.triangles)).long()

        trg_mesh = Meshes(verts=[v], faces=[f])
        return trg_mesh

    def __getitem__(self, idx):
        pcd_path = self.files[idx]['mesh_path']
        category = self.files[idx]['category']
        g_mesh = self.__preproc__(pcd_path)
        return g_mesh, self.classes[category]


