import os
import yaml
import time
import torch
from mesh_data import PointCloudData
from pathlib import Path
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import numpy as np
from random import choice
from torch.utils.data import DataLoader
from models.pointnet import PointNetCls, feature_transform_regularizer
# from models.pointnet2 import PointNet2ClsMsg
# from models.dgcnn import DGCNN
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from pytorch3d.structures import join_meshes_as_batch
from pytorch3d.utils import ico_sphere
from torch.autograd import Variable
from torch_scatter import scatter_add
import open3d as o3d
from tqdm import tqdm
from pytorch3d.io import load_obj, save_obj
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)

# set path
path = Path("Manifold40/")
valid_ds = PointCloudData(path, valid=True, folder='test')


class CrossEntropyAdvLoss(nn.Module):

    def __init__(self):
        """Adversarial function on output probabilities.
        """
        super(CrossEntropyAdvLoss, self).__init__()

    def forward(self, logits, targets):
        """Adversarial loss function using cross entropy.

        Args:
            logits (torch.FloatTensor): output logits from network, [B, K]
            targets (torch.LongTensor): attack target class
        """
        loss = F.cross_entropy(logits, targets)
        return loss


def my_collate(batch):
    ## load unregular mesh within a batch
    meshes, label = zip(*batch)
    meshes = join_meshes_as_batch(meshes, include_textures=False)
    label = torch.tensor(label)
    return [meshes, label]

class ClipMeshv_Linf(nn.Module):

    def __init__(self, budget):
        """Clip mesh vertices with a given l_inf budget.

        Args:
            budget (float): perturbation budget
        """
        super(ClipMeshv_Linf, self).__init__()

        self.budget = budget

    def forward(self, vt, ori_vt):
        """Clipping every vertice in a mesh.

        Args:
            vt (torch.FloatTensor): batch vt, [B, 3, K]
            ori_vt (torch.FloatTensor): original point cloud
        """
        with torch.no_grad():
            diff = vt - ori_vt  # [B, 3, K]
            norm = torch.sum(diff ** 2, dim=1) ** 0.5  # [B, K]
            scale_factor = self.budget / (norm + 1e-9)  # [B, K]
            scale_factor = torch.clamp(scale_factor, max=1.)  # [B, K]
            diff = diff * scale_factor[:, None]
            vt = ori_vt + diff
        return vt


class MeshAttack:
    """Class for Mesh attack.
    """

    def __init__(self, model, adv_func, attack_lr=1e-2,
                 init_weight=10., max_weight=80., binary_step=10, num_iter=1500):
        """Mesh attack by perturbing vertice.

        Args:
            model (torch.nn.Module): victim model
            adv_func (function): adversarial loss function
            attack_lr (float, optional): lr for optimization. Defaults to 1e-2.
            init_weight (float, optional): weight factor init. Defaults to 10.
            max_weight (float, optional): max weight factor. Defaults to 80.
            binary_step (int, optional): binary search step. Defaults to 10.
            num_iter (int, optional): max iter num in every search step. Defaults to 500.
        """

        self.model = model.cuda()
        self.model.eval()

        self.adv_func = adv_func
        self.attack_lr = attack_lr
        self.init_weight = init_weight
        self.max_weight = max_weight
        self.binary_step = binary_step
        self.num_iter = num_iter
        self.clip = ClipMeshv_Linf(budget=0.1)

    def attack(self, data, target,label):
        """Attack on given data to target.

        Args:
            data (torch.FloatTensor): victim data, [B, num_vertices, 3]
            target (torch.LongTensor): target output, [B]
        """
        B, K = len(data), 1024
        global bas
        data = data.cuda()
        label_val = target.detach().cpu().numpy()  # [B]

        label = label.long().cuda().detach()
        label_true = label.detach().cpu().numpy()

        deform_ori = data.clone()

        # weight factor for budget regularization
        lower_bound = np.zeros((B,))
        upper_bound = np.ones((B,)) * self.max_weight
        current_weight = np.ones((B,)) * self.init_weight

        # record best results in binary search
        o_bestdist = np.array([1e10] * B)
        o_bestscore = np.array([-1] * B)
        o_bestattack = np.zeros((B, 3, K))
        # Weight for the chamfer loss
        w_chamfer = 1.0
        # Weight for mesh edge loss
        w_edge = 0.2
        # Weight for mesh laplacian smoothing
        w_laplacian = 0.5

        # perform binary search
        for binary_step in range(self.binary_step):
            deform_verts = torch.full(deform_ori.verts_packed().shape, 0.000001, device='cuda:%s'%args.local_rank, requires_grad=True)
            ori_def = deform_verts.detach().clone()

            bestdist = np.array([1e10] * B)
            bestscore = np.array([-1] * B)
            dist_val = 0
            opt = optim.Adam([deform_verts], lr=self.attack_lr, weight_decay=0.)
            # opt = optim.SGD([deform_verts], lr=1.0, momentum=0.9) #optim.Adam([deform_verts], lr=self.attack_lr, weight_decay=0.)

            adv_loss = torch.tensor(0.).cuda()
            dist_loss = torch.tensor(0.).cuda()

            total_time = 0.
            forward_time = 0.
            backward_time = 0.
            update_time = 0.

            # one step in binary search
            for iteration in range(self.num_iter):
                t1 = time.time()
                opt.zero_grad()
                new_defrom_mesh = deform_ori.offset_verts(deform_verts)

                # forward passing
                ori_data = sample_points_from_meshes(data, 1024)
                adv_pl = sample_points_from_meshes(new_defrom_mesh, 1024)
                adv_pl1 = adv_pl.transpose(1, 2).contiguous()
                logits = self.model(adv_pl1)  # [B, num_classes]
                if isinstance(logits, tuple):  # PointNet
                    logits = logits[0]

                t2 = time.time()
                forward_time += t2 - t1

                pred = torch.argmax(logits, dim=1)  # [B]
                success_num = (pred == target).sum().item()
                if iteration % (self.num_iter // 5) == 0:
                    print('Step {}, iteration {}, current_c {},success {}/{}\n'
                          'adv_loss: {:.4f}'.
                          format(binary_step, iteration, torch.from_numpy(current_weight).mean(), success_num, B,
                                 adv_loss.item()))
                dist_val = torch.sqrt(torch.sum(
                    (adv_pl - ori_data) ** 2, dim=[1, 2])).\
                    detach().cpu().numpy()  # [B]
                pred_val = pred.detach().cpu().numpy()  # [B]
                input_val = adv_pl1.detach().cpu().numpy()  # [B, 3, K]

                # update
                for e, (dist, pred, label, ii) in \
                        enumerate(zip(dist_val, pred_val, label_val, input_val)):
                    if dist < bestdist[e] and pred == label:
                        bestdist[e] = dist
                        bestscore[e] = pred
                    if dist < o_bestdist[e] and pred == label:
                        o_bestdist[e] = dist
                        o_bestscore[e] = pred
                        o_bestattack[e] = ii

                t3 = time.time()
                # compute loss and backward
                adv_loss = self.adv_func(logits, target).mean()
                loss_chamfer, _ = chamfer_distance(ori_data, adv_pl)
                loss_edge = mesh_edge_loss(new_defrom_mesh)
                loss_laplacian = mesh_laplacian_smoothing(new_defrom_mesh, method="uniform")

                loss = adv_loss + torch.from_numpy(current_weight).mean()*(loss_chamfer * w_chamfer + loss_edge * w_edge  + loss_laplacian * w_laplacian)
                loss.backward()
                opt.step()

                deform_verts.data = self.clip(deform_verts.clone().detach(),
                                               ori_def)

                t4 = time.time()
                backward_time += t4 - t3
                total_time += t4 - t1

                if iteration % 100 == 0:
                    print('total time: {:.2f}, for: {:.2f}, '
                          'back: {:.6f}, update: {:.2f}, total loss: {:.6f}, chamfer loss: {:.6f}'.
                          format(total_time, forward_time,
                                 backward_time, update_time,loss, loss_chamfer))
                    total_time = 0.
                    forward_time = 0.
                    backward_time = 0.
                    update_time = 0.
                    torch.cuda.empty_cache()

            # adjust weight factor
            for e, label in enumerate(label_val):
                if bestscore[e] == label and bestscore[e] != -1 and bestdist[e] <= o_bestdist[e]:
                    # success
                    lower_bound[e] = max(lower_bound[e], current_weight[e])
                    current_weight[e] = (lower_bound[e] + upper_bound[e]) / 2.
                else:
                    # failure
                    upper_bound[e] = min(upper_bound[e], current_weight[e])
                    current_weight[e] = (lower_bound[e] + upper_bound[e]) / 2.

        bas += 1
        ## save the mesh
        new_defrom_mesh = deform_ori.offset_verts(deform_verts)
        for e1 in range(B):
            final_verts, final_faces = new_defrom_mesh.get_mesh_verts_faces(e1)
            final_obj = os.path.join('./p1_manifold_random_target01', 'result_model%s_%s_%s_%s.obj'%(bas,e1,label_val[e1],label_true[e1]))
            save_obj(final_obj, final_verts, final_faces)

        fail_idx = (lower_bound == 0.)
        o_bestattack[fail_idx] = input_val[fail_idx]

        # return final results
        success_num = (lower_bound > 0.).sum()
        print('Successfully attack {}/{}'.format(success_num, B))
        return o_bestdist, o_bestattack.transpose((0, 2, 1)), success_num



def get_random_labels(label):
    ret = []
    for j in range(len(label)):
        random_taget = choice([i for i in range(0,40) if i not in [label[j]]])
        ret.append(random_taget)
    return torch.Tensor(np.array(ret))


def attack():
    model.eval()
    all_adv_pc = []
    all_real_lbl = []
    all_target_lbl = []
    global bas
    bas = 0
    num = 0
    for mesh, label in tqdm(test_loader):
        target_label = get_random_labels(label).long().cuda(non_blocking=True)
        label = label.long().cuda(non_blocking=True)
        

        # attack!
        _, best_pc, success_num = attacker.attack(mesh, target_label,label)
        # results
        num += success_num
        all_adv_pc.append(best_pc)
        all_real_lbl.append(label.detach().cpu().numpy())
        all_target_lbl.append(target_label.detach().cpu().numpy())

    # accumulate results
    all_adv_pc = np.concatenate(all_adv_pc, axis=0)  # [num_data, K, 3]
    all_real_lbl = np.concatenate(all_real_lbl, axis=0)  # [num_data]
    all_target_lbl = np.concatenate(all_target_lbl, axis=0)  # [num_data]
    return all_adv_pc, all_real_lbl, all_target_lbl, num

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--data_root', type=str,
                        default='')
    parser.add_argument('--model', type=str, default='pointnet', metavar='MODEL',
                        choices=['pointnet', 'pointnet2',
                                 'dgcnn', 'pointconv', ''],
                        help='Model to use, [pointnet, pointnet++, dgcnn, pointconv]. '
                             'If not specified, judge from data_root')
    parser.add_argument('--feature_transform', type=int, default=1, help="use feature transform")
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    args = parser.parse_args()

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    cudnn.benchmark = True
    num_classes = 40

    if args.model == 'pointnet':
        model = PointNetCls(num_classes, args.feature_transform)
    #load pretrain model
    state_dict = torch.load('model/p1.pth', map_location='cpu')
    try:
        model.load_state_dict(state_dict['model_state_dict'])
    except RuntimeError:
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    model = DistributedDataParallel(
        model.cuda(), device_ids=[args.local_rank])


    print('======> Loading data')
    train_sampler = torch.utils.data.distributed.DistributedSampler(valid_ds)
    test_loader = torch.utils.data.DataLoader(valid_ds, batch_size=args.batch_size,
                                              num_workers=4, pin_memory=True, drop_last=False, collate_fn=my_collate,
                                              sampler=train_sampler)
    print('======> Successfully loaded!')
    
    # run attack
    adv_func = CrossEntropyAdvLoss()

    attacker = MeshAttack(model, adv_func)

    attacked_data, real_label, target_label, success_num = attack()
