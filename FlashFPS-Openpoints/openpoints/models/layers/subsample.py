# subsample layer for 3d processing.
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.autograd import Function
import math

import sys
# sys.path.insert(0, "/workspace/FlashFPS-Openpoints")
import os
_this_file = os.path.abspath(__file__)
_pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_this_file))))  # .../FlashFPS-Openpoints
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)
from openpoints.cpp.pointnet2_batch import pointnet2_cuda
print(pointnet2_cuda.__file__)
print('QuickFPS_wrapper' in dir(pointnet2_cuda))
# import pdb; pdb.set_trace()
import asyncio
from concurrent.futures import ThreadPoolExecutor
import  time 

class BaseSampler(ABC):
    """If num_to_sample is provided, sample exactly
        num_to_sample points. Otherwise sample floor(pos[0] * ratio) points
    """

    def __init__(self, ratio=None, num_to_sample=None, subsampling_param=None):
        if num_to_sample is not None:
            if (ratio is not None) or (subsampling_param is not None):
                raise ValueError(
                    "Can only specify ratio or num_to_sample or subsampling_param, not several !")
            self._num_to_sample = num_to_sample

        elif ratio is not None:
            self._ratio = ratio

        elif subsampling_param is not None:
            self._subsampling_param = subsampling_param

        else:
            raise Exception(
                'At least ["ratio, num_to_sample, subsampling_param"] should be defined')

    def __call__(self, xyz):
        return self.sample(xyz)

    def _get_num_to_sample(self, npoints) -> int:
        if hasattr(self, "_num_to_sample"):
            return self._num_to_sample
        else:
            return math.floor(npoints * self._ratio)

    def _get_ratio_to_sample(self, batch_size) -> float:
        if hasattr(self, "_ratio"):
            return self._ratio
        else:
            return self._num_to_sample / float(batch_size)

    @abstractmethod
    def sample(self, xyz, feature=None, batch=None):
        pass


class RandomSample(BaseSampler):
    """Random Sample for dense data
        Arguments:
            xyz -- [B, N, 3]
    """

    def sample(self, xyz, **kwargs):
        if len(xyz.shape) != 3:
            raise ValueError(" Expects the xyz tensor to be of dimension 3")
        B, N, _ = xyz.shape
        idx = torch.randint(
            0, N, (B, self._get_num_to_sample(N)), device=xyz.device)
        sampled_xyz = torch.gather(xyz, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        # sampled_feature = torch.gather(feature, 2, idx.unsqueeze(1).repeat(1, C, 1))
        return sampled_xyz, idx


def standard_fps(xyz, npoint):
    B, N, _ = xyz.size()
    output = torch.cuda.IntTensor(B, npoint)
    temp = torch.cuda.FloatTensor(B, N).fill_(1e10)
    pointnet2_cuda.furthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output)
    # output = random_sample(xyz, npoint)
    return output

def random_sample(xyz, npoint):
    B, N, _ = xyz.shape
    idx = torch.randint(0, N, (B, npoint), device=xyz.device)
    return idx



def rearrange_indices(fps_idx: torch.Tensor, a: int, c: int, xyz: torch.Tensor):
    """
    fps_idx: [B, b]  (each batch's FPS output)
    a: int，total points number
    c: actual sampled points number
    return: B_out [B, c]
    """
    B, b = fps_idx.shape
    device = fps_idx.device

    # all indices [0,1,2,...,a-1]
    all_idx = torch.arange(a, device=device).unsqueeze(0).repeat(B, 1)  # [B, a]

    # construct mask（True means not sampled）
    mask = torch.ones((B, a), dtype=torch.bool, device=device)
    mask.scatter_(1, fps_idx.long(), False)  # mark the sampled points as False

    # introduce random to introduce diversity
    # output = random_sample(xyz, a-b)
    # B_out = torch.cat([fps_idx, output], dim=1)

    remaining_idx = all_idx[mask].reshape(B, a - b)

    B_out = torch.cat([fps_idx, remaining_idx], dim=1)
    B_out = B_out[:, :c]  # ensure the output size is [B, c]
    return B_out



def FPS_Prune(xyz, npoint, PruneRage=0.5):
    B, N, _ = xyz.size()
    # step = int(N / npoint)
    N_points = int(N* PruneRage)
    sample_rate = int(N/npoint)
    num_points = int(N*PruneRage/sample_rate)
    actual_npoint = int(N/sample_rate)
    
    if num_points != 0:
        idx = standard_fps(xyz[:, :N_points, :].contiguous(), num_points)

    if num_points < actual_npoint:
        if num_points > 0:
            idx = rearrange_indices(idx, N_points, actual_npoint, xyz)
        else:
            idx = torch.arange(actual_npoint).unsqueeze(0).expand(B, -1).contiguous().to(xyz.device)

    return idx



class FurthestPointSampling(Function):
    counter = 0
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int, PruneRage=0.25, stage: int = None, enable_quick_fps: bool = False) -> torch.Tensor:
        """
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance
        :param ctx:
        :param xyz: (B, N, 3) where N > npoint
        :param npoint: int, number of features in the sampled set
        :return:
             output: (B, npoint) tensor containing the set (idx)
        """
        # pdb.set_trace()
        assert xyz.is_contiguous()
        # ReducedFPS = [47358, 43015, 101941, 224048, 78229, 51502, 51161, 84290, 48350, 86865, 289873, 82823, 183054, 71698, 78188, 78637, 71329, 59838, 54029, 61117, 58163, 47259, 46347, 13114, 49132]
        ReducedFPS = []
        B, N, _ = xyz.size()

        if stage == 1: 
            output = FPS_Prune(xyz, npoint, PruneRage=PruneRage)
        else:
            if enable_quick_fps: # use quick fps
                output = quick_fps(xyz, npoint)
            else: # original fps
                output = torch.cuda.IntTensor(B, npoint)
                temp = torch.cuda.FloatTensor(B, N).fill_(1e10)
                pointnet2_cuda.furthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output)

        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply

# The quick fps implementation is adapted from [FastPoint](https://github.com/SNU-ARC/FastPoint)
class QuickFPS(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """
        Uses QuickFPS to select a set of npoint features that have the largest
        minimum distance
        :param ctx:
        :param xyz: (B, N, 3) where N > npoint
        :param npoint: int, number of features in the sampled set
        :return:
             output: (B, npoint) tensor containing the set (idx)
        """
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
    
        kd_high = 8
        bucket_size = 1 << kd_high
        
        bucketIndex = torch.cuda.IntTensor(bucket_size).fill_(0)
        bucketLength = torch.cuda.IntTensor(bucket_size).fill_(N)
        output = torch.cuda.FloatTensor(npoint, 5).fill_(0)
        
        idx = torch.arange(N).unsqueeze(0).unsqueeze(-1).cuda()
        xyzi = torch.concat((xyz, idx), dim=-1)
        pointnet2_cuda.QuickFPS_wrapper(B, N, npoint, kd_high, xyzi, output, bucketIndex, bucketLength)
        
        return output
    
    @staticmethod
    def backward(xyz, a=None):
        return None, None

quick_fps = QuickFPS.apply


class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param features: (B, C, N)
        :param idx: (B, npoint) index tensor of the features to gather
        :return:
            output: (B, C, npoint)
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, npoint = idx.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B, C, npoint, device=features.device)

        pointnet2_cuda.gather_points_wrapper(
            B, C, N, npoint, features, idx, output)

        ctx.for_backwards = (idx, C, N)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards
        B, npoint = idx.size()

        grad_features = torch.zeros(
            [B, C, N], dtype=torch.float, device=grad_out.device, requires_grad=True)
        grad_out_data = grad_out.data.contiguous()
        pointnet2_cuda.gather_points_grad_wrapper(
            B, C, N, npoint, grad_out_data, idx, grad_features.data)
        return grad_features, None


gather_operation = GatherOperation.apply
# mark: torch gather is even faster. sampled_xyz = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, 3))


def fps(data, number):
    '''
        data B N C
        number int
    '''
    fps_idx = furthest_point_sample(data[:, :, :3].contiguous(), number)
    fps_data = torch.gather(
        data, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, data.shape[-1]))
    return fps_data


if __name__ == '__main__':
    import time

    B, C, N = 2, 3, 10000
    K = 16
    device = 'cuda'
    points = torch.randn([B, N, 3], device=device, dtype=torch.float)
    print(points.shape, '\n', points)

    nsample = 4096
    idx = furthest_point_sample(points, nsample)

    st = time.time()
    for _ in range(100):
        query1 = torch.gather(
            points, 1, idx.long().unsqueeze(-1).expand(-1, -1, 3))
    print(time.time() - st)
    print(query1.shape)

    st = time.time()
    for _ in range(100):
        query2 = gather_operation(points.transpose(
            1, 2).contiguous(), idx).transpose(1, 2).contiguous()
    print(time.time() - st)
    print(query2.shape)

    print(torch.allclose(query1, query2))
