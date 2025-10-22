import unittest
import torch
from __init__ import *
import itertools
import random
import time


class TestMain(unittest.TestCase):

    def test_index_select(self):

        for i in range(10):
            ndim = random.randint(128, 512)
            topk = random.randint(10, 512)
            src = torch.randn(64000, ndim)
            indices = torch.randint(0, 64000, (topk,))

            target = torch.index_select(src, 0, indices)

            predict = d0_index_select(src, indices)

            self.assertTrue(torch.allclose(target, predict))

    def test_index_select_with_mask(self):

        python_time = []
        trition_time = []

        topk = 4096

        for i in range(1000):
            indices = torch.topk(torch.randn(8, 28, topk*2, 1).cuda(), topk, dim=2).indices
            mask = torch.randn(indices.shape).cuda() < 0.4

            # indices = indices.cuda()
            # mask = mask.cuda()

            tstart = time.time()
            target = take_along_dim_with_mask_python_linear_indices(topk*2, indices, mask)
            python_time.append(time.time() - tstart)

            tstart = time.time()
            out = take_along_dim_with_mask_triton(topk*2, indices, mask)
            trition_time.append(time.time() - tstart)

            self.assertTrue(torch.allclose(out, target))

        print(f"python time: {sum(python_time)}")
        print(f"triton time: {sum(trition_time)}")

        

        