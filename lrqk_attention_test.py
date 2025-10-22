import unittest
import random
from lrqk_attention import *
from torch.profiler import profile, record_function, ProfilerActivity
import itertools


class TestCPUAutoIncreaseTensor(unittest.TestCase):

    def test_append(self):
        ait = AutoIncreaseTensor(capacity=16, dim=0)

        s_sum = 0
        for i in range(10):
            s = random.randint(1, 16)
            s_sum += s
            ait.append(torch.randn(s, 128))
            self.assertEqual(len(ait), s_sum)
        print(ait)

    def test_getitem(self):
        ait = AutoIncreaseTensor(capacity=16, dim=0, device='cuda:0')

        target = torch.randn(512, 128)

        ll = 0

        while ll < 512:
            s = random.randint(1, 16)
            rr = min(ll+s, 512)
            ait.append(target[ll:rr])
            ll = rr

            s = random.randint(1, rr-1)
            self.assertTrue(torch.allclose(ait[:s].cpu(), target[:s]))
            self.assertTrue(torch.allclose(ait[:s, 0].cpu(), target[:s, 0]))
            self.assertTrue(torch.allclose(ait[s].cpu(), target[s]))
            self.assertTrue(torch.allclose(ait[s], target[s].to(ait.device)))

    def test_getitem_large(self):
        ait = AutoIncreaseTensor(capacity=1024, dim=0)

        target = torch.randn(512, 128)

        ll = 0

        while ll < 512:
            s = random.randint(1, 16)
            rr = min(ll+s, 512)
            ait.append(target[ll:rr])
            ll = rr

            s = random.randint(1, rr-1)
            self.assertTrue(torch.allclose(ait[:s], target[:s]))
            self.assertTrue(torch.allclose(ait[:s, 0], target[:s, 0]))
            self.assertTrue(torch.allclose(ait[s], target[s]))

    def test_getitem_enum(self):
        ait = AutoIncreaseTensor(capacity=16, dim=2)
        target = torch.randn(2, 8, 512, 128)
        ait.append(target[:, :, 0:128])

        for i in range(128, 512):
            ait.append(target[:, :, i:i+1])

            self.assertTrue(torch.allclose(ait[:, :, 0:i+1], target[:, :, 0:i+1]))
