from ser.transforms import flip, normalize, transform
import numpy as np
import torch

def test_transform():
    transform_flip = [flip]
    composition_flip = transform(*transform_flip)

    test_image = np.array([[1.0,0.],[0.,0.]])
    expectation = torch.unsqueeze(torch.DoubleTensor([[0.,0.],[0.,1.]]),0)

    flipped_image = composition_flip(test_image)

    assert torch.equal(flipped_image, expectation)