import torch
from torch.autograd import Variable
#
# Configuration.
def getTensorConfiguration():
    use_cuda = torch.cuda.is_available()
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
    ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
    #
    # Default transforms.
    def cpuTensorImg(x):
        return torch.from_numpy(x.transpose((0,3,1,2))).type(FloatTensor).div_(255)
    def cpuTensor(x):
        return torch.from_numpy(x)
    toTensorImg = cpuTensorImg
    toTensor = cpuTensor
    #
    # Cuda transforms.
    if use_cuda:
        def cudaTensorImg(x):
            ret = torch.from_numpy(x.transpose((0,3,1,2))).contiguous().cuda(async=True).type(FloatTensor).div_(255)
            return ret
        def cudaTensor(x):
            ret = torch.from_numpy(x).cuda(async=True)
            return ret
        toTensorImg = cudaTensorImg
        toTensor = cudaTensor
    return toTensorImg, toTensor, use_cuda