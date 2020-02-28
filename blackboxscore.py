import torch.nn as nn
import torch
import numpy as np


class BlackBoxScore(nn.Module):
    """
    Class 'BlackBoxScore' scores sampled input images with respective confidence scores
    and aggregates all the weighted images to return a saliency map.
    """
    
    
    def __init__(self, model, input_size, gpu_batch=100):
        """
        Intialize and seed the class 'BlackBoxScore' with the model.
        
        Attributes:
            model : Black box classifier that outputs confidence scores for a stack of images.
            input_size (tuple): Tuple of height and width of input image.
        """
        
        super(BlackBoxScore, self).__init__()
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch
    
    def generate_b_masks(self,savepath, window_size=50, stride=20, image_size=(224, 224)):
        """
        Generating the sliding window style masks.
        
        :param window_size: the block window size (with value 0, other areas with value 1)
        :type window_size: int
        
        :param stride: the sliding step
        :type stride: int
        
        :param image_size: the mask size which should be the same to the image size
        :type image_size: tuple
        
        :return: the sliding window style masks
        :rtype: numpy array
        """
        
        rows = np.arange(0 + stride - window_size, image_size[0], stride)
        cols = np.arange(0 + stride - window_size, image_size[1], stride)

        mask_num = len(rows) * len(cols)
        masks = np.ones((mask_num, image_size[0], image_size[1])\
                                              , dtype=np.float64)
        i = 0
        for r in rows:
            for c in cols:
                if r<0:
                    r1 = 0
                else:
                    r1 = r
                if r + window_size > image_size[0]:
                    r2 = image_size[0]
                else:
                    r2 = r + window_size
                if c<0:
                    c1 = 0
                else:
                    c1 = c
                if c + window_size > image_size[1]:
                    c2 = image_size[1]
                else:
                    c2 = c + window_size
                masks[i, r1:r2, c1:c2] = 0
                i += 1
        mask_shape = [-1] + [1] + [image_size[0],image_size[1]]
        self.masks = masks.reshape(mask_shape)
        self.masks = torch.from_numpy(self.masks).float()
        self.masks = self.masks.cuda()
        self.N = self.masks.shape[0]
        
    def load_masks(self, filepath):
        """
        Load cached maps from disk from path filepath.
        """
        
        self.masks = np.load(filepath)
        self.masks = torch.from_numpy(self.masks).float().cuda()
        self.N = self.masks.shape[0]

    def forward(self, x):
        """ 
        The forward function is an altered version of RISE code from 
        https://github.com/eclique/RISE and computes saliency maps by sampling regions.
        
        Attributes:
            x (tensor): The input image that needs to be explained.
            
        Returns:
            sal (tensor): Saliency map produced using weighted confidence scores. 
        """
        
        N = self.N
        _, _, H, W = x.size()
        stack = torch.mul(self.masks, x.data)
        p = []
        for i in range(0, N, self.gpu_batch):
            p.append(self.model(stack[i:min(i + self.gpu_batch, N)]))
        p = torch.cat(p)
        CL = p.size(1)
        sal = torch.matmul(p.data.transpose(0, 1), self.masks.view(N, H * W))
        sal = sal.view((CL, H, W))
        sal = sal / N 
        return sal
