import torch.nn as nn
import torch
import numpy as np
import cv2
import torchvision.models as models
import pdb
from utils import mask_sal

class LRPFSA:
    """
    Long-Range Parameter Freee Spatial Attention (LRPFSA)
    Class for computing a spatial attention map and combine it with saliency map from 
    previous iteration to generate an adjusted saliency map and sampling masks for next iteration.
    """
    
    def __init__(self, img, model_name, lambda_fac, t_thresh, net_in_size):
        """
        Initialize and seed the LRPFSA module
        
        Attributes:
            img (Tensor): normalized image tensor.
            model_name (str): Selected black box model name. Currently only supports resnet50.
            lambda_fac (0-1 float):Weight controlling the attention map and saliency 
                             map combination to obtain the adjusted saliency maps.
            t_thresh (0-1 float): Threshold used for segmenting saliency maps for 
                                evaluation to obtain highest activated region.
            net_in_size (int): Size of input image before feature extraction.            
        """
        
        self.img = img
        self.model_name = model_name
        self.lambda_fac = lambda_fac
        self.thresh = t_thresh
        self.net_in_size = net_in_size
        if self.model_name == 'resnet50':
            self.__levels__ = ['-9','-6','-4','-3','-2']
            self.base_model = models.resnet50(True)
            self.resize_val = tuple([112,112])
        if self.model_name == 'vgg16':
            #__levels__ = TODO
            print("Oops! VGG16 not supported yet!")
            pass
        self.m = nn.Upsample(size=self.resize_val)
        
    def forward(self, ini_sal):
        """
        Main LRFSA function to generate a adjusted saliency map and sampling masks for next iteration
        
        Attributes:
            ini_sal (numpy.array): Initial coarse saliency map if first iteration or saliency map obtained
                                   in last iteration.
                             
        Returns:
            adj_sal_map (numpy.array): Adjusted saliency map for itereation k.
            masks (numpy.array): Mask to select sampling regions for next iteration.            
        """

        feat_aggregate = []
        for _layers_ in self.__levels__:
            self.my_model_it = nn.Sequential(*list(self.base_model.children())[:int(_layers_)])
            self.my_model_it = self.my_model_it.eval()
            self.my_model_it = self.my_model_it.cuda()
            feats_this = self.my_model_it(self.img.cuda())
            feat_this_level = (self.m(feats_this)).squeeze()
            if len(feat_aggregate)!=0:
                feat_aggregate = torch.cat((feat_aggregate, feat_this_level),0)
            else:
                feat_aggregate = feat_this_level
        # Reshape from CxHXW to CXHW
        feat_agg_c_hw = feat_aggregate.view(feat_aggregate.shape[0], feat_aggregate.shape[2] * feat_aggregate.shape[1])        
        # Transpose from CXHW to HWXC
        feat_agg_hw_c = feat_agg_c_hw.t()
        # Multiplying HWXC with CXHW 
        feat_agg_hw_hw = torch.matmul(feat_agg_hw_c, feat_agg_c_hw)
        feat_agg_hw_hw_soft = nn.functional.softmax(feat_agg_hw_hw, dim=0)
        feat_agg_hw_hw_soft = nn.functional.softmax(feat_agg_hw_hw_soft, dim=1)
        # Downsample ini_sal from (224,224) to (112,112) or (H,W)
        ini_sal_h_w = cv2.resize(ini_sal, dsize= (self.resize_val[0], self.resize_val[1]), interpolation = cv2.INTER_NEAREST)
        ini_sal_h_w_ten = torch.from_numpy(ini_sal_h_w).float().cuda()
        # Reshap initial saliency map from HXW to 1xHW
        ini_sal_1_hw_ten = ini_sal_h_w_ten.view(1, self.resize_val[0] * self.resize_val[1])    
        # Multiplying 1xHW and HWXHW to get 1xHW 
        new_sal_map = torch.matmul(ini_sal_1_hw_ten, feat_agg_hw_hw_soft)
        # Reshaping from 1xHW to HXW
        new_sal_map = new_sal_map.view(self.resize_val[0], self.resize_val[1])
        # Upsample final combined map to original input size
        x = torch.linspace(-1, 1, self.net_in_size).repeat(self.net_in_size, 1)
        y = torch.linspace(-1, 1, self.net_in_size).view(-1, 1).repeat(1, self.net_in_size)
        grid = torch.cat((x.unsqueeze(2), y.unsqueeze(2)), 2)
        grid = grid.unsqueeze_(0).repeat(1,1,1,1)
        new_sal_map_224_224 = nn.functional.grid_sample(new_sal_map.unsqueeze(0).unsqueeze(1), grid.cuda()).squeeze()
        # Adjusted_saliency_map = βSk + (β − 1)(A × Sk)
        adj_sal_map = (self.lambda_fac)*ini_sal + (1-self.lambda_fac)*(new_sal_map_224_224.detach().cpu().numpy())
        # Threshold the adjusted saliency maps to generate a binary mask for allowable sampling regions in next iteration.
        thre = self.thresh*(np.max(adj_sal_map)-(np.min(adj_sal_map)))
        thre += np.min(adj_sal_map)
        masks = mask_sal(adj_sal_map, self.img.squeeze(0).permute(1,2,0).detach().cpu().numpy(), thre, self.net_in_size) 
        return adj_sal_map, masks