import os
import numpy as np
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import cv2
import glob
import click
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.nn.functional import conv2d
from PIL import Image
from matplotlib import pyplot as plt
from utils import find_intersection, get_iou_, pointing_game_score
from LRPF_SA import LRPFSA
from blackboxscore import BlackBoxScore
from evaluate import CausalMetric, auc, gkern
net_in_size = 224

@click.command()
@click.option('--query_path', default = "/home/bhavan.vasu/WACV/query/*.png", help='Path to folder containing images to be explained')
@click.option('--save_path', default= "/home/bhavan.vasu/WACV/save_path/", help='Path to save best saliency maps from run')
@click.option('--gt_path', default= "/home/bhavan.vasu/WACV/mask/", help='Path to annotations')

class IASSA():
    """ 
    The IASSA class implementation that takes in experimental parameters and 
    tries to iteratively explain a database of images by sampling around important
    image regions with the help of spatial attention maps
    Iterative and Adaptive Sampling with Spatial Attentionfor Black-Box Model 
    Explanations: https://arxiv.org/abs/1912.08387
    """
    
    def __init__(self,
                 query_path,
                 save_path,
                 gt_path,
                 model_name = 'resnet50',
                 iteration_k = 5,
                 init_window_size = 45,
                 init_stride = 8,
                 lambda_fac = 0.5,
                 t_thresh = 0.3,
                 top_k_sal = 1,
                 save_final_sal = False):
        """
        Intialize a ``IASSA`` instance for obtaining saliency maps
        iteratively.
        
        Attributes:
            query_path (str): Path to folder contains images to be explained.
            save_path (str): Path to folder for saving the best saliency maps
            gt_path (str): Path to folder containing sgementations annotations 
                           for input image and class
            model_name (str): Blackbox model name. Curently supports only 'resnet50'
            iteration_k (int): Number of iterations of sampling.
            init_window_size (int): Initial sampling window size for generating coarse
                                    saliency maps
            init_stride (int): Initial sampling stride for initial coarse map.
            lamba_fac (int 0-1): Weight controlling the attention map and saliency 
                             map combination to obtain the adjusted saliency maps.
            t_thresh (int 0-1): Threshold used for segmenting saliency maps for 
                                evaluation to obtain highest activated region.
            top_k_sal (int): Generate explanation for top k model decisions.
            save_final_sal (Boolean): Flag controlling storing and saving of the best 
                                      saliency map across all iterations.
                                      
        Returns:
            Instance of class 'IASSA'
        """
    
        self.query_path = query_path
        self.save_path = save_path
        self.annotation_path = gt_path
        self.k = iteration_k
        self.init_window_size = init_window_size
        self.init_stride = init_stride
        self.lambda_fac = lambda_fac
        self.t_thresh = t_thresh
        self.top_k_sal = top_k_sal
        self.model_name = model_name
        self.read_tensor = transforms.Compose([
                            lambda x: Image.open(x),
                            transforms.Resize((net_in_size, net_in_size)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225]),
                            lambda x: torch.unsqueeze(x, 0)
                        ])
        self.mask_path = './masks.npy'
        self.save_final_sal = save_final_sal
        try:
            assert int(self.init_window_size - (self.k*1.5)) > 0
        except:
            print("Increase initial window size or decrease number of iterations")
        try:
            assert (self.init_stride - (self.k*0.2)) > 0
        except:
            print("Increase initial stride or decrease number of iterations")
        cudnn.benchmark = True
        if self.model_name == 'resnet50':
            self.base_model = models.resnet50(True)
        if self.model_name == 'VGG16':
            print("Oops! VGG16 not supported yet!")
        model = nn.Sequential(self.base_model, nn.Softmax(dim=1))
        model = model.eval()
        self.model = model.cuda()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model = nn.DataParallel(self.model)
        self.explainer = BlackBoxScore(self.model, (net_in_size, net_in_size),  1)
        maps_dict = self.forward()
        self.average_eval(maps_dict)    

    def get_initial_map(self, img):
        """
        This function performs a coarse sampling of the input image
        to obtain an initial saliency map that is refined iteratively.
        
        Attributes:
            img (torch.tensor.cuda): Input image that needs to be explained.
            TODO: Make initialization class dependant to get explanations for   
            selected classes on the same input image.
        Returns:
            sal (numpy.array): Saliency map explaining the input image for top_k model prediction.
        """
        
        saliency_ten = self.explainer(img)
        saliency = saliency_ten.cpu().numpy()
        p, c = torch.topk(self.model(img.cuda()), k=self.top_k_sal)
        p, c = p[0], c[0]
        for k in range(self.top_k_sal):
            sal = saliency[c[k]]
        return sal

    def evaluate(self, img, new_sal):
        """
        This function evaluates an image and its saliency map for deletion 
        and insertion.
        
        Attributes:
            img (torch.tensor.cuda): Input image that needs to be explained.
            new_sal (numpy.array): The saliency map obtained at iteration k. 
            
        Returns:
        score_del (float): The deletion score between 0-1 range.
        score_ins (float): The insertion score between 0-1 range.
        """
        
        klen = 11
        ksig = 5
        kern = gkern(klen, ksig)
        blur = lambda x: nn.functional.conv2d(x, kern, padding=klen//2)
        insertion = CausalMetric(self.model, 'ins', net_in_size, substrate_fn=blur)
        deletion = CausalMetric(self.model, 'del', net_in_size, substrate_fn=torch.zeros_like)
        new_sal = torch.from_numpy(new_sal).float()
        score_del = deletion.single_run(img.cpu(), new_sal, verbose=0)
        score_ins = insertion.single_run(img.cpu(), new_sal, verbose=0)
        return score_del, score_ins
    
    def average_eval(self, maps_dict):
        """
        This function evaluates a dictionary of saliency map for 
        deletion and insertion by averaging across all images.
        
        Attributes:
            maps_dict (dict): Dictionary of image path:best saliency map dict. 
            
        Returns:
        Prints the average performance across all samples in maps_dict.
        """
        
        a_del_avg = []
        a_ins_avg = []
        f1_avg = []
        for jk, img_path in enumerate(maps_dict.keys()[1:]):
            img = self.read_tensor(self.im_path)
            del_stat, ins_stat = self.evaluate(img, maps_dict.values()[jk])
            a_del_avg.append(del_stat)
            a_ins_avg.append(ins_stat)
        print("average del AUC:{}".format(sum(a_del_avg)/len(a_del_avg)))
        print("average ins AUC:{}".format(sum(a_ins_avg)/len(a_ins_avg)))

    def forward(self):
        """
        This is the main function that imports other modules to achieve 
        the proposed goal of iterative and adaptive sampling. 
            
        Returns:
        all_maps_dict (dict): Dictionary of image path:best saliency map dict.
        """
        
        all_maps_dict = {}
        for img_path in  tqdm(glob.glob(self.query_path)):
            self.im_path = img_path
            img = self.read_tensor(self.im_path)
            img = img.cuda()
            self.explainer.generate_b_masks(self.mask_path, self.init_window_size, 
                                        self.init_stride)
            self.LRPF = LRPFSA(img, self.model_name, self.lambda_fac, self.t_thresh, net_in_size)
            # Annotations were named comparable to images
            im_name = img_path.split('/')[-1].split('.')[0]
            init_sal = self.get_initial_map(img)
            # Setting the best deletion and insertion area to 100000 and 0 for max and min value.
            best_del_area = 100000
            best_iou = 0
            for n,i in enumerate(range(self.k)):
                new_sal, Har_mask = self.LRPF.forward(init_sal)
                # Loading annotations for evaluation
                mask = cv2.imread("{}{}.png".format(self.annotation_path, 
                                                    im_name.split('_')[0]),0)
                mask = mask/255
                mask = cv2.resize(mask,(net_in_size, net_in_size))
                # Calculate the IoU between thresholded saliency map and segmentation annotation
                # and evaluating to store the best saliency map for viewing
                if self.save_final_sal == True:
                    iou = get_iou_(mask, Har_mask)
                    f1 = f1_score(mask.flatten().round(), Har_mask.flatten().astype(float).round()
                              ,average='micro')
                    point_game = pointing_game_score(mask, Har_mask)
                    area_del, area_ins = self.evaluate(img,new_sal)
                    # IoU set as selection criteria
                    if iou > best_iou:
                        best_iou = iou
                        best_sal = new_sal
                # Attentuation factor for window size and stride.
                self.explainer.generate_b_masks(window_size = 
                                                int(self.init_window_size -((n+1)*1.5)), 
                                                stride=int(self.init_stride -(0.2*n)), 
                                                savepath=self.mask_path)
                # Obtaining sampling regions for next iteration.
                new_Har_masks = torch.cat(find_intersection(self.explainer.masks, Har_mask)
                                          , dim=0).unsqueeze(1)
                # Loading new masks for use in next iteration
                self.explainer.masks = new_Har_masks.cuda()
                self.explainer.N = new_Har_masks.size(0)
                init_sal = self.get_initial_map(img)
            # Saving the best saliency map for each query image.
            if self.save_final_sal == True:
                all_maps_dict[self.im_path] = best_sal
                np.save((self.save_path + self.im_path.split('/')[-1].split('.')[0]+'.npy'), best_sal)
                plt.figure(); plt.axis('off'); plt.imshow(best_sal,cmap='jet')
                plt.savefig(self.save_path + self.im_path.split('/')[-1])
        return all_maps_dict
   
if __name__ == '__main__':
    IASSA()