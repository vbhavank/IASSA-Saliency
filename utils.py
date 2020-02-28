import numpy as np
from matplotlib import pyplot as plt
import cv2
import os

        
def get_class_name(c):
    """
    Given label number returns class name 
    """
    labels = np.loadtxt('synset_words.txt', str, delimiter='\t')
    return ' '.join(labels[c].split(',')[0].split()[1:])

def mask_sal(sal, img, thre, net_in_size):
    """
    Threshold the saliency map at threshold thre.
    """
    
    im = cv2.resize(cv2.cvtColor(img,cv2.COLOR_BGR2RGB), (net_in_size, net_in_size))
    masks = np.empty([im.shape[0], im.shape[1]], dtype = bool)
    masks[sal > thre] = 1
    masks[sal <= thre] = 0
    return masks

def pointing_game_score(m, sal):
    """
    Computes the pointing game score as either 0-1. 
    """
    
    return m(sal.index(max(sal)))
                        
def find_intersection(masks, region):
    """
    Select sampling masks for next iteration
    """    
    
    masked_masks = []
    for mask in masks:
        intersection = np.max(np.multiply((mask.squeeze().cpu().numpy()), (region)))
        if intersection > 0:
            masked_masks.append(mask)
    return masked_masks

def get_iou_(m,sal):
    """
    Computes Intersection-Over-Union between m and sal
    """

    intersection = np.logical_and(m, sal)
    union = np.logical_or(m, sal)
    return np.sum(intersection) / np.sum(union)