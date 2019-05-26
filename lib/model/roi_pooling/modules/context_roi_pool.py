#############################################################
# Edited by fangfang 2019/3/4
# context_roi_pool.py
#############################################################
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torchvision.models as models
from torch.nn.modules.module import Module
from ..functions.roi_pool import RoIPoolFunction
from model.rpn.bbox_transform import clip_boxes, bbox_overlaps_batch, bbox_transform_batch
from model.rpn.anchor_target_layer import _AnchorTargetLayer
#from model.faster_rcnn.faster_rcnn import resnet
from model.rpn.proposal_layer import _ProposalLayer

import ipdb
def context_anchor(rois,features,hh,hw):
   # topleft =     [2*x1 - x2, 2*y1 y2, x1,        y1]
   # top =         [x1,        2*y1 - y2, x2,        y1]
   # topright =    [x2,        2*y1 - y2, 2*x2 - x1, y1]
   # left =        [2*x1 - x2, y1,        x1,        y2]
   # right =       [x2,        y1,        2*x2 - x1, y2]
   # bottomleft =  [2*x1 - x2, y2,        x1,        2*y2 - y1]
   # bottom =      [x1,        y2,        x2,        2*y2 - y1]
   # bottomright = [x2,        y2,        2*x2 - x1, 2*y2 - y1]
    """
    rois[:,0],rois[:,1],rois[:,2],rois[:,3]
    x1,       y1,       x2,       y2

    """
    batch_size = features.size(0)
    num_channels = features.size(1)
    H = features.size(2)
    W = features.size(3)
    #rois:[128,5]
    x1 = rois[:,1].cpu().numpy().reshape(rois.size(0),1)#[128,1]
    y1 = rois[:,2].cpu().numpy().reshape(rois.size(0),1)
    x2 = rois[:,3].cpu().numpy().reshape(rois.size(0),1)
    y2 = rois[:,4].cpu().numpy().reshape(rois.size(0),1)
    _w = (x2 - x1)#[128,1]
    _h = (y2 - y1)#[128,1]
    # cell center  
    shift_x = (x1 - _w) + _w  * np.arange(0,3)+_w/2#[128,3]
    shift_y = (y1 - _h) + _h  * np.arange(0,3)+_h/2
    offset = torch.from_numpy(np.hstack((-_w/4,-_h/4,_w/4,_h/4)))#[128,4]
    offset = offset.type_as(rois).float()

    #rois[0] anchor
    shift_xx,shift_yy = np.meshgrid(shift_x[0],shift_y[0])#[3,3]
    
    offset0 = offset[0]#[1,4]
    shifts0 = torch.from_numpy(np.vstack((shift_xx.ravel(),shift_yy.ravel(),
                                    shift_xx.ravel(),shift_yy.ravel())).transpose())#[4,9]----->[9,4]
    shifts0 = shifts0.contiguous().type_as(rois).float()
    gt =offset0 + shifts0#[9,4]
    
    import ipdb
    #ipdb.set_trace()
    ww = gt[:,2]-gt[:,0]+1
    hhh = gt[:,3]-gt[:,1]+1
    min_size = 16

    keep = ((gt[:,0]<0) |
            (gt[:,1]<0) |
            (gt[:,2]>=hw)|
            (gt[:,3]>=hh)|
            (ww<min_size)|
            (hhh<min_size))
    if torch.sum(keep) >0:
        gt[keep] = rois[0][1:5]
    
    gt = torch.cat((gt[:4,:],gt[5:,:]),0)
    
    #gt = np.delete(gt,4,0)#[8,4]
    A = rois.size(0)#128
    K = gt.shape[0]#8
    for i in range(1,rois.size(0)):
        shift_xx,shift_yy = np.meshgrid(shift_x[i],shift_y[i])#[3,3]
       
        shifts =  torch.from_numpy(np.vstack((shift_xx.ravel(),shift_yy.ravel(),\
                                        shift_xx.ravel(),shift_yy.ravel())).transpose())#[4,9]----->[9,4]
        shifts = shifts.contiguous().type_as(rois).float()
        gti = offset[i] + shifts#[9,4]
        ww = gti[:,2]-gti[:,0]+1
        hhh = gti[:,3]-gti[:,1]+1
 
        keep = ((gti[:,0]<0) |
                (gti[:,1]<0) |
                (gti[:,2]>=hw)|
                (gti[:,3]>=hh)|
                (ww<min_size)|
                (hhh<min_size))
        if torch.sum(keep)>0:
            gti[keep] = rois[i][1:5]
        gti = torch.cat((gti[:4,:],gti[5:,:]),0)
        gt = torch.cat((gt,gti),0)#[1024,4]
   
    gt = gt.view(batch_size,-1,4)#[1,1024,4]
    # check if surpass the bound

    all_anchors = rois[:,1:5]#[128,5]]
    total_anchors = int(K*A)#128*9=1024
   
    overlaps = bbox_overlaps_batch(all_anchors,gt)#[1,128,1024]
    max_overlaps, argmax_overlaps = torch.max(overlaps,2)#[1,128]
    gt_max_overlaps,_ = torch.max(overlaps,1)#[1,1024]

    inds_inside = torch.zeros(gt.size(1)).view(-1)#[1024,1]
    labels = gt.new(batch_size, inds_inside.size(0)).fill_(0)#[1,1024]
    gt_max_overlaps[gt_max_overlaps==0] = 1e-5
    keep = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size,1,-1).expand_as(overlaps)), 2)
    
    width_cell = gt[:,:,2]-gt[:,:,0]
    height_cell = gt[:,:,3]- gt[:,:,1]
    max_cell = np.maximum(width_cell,height_cell)    
    min_cell = np.minimum(width_cell,height_cell)
    # intersect with rois >0.3 
    # width height cell
    
    labels[gt_max_overlaps >= 0.3] = 1

    import ipdb
#    ipdb.set_trace()
    if torch.sum(labels==1) > 0:
        width = gt.new(batch_size,inds_inside.size(0)).fill_(0)
        height = gt.new(batch_size,inds_inside.size(0)).fill_(0)
        width[labels==1] = rois[_[labels==1]][:,3]-rois[_[labels==1]][:,1]
        height[labels == 1] = rois[_[labels==1]][:,4]-rois[_[labels == 1]][:,2]
        max = np.maximum(width,height)
        min = np.minimum(width,height)
        labels[max>=max_cell]=0
        labels[min< 1/3*min_cell] = 0
    if torch.sum(labels==1) >0:
        gt[labels == 1] = rois[_[labels == 1]][:,1:5]
    labels[:]=0
    #gt = np.insert(gt,0,values = labels,axis = 2).view(-1,5)
    gt = torch.cat((labels.view(batch_size,-1,1),gt),-1)
    return gt

class _RoIPooling(Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale):
        super(_RoIPooling, self).__init__()

        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)
    def forward(self, features, rois, hh,hw):#300,5
        """
        Train:
            rois = [batch,128,5]

        Test:
            rois = [batch,300,5]

        """
        gt_anchors = context_anchor(rois,features,hh,hw).view(-1,8,5)
        cat_rois =  torch.cat([rois.view(-1,1,5),gt_anchors],1).view(-1,5)
        cat_rois = cat_rois.contiguous().type_as(rois) 
        pool_feat = RoIPoolFunction(self.pooled_height, self.pooled_width, self.spatial_scale)(features,cat_rois).contiguous().view(rois.size(0),-1,self.pooled_width,self.pooled_height)
        return pool_feat,cat_rois 
