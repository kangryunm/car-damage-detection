import os
import json
import glob
import json
import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Optional
from collections import Counter

# data_dir 받아서 coco format 수정
class RemakeCOCOformat():
    def __init__(self, img_dir, ann_dir, data_lst=None, n_sample=None, alis=None, ratio=0.05, labeling_schme=None, task=None):
        self.base_img_path = img_dir
        self.base_label_path = ann_dir
        self.images = glob.glob(os.path.join(self.base_img_path, r"*.jpg"))
        self.annotations = glob.glob(os.path.join(self.base_label_path ,r"*.json"))
        
        ## label 리스트
        self.labeling_schme = labeling_schme
        ## task == 'parts' -> 정의한 파손 부위 리스트 ['front bumper', ... ]
        ## task == 'damage' -> 정의한 파손 종류 리스트 ['Scratch', ... ]

        self.task = task
        
        self.img_id = 0
        self.ann_id = 0

        self.ratio = ratio
        
        if data_lst:
            self.images = [ os.path.join(self.base_img_path,f.replace('.json', '.jpg')) for f in data_lst ]
            self.annotations = [ os.path.join(self.base_label_path,f.replace('.jpg', '.json')) for f in data_lst ]
            self.train_fn = alis
            
        if n_sample:
            self.n_sample = n_sample

    def load_json(self, file_name):
        with open(file_name, "r") as f:
            ann = json.load(f)
        return ann

    def save_json(self, file, file_name):
        with open(file_name, "w") as f:
            json.dump(file, f)

    ## annotation 형식 바꾸는 함수   
    def rebuilding(self, d, img_lst):
        for i in img_lst:
            self.img_id += 1  # 사진마다 'images'의 'id' 1씩 증가
            ann = self.load_json(i)

            ann['images']['id'] = self.img_id
            img_info = ann['images']  # id, width, height, file_name
            ann_info = ann['annotations']  # id, image_id, category_id, segmentation, ...  

            d['images'].append(img_info)

            for a in ann_info:
                if a[self.task] != '':  # all / part / damage
                    self.ann_id += 1  # 'annotations' 'id' 1씩 증가
                    a['id'] = self.ann_id
                    a['image_id'] = self.img_id  # 동일 사진에 대한 ann끼리는 image_id 동일 
                    
                    if self.labeling_schme:
                        ## if a[self.task] in self.labeling_schme:
                        if len(self.labeling_schme) == 4:  ## damage
                            
                            # 문제: ann_info에 여러 annotation이 있기 때문에 a['damage']가 원하는 damage가 아닐수도 있다. 또 null인 경우는?
                            # 일단 null인 경우만 제외해서 그대로 다 가져가고 나중에 category_id로 선별해서 학습 진행하자. Sample 데이터 중에서는 null인 경우가 없는 듯하다.
                            if a['damage']:
                              a['category_id'] = self.labeling_schme.index(a['damage'])  # index() : 지정한 element의 index를 반환
                            else: # null
                              print('damage is null')
                              a['category_id'] = -1

                        else:  ## part
                            # a['category_id'] = len(self.labeling_schme) # 항상 else로 넘어가서 리스트의 길이인 14가 나왔다
                            try:
                              a['category_id'] = self.labeling_schme.index(a['part'])
                            except:
                              continue

                    ### adjust format of a['segmentation'] ###
                    raw_seg_list = a['segmentation'][0][0]
                    processed_seg_list = []
                    for raw_seg in raw_seg_list:
                        x = raw_seg[0]
                        y = raw_seg[1] / pow( 10, len(str(raw_seg[1])) )
                        processed_seg = x + y
                        processed_seg_list.append(processed_seg)

                    ## len = 4 일때 바운딩 박스로 인식되어 numpy.ndarray를 요구하는 frPyobjects 때문에 임의의 점 추가
                    ## segmentation의 좌표들은 시작점과 끝점이 동일하다. 즉 len=4이면 실제로는 삼각형인 셈이다. 따라서 마지막 점을 변경하면 안 된다.
                    if len(processed_seg_list) == 4:
                        additional_point_x = raw_seg_list[2][0] + 1
                        additional_point_y = (raw_seg_list[2][1] + 1) / pow( 10, len(str(raw_seg_list[2][1])) )
                        additional_point = additional_point_x + additional_point_y
                        processed_seg_list.append(additional_point)

                    a['segmentation'] = [processed_seg_list]
                    d['annotations'].append(a)
          
        print(d)

        return d
            
    
    def coco_json(self):
        train = self.load_json(self.annotations[0])
        train['images'] = []
        train['annotations'] = []
                

        if self.labeling_schme:
            cates = [{"id":i+1, "name":v}for i,v in enumerate(self.labeling_schme)]
            # [{'id': 1, 'name': 'Scratched'}, {'id': 2, 'name': 'Separated'}, {'id': 3, 'name': 'Crushed'}, {'id': 4, 'name': 'Breakage'}]
            if self.task == 'part':
                cates.append({"id":len(self.labeling_schme)+1, "name":'etc'})
            train['categories']= cates
                

        
        train_imgs = [] 

        for i in self.annotations:
            ann = self.load_json(i)
            ann_info = ann['annotations']

            if len(ann_info) != 0:
                train_imgs.append(i)
        

      
        train = self.rebuilding(train, train_imgs)
        print(len(train['images'])) 
        
        if not os.path.exists("data/datainfo"):
            os.makedirs("data/datainfo")
            
        if not os.path.exists("data/result_log"):
            os.makedirs("data/result_log")
            
        if not os.path.exists("data/weight"):
            os.makedirs("data/weight")

        if not os.path.exists("data/Dataset/1.원천데이터/damage"):
            os.makedirs("data/Dataset/1.원천데이터/damage")
        
        if not os.path.exists("data/Dataset/1.원천데이터/damage_part"):
            os.makedirs("data/Dataset/1.원천데이터/damage_part")
        
        if not os.path.exists("data/Dataset/2.라벨링데이터/damage"):
            os.makedirs("data/Dataset/2.라벨링데이터/damage")
        
        if not os.path.exists("data/Dataset/2.라벨링데이터/damage_part"):
            os.makedirs("data/Dataset/2.라벨링데이터/damage_part")
        
        self.save_json(train, os.path.join("data/datainfo" ,self.train_fn + ".json"))



def label_split(data_dir):
    annotations = glob.glob(os.path.join(data_dir, r"*.json"))

    def load_json(file_name):
        with open(file_name, "r") as f:
            ann = json.load(f)
        return ann
    
    label_schme = {
    1:{"files":[],"label_info":'스크래치'},
    2:{"files":[],"label_info":'파손'},
    3:{"files":[],"label_info":'찌그러짐'},
    4:{"files":[],"label_info":'이격'},    
    }

    for ann in annotations:
        parse = load_json(ann)
        for a in parse['annotations']:
            label_schme[a['category_id']]['files'].append(ann)
    
    for i in label_schme:
        label_schme[i]['files'] = np.random.choice(list(set(label_schme[i]['files'])), 10, replace = False)
    
    for i in label_schme:
        coco = RemakeCOCOformat('rst', data_lst=label_schme[i]['files'], alis = f"_label{i}")
        coco.coco_json()

    return label_schme
        


# def label_accuracy_score(hist):
#     """
#     Returns accuracy score evaluation result.
#       - [acc]: overall accuracy
#       - [acc_cls]: mean accuracy
#       - [mean_iu]: mean IU
#       - [fwavacc]: fwavacc
#     """
#     acc = np.diag(hist).sum() / hist.sum() # 정확도
#     with np.errstate(divide='ignore', invalid='ignore'):
#         acc_cls = np.diag(hist) / hist.sum(axis=1) # class별 정확도
#     acc_cls = np.nanmean(acc_cls) # np.nanmean : nan 무시하고 산술평균
#     # acc_cls : class별 accuracy의 평균

#     with np.errstate(divide='ignore', invalid='ignore'):
#         iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
#         # iu : class별 IoU
#     if sum(np.isnan(iu)) == len(iu): # IoU값이 전부 nan인 경우 
#         mean_iu = np.mean([0,0])
#     else:
#         mean_iu = np.nanmean(iu) # class별 IoU값의 평균 = mIoU
    
#     # add class iu
#     cls_iu = iu
#     cls_iu[np.isnan(cls_iu)] = -1 # class별 IoU 중 nan값이 있는 경우 : -1
#     # IoU가 nan값인 경우 : 분모가 0 
#     freq = hist.sum(axis=1) / hist.sum() # class별 실측 개수/전체개수
#     fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

#     return acc, acc_cls, mean_iu, fwavacc, cls_iu


# 수정
def label_accuracy_score(hist):
    """
    Returns accuracy score evaluation result.
      - [acc]: overall accuracy
      - [acc_cls]: mean accuracy
      - [mean_iu]: mean IU
      - [fwavacc]: fwavacc
    """
    acc = np.diag(hist).sum() / hist.sum() # 정확도
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1) # class별 정확도
    acc_cls = np.nanmean(acc_cls) # np.nanmean : nan 무시하고 산술평균
    # acc_cls : class별 accuracy의 평균

    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        # iu : class별 IoU
    if sum(np.isnan(iu)) == len(iu): # IoU값이 전부 nan인 경우 
        mean_iu = np.mean([0,0])
    else:
        mean_iu = np.nanmean(iu) # class별 IoU값의 평균 = mIoU
    
    # add class iu
    cls_iu = iu
    cls_iu[np.isnan(cls_iu)] = -1 # class별 IoU 중 nan값이 있는 경우 : -1
    # IoU가 nan값인 경우 : 분모가 0 
    freq = hist.sum(axis=1) / hist.sum() # class별 실측 개수/전체개수
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    ## precision = TP/(TP+FP)
    with np.errstate(divide='ignore', invalid='ignore'):
      precision = np.diag(hist) / hist.sum(axis=0) 

    ## recall(sensitivity) = TP/(TP+FN)
    with np.errstate(divide='ignore', invalid='ignore'):
      recall = np.diag(hist) / hist.sum(axis=1) 

    # F1 score
    with np.errstate(divide='ignore', invalid='ignore'):
      F1_score = (2 * precision * recall) / (precision + recall)

    # balanced accuracy
    with np.errstate(divide='ignore', invalid='ignore'):
      specificity = []
      for n in range(2):
        B = hist.sum(axis=1).sum() - hist.sum(axis= 1)[n]
        T = hist.sum(axis=0).sum() - hist.sum(axis= 0)[n] - hist.sum(axis= 1)[n] + hist[n][n]
        specificity.append(B / T)
    
    balanced_acc = (recall[1] + specificity[1])/2

    return acc, acc_cls, mean_iu, fwavacc, cls_iu, precision, recall, F1_score, balanced_acc



def customizedAnnToMask(ann, image_info):  # 'segmentation': [[ [x.y], [x.y], ...  , [x.y] ]]

  coco_seg = ann['segmentation'][0]
  contours = []

  for point in coco_seg:
    point = str(point)
    x, y = point.split('.')
    revised_point = [int(x), int(y)]
    contours.append(revised_point)
  
  ann_mask = np.zeros((image_info['height'], image_info['width']))
  contours = np.asarray(contours)
  cv2.fillPoly( ann_mask, pts=[contours], color=(1,1,1) )
  
  return ann_mask


# label_true(mask) : 0 or 1
# label_pred(output) : [0,1]

# def _fast_hist(label_true, label_pred, n_class):
#     mask = (label_true >= 0) & (label_true < n_class)
#     hist = np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask],
#                         minlength=n_class ** 2).reshape(n_class, n_class)
#     return hist

def _fast_hist(label_true, label_pred, n_class):

    th = 0.5 ## 0.0
    final_layer = nn.Sigmoid() ##

    pred_np = final_layer(label_pred) ##
    # pred_np = label_pred.cpu().numpy()
    pred_np = pred_np.cpu().numpy()
    pred_np = np.where(pred_np < th, 0, 1)

    true_np = label_true.cpu().numpy()
    # true_np = final_layer(true_np)
    true_np *= 2 ##

    pred_np = pred_np.astype(np.int64)
    true_np = true_np.astype(np.int64)

    # hist = np.bincount(pred_np + true_np)
    hist = pred_np + true_np

    hist_bin = np.zeros((2,2), dtype=np.int64)
    hist_bin[0,0] = len(hist[hist == 0])
    hist_bin[0,1] = len(hist[hist == 1])
    hist_bin[1,0] = len(hist[hist == 2])
    hist_bin[1,1] = len(hist[hist == 3])
    # print('hist bin', hist_bin)
    # hist = np.bincount(pred_np + true_np).reshape(n_class, n_class)
    # hist = torch.Tensor(hist_bin)
    return hist_bin


# def add_hist(hist, label_trues, label_preds, n_class):
#     """
#         stack hist(confusion matrix)
#     """

#     for lt, lp in zip(label_trues, label_preds):
#         hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)

#     return hist

# def add_hist(hist, label_trues, label_preds, n_class):
#     hist = _fast_hist(label_trues.flatten(), label_preds.flatten(),n_class)
#     return hist

def add_hist(hist, label_trues, label_preds, n_class):
    # add_hist for original model
    # 원래의 모델에서는 outputs shape이 torch.Size([2, 256, 256]이라서
    # flatten()을 해주면 안된다.
    # print('label_preds',label_preds)
    a = label_preds.cpu().numpy()
    a = np.where(a<0, -1, np.where(a<0.5,0, np.where(a <= 1, 1, 2)))
    # print(Counter(a[0].flatten().astype(np.int64).tolist()))
    # print(Counter(a[1].flatten().astype(np.int64).tolist()))

    # print(Counter(label_preds[0].cpu().numpy().flatten().astype(np.int64).tolist()))
    # print(Counter(label_preds[1].cpu().numpy().flatten().astype(np.int64).tolist()))

    hist = _fast_hist(label_trues.flatten(), label_preds.flatten(),n_class)
    return hist

# 기존의 FocaslLoss
# class FocalLoss(nn.Module):
#     "Non weighted version of Focal Loss"
#     def __init__(self, alpha = 0.25, gamma = 2):
#         super(FocalLoss, self).__init__()
#         self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
#         self.gamma = gamma

#     def forward(self, inputs, targets):
#         BCE_loss = nn.CrossEntropyLoss()(inputs, targets)
#         targets = targets.type(torch.long)
#         at = self.alpha.gather(0, targets.data.view(-1))
#         pt = torch.exp(-BCE_loss)
#         F_loss = at*(1-pt)**self.gamma * BCE_loss
#         return F_loss.mean()


# 다시 가져온 focal loss + 관련 함수들 ##############################################################

def label_to_one_hot_label(
    labels: torch.Tensor,
    num_classes: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    eps: float = 1e-6,
    ignore_index=255,
) -> torch.Tensor:
    r"""Convert an integer label x-D tensor to a one-hot (x+1)-D tensor.

    Args:
        labels: tensor with labels of shape :math:`(N, *)`, where N is batch size.
          Each value is an integer representing correct classification.
        num_classes: number of classes in labels.
        device: the desired device of returned tensor.
        dtype: the desired data type of returned tensor.

    Returns:
        the labels in one hot tensor of shape :math:`(N, C, *)`,

    Examples:
        >>> labels = torch.LongTensor([
                [[0, 1], 
                [2, 0]]
            ])
        >>> one_hot(labels, num_classes=3)
        tensor([[[[1.0000e+00, 1.0000e-06],
                  [1.0000e-06, 1.0000e+00]],
        
                 [[1.0000e-06, 1.0000e+00],
                  [1.0000e-06, 1.0000e-06]],
        
                 [[1.0000e-06, 1.0000e-06],
                  [1.0000e+00, 1.0000e-06]]]])

    """
    shape = labels.shape
    # one hot : (B, C=ignore_index+1, H, W)
    one_hot = torch.zeros((shape[0], ignore_index+1) + shape[1:], device=device, dtype=dtype)
    
    # labels : (B, H, W)
    # labels.unsqueeze(1) : (B, C=1, H, W)
    # one_hot : (B, C=ignore_index+1, H, W)
    one_hot = one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps
    
    # ret : (B, C=num_classes, H, W)
    ret = torch.split(one_hot, [num_classes, ignore_index+1-num_classes], dim=1)[0]
    
    return ret


def focal_loss(input, target, alpha, gamma, reduction, eps, ignore_index):
    
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        input: logits tensor with shape :math:`(N, C, *)` where C = number of classes.
        target: labels tensor with shape :math:`(N, *)` where each value is :math:`0 ≤ targets[i] ≤ C−1`.
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Scalar to enforce numerical stabiliy.

    Return:
        the computed loss.

    Example:
        >>> N = 5  # num_classes
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = focal_loss(input, target, alpha=0.5, gamma=2.0, reduction='mean')
        >>> output.backward()
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) >= 2:
        raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input.shape}")

    if input.size(0) != target.size(0):
        raise ValueError(f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')

    # input : (B, C, H, W)
    n = input.size(0) # B
    
    # out_sie : (B, H, W)
    out_size = (n,) + input.size()[2:]
    
    # input : (B, C, H, W)
    # target : (B, H, W)
    if target.size()[1:] != input.size()[2:]:
        raise ValueError(f'Expected target size {out_size}, got {target.size()}')

    if not input.device == target.device:
        raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")
    
    if isinstance(alpha, float):
        pass
    elif isinstance(alpha, np.ndarray):
        alpha = torch.from_numpy(alpha)
        # alpha : (B, C, H, W)
        alpha = alpha.view(-1, len(alpha), 1, 1).expand_as(input)
    elif isinstance(alpha, torch.Tensor):
        # alpha : (B, C, H, W)
        alpha = alpha.view(-1, len(alpha), 1, 1).expand_as(input)       
        

    # compute softmax over the classes axis
    # input_soft : (B, C, H, W)
    input_soft = F.softmax(input, dim=1) + eps
    
    # create the labels one hot tensor
    # target_one_hot : (B, C, H, W)
    target_one_hot = label_to_one_hot_label(target.long(), num_classes=input.shape[1], device=input.device, dtype=input.dtype, ignore_index=ignore_index)

    # compute the actual focal loss
    weight = torch.pow(1.0 - input_soft, gamma)
    
    # alpha, weight, input_soft : (B, C, H, W)
    # focal : (B, C, H, W)
    focal = -alpha * weight * torch.log(input_soft)
    
    # loss_tmp : (B, H, W)
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)

    if reduction == 'none':
        # loss : (B, H, W)
        loss = loss_tmp
    elif reduction == 'mean':
        # loss : scalar
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        # loss : scalar
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss


class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math:

        FL(p_t) = -alpha_t(1 - p_t)^{gamma}, log(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Scalar to enforce numerical stabiliy.

    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Example:
        >>> N = 5  # num_classes
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> criterion = FocalLoss(**kwargs)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(input, target)
        >>> output.backward()
    """

    def __init__(self, alpha = 0.25, gamma = 2.0, reduction = 'mean', eps = 1e-8, ignore_index=30):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, input, target):
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.eps, self.ignore_index)

###################################################################################################

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--make_cocoformat',  help='make_cocoformat')
    parser.add_argument('--task', help = "all / damage / part")

    arg = parser.parse_args()

    if arg.make_cocoformat :
        if (arg.task == "all" or arg.task == "part"):
            # part
            print("make_cocoformat[part]")
            label_df = pd.read_csv('code/part_labeling.csv')
            ## label_df = pd.read_csv('code/part_labeling_sample.csv')

            dir_name_img = 'data/Dataset/1.원천데이터/damage_part'
            dir_name_label = 'data/Dataset/2.라벨링데이터/damage_part'
            ## dir_name_img = 'data/Sample/1.원천데이터/damage_part'
            ## dir_name_label = 'data/Sample/2.라벨링데이터/damage_part'
            
            ## 이 변수 중요할 듯! 분류할 클래스 여기서 정의하는 것 같다.
            ## 14 + background + etc = 16 해서 part에서 '--cls 16' 넣는듯
            ## l_sch = ["Front bumper","Rear bumper","Front fender(R)","Front fender(L)","Rear fender(R)","Trunk lid","Bonnet","Rear fender(L)","Rear door(R)","Head lights(R)","Head lights(L)","Front Wheel(R)","Front door(R)","Side mirror(R)"]
            
            ######
            # l_sch = [ "Front bumper","Rear bumper","Front fender(R)","Front fender(L)","Rear fender(R)","Trunk lid","Bonnet","Rear fender(L)",
            # "Rear door(R)","Head lights(R)","Head lights(L)","Front Wheel(R)","Front door(R)","Rocker panel(R)","Side mirror(R)",
            # "Rear door(L)","Front door(L)","Side mirror(L)","Front Wheel(L)","Rear lamp(L)","Rear lamp(R)","Rocker panel(L)",
            # "Rear Wheel(R)","Rear Wheel(L)"]
            ######

            
            l_sch = [ # from damage_part_columns
                      'Front bumper', 'Trunk lid', 'Bonnet', 'Head lights(L)',
                      'Rear bumper', 'Rear door(R)', 'Front door(R)', 'Rear fender(R)',
                      'Rear fender(L)', 'Rear lamp(R)', 'Side mirror(R)', 'A pillar(L)',
                      'Rear door(L)', 'Front door(L)', 'Front fender(R)', 'Front Wheel(L)',
                      'Front Wheel(R)','Rear lamp(L)', 'Front fender(L)', 'Rocker panel(R)',
                      'Head lights(R)', 'Rear Wheel(R)', 'Rear Wheel(L)', 'C pillar(L)',
                      'Side mirror(L)', 'Rocker panel(L)', 'Windshield', 'C pillar(R)',
                      'Rear windshield', 'Undercarriage', 'Roof', 'A pillar(R)', 'B pillar(R)',
                    ]


            
            for dt in ['train','val','test']:
                tmp = list(label_df.loc[label_df.dataset == dt]['img_id'])
                tmp = RemakeCOCOformat(img_dir = dir_name_img, ann_dir = dir_name_label, data_lst = tmp, alis= f'part_{dt}', ratio=0.1, labeling_schme=l_sch, task='part')
                # dt_25cls 수정 요망
                tmp.coco_json()
            print('Done part')

        if (arg.task == "all" or arg.task == "damage"):
            # damage
            print("make_cocoformat[damage]")
            label_df = pd.read_csv('/content/drive/MyDrive/Co-deep/code/damage_labeling.csv')
            ## label_df = pd.read_csv('code/damage_labeling.csv')
            ## label_df = pd.read_csv('code/damage_labeling_sample.csv')
            label_df = label_df.loc[label_df.total_anns > 0]
            print(f'Loaded: {len(label_df)}')

            idx = 0
            
            dir_name_img = 'data/Dataset/1.원천데이터/damage'
            dir_name_label = 'data/Dataset/2.라벨링데이터/damage'

            # l_sch = ["Scratched","Separated","Crushed","Breakage"]
            # csv column 순서랑 맞추어야지 category_id랑 맞는다!
            l_sch = ["Scratched", "Breakage", "Separated", "Crushed"]

            # training_data
            for l in l_sch:
                ### 여기서 파일 분류
                ## damage의 경우 damage_labeling_sample.csv의 damage 종류 중에서
                #3 (ann[annotations][part]에서 가져온 값) 0이 아닌 값들에 대한 사진들을 가져온다.
                ## label_df[l]>0 : l="Scratched"이면 Scratched 열의 값이 0보다 큰 경우
                tmp = list(label_df.loc[ (label_df['dataset'] == 'train') & (label_df[l]>0) ]['index'].values)  ## 두 조건을 만족하는 파일명을 리스트에 담음
                test = RemakeCOCOformat(img_dir = dir_name_img, ann_dir = dir_name_label, data_lst = tmp, alis=f'damage_{l}_train', ratio=0., labeling_schme=l_sch, task='damage')
                test.coco_json()

            # test, val
            ## test, val은 damage 종류와 관계 없이 동일
            for dt in ['val','test']:
                tmp = list(label_df.loc[label_df['dataset']==dt]['index'].values)
                test = RemakeCOCOformat(img_dir = dir_name_img, ann_dir = dir_name_label, data_lst = tmp, alis=f'damage_{dt}', ratio=0, labeling_schme=l_sch, task='damage')
                test.coco_json()

            print("Done damaage")


    
    