# -*- coding: utf-8 -*-
import torch
import numpy as np
import os
import random
import argparse
from functools import partial

from src.Models import Unet
from src.Train import Trainer
from src.Evaluation import Evaluation
from src.Utils import FocalLoss
from torch.optim.lr_scheduler import StepLR
import albumentations as A
import segmentation_models_pytorch as smp


if __name__ == "__main__":
    
    ## weight 초기화
    weights = torch.tensor([100,1], dtype=torch.float32)
    weights = weights / weights.sum() ## weights = [100/101, 1/101]
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',  help='train')
    parser.add_argument('--eval', help='evaluation')
    parser.add_argument('--task', help='damage vs part')
    parser.add_argument('--method', help='multi : multi models, single : single model')
    parser.add_argument('--label', help="all, 2,3,4,5")
    parser.add_argument('--cls', type = int, help = "n_class")
    parser.add_argument('--dataset', help = "val vs test")
    parser.add_argument('--weight_file', help = "weight file name")
    
    arg = parser.parse_args()
    ## 입력 예시) python main.py --train 1 --task part --cls 16
    
    # model
    
    if arg.task == 'damage':
        n_cls = 1 # 2, 4?
    elif arg.eval:
        n_cls = 16
    else:
        n_cls = arg.cls  ## arg.part == 'part'인 경우. 현재로서는 16이 들어가게 되는데 분류하려는 클래스 수로 맞춰야 됨
    print('gpu device num')    
    print(torch.cuda.current_device())

    model = Unet(encoder="resnet34",pre_weight='imagenet',num_classes=n_cls)
    
    # model = Unet(
    #              encoder_name='resnet34',
    #              encoder_weights='imagenet',
    #              in_channels=3,
    #              out_channels=1, # out_classes
    #              activation='sigmoid'
    #             )

    # model = Unet()


    # set seed
    def set_seed(seed:int):
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)  # type: ignore
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = True  # type: ignore
    set_seed(1230)
    
    
    # model load
    def load_model(model, weight_path, strict):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        try:
            model.model.load_state_dict(torch.load(weight_path, map_location=torch.device(device)),strict=strict)
            return model
        except:
            model.load_state_dict(torch.load(weight_path, map_location=torch.device(device)),strict=strict)
            return model
    
    
    # train
    if arg.train:
        # train damage
        if (arg.label == "all") & (arg.task == "damage"):
            # label_schme = ["Scratched","Separated","Crushed","Breakage"]
            # csv column 순서랑 맞추기
            label_schme = ["Scratched", "Breakage", "Separated", "Crushed"]
            epochs = [1,8,5,9]  ## 각 label에 대한 epoch. label마다 epoch 횟수를 달리 하고 있다. 현재는 사용하지 않음
                                ## 11/26) 데이터 수 차이로 인해 epoch 수를 달리한 듯 

            ## label에 따라 별도로 학습 진행
            for i in range(4):
                trainer = Trainer(
                            ails = f"{arg.task}",
                            train_dir = f"../data/datainfo/{arg.task}_{label_schme[i]}_train.json",
                            val_dir = f"../data/datainfo/{arg.task}_val.json",
                            img_base_path = '../data/Dataset/1.원천데이터/damage',
                            ## img_base_path = '../data/Sample/1.원천데이터/damage',
                            size = 256,
                            model = model,
                            label = i,
                            n_class = n_cls,
                            optimizer = torch.optim.Adam,
                            # criterion = torch.nn.CrossEntropyLoss(),
                            # criterion = torch.nn.BCEWithLogitsLoss(),
                            # criterion = torch.nn.MSELoss(),
                            # criterion = FocalLoss(),
                            criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True),
                            # epochs = epochs[i],
                            epochs = 10,
                            batch_size = 16, # 64
                            encoder_lr = 1e-06,
                            decoder_lr = 3e-04,
                            weight_decay = 0,
                            device = "cuda")
                trainer.train()
        # train damage one label
        elif (arg.task == 'demage') & (arg.method == 'multi'):
            trainer = Trainer(
                        ails = f"{arg.task}_label{arg.label}",
                        train_dir = f"../data/datainfo/{arg.task}_trainsample.json",
                        val_dir = f"../data/datainfo/{arg.task}_valsample.json",
                        img_base_path = '../data/Dataset/1.원천데이터/damage_part',
                        size = 256,
                        model = model,
                        label = arg.label,
                        n_class = n_cls,
                        optimizer = torch.optim.Adam,
                        criterion = torch.nn.CrossEntropyLoss(),
                        epochs = 70,
                        batch_size = 32,
                        encoder_lr = 1e-07,
                        decoder_lr = 1e-06,
                        weight_decay = 0,
                        device = "cuda")
            trainer.train()
        
        # train part_ver2
        else:

            transform = A.Compose([
                            A.RandomRotate90(p=0.3),
                            A.Cutout(p=0.3,max_h_size=32,max_w_size=32),
                            A.Resize(256,256)])       

            scheduler = partial(StepLR, step_size=10, gamma=0.9)
            trainer = Trainer(
                ails = f"{arg.task}",
                train_dir = f"../data/datainfo/{arg.task}_train.json",
                val_dir = f"../data/datainfo/{arg.task}_val.json",
                img_base_path = '../data/Dataset/1.원천데이터/damage_part',
                ## img_base_path = '../data/Sample/1.원천데이터/damage_part',
                size = 256,
                model = model,
                label = None,
                n_class = arg.cls,
                optimizer = torch.optim.Adam,
                criterion = torch.nn.CrossEntropyLoss(),
                epochs = 20, ## 57
                batch_size = 32,
                encoder_lr = 1e-06,
                decoder_lr = 3e-04,
                weight_decay = 1e-02,
                device = "cuda",
                transform = transform,
                lr_scheduler = scheduler,
                start_epoch = None)
            trainer.train()
            
    
    if arg.eval:
        set_seed(12)
        # evaluation
        if arg.task == 'damage':
            evaluation = Evaluation(
                        eval_dir = f"../data/datainfo/damage_{arg.dataset}.json",
                        size = 256, 
                        model = model, 
                        weight_paths = ["../data/weight/"+n for n in ["[DAMAGE][Scratch_0]Unet.pt","[DAMAGE][Seperated_1]Unet.pt","[DAMAGE][Crushed_2]Unet.pt","[DAMAGE][Breakage_3]Unet.pt"]],
                        device = 'cuda',
                        batch_size = 64, 
                        ails = f"../data/result_log/[{arg.task}]_{arg.dataset}_evaluation_log.json",
                        criterion = torch.nn.CrossEntropyLoss(),
                        img_base_path = "../data/Dataset/1.원천데이터/damage"
                        ## img_base_path = "../data/Sample/1.원천데이터/damage"
            )
            evaluation.evaluation()
        else:
            model = Unet(encoder="resnet34",pre_weight='imagenet',num_classes=n_cls)
            if arg.weight_file:
                weight_path = f"../data/weight/{arg.weight_file}"
            else:
                weight_path = "../data/weight/[PART]Unet.pt"
            evaluation = Evaluation(
                        eval_dir = f"../data/datainfo/part_{arg.dataset}.json",
                        size = 256, 
                        model = model, 
                        weight_paths = [weight_path],
                        device = 'cuda',
                        batch_size = 64, 
                        ails = f"../data/result_log/[{arg.task}]_{arg.dataset}_evaluation_log.json",
                        criterion = torch.nn.CrossEntropyLoss(),
                        img_base_path = "../data/Dataset/1.원천데이터/damage_part"
                        ## img_base_path = "../data/Sample/1.원천데이터/damage_part"
            )
            evaluation.evaluation()

