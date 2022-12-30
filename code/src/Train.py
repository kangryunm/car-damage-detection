import datetime
import os
import random
from pytz import timezone
import json

import torch
from torch.optim import Adam
import cv2
import numpy as np

import src.Models as models
from src.Datasets import Datasets
from src.Utils import label_accuracy_score,add_hist
from torch.utils.data import DataLoader


class Trainer():
    def __init__(self, 
                train_dir, val_dir, size, label,
                model, n_class, criterion, optimizer, device,
                epochs, batch_size, encoder_lr, decoder_lr, weight_decay, ails, img_base_path=None, transform=None, lr_scheduler=None, start_epoch=None ):
        
        self.model = model.model # for old model
        # self.model = model # for new model
        
        self.n_class = n_class
        self.epochs = epochs
        self.batch_size = batch_size

        self.label = label  # i로 고정 ! 
        self.one_channel = False if label is None else True  ## task = 'part'이면 None 
        self.train_dataset = Datasets(train_dir, 'train', size = size, label = label, one_channel = self.one_channel, img_base_path=img_base_path, transform=transform)
        ## train_dir: data_info의 json 파일
        ## size = 256
        ## label: i (label_schme의 지정 index)

        self.val_dataset = Datasets(val_dir, 'train', size = size, label = label, one_channel = self.one_channel, img_base_path=img_base_path)
        
        # img_id = np.random.choice(self.val_dataset.img_ids, int(len(self.val_dataset)*0.003), replace = False)
        # self.sample_val_id = [f['file_name'] for f in self.val_dataset.coco.loadImgs(img_id)]

        self.device = device
        self.criterion = criterion 

        self.optimizer = optimizer([
                                    {'params': self.model.encoder.parameters()},
                                    {'params': self.model.decoder.parameters(), 'lr':decoder_lr}
                                    ], lr = encoder_lr, weight_decay = weight_decay)
        
        if lr_scheduler:
            self.lr_scheduler = lr_scheduler(optimizer=self.optimizer)
        else:
            self.lr_scheduler = False
        
            
        
        self.ails = ails # damage_{l}_train
        
        if self.one_channel:  ## damage
            self.log = {
                    "comand" : "python main.py --train train --task damage --label all",
                    "start_at_kst": 1,
                    "end_at_kst": 1,
                    "train_log": []
                }
        else:  ## part
            categories = {0:{'id':0, 'name':'Background'}}
            categories.update(self.train_dataset.coco.cats)
            ## (cls-1) 만큼 딕셔너리를 읽어온다. 0은 Background이기 때문. 마지막 index의 name은 'etc'로 고정되는 듯
            ## { 1: {'id': 1, 'name': 'Front bumper'}, ... , 15: {'id': 15, 'name': 'etc'} }
            ## print(self.train_dataset.coco.cats)
            self.log = {
                        "comand" : "python main.py --train train --task part --cls 16",
                        "start_at_kst": 1,
                        "end_at_kst": 1,
                        "train_log": [],
                        "category" : categories
                    }

        self.logging_step = 0
        
        if start_epoch:
            self.start_epoch = start_epoch
        else:
            self.start_epoch = 0

    def get_dataloader(self):
        def collate_fn(batch):
            return tuple(zip(*batch))
        
        train_loader = DataLoader(
            dataset = self.train_dataset,
            shuffle = True,
            num_workers = 4,
            collate_fn = collate_fn,
            batch_size = self.batch_size)
            
        val_loader = DataLoader(
            dataset = self.val_dataset,
            shuffle = False, 
            num_workers = 4,
            collate_fn = collate_fn,
            batch_size = self.batch_size)

        
        return train_loader, val_loader

    
    def train(self):
        print(f'--- start-training ---')
        now = datetime.datetime.now(timezone('Asia/Seoul'))
        start_time = now.strftime('%Y-%m-%d %H:%M:%S %Z%z')
        self.log['start_at_kst'] = start_time

        train_data_loader, self.val_data_loader = self.get_dataloader()
        self.model.to(self.device)

        best_loss = 999999999
        best_mIoU = 0.0
        best_cls_IoU = 0.0

        ## Load되는 데이터 개수 확인
        print(f'Loaded train data: {len(train_data_loader.dataset)}')
        print(f'Loaded val data: {len(self.val_data_loader.dataset)}')

        for epoch in range(self.epochs):
            epoch += self.start_epoch
            print(f'epoch: {epoch+1}')    
            self.model.train()

            # loging
            train_losses = []
    
            for step, (images, masks, _) in enumerate(train_data_loader):
                
                # tuple -> np array -> tensor
                images = torch.tensor(np.array(images)).float().to(self.device)
                # binary mask인 (B, 256, 256)에서 dimension을 추가한다. (model output과 loss 연산 위해)
                masks = torch.tensor(np.expand_dims(np.array(masks), axis = 1)).float().to(self.device)
                
                # inputs : (B, 3, 256, 256)
                # outputs : (B, 1, 256, 256) = (batch, number of classes, h, w) / pixelwise 확률분포
                # masks : (B, 1, 256, 256) / 0 = background, 1 = target class
                outputs = self.model(images)
              
                # calculate loss
                loss = self.criterion(outputs, masks)
                # mask : train data loader를 통해서 나온 train data masked - target
                # outputs : predicted

                # backprob
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                
                # loging
                if step % 10 == 0:
                    print(f"step {step} | loss {loss.item()}")
                    train_losses.append(loss.item())
            
            # lr_scheduler
            if self.lr_scheduler:
                self.lr_scheduler.step()

            # if epoch % 2 == 0:
                
            # logging
            base_form = {
                        "epoch": epoch+1,
                        "train_loss": [],
                        "eval": {
                            "img": [],
                            "summary": {
                                "Imou": 0.4,
                                "Average Loss": 0.2,
                                "background IoU": 0.9,
                                "target IoU": 0.3,
                                "end_at_kst" : 0
                            }
                        }
                    }
            self.log['train_log'].append(base_form)
            self.log["train_log"][self.logging_step]['train_loss'] = train_losses
            # train_losses : 한 번의 epoch에서의 train loss list

            # validation : 아래에 있는 def validation
            avrg_loss, mIoU, cls_IoU= self.validation(epoch+1, step, self.val_data_loader)
            print('avrg_loss, mIoU, cls_IoU :',avrg_loss, mIoU, cls_IoU)

            # best 경우 기록
            self.logging_step += 1
            # if (best_mIoU < mIoU):
            if best_cls_IoU < cls_IoU:
                if self.one_channel:
                    save_file_name = f"../data/weight/Unet_{self.ails}_label{self.label}_start:{start_time}_{epoch+1}_epoch_IoU_{float(cls_IoU[1]*100):.1}"
                    save_log_name = f"../data/result_log/[{self.ails}_label{self.label}]train_log.json"
                else:
                    save_file_name = f"../data/weight/Unet_{self.ails}_start:{start_time}_{epoch+1}_epoch_IoU_{float(mIoU):.1}"
                    save_log_name = f"../data/result_log/[{self.ails}]train_log.json"
                self.save_model(save_file_name) 
                # save model을 안하면 weight에 아무것도 저장이 안되어서 evaluate를 단독으로 돌릴 수가 없어서
                best_mIoU = mIoU
                best_cls_IoU = cls_IoU


    def save_model(self, file_name):
        # check_point = {'net': self.model.state_dict()}
        file_name = file_name + '.pt'
        # output_path = os.path.join('weight', file_name)
        # torch.save(self.model.state_dict(), file_name)
        print('MODEL SAVED!!')
        

    def validation(self, epoch, step, data_loader):
        n_class = self.n_class
        print(f'number of classes: {n_class}')
        print('Start validation # epoch {} # step {}'.format(epoch,step))
        self.model.eval()

        with torch.no_grad():
            total_loss = 0
            cnt = 0
            hist = np.zeros((2,2))

            mIoU_list = []
            for step, (images, masks, img_ids) in enumerate(data_loader):
                # 한 번 for문 -- torch.Size([16, 1, 256, 256]) 

                images = torch.tensor(np.array(images)).float().to(self.device)
                masks = torch.tensor(np.expand_dims(np.array(masks), axis = 1)).float().to(self.device)
                outputs = self.model(images)
                
                loss = self.criterion(outputs, masks)
                total_loss += loss
                cnt += 1
                
                # output postprocessing
                # outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                # masks = masks.detach().cpu().numpy()
                # print(img_ids) # 0507431_as-3128259.jpg 이런 파일 16(Batch size)개 tuple
                
                ## batch 횟수만큼 iterate
                for i, img_id in enumerate(img_ids):
                    # 한 번 for문 -- torch.Size([1, 256, 256])

                    h = np.zeros((2,2))
                    
                    h += add_hist(h, masks[i], outputs[i], n_class=n_class)
                    acc, acc_cls, mIoU, fwavacc, cls_IoU, precision, recall, F1_score, balanced_acc = label_accuracy_score(h)
                    
                    tmp = {"img_id": img_id,
                            "IoU" : list(cls_IoU)}
                    
                    # sample_logging
                    self.log["train_log"][self.logging_step]['eval']['img'].append(tmp)
                    

                hist += add_hist(hist, masks, outputs, n_class=n_class)

            print(hist)
            acc, acc_cls, mIoU, fwavacc, cls_IoU, precision, recall, F1_score, balanced_acc = label_accuracy_score(hist)
            avrg_loss = total_loss / cnt
         
            
            if self.one_channel: ## damage
                # logging
                now = datetime.datetime.now(timezone('Asia/Seoul'))
                end_time = now.strftime('%Y-%m-%d %H:%M:%S %Z%z')
                tmp = {"mIoU": mIoU.item(),
                        "average Loss" : avrg_loss.item(), 
                        "background IoU" : cls_IoU[0].item(),
                        "target IoU" : cls_IoU[1].item(),
                        "end_at_kst" : end_time}
                
                self.log["end_at_kst"] = end_time
                self.log["train_log"][self.logging_step]['eval']['summary'] = tmp
                
                # print(avrg_loss, mIoU, cls_IoU[0], cls_IoU[1], precision, recall, F1_score, balanced_acc) 
                # 이렇게 프린트해보면 precision, recall 이런 것들이 element 2개의 list로 나와서 이거 cls_IoU[1]처럼 지정해줘야 0

                message='Validation #{} #{} Average Loss: {:.4f}, mIoU: {:.4f}, background IoU : {:.4f}, target IoU : {:.4f}, precision : {:.4f}, recall : {:.4f}, F1_score : {:.4f}, balanced_accuracy : {:.4f}'.format(epoch, step, avrg_loss, mIoU, cls_IoU[0], cls_IoU[1], precision[1], recall[1], F1_score[1], balanced_acc)
                # message='Validation #{} #{} Average Loss: {:.4f}, mIoU: {:.4f}, background IoU : {:.4f}, target IoU : {:.4f}'.format(epoch, step, avrg_loss, mIoU, cls_IoU[0], cls_IoU[1] )
            else: ## part
                message='Validation #{} #{} Average Loss: {:.4f}, mIoU: {:.4f}, background IoU : {:.4f}, target IoU : {:.4f}, precision : {:.4f}, recall : {:.4f}, F1_score : {:.4f}, balanced_accuracy : {:.4f}'.format(epoch, step, avrg_loss, mIoU, cls_IoU[0], cls_IoU[1], precision[1], recall[1], F1_score[1], balanced_acc)
                # message='Validation #{} #{} Average Loss: {:.4f}, mIoU: {:.4f}, background IoU : {:.4f}, target 1IoU : {:.4f}, target 2IoU : {:.4f}, target 3IoU : {:.4f}, target 3IoU : {:.4f}'.format(epoch, step, avrg_loss, mIoU, cls_IoU[0], cls_IoU[1], cls_IoU[2], cls_IoU[3], cls_IoU[4] )
                now = datetime.datetime.now(timezone('Asia/Seoul'))
                end_time = now.strftime('%Y-%m-%d %H:%M:%S %Z%z')
                tmp = {"mIoU": mIoU.item(),
                        "average Loss" : avrg_loss.item(), 
                        "background IoU" : cls_IoU[0].item(),
                        "target IoU" : list(cls_IoU[1:]),
                        "end_at_kst" : end_time}
                
                self.log["end_at_kst"] = end_time
                self.log["train_log"][self.logging_step]['eval']['summary'] = tmp

            print(message)
            
            if self.one_channel:
                save_log_name = f"../data/result_log/[{self.ails}_label{self.label}]train_log.json"
            else:
                save_log_name = f"../data/result_log/[{self.ails}]train_log.json"

            with open( f"{save_log_name}", "w" ) as f:
                json.dump(self.log,f)

        return avrg_loss, mIoU, cls_IoU