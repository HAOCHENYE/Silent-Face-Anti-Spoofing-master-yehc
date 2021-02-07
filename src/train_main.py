# -*- coding: utf-8 -*-
# @Time : 20-6-4 上午9:59
# @Author : zhuying
# @Company : Minivision
# @File : train_main.py
# @Software : PyCharm
from src.model_lib.RepVGGNet import RepVggFTNet
import torch
from torch import optim
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm
from tensorboardX import SummaryWriter


from  src.utility import WarmUpMultiStepLR
from src.utility import get_time
from src.model_lib.MultiFTNet import MultiFTNet
from src.data_io.dataset_loader import get_train_loader,get_eval_loader
from src.utility import EstimatorCV, ISDALoss

class TrainMain:
    def __init__(self, conf):
        self.conf = conf
        self.board_loss_every = conf.board_loss_every
        self.save_every = conf.save_every
        self.step = 0
        self.start_epoch = 0
        self.train_loader = get_train_loader(self.conf)
        self.eval_loader = get_eval_loader(self.conf)
        self.ISA = ISDALoss(128, 3)
    def train_model(self):
        self._init_model_param()
        self._train_eval_stage()


    def _init_model_param(self):
        self.cls_criterion = CrossEntropyLoss()
        self.ft_criterion = MSELoss()
        self.model = self._define_network()
        self.optimizer = optim.SGD(self.model.module.parameters(),
                                   lr=self.conf.lr,
                                   weight_decay=4e-4,
                                   momentum=self.conf.momentum)
        # self.optimizer = optim.AdamW(self.model.module.parameters(), lr=0.1)
        # self.schedule_lr = optim.lr_scheduler.MultiStepLR(
        #     self.optimizer, self.conf.milestones, self.conf.gamma, - 1)

        self.schedule_lr = WarmUpMultiStepLR(self.optimizer,
                                             self.conf.milestones,
                                             warm_up={"warmup_iter": 5000,
                                                      "warmup_ratio": 0.01})

        print("lr: ", self.conf.lr)
        print("epochs: ", self.conf.epochs)
        print("milestones: ", self.conf.milestones)

    def _train_eval_stage(self):

        running_loss = 0.
        running_acc = 0.
        running_loss_cls = 0.
        running_loss_ft = 0.
        is_first = True
        self.total_iter = len(self.train_loader) * self.conf.epochs
        for e in range(self.start_epoch, self.conf.epochs):
            self.model.train()
            eval = False
            time_stamp = get_time()
            if is_first:
                self.writer = SummaryWriter(self.conf.log_path, self.conf.time)
                is_first = False
            print('epoch {} started'.format(e))
            print("lr: ", self.schedule_lr.optimizer.param_groups[0]['lr'])
            self.ratio = self.step / self.total_iter

            count = 1
            for sample, ft_sample, target in tqdm(iter(self.train_loader)):
                # count += 1
                # if count == 4:
                #     break
                imgs = [sample, ft_sample]
                labels = target
                loss, acc, loss_cls, loss_ft = self._load_batch_data(imgs, labels, eval)
                running_loss_cls += loss_cls
                running_loss_ft += loss_ft
                running_loss += loss
                running_acc += acc

                self.step += 1
                self.optimizer.step()
                self.schedule_lr.step_iter(self.step)
                # import pdb
                # pdb.set_trace()
                if self.step % self.board_loss_every == 0 and self.step != 0:
                    loss_board = running_loss / self.board_loss_every
                    self.writer.add_scalar(
                        'Training/Loss', loss_board, self.step)
                    acc_board = running_acc / self.board_loss_every
                    self.writer.add_scalar(
                        'Training/Acc', acc_board, self.step)
                    lr = self.optimizer.param_groups[0]['lr']
                    self.writer.add_scalar(
                        'Training/Learning_rate', lr, self.step)
                    loss_cls_board = running_loss_cls / self.board_loss_every
                    self.writer.add_scalar(
                        'Training/Loss_cls', loss_cls_board, self.step)
                    loss_ft_board = running_loss_ft / self.board_loss_every
                    self.writer.add_scalar(
                        'Training/Loss_ft', loss_ft_board, self.step)
                    print("Step:{}  Training/lr:{}  Loss:{}  Loss_cls:{}  Loss_ft:{}  Acc:{}".format(
                          str(self.step),
                          str(float("%.4f"%lr)),
                          str(float("%.2f"%loss_board)),
                          str(float("%.2f"%loss_cls_board)),
                          str(float("%.4f"%loss_ft_board)),
                          str(float("%.2f"%acc_board))))

                    running_loss = 0.
                    running_acc = 0.
                    running_loss_cls = 0.
                    running_loss_ft = 0.



            torch.save(self.model.state_dict(), self.conf.model_path+"/epoch_{}.pth".format(e))

            if e % 1 == 0:
                eval = True
                self.model.eval()
                total_val_iter = 0
                eval_loss = 0.
                eval_acc = 0.
                eval_loss_cls = 0.
                eval_loss_ft = 0.
                with torch.no_grad():
                    for sample, ft_sample, target in tqdm(iter(self.eval_loader)):
                        total_val_iter += 1
                        imgs = [sample, ft_sample]
                        labels = target
                        loss, acc, loss_cls, loss_ft = self._load_batch_data(imgs, labels,eval)
                        eval_loss_cls += loss_cls
                        eval_loss_ft += loss_ft
                        eval_loss += loss
                        eval_acc += acc


                    loss_board = eval_loss / total_val_iter
                    self.writer.add_scalar(
                        'Eval/Loss', loss_board, self.step)

                    acc_board = eval_acc / total_val_iter
                    self.writer.add_scalar(
                        'Eval/Acc', acc_board, self.step)

                    loss_cls_board = eval_loss_cls / total_val_iter
                    self.writer.add_scalar(
                        'Eval/Loss_cls', loss_cls_board, self.step)
                    loss_ft_board = eval_loss_ft / total_val_iter
                    self.writer.add_scalar(
                        'Eval/Loss_ft', loss_ft_board, self.step)
                    self.writer.close()
            # self.schedule_lr.step()

            self.schedule_lr.step_epoch()

        self.writer.close()




    def _load_batch_data(self, imgs, labels, eval):

        labels = labels.to(self.conf.device)
        embeddings, feature_map, feature, fc = self.model.forward(imgs[0].to(self.conf.device))
        if eval == False:
            self.optimizer.zero_grad()
            # embeddings = self.ISA(embeddings, feature, fc, labels, self.ratio)
        # import pdb
        # pdb.set_trace()
        loss_cls = self.cls_criterion(embeddings, labels)
        loss_fea = self.ft_criterion(feature_map, imgs[1].to(self.conf.device))

        loss = 0.5*loss_cls + 0.5*loss_fea
        acc = self._get_accuracy(embeddings, labels)[0]
        if eval == False:
            loss.backward()


        return loss.item(), acc, loss_cls.item(), loss_fea.item()

    def _define_network(self):
        # param = {
        #     'num_classes': self.conf.num_classes,
        #     'img_channel': self.conf.input_channel,
        #     'embedding_size': self.conf.embedding_size,
        #     'conv6_kernel': self.conf.kernel_size}

        # model = MultiFTNet(**param).to(self.conf.device)
        model = RepVggFTNet(32, [48, 64, 128], [4, 10, 8], num_out=2, num_classes=self.conf.num_classes)
        model = torch.nn.DataParallel(model, self.conf.devices)
        model.to(self.conf.device)
        return model

    def _get_accuracy(self, output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        ret = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
            ret.append(correct_k.mul_(1. / batch_size))
        return ret

    def _save_state(self, time_stamp, epoch, extra=None):
        save_path = self.conf.model_path
        torch.save(self.model.state_dict(), save_path + '/' +
                   ('{}_{}_model_epoch-{}.pth'.format(time_stamp, extra, epoch)))

    def _save_eval_state(self, time_stamp, epoch, extra=None):
        save_path = self.conf.eval_model_path
        torch.save(self.model.state_dict(), save_path + '/' +
                   ('{}_{}_model_epoch-{}.pth'.format(time_stamp, extra, epoch)))