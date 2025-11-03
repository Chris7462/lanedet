import time
import torch
from tqdm import tqdm
import pytorch_warmup as warmup
import numpy as np
import random
import cv2

from lanedet.models.registry import build_net
from .registry import build_trainer, build_evaluator
from .optimizer import build_optimizer
from .scheduler import build_scheduler
from lanedet.datasets import build_dataloader
from lanedet.utils.recorder import build_recorder
from lanedet.utils.net_utils import save_model, load_network


class Runner(object):
    def __init__(self, cfg):
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        self.cfg = cfg
        self.recorder = build_recorder(self.cfg)
        self.net = build_net(self.cfg)
        # Use standard PyTorch DataParallel instead of mmcv's MMDataParallel
        self.net = torch.nn.DataParallel(
                self.net, device_ids=list(range(self.cfg.gpus))).cuda()
        self.recorder.logger.info('Network: \n' + str(self.net))
        self.resume()
        self.optimizer = build_optimizer(self.cfg, self.net)
        self.scheduler = build_scheduler(self.cfg, self.optimizer)
        self.warmup_scheduler = None
        # TODO(zhengtu): remove this hard code
        if self.cfg.optimizer.type == 'SGD':
            self.warmup_scheduler = warmup.LinearWarmup(
                self.optimizer, warmup_period=5000)
        self.metric = 0.
        self.val_loader = None

    def resume(self):
        if not self.cfg.load_from and not self.cfg.finetune_from:
            return
        load_network(self.net, self.cfg.load_from, 
                finetune_from=self.cfg.finetune_from,
                logger=self.recorder.logger)

    def to_cuda(self, batch):
        for k in batch:
            if not isinstance(batch[k], torch.Tensor):
                continue
            batch[k] = batch[k].cuda()
        return batch

    def train_epoch(self, epoch, train_loader):
        self.net.train()
        end = time.time()
        max_iter = len(train_loader)
        for i, data in enumerate(train_loader):
            if self.recorder.step >= self.cfg.total_iter:
                break
            date = self.to_cuda(data)
            self.recorder.step += 1
            self.recorder.record('train')

    def train(self):
        self.recorder.logger.info('Start training...')
        self.train_loader = build_dataloader(self.cfg.dataset.train, self.cfg, is_train=True)
        self.val_loader = build_dataloader(self.cfg.dataset.val, self.cfg, is_train=False)
        self.trainer = build_trainer(self.cfg)
        self.evaluator = build_evaluator(self.cfg)
        for epoch in range(self.cfg.epochs):
            self.recorder.epoch = epoch
            self.train_epoch(epoch, self.train_loader)
            if (epoch + 1) % self.cfg.eval_ep == 0 or epoch == self.cfg.epochs - 1:
                self.validate(epoch)
            self.save_ckpt(is_best=False)
    
    def validate(self, epoch=None):
        self.net.eval()
        self.evaluator.run(self.net, self.val_loader, self.cfg.work_dir, epoch=epoch)

    def save_ckpt(self, is_best=False):
        save_model(self.net, self.optimizer, self.cfg.work_dir, 
                is_best=is_best, recorder=self.recorder)
