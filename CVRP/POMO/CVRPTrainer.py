from typing import Any
import torch
from logging import getLogger

from CVRPEnv import CVRPEnv as Env
from CVRPModel import CVRPModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils.utils import *

from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl


class CVRPTrainer(pl.LightningModule):
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):
        super().__init__()
        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        self.batch_size = self.trainer_params['train_batch_size']
        self.baseline = self.trainer_params['baseline']

        # Main Components
        self.model = Model(**self.model_params)
        self.env = Env(**self.env_params)

    def configure_optimizers(self) -> Any:
        opt = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        sch = Scheduler(opt, **self.optimizer_params['scheduler'])

        return {'optimizer': opt, 'lr_scheduler': sch}

    def training_step(self, batch, batch_idx):
        self.model.device = self.device
        self.model.encoder.device = self.device
        self.model.decoder.device = self.device

        # Prep
        ###############################################
        self.model.train()
        self.env.device = self.device
        self.env.load_problems(self.batch_size)

        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(self.batch_size, self.env.pomo_size, 0)).to(self.device)
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        vals = []

        while not done:
            selected, prob, val = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
            vals.append(val)

        log_prob = prob_list.log().sum(dim=2)

        reward = -reward
        # Loss
        ###############################################
        val_tensor = torch.cat(vals, dim=-1)  # shape: (batch, pomo, T)
        reward_broadcasted = torch.broadcast_to(reward[:, :, None], val_tensor.shape)
        val_loss = torch.nn.functional.mse_loss(val_tensor, reward_broadcasted)

        if self.baseline == 'val':
            baseline = val_tensor
            advantage = reward_broadcasted - baseline.detach()
            loss = advantage * log_prob[:, :, None] + 0.5 * val_loss  # Minus Sign: To Increase REWARD

        elif self.baseline == 'mean':
            baseline = reward.float().mean(dim=1, keepdims=True)  # shape: (batch, 1), original
            # shape: (batch, 1)
            advantage = reward - baseline.detach()
            # shape: (batch, pomo)
            loss = advantage * log_prob + 0.5 * val_loss  # Minus Sign: To Increase REWARD
            # shape: (batch, pomo)

        loss_mean = loss.mean()

        # Score
        ###############################################

        max_pomo_reward, _ = reward.min(dim=1)  # get best results from pomo
        score_mean = max_pomo_reward.float().mean()  # negative sign to make positive value

        lr = self.trainer.optimizers[0].param_groups[0]['lr']

        self.log('train_loss', loss_mean, on_epoch=True, prog_bar=True, logger=True)
        self.log('score/train_score', score_mean, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_score', score_mean, on_epoch=True, prog_bar=True, logger=True)
        self.log('debug/lr', lr, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True)

        return loss_mean