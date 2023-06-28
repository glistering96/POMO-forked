##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src, get_result_folder, set_result_folder

from CVRPTrainer import CVRPTrainer as Model
import pytorch_lightning as pl
from lightning.pytorch import loggers as pl_loggers
import torch


##########################################################################################
# parameters

env_params = {
    'problem_size': 100,
    'pomo_size': 100,
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 4,
    'qkv_dim': 16,
    'head_num': 4,
    'logit_clipping': 10,
    'eval_type': 'argmax',
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [8001, 8051],
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 8100,
    'train_episodes': 10 * 1000,
    'train_batch_size': 64,
    'prev_model_path': None,
    'baseline': 'mean',
    'logging': {
        'model_save_interval': 2,
    },
    'model_load': {
        'enable': False,  # enable loading pre-trained model
        # 'path': './result/saved_CVRP20_model',  # directory path of pre-trained model and log files saved.
        # 'epoch': 2000,  # epoch version of pre-trained model to laod.

    }
}

desc = f"cvrp_n{env_params['problem_size']}-{trainer_params['baseline']}"

logger_params = {
    'log_file': {
        'desc': desc,
        'filename': 'log.txt'
    }
}


##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    model = Model(env_params=env_params,
                  model_params=model_params,
                  optimizer_params=optimizer_params,
                  trainer_params=trainer_params)

    epochs = trainer_params['epochs']

    result_folder = get_result_folder()

    logger = pl_loggers.TensorBoardLogger(save_dir=result_folder,
                                          name='', max_queue=500)

    default_root_dir = logger.log_dir

    score_cp_callback = pl.callbacks.ModelCheckpoint(
        dirpath=default_root_dir,
        monitor="train_score",
        every_n_epochs=trainer_params['logging']['model_save_interval'],
        mode="min",
        filename="{epoch}-{train_score:.5f}",
        save_on_train_epoch_end=True,
        save_top_k=5
    )

    trainer = pl.Trainer(
        accumulate_grad_batches=1,
        logger=logger,
        # log_every_n_epoch=1,
        check_val_every_n_epoch=0,
        max_epochs=epochs,
        default_root_dir=default_root_dir,
        precision="16-mixed",
        callbacks=[score_cp_callback],
        gradient_clip_val=1.0
    )
    num_steps_in_epoch = trainer_params['train_episodes'] // trainer_params['train_batch_size'] + 1
    dummy_dl = torch.utils.data.DataLoader(torch.zeros((num_steps_in_epoch, 1, 1, 1)), batch_size=1)
    trainer.fit(model, train_dataloaders=dummy_dl)

    # copy_all_src(trainer.result_folder)

    trainer.run()


def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    model = Model(env_params=env_params,
                  model_params=model_params,
                  optimizer_params=optimizer_params,
                  trainer_params=trainer_params)

    epochs = trainer_params['epochs']

    result_folder = get_result_folder()

    logger = pl_loggers.TensorBoardLogger(save_dir=result_folder,
                                          name='', max_queue=500)

    default_root_dir = logger.log_dir

    score_cp_callback = pl.callbacks.ModelCheckpoint(
        dirpath=default_root_dir,
        monitor="train_score",
        every_n_epochs=trainer_params['logging']['model_save_interval'],
        mode="min",
        filename="{epoch}-{train_score:.5f}",
        save_on_train_epoch_end=True,
        save_top_k=5
    )

    trainer = pl.Trainer(
        accumulate_grad_batches=1,
        logger=logger,
        # log_every_n_epoch=1,
        check_val_every_n_epoch=0,
        max_epochs=epochs,
        default_root_dir=default_root_dir,
        precision="16-mixed",
        callbacks=[score_cp_callback],
        gradient_clip_val=1.0
    )
    num_steps_in_epoch = trainer_params['train_episodes'] // trainer_params['train_batch_size'] + 1
    dummy_dl = torch.utils.data.DataLoader(torch.zeros((num_steps_in_epoch, 1, 1, 1)), batch_size=1)
    trainer.fit(model, train_dataloaders=dummy_dl)

    # copy_all_src(trainer.result_folder)

    trainer.run()


def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 4
    trainer_params['train_batch_size'] = 2


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    main()
