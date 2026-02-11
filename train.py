# Imports
import datetime
import json
from pathlib import Path
import torch
import lightning.pytorch as pl 
import argparse
import os
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor

from data.cta_dataloader import CTADataModule
from models.AutoDecoder import LitAutoDecoder
from models.utils.seed_everything import seed_everything

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TRAIN & VALIDATE THE MODEL
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~

def train(config:dict, exp_name:str):
    """ 
    Train and Validate the model.
    
    Args:
        -- config (dict): Configuration dictionary
        -- exp_name (str): Name of the experiment
    """
    pl.seed_everything(config["GENERAL"]["seed"], workers=True)

    # Set up Dataloader
    dm = CTADataModule(config)  
    dm.setup()
    
    # Set up experiment name, model checkpoints and logger
    dt = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    exp_name = f'{exp_name}_{dt}'
    checkpoint_last = ModelCheckpoint(dirpath=Path(__file__).resolve().parent / "logs" / exp_name / "checkpoints/",	 
                                      every_n_epochs=config["EVAL"]["validate_every"])
    checkpoint_bestval = ModelCheckpoint(dirpath=Path(__file__).resolve().parent / "logs" / exp_name / "checkpoints/", 
                                         monitor="loss/validation_loss", save_top_k=3, mode="min")
    wandb_logger = WandbLogger(save_dir=Path(__file__).resolve().parent / "logs" / exp_name, 
                               project='neural_shapes', id=exp_name, name=exp_name)    
    
    # Set up model
    model = LitAutoDecoder(config, n_train_samples=len(dm.train_ds), exp_name=exp_name)
    
    # Set up Trainer
    acc = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(devices=1, accelerator=acc, 
                         log_every_n_steps=10,
                         max_epochs=config["OPTIMIZER"]["max_epochs"],
                         check_val_every_n_epoch=config["EVAL"]["validate_every"],
                         num_sanity_val_steps=0,
                         logger=wandb_logger, 
                         callbacks=[LearningRateMonitor(), checkpoint_last, checkpoint_bestval]) 
    
    trainer.fit(model, dm)

    # Resave config dict (.json) with the name, date and time of the experiment  
    config["GENERAL"]["exp_name"] = exp_name
    with open(Path(trainer.log_dir) / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
# ============================================================================


if __name__ == '__main__':
    # Parse train config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='config_json_file', default="configs/config_train.json", help='The configuration file')
    parser.add_argument('--exp_name', type=str, help='The name of the experiment')
    args = parser.parse_args()
    config_train = json.load(open(args.config))
    
    # Set GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config_train["GENERAL"]["gpu"])
    
    # Set seed
    seed_everything(config_train["GENERAL"]["seed"])
    
    # Let's go!!!
    train(config=config_train, exp_name=args.exp_name)
