# Imports
import math
from importlib import import_module
from typing import Optional
import numpy as np
import torch
import tqdm
from torch import nn
import lightning.pytorch as pl 

from .utils.loss import DecoderLoss, compute_chamfer

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~	
# DEFINING THE MODEL: NETWORK, LOSS, OPTIMIZER, TRAINING AND VALIDATION STEPS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class LitAutoDecoder(pl.LightningModule):
    def __init__(self, config: dict, n_train_samples:int, exp_name:Optional[str]=None):
        super().__init__()
        
        self.save_hyperparameters()

        self.config = config
        self.surface_only = config["DATALOADER"]["surface_only"]                # True if only surface points are used	
        self.n_samples_per_mesh = config["DATALOADER"]["n_samples_per_mesh"]    # Nr. of surface points per mesh
        self.n_train_samples = n_train_samples                                  # Nr. of train meshes
        self.latent_size = config["MODEL"]["shape_latent_size"]
        self.clamp = config["MODEL"]["clamp"]
        self.clamp_fn = lambda x: x.clamp_(-self.clamp, self.clamp) if self.clamp else lambda x: x
        self.setup_labels = config["DATALOADER"]["setup_labels"]
        network_class = getattr(import_module('models.utils.networks'), config['MODEL']['network'])  # Convert network name from str to class obj
        
        print(f'INFO - Loading network: {config["MODEL"]["network"]}')
        
        # NETWORK - MLP Decoder
        self.decoder = network_class(config["MODEL"], out_channels=len(self.setup_labels))
        
        # EMBEDDING LAYER:
        # -- Initialize each latent shape vector as Univariate Normal distribution -- N(0, 0.01),
        #    constrained to have L2norm <= 1.0
        shape_embedding = nn.Embedding(n_train_samples, self.latent_size, max_norm=1.0)
        nn.init.normal_(shape_embedding.weight.data, 0.0, 1.0 / math.sqrt(self.latent_size))
        self.shape_embedding = shape_embedding
        
        # Loss function
        self.criterion = DecoderLoss(config["LOSS"]) 
        self.loss_name = config["LOSS"]["loss_name"]
        self.reg_lambda = config["LOSS"]["reg_lambda"]
        """self.eikonal_lambda = config["LOSS"]["eikonal_lambda"]
        self.normals_lambda = config["LOSS"]["normals_lambda"]
        self.cosine_lambda = config["LOSS"]["cosine_lambda"]"""
        #   -- List of weights for each loss term
        if config["LOSS"]["weights"]:
            assert len(config["LOSS"]["weights"]) == len(self.setup_labels), "Number of loss weights must match number of labels"
            self.loss_weights = config["LOSS"]["weights"]
        else:
            # if false --> default: equal weights for each label ~ avg of loss terms
            self.loss_weights = [1.0 / len(self.setup_labels)] * len(self.setup_labels) 
    	
        # Optimizer parameters
        self.optimizer = config["OPTIMIZER"]["optimizer"]
        self.lr = config["OPTIMIZER"]["lr"]
        self.lr_policy = config["OPTIMIZER"]["lr_policy"]
        self.lr_steps = config["OPTIMIZER"]["lr_steps"]
        
        self.rng = np.random.default_rng(config["GENERAL"]["seed"])
        self.exper_name = exp_name 
    # ---------------------
    
    def forward(self, latent_z, coords):
        """ 
        Forward pass of the model,
        
        Args:
            -- latent_z (torch.tensor): batch of latent shape vectors (N, latent_size)
            -- coords (torch.tensor): batch of input coordinates (N, #points, 3xyz)
        
        Returns:
            -- sdf (torch.tensor): batch of predicted SDF values (N, #points, 1 or 2 or 3 sdfs)
        """
        sdf = self.decoder(latent_z, coords)
        return sdf                  # (N, #points, 1 or 2 or 3sdfs) 
    # ---------------------
    
    def configure_optimizers(self):
        """ Configure optimizer and scheduler for training. """ 
        lr = self.lr
        optim = self.str_to_class(self.optimizer, torch.optim) (self.parameters(), lr=lr, amsgrad=True)
        scheduler = self.str_to_class(self.lr_policy, torch.optim.lr_scheduler) (optim, T_0=self.lr_steps)
        return [optim], [scheduler]
    # ---------------------
    
    def training_step(self, batch, batch_idx):
        """ 
        Training step for the model - defines every training iteration: 
            -- Updates the weights of the model every batch, i.e., every iteration.
        """
        # 1. Get batch data
        idxs = batch['idx']                             # (N, 1), mesh indexes in the bacth
        coords = batch['coords'].to(torch.float32)      # (N, #points, 3 xyz)
        gt_sdf = batch['sdf'].to(torch.float32)         # (N, #points, 1 or 2 or 3 sdfs)
        cta_ids = batch['cta_id']

        # 2. Get shape embeddings and predict SDFs
        shape_emb = self.shape_embedding(idxs)          # (N, latent_size), shape latents for each mesh in the batch
        y_hat = self(shape_emb, coords)                 # (N, #points, 1 or 2 or 3 sdfs), auto-decoder sdf outputs

        # 3. Compute loss
        reg_slope_factor = min(1, self.current_epoch / 10)
        reg = self.reg_lambda * reg_slope_factor * torch.mean(torch.norm(shape_emb, dim=1))
        
        loss, chamfer = 0, 0
        for i in range(len(self.setup_labels)):
            if self.loss_name == "L1":
                # L1 loss
                loss_i = (abs(y_hat[:,:,i] - gt_sdf[:,:,i])).mean()
            elif self.loss_name == "L2":
                # L2 loss
                loss_i = ((y_hat[:,:,i] - gt_sdf[:,:,i]) ** 2).mean()
            # ---
            loss = loss + self.loss_weights[i] * loss_i
            self.log(f'loss/tr_sdf_{self.setup_labels[i]}', loss_i)
            
            if batch_idx % 50 == 0:
            # Computes the chamfer distance between two sets of points:
            # (used to evaluate the similarity between predicted (from pred sdfs) and ground truth surface points)
                chamfer_i = self._batched_chamfer_loss(batch['gt_coords'][self.setup_labels[i]], coords, y_hat[:,:,i])
                chamfer = chamfer + chamfer_i
        
        combined_loss = loss + reg 
        
        self.log('loss/training_loss', combined_loss)
        self.log('loss/tr_sdf', loss)
        self.log('loss/tr_reg', reg)
        if batch_idx % 50 == 0:
            self.log('loss/tr_chamfer', torch.tensor(chamfer, dtype=torch.float32))

        return combined_loss
    # ---------------------
    
    def validation_step(self, batch, batch_idx):
        """  """
        # 1. Get batch data
        #idxs = batch['idx']                             # (N, 1), mesh indexes in the bacth
        coords = batch['coords'].to(torch.float32)       # (N, #points, 3 xyz)
        gt_sdf = batch['sdf'].to(torch.float32)          # (N, #points, 1 or 3 sdfs)
        
        # NOTE: -- per mesh, half of the surface points will be used to find the optimal latent
        #       vector and the other half (with or w/o volume points) to validate the model
        #       -- coords = [coords_surface, coords_volume] if surface_only=False
        if self.surface_only:
            coords_latent, coords_ = torch.chunk(coords, 2, dim=1)
            gt_sdf_latent, gt_sdf_ = torch.chunk(gt_sdf, 2, dim=1)
        else:
            total_surface = self.n_samples_per_mesh*len(self.setup_labels)
            coords_latent, coords_ = torch.split(coords, [total_surface, coords.shape[1]-total_surface], dim=1)
            gt_sdf_latent, gt_sdf_ = torch.split(gt_sdf, [total_surface, coords.shape[1]-total_surface], dim=1)
        
        # 2. Get optimal shape embeddings (optimizing latent vectors)
        latent_vecs, _, _ = self.find_latent_vector(coords_latent, gt_sdf_latent, 
                                                    cfg_latent=self.config["OPTIMIZER"]["FIND_LATENT"], latent_in=None) 

        # 3. Predict SDFs
        with torch.no_grad():
            y_hat = self(latent_vecs, coords_)
        
        # 4. Compute loss
        loss, chamfer = 0, 0
        for i in range(len(self.setup_labels)):
            if self.loss_name == "L1":
                # L1 loss
                loss_i = (abs(y_hat[:,:,i] - gt_sdf_[:,:,i])).mean()
            elif self.loss_name == "L2":
                # L2 loss
                loss_i = ((y_hat[:,:,i] - gt_sdf_[:,:,i]) ** 2).mean()
            # ---
            loss = loss + self.loss_weights[i] * loss_i
            self.log(f'loss/val_sdf_{self.setup_labels[i]}', loss_i)
            
            # Computes the chamfer distance between two sets of points:
            # (used to evaluate the similarity between predicted (from pred sdfs) and ground truth surface points)
            chamfer_i = self._batched_chamfer_loss(batch['gt_coords'][self.setup_labels[i]], coords_, y_hat[:,:,i])
            chamfer = chamfer + chamfer_i
               
        self.log('loss/validation_loss', loss)
        self.log('loss/val_chamfer', torch.tensor(chamfer, dtype=torch.float32))

        return loss
    # ---------------------
     
    def _batched_chamfer_loss(self, batch_gt_coords, batch_observed_coords, batch_pred_sdf):
        # gt_coords [b, ground truth n_train, 3], batch_observed_coords [b, train num samples, 3]
        # batch_pred_sdf [b, train num samples]
        loss_cham, num_s = [], 0
        for gt_coord, observ_coord, pred_sdf in zip(batch_gt_coords, batch_observed_coords, batch_pred_sdf):  # loop over samples
            ll = compute_chamfer(gt_coord, observed_coords=observ_coord, pred_sdf=pred_sdf)
            if ll != -1:
                loss_cham.append(ll)
                num_s += 1
        if num_s > 0:
            return np.stack(loss_cham).mean()
        return -1
    # ---------------------
    
    def find_latent_vector(self, coords, gt_sdf, cfg_latent, n_iters=None, latent_in=None,):
        """ 
        Find optimal latent shape vector (z_i) for each shape in the batch.
        
        Args:
            -- coords (torch.tensor): input coordinates (N, #points, 3xyz)
            -- gt_sdf (torch.tensor): ground truth SDF values (N, #points, 1 or 3 sdfs)
            -- cfg_latent (dict): configuration dictionary for latent vector optimization
            -- latent_in (torch.tensor): initial latent vector (N, latent_size)
        
        Returns:
            -- latent (torch.tensor): optimized latent vector (N, latent_size)
            -- losses (list): list of loss values (without reg term) during optimization
            -- total_losses (list): list of total loss values during optimization
        """
        torch.set_grad_enabled(True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if latent_in is not None:
            latent = latent_in.data.clone()
        else:
            # Init latent as z_i~N(0,0.01)
            latent = torch.empty(coords.shape[0], self.latent_size, dtype=torch.float32, 
                                 device=device).normal_(mean=cfg_latent["init_mean"], std=cfg_latent["init_std"])     
        
        latent.requires_grad = True
        n_iters = cfg_latent["find_latent_niters"] if n_iters is None else n_iters
        optim = self.str_to_class(cfg_latent["optimizer"], torch.optim)([latent], lr=cfg_latent["lr"])
        scheduler = self.str_to_class(cfg_latent["lr_policy"], 
                                          torch.optim.lr_scheduler) (optim, step_size=n_iters // 2, gamma=cfg_latent["lr_gamma"])
        
        loss_weights = cfg_latent["weights"]
        if loss_weights:
            assert len(loss_weights) == len(self.setup_labels), "Number of loss weights must match number of labels"
        
        self.decoder.eval()
        total_losses, losses, regs = [], [], []
        pbar = tqdm.tqdm(np.arange(n_iters), desc="Optimizing latents")
        for idx in pbar:
            ref_coords = coords.clone()
            optim.zero_grad()
            pred_sdf = self.decoder(latent, ref_coords)   # pred_sdf = f(z_i, x_i), MLP decoder with fixed weights

            # Compute loss -- L1 loss
            reg = cfg_latent["reg_lambda"] * torch.mean(latent.pow(2))
            if loss_weights:
                loss, chamfer = 0, 0
                for i in range(len(self.setup_labels)):
                    loss_i = (abs(pred_sdf[:,:,i] - gt_sdf[:,:,i])).mean()
                    loss = loss + loss_weights[i] * loss_i
            else:
                loss = (abs(pred_sdf - gt_sdf)).mean()
            
            total_loss = loss + reg   
            
            total_loss.backward()
            optim.step()
            scheduler.step()
            
            losses.append(loss.item())
            total_losses.append(total_loss.item())
            
            pbar.set_description("Loss: {:.3f}".format(total_loss.item()))
            self.log(f'loss/latent_optim', total_loss)
    
        return latent.detach(), losses, total_losses
    # ---------------------
    
    @staticmethod
    def str_to_class(classname, module):
        """ Convert string to class object """
        #return getattr(sys.modules[__name__], classname)
        return getattr(module, classname)
    
    