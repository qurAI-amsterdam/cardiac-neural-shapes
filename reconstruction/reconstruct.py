# Imports
import os
from typing import Optional
import numpy as np
from pathlib import Path
import json
import tqdm
import torch
import pyvista as pv
import lightning.pytorch as pl
import mcubes

from models.utils.seed_everything import seed_everything
from models.AutoDecoder import LitAutoDecoder

# ~~~~~~~~~~~~~~~~~~~~
# MESH RECONSTRUCTION 
# ~~~~~~~~~~~~~~~~~~~~

# ----------------------------
# 1) Load model weights
# 2) Sample GT coords and SDFs
# 3) Optimize latent shape vector
# 4) Grid Coords --> Pred SDFs Volume --> Zero Iso-Surface Point Clouds --> Mesh Extraction (Marching Cubes)
# ----------------------------


class ReconstructMesh():
    """
    MESH RECONSTRUCTION CLASS
    """
    def __init__(self, config_recon:dict):
        # From config_reconstruct.json file:
        outputs_dir = Path(config_recon["DATA"]["outputs_dir"])
        project_folder = config_recon["DATA"]["project_folder"]
        modality = config_recon["DATA"]["modality"]
        dataset = config_recon["DATA"]["dataset"]
        self.phase = config_recon["DATA"]["phase"]
        self.out_folder = config_recon["DATA"]["out_folder"]
        
        chkpnt_relative_path = Path(config_recon["RECONSTRUCTION"]["chkpnt_relative_path"])
        self.grid_res = config_recon["RECONSTRUCTION"]["grid_res"]
        self.grid_lims = config_recon["RECONSTRUCTION"]["grid_lims"]
        self.level_set = config_recon["RECONSTRUCTION"]["level_set"]
        self.n_iters_latent = config_recon["RECONSTRUCTION"]["n_iters_latent"]
        self.smooth_iters = config_recon["RECONSTRUCTION"]["smooth_iters"]
        # ----------

        # Load config used for training 
        config = json.load(open(chkpnt_relative_path.parent.parent / 'config.json'))
        self.config_latent = config['OPTIMIZER']['FIND_LATENT']
        self.setup_labels = config['DATALOADER']['setup_labels']
        exper_name = config["GENERAL"]["exp_name"] #chkpnt_relative_path.parent.parent.name
        self.process_dir = outputs_dir / project_folder / exper_name / modality / dataset / self.out_folder / self.phase
        if not self.process_dir.exists():
            self.process_dir.mkdir(parents=True, exist_ok=False)
        # ----------
        
        # Set GPU
        gpu = config_recon["GENERAL"]["gpu"]
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set seed
        seed_everything(config_recon["GENERAL"]["seed"])
        pl.seed_everything(config_recon["GENERAL"]["seed"], workers=True)
        
        # Load model    
        self.model = LitAutoDecoder.load_from_checkpoint(chkpnt_relative_path)
        self.model.eval()
        print(f"INFO -- Model loaded from: {exper_name}")
        print(f"INFO -- Reconstruction resolution: {self.grid_res}")
        
        # To be set in child class (e.g., ReconstructMeshCMRI)
        self.id_key = None
        self.loader = None
        self.max_norm_coords = None
        self.scale_factor = None
        
        self.config_train = config
        self.config_recon = config_recon
    # ------------------------
    
    def get_sdf_volumes(self):
        """ 
        Predict SDF(s) volume for each point cloud (i.e., predict SDF(s) on grid coords)
        Create a dictionary with key: pat_id and value: predicted SDF(s) volume
        """
        self.sdf_volumes, self.samples_dict, self.latents_dict = {}, {}, {}
        for single_batch in tqdm.tqdm(self.loader, desc="Getting SDF volumes"):
            # Coords to optimize latent shape vector
            coords = single_batch['coords'].to(torch.float32).to(self.device)    # (1, #points, 3 xyz)
            gt_sdf = single_batch['sdf'].to(torch.float32).to(self.device)       # (1, #points, 1 or 2 or 3 sdfs)
            pat_id = single_batch[self.id_key][0]                                # (1,)
            
            self.samples_dict[pat_id] = {'coords': coords.detach().cpu().numpy()[0], 'sdf': gt_sdf.detach().cpu().numpy()[0]}
            
            shape_latent, _, total_losses = self.model.find_latent_vector(coords, gt_sdf, self.config_latent, 
                                                               n_iters=self.n_iters_latent)          # (1, latent_size)
            
            self.latents_dict[pat_id] = {"dim": shape_latent.size(dim=1), 
                                         "loss": total_losses[-1], 
                                         "latent": shape_latent.detach().cpu().numpy()[0].tolist()}
            
            xx, yy, zz = np.meshgrid(np.linspace(self.grid_lims[0], self.grid_lims[1], self.grid_res), 
                                     np.linspace(self.grid_lims[0], self.grid_lims[1], self.grid_res),
                                     np.linspace(self.grid_lims[0], self.grid_lims[1], self.grid_res))
            coords_grid = np.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)]).T               # (grid_res^3, 3 xyz) 
            coords_grid = torch.tensor(coords_grid, dtype=torch.float32, device=shape_latent.device) # (#grid_points, 3 xyz)
            coords_grid = coords_grid.unsqueeze(0)                                                   # (1, #grid_points, 3 xyz)
            
            if self.grid_res > 100:
                n_splits_grid = (self.grid_res**3) // (100**3)
                n_coords_grid = torch.chunk(coords_grid, n_splits_grid, dim=1)
                with torch.no_grad():
                    pred_sdf = torch.cat([self.model(shape_latent, coords_grid) for coords_grid in n_coords_grid], dim=1) 
            else:
                with torch.no_grad():
                    pred_sdf = self.model(shape_latent, coords_grid)                                 # (1, #grid_points, 1 or 2 or 3 sdfs)
            
            # Predicted SDF volume: (grid_res, grid_res, grid_res, 1 or 2 or 3 sdfs)
            pred_sdf_vol = pred_sdf.reshape(self.grid_res, self.grid_res, self.grid_res, len(self.setup_labels))
            self.sdf_volumes[pat_id] = pred_sdf_vol.detach().cpu().numpy()
            
            del coords, gt_sdf, shape_latent, total_losses
            torch.cuda.empty_cache()  # Empties the GPU cache after each iteration
    # ------------------------
        
    def extract_mesh(self, pat_id:Optional[str]=None):
        """ 
        Extract mesh(es) from predicted SDF(s) volume(s), using Lewiner Marching Cubes algorithm. 
        It additionally:
            - Transforms the mesh(es) to the original image space
            - Extracts the largest connected component of the mesh(es)
            - Smooths the mesh(es) if required
            - Saves the mesh(es) in .ply format
        
        Args:
            -- pat_id (str, optional): Patient id for mesh reconstruction. If None, all ids are considered.
        """
        if pat_id is not None:
            sdf_volumes = {pat_id: self.sdf_volumes[pat_id]}
        else:
            sdf_volumes = self.sdf_volumes
            
        for pat_id, pred_sdf_vol in tqdm.tqdm(sdf_volumes.items(), desc="Extracting meshes"):
            meshes, meshes_taubin = [], []
            for i in range(len(self.setup_labels)):
                # Get mesh vertices and faces (extract iso-surface, i.e., 0-level-set, from SDF volume)
                vertices, faces = mcubes.marching_cubes(pred_sdf_vol[..., i], self.level_set)
                
                # Order from xyz to yxz (GT orientation)
                vertices = vertices[:, [1, 0, 2]]
                faces = faces[:, [1, 0, 2]]
                
                # Add vertex counts (required format for Pyvista object)
                faces = np.pad(faces, ((0,0), (1,0)) , constant_values=3) 
                faces = faces.ravel()
                
                # Back to [-1, 1]^3 train space
                vertices -= self.grid_lims[1]
                vertices -= self.grid_res // 2
                vertices /= self.grid_res // 2 
                
                # Back to original image space
                vertices /= self.scale_factor
                vertices *= self.max_norm_coords
                
                # Get mesh and its largest connected component
                mesh_i = pv.PolyData(vertices, faces)
                mesh_i = mesh_i.extract_largest()
                mesh_i.compute_normals(flip_normals=True, inplace=True)
                meshes.append(mesh_i)
                
                # Smooth mesh (if required)
                if self.smooth_iters > 0:
                    mesh_taubin_i = mesh_i.smooth_taubin(n_iter=self.smooth_iters, pass_band=0.1)
                    meshes_taubin.append(mesh_taubin_i) 
            
            for j, mesh in enumerate(meshes):
                # Save mesh
                mesh.save(Path(self.process_dir) / f'{pat_id}_{self.setup_labels[j]}.ply')
                if self.smooth_iters > 0:
                    #meshes_smooth[j].save(Path(self.process_dir) / f'{pat_id}_{self.setup_labels[j]}_smooth.ply')
                    meshes_taubin[j].save(Path(self.process_dir) / f'{pat_id}_{self.setup_labels[j]}_taubin.ply')
            
            # Save sampled coords and SDFs as dict in .npz
            np.savez(Path(self.process_dir) / f'{pat_id}.npz', 
                     coords=self.samples_dict[pat_id]['coords'],
                     sdf=self.samples_dict[pat_id]['sdf'])
           
        # Save recontruction config file
        config_recon = self.config_recon
        config_recon['PREPROCESSING'] = {}
        config_recon['PREPROCESSING']['max_norm_coords'] = self.max_norm_coords
        config_recon['PREPROCESSING']['scale_factor'] = self.scale_factor
        with open(self.process_dir / 'config_reconstruct.json', 'w') as f:
            json.dump(config_recon, f, indent=4)
        
        # Save latents dict
        with open(self.process_dir / 'latents.json', 'w') as f:
            # One patient per line
            f.write("{\n")
            for i, (key, value) in enumerate(self.latents_dict.items()):
                json_str = json.dumps(value)
                f.write(f'    "{key}": {json_str}')
                if i < len(self.latents_dict) - 1:
                    f.write(",\n")  # comma at end of line for all but last
                else:
                    f.write("\n")
            f.write("}\n")
    
        # Save LAX counts dict if it is available (if MRI data)
        if (self.id_key == 'pat_id') and (len(self.ds.lax_counts_dict) > 0):
            with open(self.process_dir / 'lax_counts.json', 'w') as f:
                json.dump(self.ds.lax_counts_dict, f, indent=4)