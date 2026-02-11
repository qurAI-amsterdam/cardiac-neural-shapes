# Imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent)) # Add project root to sys.path

import numpy as np
import tqdm
import pickle
import json
from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset, DataLoader, RandomSampler
import lightning.pytorch as pl 
import pyvista as pv


# ===============================================================
# CTA DATALOADER - MULTI MESHES AND SDFs - LVBP, LV (MYO), RVBP
# ===============================================================

# Loading a 'cta_id'.npz file per patient, describing their multi-mesh with the following fields: 
#   mesh_data = (np.load(f'{cta_id}.npz')), where mesh_data.files:
#   -- surface_coords (3 * #points, 3)
#   -- normals (3 * #points, 3)
#   -- volume_coords (#points, 3)
#   -- sdfs_surface (3 * #points, 3)
#   -- sdfs_volume (#points, 3)

# get_item method: returns a dict with the above mentioned fields with a given sample_size per mesh,
#                  randomly sampled using np.choice.

# --------------------
# Idx order of each label on the GT vectors with coords and SDFs (from .npz)
IDX_MAP_LABEL = {"LVBP": 0, "LV": 1, "RVBP": 2}
# --------------------

class MeshDataset(Dataset):
    """
    Represents single or multi-mesh CTA data by loading their point clouds, 
    SDF values and normals from pre-saved .npz files.
    
    Args:
        -- config (dict): Configuration dictionary
        -- mode (str): 'train', 'val' or 'test' mode
        -- load_first (int): Load only the first 'load_first' meshes
    """
    
    def __init__(self, config:dict, mode:Optional[list]=None, load_first:Optional[int]=None, 
                 n_samples_per_mesh:Optional[int]=None, surface_only:Optional[bool]=None, sample_sax:bool=False):
        super().__init__()
        
        self.config = config
        self.setup_labels = config["DATALOADER"]["setup_labels"]
        label = f'_{self.setup_labels[0]}' if len(self.setup_labels) == 1 else ''
        
        with open(config["DATALOADER"]["data_split_path"], 'rb') as f:
                data_split_dict = json.load(f)
        self.mode = mode
        if mode:
            assert mode in ['train', 'val', 'test'], 'Mode must be either train, val or test'
            # Get train, val or test CTA data
            self.cta_ids = data_split_dict[mode]
            self.fnames = [Path(config["DATA"]["data_dir"]) / 
                           Path(config["DATALOADER"]["load_folder"]) / 
                           f'{cta_id}{label}.npz' for cta_id in self.cta_ids]
        else:
            # Load all CTA data
            self.fnames = list((Path(config["DATA"]["data_dir"]) / Path(config["DATALOADER"]["load_folder"])).glob('*.npz'))
            self.cta_ids = []
            for mode in ['train', 'val', 'test']:
                self.cta_ids += data_split_dict[mode]
        
        if load_first:
            self.fnames = self.fnames[:load_first]
            self.cta_ids = self.cta_ids[:load_first]
        
        self.meshes = [np.load(fname) for fname in tqdm.tqdm(self.fnames, desc="Loading meshes data (.npz)")]  # List of dicts, each representing multi-mesh data
        
        if n_samples_per_mesh:
            self.n_samples_per_mesh = n_samples_per_mesh
        else:
            if mode == "val":
                self.n_samples_per_mesh = 2 * config["DATALOADER"]["n_samples_per_mesh"] # Nr. of points to sample per mesh
            else:
                self.n_samples_per_mesh = config["DATALOADER"]["n_samples_per_mesh"]
    
        self.n_surface = self.meshes[0]['surface_coords'].shape[0]                   # Total nr. of surface points of each multi-mesh
        self.n_volume = self.meshes[0]['volume_coords'].shape[0]                     # Total nr. of volume points of each multi-mesh
        self.surface_only = surface_only if surface_only is not None else self.config["DATALOADER"]["surface_only"]
        self.sample_sax = sample_sax
        self.load_normals = self.config["DATALOADER"]["load_normals"]
        self.rng = np.random.default_rng(seed=config["GENERAL"]["seed"])
        np.random.seed(seed=config["GENERAL"]["seed"])
        
        if (Path(config["DATA"]["data_dir"]) / Path(config["DATALOADER"]["load_folder"]) / "bounds.pkl").exists():
            # Read bounds from .pkl file
            with open(Path(config["DATA"]["data_dir"]) / Path(config["DATALOADER"]["load_folder"]) / "bounds.pkl", 'rb') as f:
                self.bounds_dict = pickle.load(f)
        else:
            self.bounds_dict = None
            print("WARNING -- Bounds file not found. Run save_bounds() method to save bounds.")
    # --------------------  
     
    def save_bounds(self):
        meshes_dir = Path(self.config["DATA"]["data_dir"]) / Path(self.config["DATA"]["mesh_folder"])
        bounds = {}
        for cta_id in tqdm.tqdm(self.cta_ids, desc="Saving bounds"):
            bounds_ = {}
            for label in self.setup_labels:
                mesh_ply = pv.read(meshes_dir / label / f"{cta_id}_{label}.ply")
                bounds_[label] = np.array(mesh_ply.bounds).reshape(3,2)
            bounds[cta_id] = bounds_
        
        self.bounds_dict = bounds  # Dict: {cta_id: {label: (xmin, xmax, ymin, ymax, zmin, zmax), ...}, ...}
        
        # Save bounds as pickle file
        with open(Path(self.config["DATA"]["data_dir"]) / Path(self.config["DATALOADER"]["load_folder"]) / "bounds.pkl", 'wb') as f:
            pickle.dump(self.bounds_dict, f) 
    # --------------------    
    
    def sample_like_sax(self, coords_surface:np.ndarray, sdfs_surface:np.ndarray, 
                        slice_thick:int=10, decimals:int=2):
        """
        Sample CTA mesh surface points like in SAX-MRI view, considering a given
        slice thickness and rounding decimals for the z-coordinate. 
        
        Args:
            coords_surface (np.ndarray): Surface coordinates of the mesh (#points, 3xyz).
            sdfs_surface (np.ndarray): SDF values of the surface coordinates (#points, 1 or 2 or 3 sdfs).
            slice_thick (int): Slice thickness in mm to sample the SAX-like surface points.
            decimals (int): Number of decimals to round the z-coordinates.
        
        Returns:
            surface_coords (np.ndarray): SAX-sampled surface coordinates.
            surface_sdfs (np.ndarray): Sax-sampled SDF values of the surface coordinates.
        """
        with open((Path(self.config["DATA"]["data_dir"]) / 
                   Path(self.config["DATALOADER"]["load_folder"])).parent.parent /
                   "config_preprocess_cta.json", 'r') as file:
            preprocess_dict = json.load(file)
        
        # Slice thickness (z-spacing) from mm to [-1,1]^3 domain
        z_spacing = np.round(slice_thick / preprocess_dict["PREPROCESSING"]["max_norm_coords"] \
                                * preprocess_dict["PREPROCESSING"]["scale_factor"], decimals)
        
        # Round z-coordinates to the given decimals
        coords_z = coords_surface[:,2].round(decimals)
        
        # Simulate z-coordinates based on the given SAX-like z-spacing 
        z_filter_coords = np.arange(np.min(coords_z), np.max(coords_z), z_spacing).round(decimals)
        
        coords_filtered, sdfs_filtered = [], []
        for z_i in z_filter_coords[1:-1]:
            coords_z_i = coords_surface[coords_z == z_i]
            sdfs_z_i = sdfs_surface[coords_z == z_i]
            coords_filtered.append(coords_z_i)
            sdfs_filtered.append(sdfs_z_i)
            
        coords = np.concatenate(coords_filtered, axis=0)
        sdfs= np.concatenate(sdfs_filtered, axis=0)
        
        return coords, sdfs  
    # --------------------
     
    def __len__(self):
        return len(self.meshes)
    # --------------------
    
    def __getitem__(self, idx):
        mesh_data = self.meshes[idx]
        
        # -----
        if len(self.setup_labels) == 2:
            # TRAINING ONLY WITH LV + RVBP
            assert("LV" in self.setup_labels) and ("RVBP" in self.setup_labels), \
            "WARNING -- Only [LV, RVBP] is supported as a DUO for now..."
            
            n_per_surface = self.n_surface // 3
            n_per_volume = self.n_volume // 3
            n_range_surface = np.arange(n_per_surface, self.n_surface)
            n_range_volume = np.arange(n_per_volume, self.n_volume)
            sdf_i = 1
        else:
            n_range_surface = np.arange(self.n_surface)
            n_range_volume = np.arange(self.n_volume)
            sdf_i = 0
        # -----
        
        if self.sample_sax:
            assert self.surface_only, "SAX sampling is only supported for surface-only mode."
            
            # Sample surface points like in SAX-MRI view
            coords_sax, sdfs_sax = self.sample_like_sax(coords_surface=mesh_data['surface_coords'][n_range_surface,:],
                                                        sdfs_surface=mesh_data['sdfs_surface'][n_range_surface, sdf_i:])
            sax_idxs = self.rng.choice(np.arange(coords_sax.shape[0]), 
                                        self.n_samples_per_mesh * len(self.setup_labels), 
                                        replace=False)
            coords = coords_sax[sax_idxs, :]                                                # Sampled SAX surface points
            sdf = sdfs_sax[sax_idxs, :]                                                     # SDF values of sampled SAX surface points
        else:      
            surface_idxs = self.rng.choice(n_range_surface, 
                                        self.n_samples_per_mesh * len(self.setup_labels), 
                                        replace=False)
            coords = mesh_data['surface_coords'][surface_idxs, :]                           # Sampled surface points
            sdf = mesh_data['sdfs_surface'][surface_idxs, sdf_i:]                           # SDF values of sampled surface points
        
        gt_coords = np.split(mesh_data['surface_coords'][n_range_surface, :], 
                             len(self.setup_labels), axis=0)                                # Ground truth ALL SURFACE points per label
        gt_coords = {label: gt_coords[i] for i, label in enumerate(self.setup_labels)}      # Dict: {label_i: gt_coords_i, ...}
        
        if not(self.surface_only):
            # Sample additional volume points
            n_volume_per_mesh = self.n_samples_per_mesh // 2 if self.mode=='val' else self.n_samples_per_mesh
            volume_idxs = self.rng.choice(n_range_volume, 
                                          n_volume_per_mesh * len(self.setup_labels),
                                          replace=False)
            coords = np.concatenate((coords, mesh_data['volume_coords'][volume_idxs, :]), axis=0)
            sdf = np.concatenate((sdf, mesh_data['sdfs_volume'][volume_idxs, sdf_i:]), axis=0)

        if self.load_normals and self.surface_only:
            normals = mesh_data['normals'][surface_idxs]
            return dict(idx=idx, coords=coords, sdf=sdf, normals=normals, 
                        cta_id=self.cta_ids[idx], gt_coords=gt_coords)
        else:
            return dict(idx=idx, coords=coords, sdf=sdf, 
                        cta_id=self.cta_ids[idx], gt_coords=gt_coords)   
# ==============================


class CTADataModule(pl.LightningDataModule):
    def __init__(self, config: dict, load_first: Optional[int] = None):
        super().__init__()
        
        self.config = config
        self.load_first = load_first
        self.setup_labels = config["DATALOADER"]["setup_labels"]
        self.batch_size = config["DATALOADER"]["batch_size"]
        self.train_ds, self.val_ds = None, None
    # ------------------------
        
    def setup(self, stage: Optional[str] = None):
    
        self.train_ds = MeshDataset(self.config, mode='train', load_first=self.load_first)
        self.val_ds = MeshDataset(self.config, mode='val', load_first=self.load_first)
        
        print(f"INFO -- Setup Dataset: {self.setup_labels} -- #train meshes: {len(self.train_ds)}, #val meshes: {len(self.val_ds)}")
    # ------------------------
    
    def train_dataloader(self):
        #train_iters_per_epoch = 100
        #train_sampler = RandomSampler(self.train_ds, replacement=True, num_samples=self.batch_size * train_iters_per_epoch)
        train_sampler = RandomSampler(self.train_ds, replacement=False, num_samples=len(self.train_ds))
        train_loader = DataLoader(self.train_ds, batch_size=self.batch_size, sampler=train_sampler, num_workers=6)
        return train_loader
    # ------------------------
    
    def val_dataloader(self):
        val_loader = DataLoader(self.val_ds, batch_size=len(self.val_ds), shuffle=False, num_workers=6)
        return val_loader
# ==============================


