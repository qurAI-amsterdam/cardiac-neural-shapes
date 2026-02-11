# Imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent)) # Add project root to sys.path

import numpy as np
import tqdm
import json
from pathlib import Path
from typing import Optional, Union
from torch.utils.data import Dataset


# ===============================================================
# MRI DATALOADER - MULTI MESHES AND SDFs - LVBP, LV (MYO), RVBP
# ===============================================================

# Loading a 'pat_id'.npz file per patient, describing their multi-mesh with the following fields: 
#   mesh_data = (np.load(f'{pat_id}.npz')), where mesh_data.files:
#   -- 'sax_surface_coords',
#   -- 'sax_tolerance_coords',
#   -- 'sax_surface_sdfs',
#   -- 'sax_tolerance_sdfs'
#   -- 'lax_coords'
#   -- 'lax_sdfs'
#   -- 'tolerance'
# --------------------

class MeshDatasetCMRI(Dataset):
    """
    Represents single or multi-mesh CMRI data by loading their point clouds, 
    SDF values and normals from pre-saved .npz files.
    
    Args:
        -- config (dict): Configuration dictionary (reconstruction config)
        -- load_first (int): Load only the first 'load_first' meshes
    """
    def __init__(self, 
                 config:dict, 
                 load_first:Optional[int]=None, 
                 load_ids:Optional[list]=None,
                 n_samples_sax:Optional[Union[int,str]]=None,   # int or 'all'
                 samples_type:Optional[bool]=None,              # 'surface' or 'tolerance'
                 use_lax:Optional[bool]=None):
        super().__init__()
        
        self.config = config
        self.setup_labels = config["DATA"]["setup_labels"]
        load_folder = f'single/{self.setup_labels[0]}' if len(self.setup_labels) == 1 else 'multi/multi'
        
        # Get all CMRI .npz data files & List of patient IDs
        self.fnames = sorted(list((Path(config["DATA"]["input_dir"]) / load_folder).glob('*.npz')))
        pat_id_digits = config["DATA"]["pat_id_digits"]
        self.pat_ids = [fname.stem[0:pat_id_digits] for fname in self.fnames]
     
        if load_first:
            self.fnames = self.fnames[:load_first]
            self.pat_ids = self.pat_ids[:load_first]
        
        if load_ids:
            if isinstance(pat_id_digits, str):
                self.fnames = [fname for fname in self.fnames if fname.stem in load_ids]
                self.pat_ids = load_ids
            else:
                self.fnames = [fname for fname in self.fnames if fname.stem[0:pat_id_digits] in load_ids]
                self.pat_ids = [fname.stem[0:pat_id_digits] for fname in self.fnames]
            
        # {pat_id: {dict representing single- or multi-mesh data}, ...}  
        self.data_dict = {self.pat_ids[i]:np.load(fname) for i, fname in tqdm.tqdm(enumerate(self.fnames), desc="Loading .npz files")}
        
        # Sampling parameters
        self.n_samples_sax = n_samples_sax if n_samples_sax is not None else self.config["RECONSTRUCTION"]["n_samples_sax_per_mesh"]    
        self.samples_type = samples_type if samples_type is not None else self.config["RECONSTRUCTION"]["samples_type"]
        self.use_lax = use_lax if use_lax is not None else self.config["RECONSTRUCTION"]["use_lax"]
        self.load_normals = self.config["RECONSTRUCTION"]["load_normals"]
        
        assert self.samples_type in ['surface', 'tolerance'], \
        f"Invalid samples_type: {self.samples_type}. Must be 'surface' or 'tolerance'."
        
        # Load config used for training to set the same seed
        config_train = json.load(open(Path(self.config["RECONSTRUCTION"]["chkpnt_relative_path"]).parent.parent / 'config.json'))
        seed = config_train["GENERAL"]["seed"]
        self.rng = np.random.default_rng(seed)
        np.random.seed(seed)
        
       # Count total LAX coords per patient
        self.lax_counts_dict = {}
    # --------------------  
            
    def __len__(self):
        return len(self.data_dict)
    # --------------------
    
    def __getitem__(self, idx):
        #   Data fields:
        #   -- 'sax_surface_coords',
        #   -- 'sax_tolerance_coords',
        #   -- 'sax_surface_sdfs',
        #   -- 'sax_tolerance_sdfs',
        #   -- 'lax_contour_coords',
        #   -- 'lax_surface_coords'
        #   -- 'lax_contour_sdfs'
        #   -- 'lax_surface_sdfs'
        #   -- 'tolerance'
        pat_id = self.pat_ids[idx]
        data = self.data_dict[pat_id]
        
        sax_coords = data[f'sax_{self.samples_type}_coords']                        # (#points, 3xyz)
        n_samples_sax = len(sax_coords) if self.n_samples_sax == 'all' else self.n_samples_sax * len(self.setup_labels)
        idxs_sax_coords = self.rng.choice(range(len(sax_coords)), n_samples_sax, replace=False)
        
        if (len(self.setup_labels) == 2 and ("LV" in self.setup_labels) and ("RVBP" in self.setup_labels)):
            # IF INFERENCE BASED ONLY ON LV + RVBP
            sdf_i = 1
        else:
            sdf_i = 0
        
        coords = data[f'sax_{self.samples_type}_coords'][idxs_sax_coords]           # Sampled SAX coords
        sdf = data[f'sax_{self.samples_type}_sdfs'][idxs_sax_coords, sdf_i:]        # SDF values of sampled SAX coords
            
        # IF use extra coords from LAX contour:
        if self.use_lax:
            self.lax_counts_dict[pat_id] = len(data['lax_surface_coords'])
            coords = np.concatenate((coords, data['lax_surface_coords']), axis=0)   # (#points, 3xyz)
            sdf = np.concatenate((sdf, data['lax_surface_sdfs'][:, sdf_i:]), axis=0)# (#points, 1 or 2 or  3 sdfs)
        
        if self.load_normals:
            normals = data['normals'][idxs_sax_coords]
            return dict(idx=idx, coords=coords, sdf=sdf, normals=normals, 
                        pat_id=pat_id) 
        else:
            return dict(idx=idx, coords=coords, sdf=sdf, 
                        pat_id=pat_id)  
# ==============================

