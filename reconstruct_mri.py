# Imports
import sys
from typing import Optional
import numpy as np
from pathlib import Path
import json
import tqdm
import nibabel as nib
from skimage.morphology import convex_hull_image
import torch
from torch.utils.data import DataLoader

from data.mri_dataloader import MeshDatasetCMRI
from data_preprocessing.canonical_transform import get_affine, get_canonical_transform_mri, getLargestCC
from data_preprocessing.data_labels import *
from reconstruction.reconstruct import ReconstructMesh
    
# ~~~~~~~~~~~~~~~~~~~~~~~~
# CMRI MESH RECONSTRUCTION 
# ~~~~~~~~~~~~~~~~~~~~~~~~

# ----------------------------
# 1) Load model weights
# 2) Sample GT coords and SDFs
# 3) Optimize latent shape vector
# 4) Grid Coords --> Pred SDFs Volume --> Zero Iso-Surface Point Clouds --> Mesh Extraction (Marching Cubes)
# ----------------------------

class ReconstructMeshCMRI(ReconstructMesh):
    """
    MESH RECONSTRUCTION CLASS - CMRI
    
    Inherited methods from 'ReconstructMesh':
    -- get_sdf_volumes()
    -- extract_mesh()
    
    Added methods:
    -- get_lax_plane_coords()
    -- save_cross_section()
    -- fill_cross_lvbp()
    """
    def __init__(self, config_recon:dict, load_first:Optional[int]=None, load_ids:Optional[list]=None):
        super().__init__(config_recon)
        
        # Load CMRI data
        self.ds = MeshDatasetCMRI(config_recon, load_first=load_first, load_ids=load_ids)
        self.loader = DataLoader(self.ds, batch_size=1, shuffle=False, num_workers=0)
        print(f"INFO -- #Meshes-CMRI: {len(self.ds)}")
        
        # Get CMRI preprocessing parameters
        config_preprocess = json.load(open(Path(config_recon["DATA"]["input_dir"]) / "config_preprocess_mri.json"))
        self.max_norm_coords = config_preprocess['PREPROCESSING']['max_norm_coords']
        self.scale_factor = config_preprocess['PREPROCESSING']['scale_factor']
        
        # Directories for GT meshes and segmentations 
        self.meshes_dir = Path(config_recon["DATA"]["input_dir"])
        self.segm_dir = Path(config_recon["DATA"]["input_dir"]).parent / "segmentations"
        
        # Set ID key for CMRI 
        self.id_key = 'pat_id'
        
        # Enum with labels for MRI segmentations
        self.MRILabels = self.str_to_class(config_recon["DATA"]["labels_class"])  
    # ------------------------
    
    def str_to_class(self, classname:str):
        return getattr(sys.modules[__name__], classname)
    # -------------------- 
    
    def get_lax_plane_coords(self, lax_path:Path) -> np.ndarray:
        """ 
        Get LAX plane coordinates (xyz) in both voxel and world-canonical spaces.
        
        Args:
        -- lax_path (Path): Path to LAX segmentation NIfTI file.
            
        Returns:
        -- plane_dict (dict): Dictionary with LAX plane coordinates in voxel 
        ('coords_v') and world-canonical ('coords_cano') spaces.
        """
        # Get transformation matrices: canonical rotation & c.o.m. translation to origin (0,0,0)
        M_cano_sa, M_cano_la, M_cm_dict = get_canonical_transform_mri(str(lax_path).replace("LA", "SA"), self.MRILabels,
                                                                      align_x=True, flip_y=False)
        # Get LAX shape (xy)
        nii_la = nib.load(lax_path)
        la_shape = nii_la.header.get_data_shape()[:2]  # xyz --> xy
        self.la_shape_xy = la_shape
        self.nii_la = nii_la
    
        # Get all voxels index positions (1D vector)
        idxs = np.arange(0, la_shape[0] * la_shape[1])
        
        # Get LAX plane voxel coords (xy0 order)
        coords_v = np.unravel_index(idxs, la_shape)
        coords_v = np.stack(coords_v, -1)     # Stack lists in array (#points, 2xy)
        # Add z=0 to all coords: (#points, 3 xy0)
        coords_v = np.concatenate((coords_v, np.zeros((len(coords_v), 1))), axis=-1).astype(np.int32)
        
        # Add homogeneous coordinate + Apply canonical transformation
        coords_v_h = np.concatenate((coords_v, np.ones((len(coords_v), 1))), axis=-1)
        coords_cano = (coords_v_h @ M_cano_la.T @ np.linalg.inv(M_cm_dict["multi"]).T)[..., :3]
        
        coords_cano = (coords_cano / self.max_norm_coords) * self.scale_factor
        plane_dict = {'coords_v': coords_v, 'coords_cano': coords_cano}   
        
        return plane_dict
    # --------------------
    
    def save_cross_section(self, lax_dir:str, lax_name:str, optimize_latent:bool=False):
        """ 
        Get and save LAX cross-sections from reconstructed shapes
        (i.e., running the reconstruction model on LAX plane coordinates)
        
        Args:
        -- lax_dir (str): Directory with LAX segmentation NIfTI files.
        -- lax_name (str): Name of LAX segmentation NIfTI files, assuming format: pat_id_{lax_name}.
        -- optimize_latent (bool): Whether to optimize latent shape vectors or load 
                                   pre-optimized ones.
        """
        if not optimize_latent:
            # Load pre-optimized latent shape vectors
            latents_ = json.load(open(self.process_dir / "latents.json"))
            latents_dict = {k:v for k,v in latents_.items()}
        
        self.sdf_planes = {} 
        for single_batch in tqdm.tqdm(self.loader, desc="Saving LAX cross-sections"):
            # Coords to optimize latent shape vector
            coords = single_batch['coords'].to(torch.float32).to(self.device)    # (1, #points, 3 xyz)
            gt_sdf = single_batch['sdf'].to(torch.float32).to(self.device)       # (1, #points, 1 or 2 or 3 sdfs)
            pat_id = single_batch[self.id_key][0]                                # (1,)
            
            if optimize_latent:
                shape_latent, _, _ = self.model.find_latent_vector(coords, gt_sdf, self.config_latent, 
                                                                   n_iters=self.n_iters_latent)                     # (1, latent_size)
            else:
                shape_latent = torch.tensor(latents_dict[pat_id]["latent"], device=coords.device).unsqueeze(0)
                
            # Coordinate grid = LAX plane coords
            plane_dict = self.get_lax_plane_coords(f"{lax_dir}/{pat_id}{lax_name}")                                # Get LAX plane coords
            coords_grid = torch.tensor(plane_dict["coords_cano"], dtype=torch.float32, device=shape_latent.device)  # (#grid_points, 3 xyz)
            coords_grid = coords_grid.unsqueeze(0)                                                                  # (1, #grid_points, 3 xyz)
            
            with torch.no_grad():
                pred_sdf = self.model(shape_latent, coords_grid)                 # (1, #grid_points, 1 or 2 or 3 sdfs)
            
            pred_sdf = pred_sdf.squeeze(0)                                       # (#grid_points, 1 or 2 or 3 sdfs)
            pred_sdf = pred_sdf.detach().cpu().numpy()
            
            self.sdf_planes[pat_id] = {"coords_v": plane_dict["coords_v"],
                                       "coords_cano": plane_dict["coords_cano"],
                                       "sdf": pred_sdf}  
            
            # Predicted SDF volume: (grid_res, grid_res, 1, 1 or 2 or 3 sdfs)
            pred_sdf_plane = pred_sdf.reshape(self.la_shape_xy[0], self.la_shape_xy[1], 1, len(self.setup_labels))
            
            bool_lv = pred_sdf_plane[:, :, :, 0] <= 0
            bool_rv = pred_sdf_plane[:, :, :, 1] <= 0
            
            lax_hr = np.zeros((self.la_shape_xy[0], self.la_shape_xy[1], 1), dtype=np.float32)
    
            lax_hr[bool_lv] = self.MRILabels.LV.value
            lax_hr[bool_rv] = self.MRILabels.RVBP.value
            
            # Save with nibabel
            lax_hr_img = nib.Nifti1Image(lax_hr, affine=get_affine(self.nii_la))
            lax_hr_dir = self.process_dir.parent.parent / f"cross_{self.out_folder}" / self.phase
            if not (lax_hr_dir).exists():
                (lax_hr_dir).mkdir(parents=True, exist_ok=True)
                self.lax_hr_dir = lax_hr_dir
            nib.save(lax_hr_img, lax_hr_dir / f"{pat_id}.nii.gz")
        
            del coords, gt_sdf, shape_latent
            torch.cuda.empty_cache()  # Empties the GPU cache after each iteration
    # -------------------- 
    
    def fill_cross_lvbp(self):
        """
        Fill LVBP region in reconstructed cross-section segmentations.
        """
        new_dir = self.lax_hr_dir / f"lvbp_filled"
        if not new_dir.exists():
            new_dir.mkdir(parents=True, exist_ok=True)
            
        for cross_path in tqdm.tqdm(sorted(self.lax_hr_dir.glob("*.nii.gz")), desc="Filling LVBP in cross-sections"):
            segm_nii = nib.load(cross_path)
            segm = np.squeeze(segm_nii.get_fdata())
            mask_rv = segm == self.MRILabels.RVBP.value
            mask_lv = segm == self.MRILabels.LV.value
            mask_hull = convex_hull_image(mask_lv)
            mask_lvbp = getLargestCC(mask_hull & (~mask_lv), ndim=2)
            
            new_segm = np.zeros((segm.shape[0], segm.shape[1], 1), dtype=np.float32)    # LAX shape: (H,W,1)
            new_segm[mask_lvbp] = self.MRILabels.LVBP.value
            new_segm[mask_lv] = self.MRILabels.LV.value
            new_segm[mask_rv] = self.MRILabels.RVBP.value
            
            new_segm_nii = nib.Nifti1Image(new_segm, affine=get_affine(segm_nii))
            nib.save(new_segm_nii, new_dir / cross_path.name)          
# ============================================================================
        

if __name__ == "__main__":
    # Load CMRI reconstruction config file
    config_recon = json.load(open('configs/config_reconstruct_mri.json'))
    
    # Reconstruct meshes
    mesh_reconstructor = ReconstructMeshCMRI(config_recon)
    mesh_reconstructor.get_sdf_volumes()
    mesh_reconstructor.extract_mesh()
    
    # Reconstruct 4CH-LAX cross-sections
    # TODO: set 'lax_dir' and 'lax_name'(assuming format: pat_id_{lax_name})
    mesh_reconstructor.save_cross_section(lax_dir="path_to_lax_segmentations_directory",
                                          lax_name="_lax_name.nii.gz", optimize_latent=False)
    # Fill LVBP in cross-sections
    mesh_reconstructor.fill_cross_lvbp()

        
        
    
    