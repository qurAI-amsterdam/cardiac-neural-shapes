# Imports
import sys 
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent)) # Add project root to sys.path
import pickle
from typing import Optional
import pyvista as pv
import numpy as np
import os
from scipy import ndimage
import tqdm
from pathlib import Path
import json
import glob
import traceback
import datetime
from skimage.measure import marching_cubes, find_contours

from canonical_transform import *

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CONVERT CMRI SEGMENTATIONS TO MESHES - LVBP (LV ENDOCARDIUM), LV (LV EPICARDIUM) and RVBP
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 	

def norm(vec):
    """ Return the norm of a vector. """
    return np.sqrt(np.sum(vec**2))
# ===============================================


class PreprocessMultiMeshCMRI:
    """ 
    Preprocess MRI-SAX segmentations to meshes of LVBP (LV Endocardium), LV (LV Epicardium)
    and RVBP, with the following transformations:
        -- self.preprocess():
            - Rotation into the canonical orientation of the reference world coordinate system + apical-basal direction;
            - Alignment of multi-mesh center of mass with the origin of the reference coordinate system
            - Computing mesh normals so that they point outwards.
            - Get coords and corresponding SDFs from SAX and LAX segmentations.

        -- self.normalize_coords():    
            - Normalization of all mesh coordinates to [-1,1]^3 space, considering the max norm of coordinates across all meshes
            and a scaling factor.
        
        -- self.save_meshes():   
            - Save meshes in .ply format.    
    Args:
        -- config (dict): preprocessing configuration parameters.
        -- load_first_segm (int): number of segmentations to process.
        -- load_segm_ids (list): list of segmentation CTA IDs to process.   
        
        .npz files contain the following data:
            'sax_surface_coords',
            'sax_tolerance_coords',
            'sax_surface_sdfs',
            'sax_tolerance_sdfs'
            'lax_contour_coords'
            'lax_surface_coords'
            'lax_contour_sdfs'
            'lax_surface_sdfs'
            'tolerance' 
    """
    def __init__(self, config: dict, load_first_segms:Optional[int]=None, load_segm_ids:Optional[list[str]]=None):
        self.config = config
        self.pat_id_digits = config["DATA"]["pat_id_digits"]                # Number of digits in patient ID
        self.segm_dirs = [os.path.join(config["DATA"]["data_dir"], segm_folder) for segm_folder in config["DATA"]["segm_folders"]]
        self.sax_segm_paths = []
        for segm_dir in self.segm_dirs:
            self.sax_segm_paths += sorted(glob.glob(os.path.join(segm_dir, "SA", "*.nii.gz")))  
        if load_first_segms:
            self.sax_segm_paths = self.sax_segm_paths[:load_first_segms]
        if load_segm_ids:
            self.sax_segm_paths = [p for p in self.sax_segm_paths if Path(p).stem[:self.pat_id_digits] in load_segm_ids]
      
        self.meshes_dir = os.path.join(config["DATA"]["data_dir"], config["DATA"]["mesh_folder"])
        self.meshes_labels =config["DATA"]["mesh_labels"]
        self.max_norm = 0           # Max norm of mesh coordinates
        
        self.meshes_single_dict= {} # Dict: {pat_id: [mesh_LVBP, mesh_LV, mesh_RVBP], ...}, each centered on single-shape c.o.m.
        self.meshes_dict = {}       # Dict: {pat_id: [mesh_LVBP, mesh_LV, mesh_RVBP], ...}, each centered on multi-shape c.o.m.
        
        self.scale_factor = config["PREPROCESSING"]["scale_factor"]
        self.segm_padding = config["PREPROCESSING"]["segm_padding"]
        self.MRILabels = self.str_to_class(config["DATA"]["labels_class"])   # Enum with labels for MRI segmentations
        self.canonical_flip_y = config["PREPROCESSING"]["canonical_flip_y"]
        self.canonical_align_x = config["PREPROCESSING"]["canonical_align_x"]
        self.phase = config["PREPROCESSING"]["phase"]
        self.level = config["PREPROCESSING"]["level_set"]
        self.surface_tolerance = config["PREPROCESSING"]["surface_tolerance"]
        self.lv_epi = config["PREPROCESSING"]["lv_epi"]                       # Whether to model MYO (label="LV") as LV Epi or (LV Epi + LV Endo)
        
        self.save_transforms = config["PREPROCESSING"]["save_transforms"]     # Whether to save orientation transforms to .pkl files
        transforms_path = config["PREPROCESSING"]["use_transforms_from"]
        if len(transforms_path) > 0:
            # Load pre-computed transforms from .pkl file
            with open(transforms_path, "rb") as f:
                self.M_dict = pickle.load(f)
        else:
            self.M_dict = {}
        
        self.seed = config["GENERAL"]["seed"]
        self.rng = np.random.default_rng(seed=self.seed)
        np.random.seed(seed=self.seed)
    # --------------------

    def str_to_class(self, classname:str):
        return getattr(sys.modules[__name__], classname)
    # -------------------- 
    
    def get_sdf_field(self, bin_segmask: np.ndarray, spacing_zyx):
        """
        Compute the signed distance field for a given binary segmentation mask,
        in world spacing. 
        """
        # ~bin_segmask
        bg_bin_segmask = np.ones(bin_segmask.shape) - bin_segmask
        edt_bg = ndimage.distance_transform_edt(bg_bin_segmask, sampling=spacing_zyx) # Eucledian distance background
        edt_fg = ndimage.distance_transform_edt(bin_segmask, sampling=spacing_zyx)    # Eucledian distance foreground
        sdf_field = edt_bg - edt_fg
        
        # Move distance to "voxel-origin" (given in-plane spacing)
        sdf_field[sdf_field > 0] -= spacing_zyx[-1] / 2
        sdf_field[sdf_field < 0] += spacing_zyx[-1] / 2

        return sdf_field    # (bin_segmask.shape)
    # --------------------

    def sample_voxel_coords(self, bin_segmask, sdf_field, sdf_level):
        """ 
        Sample surface or tolerance voxel coords from binary segmentation mask,
        based on signed distance field and given SDF-level.
        Assumes bin_segmask in zyx order.
        """
        # Get surface or tolerance index positions (1D vector)
        idxs = np.where(np.abs(sdf_field.ravel()) <= sdf_level)[0]  # In-plane, assuming iso in plane; e.g., sdf_field <= 1.5 mm
        
        # Get segmentation voxel coords (xyz order)
        coords_v = np.unravel_index(idxs, bin_segmask.shape)
        coords_v = np.stack(coords_v, -1)     # Stack lists in array (#points, 3)
        coords_v = np.flip(coords_v, -1)      # To x, y, z order
        
        return coords_v, idxs
    # --------------------
    
    def sample_voxel_contour(self, bin_segmask):
        """ 
        Sample 2D contour voxel coords from binary segmentation mask,
        in (x,y,z=0) order, based on Marching Squares algorithm.
        It assumes bin_segmask in yx order.
        """
        # yx --> xy order
        bin_segmask = np.swapaxes(bin_segmask, 0, 1)
    
        # Get 2D shape contour coords, using Marching Squares
        coords_contour_v = find_contours(bin_segmask.astype(np.uint8), 0.5)[0]  # (#points, 2 xy)   
        coords_contour_v = coords_contour_v.astype(np.int32)
        
        # Add z=0 to all coords: (#points_contour, 3 xy0)
        coords_contour_v = np.concatenate((coords_contour_v, np.zeros((len(coords_contour_v), 1))), axis=-1).astype(np.int32)
  
        return coords_contour_v
    # --------------------
    
    def get_mesh_from_bin_sax(self, bin_segmask:np.ndarray):
        """
        Extract mesh from binary segmentation mask, in voxel space, using Lewiner Marching Cubes.
        """    
        # Bool z-mask (which z-slices contain the shape)
        zmask = bin_segmask.any((1, 2))
    
        # Start z-slice of each shape (in voxel coords)
        # NOTE: it is needed to further correctly align the meshes in the z-direction
        z_init_v = np.where(zmask)[0][0] 
        
        # Extract mesh from binary segmentation mask, using Lewiner Marching Cubes
        verts, faces, _, _ = marching_cubes(bin_segmask[zmask], level=self.level)

        # Set vertices and faces in Pyvista PolyData format
        verts = verts[:, ::-1]                                    # Order from zyx to xyz
        faces = faces[:, ::-1]
        faces = np.pad(faces, ((0,0), (1,0)) , constant_values=3) # Add vertex counts
        faces = faces.ravel()
        mesh_v = pv.PolyData(verts, faces)                        # Mesh in voxel coords
        mesh_v.points = mesh_v.points + (z_init_v * np.array([0, 0, 1]))
        
        return mesh_v
    # --------------------
    
    def preprocess_sax(self, nii_sa:nib.Nifti1Image):
        """ 
        Preprocess a given SAX segmentation volume to sample surface and tolerance coords,
        with corresponding SDFs, in world-canonical orientation with referential's origin
        aligned with centers of mass, for each single- and multi-shape. 
        """
        # Check if 4D nifti
        if len(nii_sa.shape) == 4:
            # If so, get segmask for given phase
            segmask_sa = np.swapaxes(nii_sa.get_fdata()[..., PHASE_DICT[self.phase]], 0, 2)
            spacing_sa_zyx = nii_sa.header.get_zooms()[::-1][1::]   # tzyx order --> zyx
        else:
            segmask_sa = np.swapaxes(nii_sa.get_fdata(), 0, 2)      # xyz --> zyx order
            spacing_sa_zyx = nii_sa.header.get_zooms()[::-1]        # zyx order
        tolerance = self.surface_tolerance * np.sqrt(spacing_sa_zyx[2]**2 + spacing_sa_zyx[1]**2)

        # Dicts: {label: values, ...}
        sdf_fields_dict, idxs_surface_dict, idxs_tolerance_dict = {}, {}, {}
        
        for i, label in enumerate(self.meshes_labels):
            # Get binary segmask (zyx)
            segmask_sa_i = get_binary_segmask(segmask_sa, self.MRILabels[label])
            
            # Extract mesh (in voxel space) from binary segmentation mask, using Lewiner Marching Cubes 
            mesh_v_i = self.get_mesh_from_bin_sax(segmask_sa_i)
            
            # Get SDF field
            sdf_field_i = self.get_sdf_field(segmask_sa_i, spacing_sa_zyx)
            
            # Get SAX-segmask surface and tolerance voxel coords
            coords_surface_v_i, idxs_surface_i = self.sample_voxel_coords(segmask_sa_i, sdf_field_i, sdf_level=spacing_sa_zyx[2])
            coords_tolerance_v_i, idxs_tolerance_i = self.sample_voxel_coords(segmask_sa_i, sdf_field_i, sdf_level=tolerance)
            
            sdf_fields_dict[label] = sdf_field_i 
            idxs_surface_dict[label] = idxs_surface_i
            idxs_tolerance_dict[label] = idxs_tolerance_i
            
            # ---
            # Apply transformations to meshes
            mesh_i_single = mesh_v_i.transform(np.linalg.inv(self.M_cm_dict[label]) @ self.M_cano_sa, inplace=False)
            mesh_i_multi = mesh_v_i.transform(np.linalg.inv(self.M_cm_dict["multi"]) @ self.M_cano_sa, inplace=False)
            
            # Ensure normals point outwards
            mesh_i_single.compute_normals(flip_normals=True, inplace=True)
            mesh_i_multi.compute_normals(flip_normals=True, inplace=True)
            
            self.meshes_single_dict[self.pat_id][label] = mesh_i_single  # Origin = single-mesh center of mass
            self.meshes_multi_dict[self.pat_id][label] = mesh_i_multi    # Origin = multi-mesh center of mass
            
            # ---
            # Add homogeneous coordinate
            coords_surface_v_i = np.concatenate((coords_surface_v_i, np.ones((len(coords_surface_v_i), 1))), axis=-1)
            coords_tolerance_v_i = np.concatenate((coords_tolerance_v_i, np.ones((len(coords_tolerance_v_i), 1))), axis=-1)
            
            # Apply transformations to sampled SAX coords
            # (+ Remove homogeneous coord)
            self.coords_sa_single_dict[self.pat_id][label], self.coords_sa_multi_dict[self.pat_id][label] = {}, {}
            
            self.coords_sa_single_dict[self.pat_id][label]["surface"] = \
                (coords_surface_v_i @ self.M_cano_sa.T @ np.linalg.inv(self.M_cm_dict[label]).T)[..., :3]
            self.coords_sa_single_dict[self.pat_id][label]["tolerance"] = \
                (coords_tolerance_v_i @ self.M_cano_sa.T @ np.linalg.inv(self.M_cm_dict[label]).T)[..., :3]
            
            self.coords_sa_multi_dict[self.pat_id][label]["surface"] = \
                (coords_surface_v_i @ self.M_cano_sa.T @ np.linalg.inv(self.M_cm_dict["multi"]).T)[..., :3]
            self.coords_sa_multi_dict[self.pat_id][label]["tolerance"] = \
                (coords_tolerance_v_i @ self.M_cano_sa.T @ np.linalg.inv(self.M_cm_dict["multi"]).T)[..., :3]
            # ---
            
            # Get SDFs for surface and tolerance coords (single-meshes)
            self.sdfs_sa_single_dict[self.pat_id][label] = {}
            self.sdfs_sa_single_dict[self.pat_id][label]["surface"] = sdf_field_i.flat[idxs_surface_i].reshape(-1, 1)
            self.sdfs_sa_single_dict[self.pat_id][label]["tolerance"] = sdf_field_i.flat[idxs_tolerance_i].reshape(-1, 1)
            
        # Get SDFs for surface and tolerance coords (multi-meshes)
        for label in self.meshes_labels:
            self.sdfs_sa_multi_dict[self.pat_id][label] = {}
            self.sdfs_sa_multi_dict[self.pat_id][label]["surface"] = \
                np.zeros((len(self.coords_sa_multi_dict[self.pat_id][label]["surface"]), 3))
            self.sdfs_sa_multi_dict[self.pat_id][label]["tolerance"] = \
                np.zeros((len(self.coords_sa_multi_dict[self.pat_id][label]["tolerance"]), 3))
            for i, label_sdf_i in enumerate(self.meshes_labels):
                self.sdfs_sa_multi_dict[self.pat_id][label]["surface"][:, i] = sdf_fields_dict[label_sdf_i].flat[idxs_surface_dict[label]]
                self.sdfs_sa_multi_dict[self.pat_id][label]["tolerance"][:, i] = sdf_fields_dict[label_sdf_i].flat[idxs_tolerance_dict[label]]
    # --------------------
    
    def preprocess_lax(self, nii_la:nib.Nifti1Image):
        """ 
        Preprocess LAX segmentations to get boundary/surface coords and corresponding SDFs. 
        """
        # Check if 4D nifti
        if len(nii_la.shape) == 4:
            # If so, get segmask for given phase
            segmask_la = np.swapaxes(nii_la.get_fdata()[..., PHASE_DICT[self.phase]], 0, 2)[0]
            spacing_la_yx = nii_la.header.get_zooms()[::-1][2::]   # tzyx order --> yx
        else:
            segmask_la = np.swapaxes(nii_la.get_fdata(), 0, 2)[0]  # xyz --> zyx order --> yx !!!
            spacing_la_yx = nii_la.header.get_zooms()[::-1][1::]   # yx order
        
        # Dicts: {label: values, ...}
        sdf_fields_dict, idxs_surface_dict, coords_contour_v_dict = {}, {}, {}
        
        for i, label in enumerate(self.meshes_labels):
            # Get binary segmask (yx)
            segmask_la_i = get_binary_segmask(segmask_la, self.MRILabels[label])
            
            # Get SDF field
            sdf_field_i = self.get_sdf_field(segmask_la_i, spacing_la_yx)
            
            # Get LAX-segmask contour voxel coords (#points_contour, 3 xyz[z=0])
            coords_contour_v_i = self.sample_voxel_contour(segmask_la_i)
            
            # Get LAX-segmask surface voxel coords
            coords_surface_v_i, idxs_surface_i = self.sample_voxel_coords(segmask_la_i, sdf_field_i, sdf_level=spacing_la_yx[-1])
            
            # Add z=0 to all coords: (#points_contour, 3 xy0)
            coords_surface_v_i = np.concatenate((coords_surface_v_i, np.zeros((len(coords_surface_v_i), 1))), axis=-1).astype(np.int32)
            
            sdf_fields_dict[label] = sdf_field_i 
            idxs_surface_dict[label] = idxs_surface_i
            coords_contour_v_dict[label] = coords_contour_v_i
            
            # ---
            # Add homogeneous coordinate
            coords_contour_v_i = np.concatenate((coords_contour_v_i, np.ones((len(coords_contour_v_i), 1))), axis=-1)
            coords_surface_v_i = np.concatenate((coords_surface_v_i, np.ones((len(coords_surface_v_i), 1))), axis=-1)
            
            # Apply transformations to sampled LAX coords
            # (+ Remove homogeneous coord)
            self.coords_la_single_dict[self.pat_id][label], self.coords_la_multi_dict[self.pat_id][label] = {}, {}
            
            self.coords_la_single_dict[self.pat_id][label]["surface"] = \
                (coords_surface_v_i @ self.M_cano_la.T @ np.linalg.inv(self.M_cm_dict[label]).T)[..., :3]
            self.coords_la_single_dict[self.pat_id][label]["contour"] = \
                (coords_contour_v_i @ self.M_cano_la.T @ np.linalg.inv(self.M_cm_dict[label]).T)[..., :3]
            
            self.coords_la_multi_dict[self.pat_id][label]["surface"] = \
                (coords_surface_v_i @ self.M_cano_la.T @ np.linalg.inv(self.M_cm_dict["multi"]).T)[..., :3]
            self.coords_la_multi_dict[self.pat_id][label]["contour"] = \
                (coords_contour_v_i @ self.M_cano_la.T @ np.linalg.inv(self.M_cm_dict["multi"]).T)[..., :3]
            # ---
            
            # Get SDFs for surface and tolerance coords (single-meshes)
            self.sdfs_la_single_dict[self.pat_id][label] = {}
            self.sdfs_la_single_dict[self.pat_id][label]["surface"] = sdf_field_i.flat[idxs_surface_i].reshape(-1, 1)
            self.sdfs_la_single_dict[self.pat_id][label]["contour"] = np.zeros((len(coords_contour_v_i), 1))
            
        # Get SDFs for surface and tolerance coords (multi-meshes)
        for label in self.meshes_labels:
            self.sdfs_la_multi_dict[self.pat_id][label] = {}
            self.sdfs_la_multi_dict[self.pat_id][label]["surface"] = \
                np.zeros((len(self.coords_la_multi_dict[self.pat_id][label]["surface"]), 3))
            self.sdfs_la_multi_dict[self.pat_id][label]["contour"] = \
                np.zeros((len(self.coords_la_multi_dict[self.pat_id][label]["contour"]), 3))
            for i, label_sdf_i in enumerate(self.meshes_labels):
                self.sdfs_la_multi_dict[self.pat_id][label]["surface"][:, i] = sdf_fields_dict[label_sdf_i].flat[idxs_surface_dict[label]]
                if label_sdf_i != label:
                    self.sdfs_la_multi_dict[self.pat_id][label]["contour"][:, i] = \
                        sdf_fields_dict[label_sdf_i][coords_contour_v_dict[label][:, 1], coords_contour_v_dict[label][:, 0]] 
    # --------------------
    
    def preprocess(self):
        """
        Sample coords and SDFs from SAX and LAX (if it exists), in world canonical orientation.
        (surface and tolerance coords; single-shape and multi-shape)
        """
        # Dicts- SAX meshes: {pat_id: {"LVBP": mesh_LVBP, "LV": mesh_LV, "RVBP": mesh_RVBP}, ...}
        # --> dict[pat_id][label]
        self.meshes_single_dict = {}    # Origin = single mesh center of mass         
        self.meshes_multi_dict = {}     # Origin = multi-mesh center of mass 
        
        # Dicts - SAX coords: {pat_id: {"LVBP": {"surface": coords_LVBP_s, "tolerance": coords_LVBP_t}, ...}, ...}
        # --> dict[pat_id][label]["surface" or "tolerance"]
        self.coords_sa_single_dict, self.coords_sa_multi_dict= {}, {}
        self.sdfs_sa_single_dict, self.sdfs_sa_multi_dict = {}, {}
        
        # Dicts - LAX coords: {pat_id: {"LVBP": {"contour": coords_LVBP_c, "surface": coords_LVBP_t}, ...}, ...}
        # --> dict[pat_id][label]["contour" or "surface"]
        self.coords_la_single_dict, self.coords_la_multi_dict= {}, {}
        self.sdfs_la_single_dict, self.sdfs_la_multi_dict = {}, {} 
        self.transforms_dict, self.max_norms_dict = {}, {}
        
        preprocess_fails = False                                                            
        for sax_segm_path in tqdm.tqdm(self.sax_segm_paths, desc="Preprocessing SAX/LAX segmentations"):
            # Get patient ID & initialize dictionaries
            if isinstance(self.pat_id_digits, str):
                pat_id = Path(sax_segm_path).name.split(self.pat_id_digits)[0]
            else:
                pat_id = Path(sax_segm_path).stem[:self.pat_id_digits]
            self.pat_id = pat_id
            
            self.meshes_single_dict[pat_id], self.meshes_multi_dict[pat_id] = {}, {}
            self.coords_sa_single_dict[pat_id], self.coords_sa_multi_dict[pat_id] = {}, {}
            self.sdfs_sa_single_dict[pat_id], self.sdfs_sa_multi_dict[pat_id] = {}, {}
            self.coords_la_single_dict[pat_id], self.coords_la_multi_dict[pat_id] = {}, {}
            self.sdfs_la_single_dict[pat_id], self.sdfs_la_multi_dict[pat_id] = {}, {}
            
            if len(self.M_dict) > 0:
                # Use pre-computed transforms
                self.M_cano_sa = self.M_dict[pat_id]["cano_sa"]
                self.M_cano_la = self.M_dict[pat_id]["cano_la"]
                self.M_cm_dict = self.M_dict[pat_id]["cm_dict"]
            else:
                # Get transformation matrices: canonical rotation & c.o.m. translation to origin (0,0,0)
                self.M_cano_sa, self.M_cano_la, self.M_cm_dict = get_canonical_transform_mri(sax_segm_path, self.MRILabels, 
                                                                          align_x=self.canonical_align_x, flip_y=self.canonical_flip_y)
            if self.save_transforms:
                self.transforms_dict[pat_id] = {"cano_sa": self.M_cano_sa, 
                                                "cano_la": self.M_cano_la,
                                                "cm_dict": self.M_cm_dict}
            try:
                # Get SAX meshes, coords and SDFs (with applied canonical transformations)
                nii_sa = nib.load(sax_segm_path)
                self.preprocess_sax(nii_sa)
                
                if self.M_cano_la is not None:
                    # Get LAX coords and SDFs (with applied canonical transformations)
                    nii_la = nib.load(sax_segm_path.replace("SA", "LA"))
                    self.preprocess_lax(nii_la)
            
                # Get max norm amongst tolerance SAX & surface LAX multi-shape coords
                coords_sa = [self.coords_sa_multi_dict[pat_id][label]["tolerance"] for label in self.meshes_labels]
                if self.M_cano_la is not None:
                    coords_la = [self.coords_la_multi_dict[pat_id][label]["surface"] for label in self.meshes_labels] 
                else:
                    coords_la = []
                total_coords = np.concatenate(coords_sa + coords_la, axis=0)
                norms = [norm(np.array(coord)) for coord in total_coords]
                max_norm = max(norms)
                self.max_norm = max(self.max_norm, max_norm)
                self.max_norms_dict[pat_id] = float(max_norm)
            
            except Exception as e:
                if not preprocess_fails:
                    preprocess_fails = True  
                with open(f"data_preprocessing/errors/failed_preprocessing_{datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.txt", "a") as log:
                    log.write(f"{pat_id}: {type(e).__name__} - {e}\n")
                    log.write(traceback.format_exc() + "\n\n")
        
        if preprocess_fails:
            print("Some segmentations could not be preprocessed! Check the log file for details.")
                
        print(f"Max norm accross dataset (multi-centered based!!!): {self.max_norm}")
        self.config["PREPROCESSING"]["true_max_norm"] = float(self.max_norm)
    # -------------------- 
    
    def normalize_coords(self):
        """ Normalize mesh coordinates to [-1,1]^3 space. """
        
        if self.config["PREPROCESSING"]["max_norm_coords"] > 0: # Use max norm from config file in case it is set
            self.max_norm = self.config["PREPROCESSING"]["max_norm_coords"]
            
        if self.max_norm > 0:
            self.config["PREPROCESSING"]["max_norm_coords"] = float(self.max_norm)
            meshes_single_norm_dict = {}
            meshes_multi_norm_dict = {}
            
            for pat_id, meshes_dict in tqdm.tqdm(self.meshes_single_dict.items(), desc="Normalizing mesh coordinates"):
                # Normalize meshes, IF they were successfully extracted
                if len(meshes_dict) != 0:
                    meshes_single_norm_dict[pat_id] = {}
                    meshes_multi_norm_dict[pat_id] = {}
                    for label, mesh in meshes_dict.items():
                        # Normalize SAX single-centered meshes
                        mesh_single_i = mesh.copy()  
                        mesh_single_i.points = (mesh_single_i.points / self.max_norm) * self.scale_factor

                        # Normalize SAX multi-centered meshes
                        mesh_multi_i = self.meshes_multi_dict[pat_id][label].copy()
                        mesh_multi_i.points = (mesh_multi_i.points / self.max_norm) * self.scale_factor
                        
                        meshes_single_norm_dict[pat_id][label] = mesh_single_i
                        meshes_multi_norm_dict[pat_id][label] = mesh_multi_i
                        
                        # ---
                        for coords_type in ["surface", "tolerance"]:
                            # Normalize SAX single-centered coords and SDFs
                            self.coords_sa_single_dict[pat_id][label][coords_type] = \
                                (self.coords_sa_single_dict[pat_id][label][coords_type] / self.max_norm) * self.scale_factor
                            self.sdfs_sa_single_dict[pat_id][label][coords_type] = \
                                (self.sdfs_sa_single_dict[pat_id][label][coords_type] / self.max_norm) * self.scale_factor
                            
                            # Normalize SAX multi-centered coords and SDFs
                            self.coords_sa_multi_dict[pat_id][label][coords_type] = \
                                (self.coords_sa_multi_dict[pat_id][label][coords_type] / self.max_norm) * self.scale_factor
                            self.sdfs_sa_multi_dict[pat_id][label][coords_type] = \
                                (self.sdfs_sa_multi_dict[pat_id][label][coords_type] / self.max_norm) * self.scale_factor
                        
                        # ---
                        if len(self.coords_la_single_dict[pat_id]) > 0:
                            for coords_type in ["surface", "contour"]:
                                # Normalize LAX single-centered coords and SDFs
                                self.coords_la_single_dict[pat_id][label][coords_type] = \
                                    (self.coords_la_single_dict[pat_id][label][coords_type] / self.max_norm) * self.scale_factor
                                self.sdfs_la_single_dict[pat_id][label][coords_type] = \
                                    (self.sdfs_la_single_dict[pat_id][label][coords_type] / self.max_norm) * self.scale_factor
                                
                                # Normalize LAX multi-centered coords and SDFs
                                self.coords_la_multi_dict[pat_id][label][coords_type] = \
                                    (self.coords_la_multi_dict[pat_id][label][coords_type] / self.max_norm) * self.scale_factor
                                self.sdfs_la_multi_dict[pat_id][label][coords_type] = \
                                    (self.sdfs_la_multi_dict[pat_id][label][coords_type] / self.max_norm) * self.scale_factor
            
            self.meshes_single_norm_dict = meshes_single_norm_dict
            self.meshes_multi_norm_dict = meshes_multi_norm_dict
        else:
            print("No max norm found! Run preprocess() first.")
        
        print(f"Max norm used for Normalization: {self.max_norm}")
    # --------------------
    
    def save_meshes(self):
        """ 
        Save meshes and point clouds in .ply format, as well as coords 
        and SDFs in .npz format, as a dict with the following keys:
            - 'sax_surface_coords'
            - 'sax_tolerance_coords'
            - 'sax_surface_sdfs'
            - 'sax_tolerance_sdfs'
            - 'lax_contour_coords'
            - 'lax_contour_sdfs'
            - 'lax_surface_coords'
            - 'lax_surface_sdfs'
            - 'tolerance' 
        It is to be noted that all data is both saved centered on the multi-shape c.o.m.
        and on each single-shape c.o.m. (for both multi and individual inference models).
        
        The file 'config_preprocess_mri.json' should also be saved.
        """	
        # Create directories to save the meshes
        # NOTE: the meshes will be organized in single- and multi-centered folders,
        #       with each mesh label having its own subdirectory.
        subdir_single = os.path.join(self.meshes_dir, "single")
        subdir_multi = os.path.join(self.meshes_dir, "multi")
        
        for subdir in [subdir_single, subdir_multi]:
            if not os.path.exists(subdir):
                Path(subdir).mkdir(exist_ok=True, parents=True)
       
            for label in self.meshes_labels:
                subdir_label = os.path.join(self.meshes_dir, subdir, label)
                subdir_label_og = os.path.join(self.meshes_dir, subdir, f"{label}_og")
                if not os.path.exists(subdir_label):
                    Path(subdir_label).mkdir(exist_ok=True, parents=True)
                if not os.path.exists(subdir_label_og):
                    Path(subdir_label_og).mkdir(exist_ok=True, parents=True)
                    
        if not os.path.exists(os.path.join(subdir_multi, "multi")):
                Path(os.path.join(subdir_multi, "multi")).mkdir(exist_ok=True, parents=True)
        # ----------
        
        # Save (multi-centered based) max norms in .json file
        with open(os.path.join(self.meshes_dir, "max_norms.json"), "w") as f:
            json.dump(self.max_norms_dict, f, indent=4)
            
        # SAVE transforms in .npz file
        if self.save_transforms:
            # np.savez(os.path.join(self.meshes_dir, "transforms.npz"), **self.transforms_dict, allow_pickle=True)
            with open(os.path.join(self.meshes_dir, "transforms.pkl"), "wb") as f:
                pickle.dump(self.transforms_dict, f)
        
        # SAVE preprocessing config file in the meshes directory
        with open(os.path.join(self.meshes_dir, "config_preprocess_mri.json"), "w") as f:
            json.dump(self.config, f, indent=4)
        # ----------
     
        for pat_id, meshes_dict in tqdm.tqdm(self.meshes_single_norm_dict.items(), desc="Saving meshes and SDFs"):
            # Save meshes, IF they were successfully extracted
            if len(meshes_dict) != 0:
                for i, label in enumerate(self.meshes_labels):
                    # Save all meshes in .ply format:
                    # -- Single-centered meshes (normalized and original space)
                    meshes_dict[label].save(os.path.join(subdir_single, label, f"{pat_id}_{label}.ply"))
                    self.meshes_single_dict[pat_id][label].save(os.path.join(subdir_single, f"{label}_og", f"{pat_id}_{label}.ply"))
                    # -- Multi-centered meshes (normalized and original space)
                    self.meshes_multi_norm_dict[pat_id][label].save(os.path.join(subdir_multi, label, f"{pat_id}_{label}.ply"))
                    self.meshes_multi_dict[pat_id][label].save(os.path.join(subdir_multi, f"{label}_og", f"{pat_id}_{label}.ply"))
                    # -----
                    
                    # Save SAX point clouds in .ply format (single- and multi-centered):
                    for coords_type in ["surface", "tolerance"]:
                        pv.PolyData(self.coords_sa_single_dict[pat_id][label][coords_type]).save(
                                os.path.join(subdir_single, label, f"{pat_id}_{label}_{coords_type}_sa.ply"))
                        pv.PolyData(self.coords_sa_multi_dict[pat_id][label][coords_type]).save(
                                os.path.join(subdir_multi, label, f"{pat_id}_{label}_{coords_type}_sa.ply"))
                    
                    if len(self.coords_la_single_dict[pat_id]) > 0:
                        # Save LAX point clouds in .ply format:
                        for coords_type in ["surface", "contour"]:
                            pv.PolyData(self.coords_la_single_dict[pat_id][label][coords_type]).save(
                                    os.path.join(subdir_single, label, f"{pat_id}_{label}_{coords_type}_la.ply"))
                            pv.PolyData(self.coords_la_multi_dict[pat_id][label][coords_type]).save(
                                    os.path.join(subdir_multi, label, f"{pat_id}_{label}_{coords_type}_la.ply"))
                    # -----
                        # SAVE single-centered mesh data as a dictionary in .npz format (with LAX data)
                        np.savez(os.path.join(subdir_single, label, f"{pat_id}_{label}.npz"),
                                sax_surface_coords=self.coords_sa_single_dict[pat_id][label]["surface"], 
                                sax_tolerance_coords=self.coords_sa_single_dict[pat_id][label]["tolerance"],
                                sax_surface_sdfs=self.sdfs_sa_single_dict[pat_id][label]["surface"], 
                                sax_tolerance_sdfs=self.sdfs_sa_single_dict[pat_id][label]["tolerance"],
                                lax_contour_coords=self.coords_la_single_dict[pat_id][label]["contour"],
                                lax_surface_coords=self.coords_la_single_dict[pat_id][label]["surface"],
                                lax_contour_sdfs=self.sdfs_la_single_dict[pat_id][label]["contour"],
                                lax_surface_sdfs=self.sdfs_la_single_dict[pat_id][label]["surface"],
                                tolerance=self.surface_tolerance)
                    else:
                        # SAVE single-centered mesh data as a dictionary in .npz format (w/o LAX data)
                        np.savez(os.path.join(subdir_single, label, f"{pat_id}_{label}.npz"),
                                sax_surface_coords=self.coords_sa_single_dict[pat_id][label]["surface"], 
                                sax_tolerance_coords=self.coords_sa_single_dict[pat_id][label]["tolerance"],
                                sax_surface_sdfs=self.sdfs_sa_single_dict[pat_id][label]["surface"], 
                                sax_tolerance_sdfs=self.sdfs_sa_single_dict[pat_id][label]["tolerance"],
                                tolerance=self.surface_tolerance)
                        
                if len(self.coords_la_multi_dict[pat_id]) > 0:      
                    # SAVE multi-centered mesh data as a dictionary in .npz format (with LAX data)
                    np.savez(os.path.join(subdir_multi, "multi", f"{pat_id}.npz"),
                            sax_surface_coords=np.concatenate([self.coords_sa_multi_dict[pat_id][label]["surface"] for label in self.meshes_labels], axis=0),
                            sax_tolerance_coords=np.concatenate([self.coords_sa_multi_dict[pat_id][label]["tolerance"] for label in self.meshes_labels], axis=0),
                            sax_surface_sdfs=np.concatenate([self.sdfs_sa_multi_dict[pat_id][label]["surface"] for label in self.meshes_labels], axis=0),
                            sax_tolerance_sdfs=np.concatenate([self.sdfs_sa_multi_dict[pat_id][label]["tolerance"] for label in self.meshes_labels], axis=0),
                            lax_contour_coords=np.concatenate([self.coords_la_multi_dict[pat_id][label]["contour"] for label in self.meshes_labels], axis=0),
                            lax_surface_coords=np.concatenate([self.coords_la_multi_dict[pat_id][label]["surface"] for label in self.meshes_labels], axis=0),
                            lax_contour_sdfs=np.concatenate([self.sdfs_la_multi_dict[pat_id][label]["contour"] for label in self.meshes_labels], axis=0),
                            lax_surface_sdfs=np.concatenate([self.sdfs_la_multi_dict[pat_id][label]["surface"] for label in self.meshes_labels], axis=0),
                            tolerance=self.surface_tolerance)
                else:
                    # SAVE multi-centered mesh data as a dictionary in .npz format (w/o LAX data)
                    np.savez(os.path.join(subdir_multi, "multi", f"{pat_id}.npz"),
                            sax_surface_coords=np.concatenate([self.coords_sa_multi_dict[pat_id][label]["surface"] for label in self.meshes_labels], axis=0),
                            sax_tolerance_coords=np.concatenate([self.coords_sa_multi_dict[pat_id][label]["tolerance"] for label in self.meshes_labels], axis=0),
                            sax_surface_sdfs=np.concatenate([self.sdfs_sa_multi_dict[pat_id][label]["surface"] for label in self.meshes_labels], axis=0),
                            sax_tolerance_sdfs=np.concatenate([self.sdfs_sa_multi_dict[pat_id][label]["tolerance"] for label in self.meshes_labels], axis=0),
                            tolerance=self.surface_tolerance)     
    # --------------------
    
    def get_matrix_coords(self, sax_path):
        """ 
        Function to get SAX and LAX matrix coordinates:
            - SAX image coords in SAX voxel space (coords_sa_v) and world space (coords_sa_w)
            - LAX image coords in LAX voxel space (coords_la_v), world space (coords_la_w) and 
              transformed to SAX voxel space (coords_la_vsa)
        """
        lax_path = sax_path.replace("SA", "LA")
        nii_sa = nib.load(sax_path)
        nii_la = nib.load(lax_path)
        M_affine_sa = get_affine(nii_sa)
        M_affine_la = get_affine(nii_la)
        self.M_affine_la = M_affine_la
        self.img_sa = nii_sa.get_fdata()
        self.spacing_la = nii_la.header.get_zooms()  # Get LAX in-plane spacing 
        self.spacing_sa = nii_sa.header.get_zooms()  # Get SAX spacing
        
        # Get SAX (xyz) and LAX shapes (xy)
        sa_shape = nii_sa.header.get_data_shape()      # xyz  
        la_shape = nii_la.header.get_data_shape()[:2]  # xyz --> xy
        self.la_shape_xy = la_shape
        self.sa_shape = sa_shape
        
        # Get all voxels index positions (1D vector)
        idxs_sa = np.arange(0, sa_shape[0] * sa_shape[1] * sa_shape[2]) 
        idxs_la = np.arange(0, la_shape[0] * la_shape[1])
        
        # Get SAX matrix voxel coords (xyz 
        # order)
        coords_sa_v = np.unravel_index(idxs_sa, sa_shape)
        coords_sa_v = np.stack(coords_sa_v, -1)     # Stack lists in array (#points, 3xyz)
        
        # Get LAX plane voxel coords (xy0 order)
        coords_la_v = np.unravel_index(idxs_la, la_shape)
        coords_la_v = np.stack(coords_la_v, -1)     # Stack lists in array (#points, 2xy)
        # Add z=0 to all coords: (#points, 3 xy0)
        coords_la_v = np.concatenate((coords_la_v, np.zeros((len(coords_la_v), 1))), axis=-1).astype(np.int32)
        
        # Add homogeneous coordinate + Apply affine transformation
        coords_sa_v_h = np.concatenate((coords_sa_v, np.ones((len(coords_sa_v), 1))), axis=-1)
        coords_sa_w = (coords_sa_v_h @ M_affine_sa.T)[..., :3]  
        
        coords_la_v_h = np.concatenate((coords_la_v, np.ones((len(coords_la_v), 1))), axis=-1)
        coords_la_w_h = (coords_la_v_h @ M_affine_la.T)
        coords_la_w = coords_la_w_h[..., :3]
        
        # Transform LAX world space coords to SAX voxel space
        coords_la_vsa = (coords_la_w_h @ np.linalg.inv(M_affine_sa.T))[..., :3]
        coords_la_vsa = np.round(coords_la_vsa).astype(np.int32)
        
        sax_dict = {'coords_v': coords_sa_v, 'coords_w': coords_sa_w}   
        lax_dict = {'coords_v': coords_la_v, 'coords_w': coords_la_w, 'coords_vsa': coords_la_vsa}

        return sax_dict, lax_dict
    # --------------------
    
    def save_cross_sections(self):
        for sax_segm_path in tqdm.tqdm(self.sax_segm_paths, desc="Saving cross-sections"):
            if isinstance(self.pat_id_digits, str):
                pat_id = Path(sax_segm_path).name.split(self.pat_id_digits)[0]
            else:
                pat_id = Path(sax_segm_path).stem[:self.pat_id_digits]
            
            sax_dict, lax_dict = self.get_matrix_coords(sax_segm_path)
            
            # Build the k-d tree with the reference set (sax)
            # tree = cKDTree(sax_dict["coords_w"])
            # tree = cKDTree(sax_dict["coords_v"])

            # For each point in lax, find the nearest neighbor in sax
            # distances, indices = tree.query(lax_dict["coords_w"], k=1)  
            # distances, indices = tree.query(lax_dict["coords_vsa"], k=1) 
            # distances (M,): the Euclidean distance to the nearest neighbor
            # indices (M,): the index in sax_coords of the nearest neighbor for each lax_coords  
            
            # -------
            # 1) Get LAX coords in SAX voxel space (LAX_vsa)
            # 2) Get valid LAX_vsa voxels that fall within the SAX volume
            # 3) Get SAX values at those valid LAX_vsa voxels
            
            coords_la_v = lax_dict["coords_v"]
            coords_la_vsa = lax_dict["coords_vsa"]
            
            min_vsa_values = np.array([0, 0, 0])
            max_vsa_values = np.array([self.sa_shape[0], self.sa_shape[1], self.sa_shape[2]])
            mask_vsa = np.all((coords_la_vsa >= min_vsa_values) & (coords_la_vsa < max_vsa_values), axis=1)
            
            coords_la_v_valid = coords_la_v[mask_vsa]
            coords_la_vsa_valid = coords_la_vsa[mask_vsa]
            
            sax_values = self.img_sa[coords_la_vsa_valid[:, 0], coords_la_vsa_valid[:, 1], coords_la_vsa_valid[:, 2]]
            lax_cross = np.zeros((self.la_shape_xy[0], self.la_shape_xy[1], 1), dtype=np.float32)
            lax_cross[coords_la_v_valid[:, 0], coords_la_v_valid[:, 1], coords_la_v_valid[:, 2]] = sax_values
            # -------
            
            # diag_sa = np.linalg.norm(self.spacing_sa)  # Diagonal of SAX voxel (in mm)
            # for i, idx_sax in enumerate(indices):
                # dist = distances[i]
                # if dist < diag_sa * 0.5:  # If the distance is within the diagonal of the SAX voxel
                #     sax_value = self.img_sa[sax_dict["coords_v"][idx_sax][0], sax_dict["coords_v"][idx_sax][1], sax_dict["coords_v"][idx_sax][2]]
                # else:
                #     sax_value = 0
                # lax_cross[lax_dict["coords_v"][i][0], lax_dict["coords_v"][i][1], lax_dict["coords_v"][i][2]] = sax_value
            
            # Save with nibabel
            lax_cross_img = nib.Nifti1Image(lax_cross, affine=self.M_affine_la)
            if not (Path(sax_segm_path).parent.parent / "cross").exists():
                (Path(sax_segm_path).parent.parent / "cross").mkdir(parents=True, exist_ok=True)
            nib.save(lax_cross_img, Path(sax_segm_path).parent.parent / "cross" / f"{pat_id}.nii.gz")      
# =============================================== 


if __name__=="__main__":
    # TODO: Define phase Dict
    PHASE_DICT = {"ED":0, "ES":1}
    
    # Load CMRI preprocessing config file
    config_pre = json.load(open('configs/config_preprocess_mri.json'))
    
    multi_mesh_pre = PreprocessMultiMeshCMRI(config_pre)
    multi_mesh_pre.preprocess()
    multi_mesh_pre.normalize_coords()
    multi_mesh_pre.save_meshes()
    multi_mesh_pre.save_cross_sections()
    
    

            