# Imports
from pathlib import Path
import pyvista as pv
import numpy as np
import os
import tqdm
import sys 
from pathlib import Path
import json
import glob
from typing import Optional, Union
import trimesh
import nibabel as nib
from skimage.measure import marching_cubes

sys.path.append(str(Path(__file__).resolve().parent.parent)) # Add project root to sys.path
from data_labels import *
from canonical_transform import get_canonical_transform_cta, get_binary_segmask

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CONVERT CTA SEGMENTATIONS TO MESHES - LVBP (LV ENDOCARDIUM), LV (LV EPICARDIUM) and RVBP
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 	


def norm(vec):
    """ Return the norm of a vector. """
    return np.sqrt(np.sum(vec**2))
# ===============================================


class PreprocessMultiMeshCTA:
    """ 
    Preprocess CTA segmentations tomeshes of LVBP (LV Endocardium), LV (LV myocardium)
    and RVBP, with the following transformations:
        -- self.preprocess():
            - Rotation into the canonical orientation of the reference world coordinate system + apical-basal direction;
            - Alignment of multi-mesh center of mass with the origin of the reference coordinate system
            - Computing mesh normals so that they point outwards.
            - Get coords and corresponding SDFs from CTA segmentations.

        -- self.normalize_coords():    
            - Normalization of all mesh coordinates to [-1,1]^3 space, considering the max norm of coordinates across all meshes
            and a scaling factor.
        
        -- self.save_meshes():   
            - Save meshes in .ply format.    
    Args:
        -- config (dict): preprocessing configuration parameters.
        -- load_first_segm (int): number of segmentations to process.
        -- load_segm_ids (list): list of segmentation CTA IDs to process.   
    """
    def __init__(self, config: dict, load_first_segms:Optional[int]=None, load_segm_ids:Optional[list[str]]=None):
        self.config = config
        self.segm_dirs = [os.path.join(config["DATA"]["data_dir"], segm_folder) for segm_folder in config["DATA"]["segm_folders"]]
        self.segm_paths = []
        for segm_dir in self.segm_dirs:
            self.segm_paths += sorted(glob.glob(os.path.join(segm_dir, "*.nii.gz")))
        if load_first_segms:
            self.segm_paths = self.segm_paths[:load_first_segms]
        if load_segm_ids:
            self.segm_paths = [p for p in self.segm_paths if Path(p).stem in load_segm_ids]
        
        self.meshes_dir = os.path.join(config["DATA"]["data_dir"], config["DATA"]["mesh_folder"])
        self.meshes_labels =config["DATA"]["mesh_labels"]
        self.samples_per_mesh = config["PREPROCESSING"]["samples_per_mesh"]
        self.max_norm = 0   # Max norm of mesh coordinates
        self.max_norm_multi = 0
        
        self.scale_factor = config["PREPROCESSING"]["scale_factor"]
        self.segm_padding = config["PREPROCESSING"]["segm_padding"]
        self.sigma = config["PREPROCESSING"]["sigma_gaussian"]
        self.CTALabels = self.str_to_class(config["DATA"]["labels_class"])  # Enum with labels for CTA segmentations
        self.pat_id_digits = config["DATA"]["pat_id_digits"]                # Number of digits in patient ID
        self.canonical_flip_y = config["PREPROCESSING"]["canonical_flip_y"]
        self.canonical_align_x = config["PREPROCESSING"]["canonical_align_x"]

        self.rng = np.random.default_rng(seed=config["GENERAL"]["seed"])
        np.random.seed(seed=config["GENERAL"]["seed"])      
    # --------------------
    
    def str_to_class(self, classname:str):
        return getattr(sys.modules[__name__], classname)
    # -------------------- 
       
    def get_meshes_from_segm(self, segm_path:Union[Path, str]) -> dict:
        """
        Extract meshes (in world spacing) from a CTA segmentation mask,
        using Lewiner Marching Cubes algorithm.
        
        Args:
            - segm_path (Path): Path to the segmentation mask file (.nii.gz).
        Returns:
            - meshes_dict (dict): Dictionary of meshes with labels as keys: {label: mesh}.
        """
        # Get segmentation mask & spacing (xyz order)
        nii_cta = nib.load(str(segm_path))
        segmask = nii_cta.get_fdata()   
        spacing_xyz = nii_cta.header.get_zooms()  
        
        # Dict of meshes: {label: mesh} 
        meshes_dict = {}
        
        for label in tqdm.tqdm(self.meshes_labels, desc="Extracting meshes"):
            # Get LCC binary segmask
            segmask_i = get_binary_segmask(segmask, self.CTALabels[label], pad=self.segm_padding)
        
            # Extract mesh from binary segmentation mask (in world spacing)
            verts, faces, _, _ = marching_cubes(segmask_i, level=0.5, spacing=spacing_xyz)
        
            # Set vertices and faces in Pyvista PolyData format
            faces = np.pad(faces, ((0,0), (1,0)) , constant_values=3) # Add vertex counts
            faces = faces.ravel()
            mesh_i = pv.PolyData(verts, faces)
            meshes_dict[label] = mesh_i
        
        return meshes_dict
    # --------------------
        
    def preprocess(self):
        """ 
        Get meshes from segmentations and preprocess them. 
        """
        # Dicts: {cta_id: {"LVBP": mesh_LVBP, "LV": mesh_LV, "RVBP": mesh_RVBP}, ...}
        self.meshes_single_dict = {}    # Origin = single mesh center of mass         
        self.meshes_multi_dict = {}     # Origin = multi-mesh center of mass       
        
        for segm_path in tqdm.tqdm(self.segm_paths, desc="Preprocessing CTA segmentations"):
            # Extract meshes from segmentation (using Lewiner Matching Cubes), in world spacing
            meshes_dict = self.get_meshes_from_segm(segm_path)
            
            # Get transformation matrices: canonical rotation & c.o.m. translation to origin (0,0,0)
            M_cano, M_cm_dict = get_canonical_transform_cta(segm_path, self.CTALabels, 
                                                           align_x=self.canonical_align_x, flip_y=self.canonical_flip_y)
            # Apply transformations to meshes
            single_dict, multi_dict = {}, {}
            for label, mesh in tqdm.tqdm(meshes_dict.items(), desc="Transforming meshes"):
                mesh_i_single = mesh.transform(np.linalg.inv(M_cm_dict[label]) @ M_cano, inplace=False)
                mesh_i_multi = mesh.transform(np.linalg.inv(M_cm_dict["multi"]) @ M_cano, inplace=False)
                
                # Ensure normals point outwards
                mesh_i_single.compute_normals(flip_normals=True, inplace=True)
                mesh_i_multi.compute_normals(flip_normals=True, inplace=True)
                
                single_dict[label] = mesh_i_single  # Origin = single mesh center of mass
                multi_dict[label] = mesh_i_multi    # Origin = multi-mesh center of mass
                
                if not(mesh_i_single.is_manifold):
                    print(f"Mesh {label} - {Path(segm_path).stem} is not watertight! Check normals!")
    
            # Get max norm of all mesh coords (from multi-centered meshes!!!)
            norms = [norm(p) for p in pv.merge(list(multi_dict.values())).points]
            max_norm = max(norms)
            self.max_norm = max(self.max_norm, max_norm)
            
            self.meshes_single_dict[Path(segm_path).stem[:self.pat_id_digits]] = single_dict
            self.meshes_multi_dict[Path(segm_path).stem[:self.pat_id_digits]] = multi_dict
                
        print(f"Max norm accross all multi-centered meshes: {self.max_norm}")
        print(f"Number of processed meshes: {len(self.meshes_single_dict)}")
    # --------------------
    
    def normalize_coords(self):
        """ Normalize mesh coordinates to [-1,1]^3 space. """
        
        if self.config["PREPROCESSING"]["max_norm_coords"] > 0: # Use max norm from config file in case it is set
            self.max_norm = self.config["PREPROCESSING"]["max_norm_coords"]
            
        if self.max_norm > 0:
            self.config["PREPROCESSING"]["max_norm_coords"] = float(self.max_norm)
            meshes_single_norm_dict = {}
            meshes_multi_norm_dict = {}
            
            for cta_id, meshes_dict in tqdm.tqdm(self.meshes_single_dict.items(), desc="Normalizing mesh coordinates"):
                meshes_single_norm_dict[cta_id] = {}
                meshes_multi_norm_dict[cta_id] = {}
                for label, mesh in meshes_dict.items():
                    # Normalize single-centered meshes
                    mesh_single_i = mesh.copy()  
                    mesh_single_i.points = (mesh_single_i.points / self.max_norm) * self.scale_factor

                    # Normalize multi-centered meshes
                    mesh_multi_i = self.meshes_multi_dict[cta_id][label].copy()
                    mesh_multi_i.points = (mesh_multi_i.points / self.max_norm) * self.scale_factor
                    
                    meshes_single_norm_dict[cta_id][label] = mesh_single_i
                    meshes_multi_norm_dict[cta_id][label] = mesh_multi_i
            
            self.meshes_single_norm_dict = meshes_single_norm_dict
            self.meshes_multi_norm_dict = meshes_multi_norm_dict
        else:
            print("No max norm found! Run preprocess() first.")
        
        # print(f"Max norm accross all meshes: {self.max_norm}")
    # --------------------
    
    def sample_normals(self, mesh:trimesh.Trimesh, surface_points:np.array, faces_idx:np.array):
        """ 
        Sample normals at given mesh surface points. 
        
        Args:
            -- mesh (trimesh.Trimesh): mesh to sample from.
            -- surface_points (np.array): surface points/vertices at which to get the normals.
            -- faces_idx (np.array): indices of the mesh faces/triangles that the surfaces points belong to.
        
        Returns:
            -- normals (np.array): sampled normals.
        """
        # Barycentric coordinates of the surface points relative to the mesh triangles (given their vertices) - (n,3)
        bary_points = trimesh.triangles.points_to_barycentric(triangles=mesh.triangles[faces_idx], 
                                                              points=surface_points) 
        normals = trimesh.unitize((mesh.vertex_normals[mesh.faces[faces_idx]] * trimesh.unitize(bary_points).reshape((-1, 3, 1))).sum(axis=1))
        return normals
    # --------------------
    
    def sample_mesh(self, mesh:pv.PolyData, n_points:int=200_000):
        """ 
        Sample mesh surface points and corresponding normals.
        Returns False if mesh is not all triangles.
        
        Args:
            -- mesh (pv.PolyData): mesh to sample.
            -- n_points (int): number of points to sample. 
        
        Returns:
            -- surface_points (np.array): sampled surface points.
            -- normals (np.array): corresponding normals.
        """
        if mesh.is_all_triangles:
            # Convert mesh from pyvista to trimesh
            mesh_tri = trimesh.Trimesh(vertices=mesh.points, faces=mesh.faces.reshape(-1, 4)[:, 1:] ) 
            surface_points, faces_idx = mesh_tri.sample(n_points, return_index=True)
            normals = self.sample_normals(mesh_tri, surface_points, faces_idx)
            return surface_points, normals
        else:
            return False, False
    # --------------------
    
    def sdf(self, coords:np.array, mesh:pv.PolyData):
        """ Compute signed distance function for a mesh at given coordinates. """
        impl_dist = pv.PolyData(coords).compute_implicit_distance(mesh)
        return np.asarray(impl_dist.active_scalars, dtype=np.float32)
    # --------------------
    
    def save_data(self):
        """ 
        -- Save all meshes in .ply format (in original (OG) and [-1,1]^3 normalized space,
          single- and multi-centered). 
        -- Sample and save meshes' volume points, surface points, normals and signed distance
          functions (SDFs) as a dict in .npz format (only for normalized meshes).
        -- Save preprocessing config file in the meshes directory.
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
        
        # SAVE preprocessing config file in the meshes directory
        with open(os.path.join(self.meshes_dir, "config_preprocess_cta.json"), "w") as f:
            json.dump(self.config, f, indent=4)
        # ----------
        
        for cta_id, meshes_dict in tqdm.tqdm(self.meshes_single_norm_dict.items(), desc="Saving meshes and SDFs"):
            # Lists of sampled data from multi-centered meshes: {label: data} 
            surface_multi, normals_multi, sdfs_surface_multi = [], [], []
            volume_multi, sdfs_volume_multi = [], []
            
            for i, label in enumerate(self.meshes_labels):
                # Save all meshes in .ply format:
                # -- Single-centered meshes (normalized and original space)
                meshes_dict[label].save(os.path.join(subdir_single, label, f"{cta_id}_{label}.ply"))
                self.meshes_single_dict[cta_id][label].save(os.path.join(subdir_single, f"{label}_og", f"{cta_id}_{label}.ply"))
                # -- Multi-centered meshes (normalized and original space)
                self.meshes_multi_norm_dict[cta_id][label].save(os.path.join(subdir_multi, label, f"{cta_id}_{label}.ply"))
                self.meshes_multi_dict[cta_id][label].save(os.path.join(subdir_multi, f"{label}_og", f"{cta_id}_{label}.ply"))
                # -----
                
                # 1. SAMPLE FROM SINGLE-CENTERED NORMALIZED MESHES
                # Sample mesh surface coords and respective normals: (#points, 3xyz)
                mesh_single_i = meshes_dict[label]
                surface_i, normals_i = self.sample_mesh(mesh_single_i, n_points=self.samples_per_mesh)
                
                if isinstance(surface_i, bool):
                    print(f"Mesh {label} - {cta_id} is not all triangles! No sampling done.")
                    continue
                else:
                    # Sample coords around surface, by adding Gaussian noise: (#points, 3xyz) - Volume coords
                    # Note: The noise is also scaled to [-1,1]^3 space
                    gauss_noise_i = self.sigma * np.random.randn(*surface_i.shape)
                    volume_i = surface_i + gauss_noise_i 
                    
                    # Get SDF value at each sampled surface coord: (#points, #sdfs=1) -- SDF_on_surface = 0
                    sdfs_surface_i = np.zeros([surface_i.shape[0], 1])
                    
                    # Get SDF value at each sampled volume coord: (#points, #sdfs=1)
                    sdfs_volume_i = self.sdf(volume_i, mesh_single_i)
                    
                    # SAVE single-centered mesh data as a dictionary in .npz format
                    np.savez(os.path.join(subdir_single, label, f"{cta_id}_{label}.npz"),
                            surface_coords=surface_i, normals=normals_i, volume_coords=volume_i,
                            sdfs_surface=sdfs_surface_i, sdfs_volume=sdfs_volume_i.reshape(-1, 1))
                # ----- 
                
                # 2. SAMPLE FROM MULTI-CENTERED NORMALIZED MESHES
                # Sample mesh surface coords and respective normals: (#points, 3xyz)
                mesh_multi_i = self.meshes_multi_norm_dict[cta_id][label]
                surface_i, normals_i = self.sample_mesh(mesh_multi_i, n_points=self.samples_per_mesh)
                
                if isinstance(surface_i, bool):
                    print(f"Mesh {label} - {cta_id} is not all triangles! No sampling done.")
                    continue
                else:
                    # Sample coords around surface, by adding Gaussian noise: (#points, 3xyz) - Volume coords
                    # Note 1: The noise is also scaled to [-1,1]^3 space
                    # Note 2: In multi-centered meshes, #total_volume_points = #surface_points_per_mesh
                    gauss_noise_i = self.sigma * np.random.randn(*surface_i.shape)
                    volume_i_ = surface_i + gauss_noise_i
                    volume_i = self.rng.choice(volume_i_, self.samples_per_mesh // len(self.meshes_labels), replace=False)  
                    
                    # Get SDF vector at each sampled surface and volume coord: (#points, #sdfs=3) 
                    sdfs_surface_i = np.zeros([surface_i.shape[0], 3])
                    sdfs_volume_i = np.zeros([volume_i.shape[0], 3])
                    for j, label_j in enumerate(self.meshes_labels):
                        sdfs_volume_i[:, j] = self.sdf(volume_i, self.meshes_multi_norm_dict[cta_id][label_j])
                        if j != i:
                            sdfs_surface_i[:, j] = self.sdf(surface_i, self.meshes_multi_norm_dict[cta_id][label_j])
                    
                    surface_multi.append(surface_i)
                    normals_multi.append(normals_i)
                    volume_multi.append(volume_i)
                    sdfs_volume_multi.append(sdfs_volume_i)
                    sdfs_surface_multi.append(sdfs_surface_i)
            # --------------------
            
            if len(surface_multi) < 3: # Skip patient if any mesh is not all triangles
                print(f"Skipping patient {cta_id} ...")
                continue   
            else:
                # Concatenate coords, normals and sdfs for all meshes: (#meshes=3 * #points, 3)
                surface_multi_ = np.concatenate(surface_multi, axis=0)
                normals_multi_ = np.concatenate(normals_multi, axis=0)
                volume_multi_ = np.concatenate(volume_multi, axis=0)
                sdfs_volume_multi_ = np.concatenate(sdfs_volume_multi, axis=0)
                sdfs_surface_multi_ = np.concatenate(sdfs_surface_multi, axis=0)
                # --------------------
                
                # SAVE multi-centered mesh data as a dictionary in .npz format
                np.savez(os.path.join(subdir_multi, "multi", f"{cta_id}.npz"),
                        surface_coords=surface_multi_, normals=normals_multi_, volume_coords=volume_multi_,
                        sdfs_surface=sdfs_surface_multi_, sdfs_volume=sdfs_volume_multi_)                   
# =============================================== 


if __name__=="__main__":
    
    # Load CTA preprocessing config file
    config_pre = json.load(open('configs/config_preprocess_cta.json'))
    
    multi_mesh_pre = PreprocessMultiMeshCTA(config_pre)
    multi_mesh_pre.preprocess()
    multi_mesh_pre.normalize_coords()
    multi_mesh_pre.save_data()

            