# Imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent)) # Add project root to sys.path
from matplotlib.pylab import Enum
import numpy as np
import nibabel as nib
from scipy.ndimage import label as scipy_label
from scipy import ndimage
from data_preprocessing.data_labels import *

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TRANSFORMING CTA & CMRI MESHES/POINT CLOUDS TO A SHARED REFERENTIAL (CANONICAL ORIENTATION)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def getLargestCC(segmentation, ndim=3):
    """ 
    Get largest connected component from binary segmentation mask. 
    """
    assert ndim in [2, 3]
    if ndim == 3:
        struc=np.ones((3, 3, 3))
    else:
        struc=np.ones((3, 3))
    labels, count = scipy_label(segmentation, structure=struc)
    assert(labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC
# ===================

def get_binary_segmask(segmask, label:Enum=None, lv_epi=False, pad=None):
    """ 
    Returns the binary segmentation mask for a given label, considering
    its largest connected component.
    
    Note: label should be Enum.name or Enum[name].
    """
    if pad is not None:
        # Zero-padding (to further avoid edge effects and thus holes in the mesh)
        segmask = np.pad(segmask, pad_width=((pad, pad), (pad, pad), (pad, pad)), mode='constant', constant_values=0)
    
    if label is not None:
        # Takes label foreground voxel
        if (label.name == "LV") and lv_epi:
            # Get LV Epi segmask for "LV" (i.e, LVBP + MYO)
            bin_segmask = np.add((segmask == label.__class__["LV"].value).astype(np.int32), 
                                 (segmask == label.__class__["LVBP"].value).astype(np.int32))
        else:
            bin_segmask = (segmask == label.value).astype(np.int32)
    else:
        
        # Takes all foreground voxels
        bin_segmask = (segmask > 0).astype(np.int32)
    new_segmask = np.zeros_like(bin_segmask).astype(np.int32)
    new_segmask[getLargestCC(bin_segmask == 1, ndim=len(segmask.shape)) == 1] = 1
    return new_segmask
# ===================   
 
def get_affine(nii_seg):
    """ 
    Get affine transformation from nibabel segmentation.
    """
    return nii_seg.affine
   
# ===================

def get_rotation_matrix(vec1, vec2):
    """
    Compute the rotation matrix that rotates vector vec1 to align with vector vec2.
    
    Parameters:
        vec1, vec2: 3D vectors (array-like). They do not need to be unit vectors.
        
    Returns:
        A 3x3 rotation matrix R such that R @ (vec1 normalized) == (vec2 normalized).
    """
    # Normalize the input vectors.
    vec1 = np.asarray(vec1, dtype=float) / np.linalg.norm(vec1)
    vec2 = np.asarray(vec2, dtype=float) / np.linalg.norm(vec2)
    
    # Compute the cross-product and dot product.
    v = np.cross(vec1, vec2)
    c = np.dot(vec1, vec2)
    
    # If the vectors are (almost) already aligned, return the identity matrix.
    if np.isclose(c, 1.0):
        return np.eye(4)
    
    # If the vectors are opposite, find an arbitrary perpendicular vector and construct the rotation.
    if np.isclose(c, -1.0):
        # Choose vec1 vector that is not parallel to vec1
        arbitrary_axis = np.array([1, 0, 0]) if not np.isclose(vec1[0], 1.0) else np.array([0, 1, 0])
        v = np.cross(vec1, arbitrary_axis)
        v = v / np.linalg.norm(v)
        # Rotation by π (180 degrees) about v.
        R_m = -np.eye(3) + 2 * np.outer(v, v)
        M_rot = np.eye(4)
        M_rot[:3,:3] = R_m
        return M_rot
    
    # Compute the sine of the angle via the norm of v.
    s = np.linalg.norm(v)
    
    # Construct the skew-symmetric cross-product matrix of v.
    K_m = np.array([[    0, -v[2],  v[1]],
                  [ v[2],     0, -v[0]],
                  [-v[1],  v[0],     0]])
    
    # Use Rodrigues' rotation formula:
    # R_m = I + K_m + K_m^2 * ((1-c)/s^2)
    R_m = np.eye(3) + K_m + K_m @ K_m * ((1 - c) / (s**2))
    
    M_rot = np.eye(4)
    M_rot[:3,:3] = R_m
    
    return M_rot
# =============================================

def get_canonical_transform_cta(path_cta_segm, LabelsMap:Enum, flip_y=False, align_x=False):
    """ 
    Function that returns the matrices for transforming voxel coords into 
    world coords, in canonical orientation, for CTA segmentations.
    
    We assume canonical orientation as having the c.o.m. LVBP-->LABP vector 
    parallel to the z-axis ([0,0,1]), in world coordinates. From a front 
    view, RV is located on the left and LV on the right (radiological convention).
    
    If align_x=True, the vector that connects the centers of mass of both ventricles 
    can be aligned with the x-axis ([1,0,0]). 
    
    If flip_y=True, the shapes can be y-flipped such that RV is located on the right 
    and LV on the left (from a front view).
    
    It additionally returns the transformation matrices to align the referential's 
    origin (0,0,0) with each individual shape c.o.m, as well as multi-shape c.o.m. 
    (in canonical world coords).
    """
    # Get segmentation mask
    nii_cta = nib.load(str(path_cta_segm))
    segmask = nii_cta.get_fdata()   # xyz order
    spacing_xyz = nii_cta.header.get_zooms()  

    # -------------------
    # WORLD COORDINATES:
    # NOTE: CTA images are already oriented in radiological convention, so we don't need to
    #       apply the affine transformation, in contrast to MRI. To go from voxel to world 
    #       coords, we only have to consider the spacing transformation!
    
    # Dict of c.o.m  (world coords): {label: c.o.m.}
    cms_dict= {}
    
    for label in ["LVBP", "LV", "RVBP", "LABP"]:
        # Get LCC binary segmask (zyx)
        segmask_i = get_binary_segmask(segmask, LabelsMap[label], pad=10)
        
        # Get c.o.m. (world coords)
        cms_dict[label] = np.multiply(ndimage.center_of_mass(segmask_i), spacing_xyz)
  
    # ----------------------
    # CANONICAL ORIENTATION:
    # NOTE: the LVBP-->LABP vector (n) will be used to rotate the shapes, 
    #       such that: n // z-axis (0,0,1)
    #       https://web.ma.utexas.edu/users/m408m/Display12-5-4.shtml
    
    # Get LVBP-->LABP vector (world coords)
    n = np.array(cms_dict["LABP"]) - np.array(cms_dict["LVBP"]) # Column vector

    # NOTE: Matrices' multiplication is written in reverse order to which transformations 
    #       are applied, i.e., matrices written in sequence from left to right are actually 
    #       applied from right to left, because w_coords = M * v_coords, v_coords: column-vec rep
    #       -- Our coords are represented as row-vecs, hence we use the transposed matrices: 
    #          w_coords = v_coords * M.T
     
    if align_x:
        # 1 -- Rotation to align ventricles c.o.m. (cm_vec) vector with x-axis (1,0,0)    
        if flip_y:
            # LVBP-->RVBP // x-axis
            cm_vec = np.array(cms_dict["RVBP"]) - np.array(cms_dict["LVBP"])
        else:
            # RVBP-->LVBP // x-axis
            cm_vec = np.array(cms_dict["LVBP"]) - np.array(cms_dict["RVBP"])
        M_rot_align_x = get_rotation_matrix(cm_vec, np.array([1,0,0])) 
        
        # 2 -- Canonical rotation to align (newly rotated) c.o.m vector with z-axis
        n_align_x = M_rot_align_x[:3, :3] @ n
        M_rot_align_z = get_rotation_matrix(n_align_x, np.array([0,0,1]))
        M_rot = M_rot_align_z @ M_rot_align_x
    # ----------
    elif flip_y:
        # 1 -- Canonical rotation to align world LVBP-->LABP vector with z-axis
        M_rot_align_z = get_rotation_matrix(n, np.array([0,0,1]))
        
        # 2 -- Rotation to flip y-axis
        M_rot_flip_y = get_rotation_matrix(np.array([0,1,0]), np.array([0,-1,0]))
        M_rot = M_rot_flip_y @ M_rot_align_z
    else:
        # 1 -- Canonical rotation to align world LVBP-->LABP vector with z-axis
        M_rot = get_rotation_matrix(n, np.array([0,0,1]))
        
    # ---------------------------------------------------
    # ALIGNMENT OF REFERENTIAL ORIGIN (0,0,0) WITH C.O.M
    # List of c.o.ms matrices: [M_lvbp, M_lv, M_rvbp, M_multi]
    M_cm_dict = {}  
    for label in ["LVBP", "LV", "RVBP", "multi"]:
        # Get c.o.m. in world coords
        if label == "multi":
            cm_i = np.mean((cms_dict["LV"], cms_dict["RVBP"]), axis=0)
        else:
            cm_i = cms_dict[label]
        
        # Apply canonical rotation
        cm_i_cano = M_rot[:3,:3] @ cm_i 
        
        # Create translation matrix
        M_cm_cano_i = np.eye(4)
        M_cm_cano_i[:3, 3] = np.asarray(cm_i_cano).astype(np.float32)
        M_cm_dict[label] = M_cm_cano_i
    
    return M_rot, M_cm_dict
# =============================================

def canonical_mri_aux(nii_sa, LabelsMap:Enum, flip_y=False, align_x=False):
    """ 
    Function to get canonical transform only for SAX, when LAX segmentation
    does not exist.
    """
    # Check whether the segmentation has a time dimension (x,y,z,t)
    if len(nii_sa.shape) == 4:
        # If so, ED time frame is used to get the canonical transform
        # NOTE: We assume ED is the first time frame (t=0)
        segmask_sa = np.swapaxes(nii_sa.get_fdata()[..., 0], 0, 2)  # xyz --> zyx order
        
    else:
        segmask_sa = np.swapaxes(nii_sa.get_fdata(), 0, 2)          # xyz --> zyx order
     
    # Get affine matrices (world transform)
    M_affine_sa = get_affine(nii_sa)
    
    # -------------------
    # WORLD COORDINATES:
    # List of coords: [coords_lvbp, coords_lv, coords_rvbp]
    coords_sa_v, coords_sa_w = [], []
    
    for i, label in enumerate(["LVBP", "LV", "RVBP"]):
        # Get binary segmask (zyx)
        segmask_sa_i = get_binary_segmask(segmask_sa, LabelsMap[label])
        
        # ----
        # Get segmask index positions (1D vector)
        idxs_sa_i = np.where(segmask_sa_i.ravel() > 0)[0]
       
        # Get segmentation voxel coords (xyz order)
        coords_sa_i = np.unravel_index(idxs_sa_i, segmask_sa_i.shape)
        coords_sa_i = np.stack(coords_sa_i, -1)     # Stack lists in array(num points, 3)
        coords_sa_i = np.flip(coords_sa_i, -1)      # To x, y, z order
        # ----
        
        # Add homogeneous coordinate
        coords_sa_i = np.concatenate((coords_sa_i, np.ones((len(coords_sa_i), 1))), axis=-1)
        
        # Apply world transform
        coords_sa_w_i = coords_sa_i @ M_affine_sa.T
        coords_sa_w_i = coords_sa_w_i[..., :3]      # Remove homogeneous coord
        
        coords_sa_v.append(coords_sa_i[...,:3])
        coords_sa_w.append(coords_sa_w_i)
    
    # ----------------------
    # CANONICAL ORIENTATION:
    # NOTE: the normal vector to all SAX cross-sections will be used to
    #       rotate the shapes, such that: n // z-axis
    #       https://web.ma.utexas.edu/users/m408m/Display12-5-4.shtml
    
    # Get all world coords of SAX mid z-slice of, e.g., LVBP
    coords_z = coords_sa_v[0][:,2] 
    mid_z = np.unique(coords_z)[-(len(np.unique(coords_z)) // -2) - 1]  # Ceil division to get approx z value of mid slice
    coords_mid = coords_sa_v[0][np.where(coords_z == mid_z)[0]]         # (#points, 3)
    coords_mid = np.concatenate((coords_mid, np.ones((len(coords_mid), 1))), axis=-1)
    coords_mid_w = coords_mid @ M_affine_sa.T
    coords_mid_w = coords_mid_w[..., :3]
   
    # Select 3 random points from LVBO mid z-slice
    p_idxs = np.random.choice(coords_mid_w.shape[0], 3, replace=False)
    p1, p2, p3 = coords_mid_w[p_idxs]
    
    # Make two unit vectors that belong to LVBO mid z-slice
    vec1 = (p2 - p1) / np.linalg.norm(p2 - p1)
    vec2 = (p3 - p1) / np.linalg.norm(p3 - p1)
    
    # Get normal unit vectors to LVBO mid z-slice (in both directions)
    n1 = np.cross(vec1, vec2)
    n2 = np.cross(vec2, vec1)
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)
    
    # Choose normal vector pointing in positive z-direction
    if n1[2] > 0:
        n = n1
    elif n2[2] > 0:
        n = n2
    else:
        print("WARNING: None of the normals is pointing in the positive z-direction!!!")
        n = n1  # Default to n1
        
    # NOTE: Matrices' multiplication is written in reverse order to which transformations 
    #       are applied, i.e., matrices written in vec1 sequence form left to right are actually 
    #       applied from right to left, because w_coords = M * v_coords, v_coords: column-vec rep
    #       -- Our coords are represented as row-vecs, hence we use the transposed matrices: 
    #          w_coords = v_coords * M.T
    
    if align_x:
        # 1 -- Rotation to align ventricles c.o.m. vector with x-axis    
        cm_lvbp_w = coords_sa_w[0].astype(np.float32).mean(0)
        cm_rvbp_w = coords_sa_w[2].astype(np.float32).mean(0)
        if flip_y:
            # LVBP-->RVBP // x-axis
            cm_vec_w = cm_rvbp_w - cm_lvbp_w
        else:
            # RVBP-->LVBP // x-axis
            cm_vec_w = cm_lvbp_w - cm_rvbp_w
        M_rot_align_x = get_rotation_matrix(cm_vec_w, np.array([1,0,0])) 
        
        # 2 -- Canonical rotation to align (newly rotated) Normal vector with z-axis
        n_align_x = (M_rot_align_x @ np.concatenate([n, [1]]))[:3]
        M_rot_align_z = get_rotation_matrix(n_align_x, np.array([0,0,1]))
        M_rot = M_rot_align_z @ M_rot_align_x 
    # ----------
    elif flip_y:
        # 1 -- Canonical rotation to align Normal vector with z-axis
        M_rot_align_z = get_rotation_matrix(n, np.array([0,0,1]))
        
        # 2 -- Rotation to flip y-axis
        M_rot_flip_y = get_rotation_matrix(np.array([0,1,0]), np.array([0,-1,0]))
        M_rot = M_rot_flip_y @ M_rot_align_z
    else:
        # 1 -- Canonical rotation to align Normal vector with z-axis
        M_rot = get_rotation_matrix(n, np.array([0,0,1]))
        
    # World coords transform + Canonical Transform (rotation) + Rotation to flip y-axis (if flip_y=True)
    M_sa_cano = M_rot @ M_affine_sa
    M_la_cano = None
    
    # ---------------------------------------------------
    # ALIGNMENT OF REFERENTIAL ORIGIN (0,0,0) WITH C.O.M
    # List of c.o.ms matrices: [M_lvbp, M_lv, M_rvbp, M_multi]
    M_cm_dict = {}  
    for j, label in enumerate(["LVBP", "LV", "RVBP", "multi"]):
        # Get c.o.m. in world coords
        if label == "multi":
            cm_i = np.concatenate((coords_sa_w[1], coords_sa_w[2])).astype(np.float32).mean(0)
        else:
            cm_i = coords_sa_w[j].astype(np.float32).mean(0)
        
        # Apply canonical rotation
        cm_i_cano = M_rot[:3,:3] @ cm_i 
        
        # Create translation matrix
        M_cm_cano_i = np.eye(4)
        M_cm_cano_i[:3, 3] = np.asarray(cm_i_cano).astype(np.float32)
        M_cm_dict[label] = M_cm_cano_i
        
    return M_sa_cano, M_la_cano, M_cm_dict
# =============================================        
  
def get_canonical_transform_mri(path_sa_segm, LabelsMap:Enum, flip_y=False, align_x=False):
    """ 
    Function that returns the matrices for transforming voxel coords into 
    world coords, in canonical orientation, for both SAX and LAX segmentations
    (if the latter exists).
    
    We assume canonical orientation as having SAX cross-sections orthogonal to
    the z-axis (top=basal and bottom=apex), in world coordinates. From a front 
    view, RV is located on the left and LV on the right.
    
    If align_x=True, the vector that connects the centers of mass of both ventricles 
    can be aligned with the x-axis. 
    
    If flip_y=True, the shapes can be y-flipped such that RV is located on the right 
    and LV on the left (from a front view).
    
    It additionally returns the transformation matrices to align the referential's 
    origin with each individual shape c.o.m, as well as multi-shape c.o.m. 
    (in canonical world coords).
    
    In case, the segmentations have a time dimension (x,y,z,t), the ED time frame
    is used to get the canonical transform. We assume ED is the first time frame (t=0).
    """
    # Get segmentation masks
    nii_sa = nib.load(str(path_sa_segm))
    
    if not Path(str(path_sa_segm).replace("SA", "LA")).exists():
        M_sa_cano, M_la_cano, M_cm_dict = canonical_mri_aux(nii_sa, LabelsMap, flip_y=flip_y, align_x=align_x)
    else:
        nii_la = nib.load(str(path_sa_segm).replace("SA", "LA"))
        
        # Check whether the segmentations have a time dimension (x,y,z,t)
        if len(nii_sa.shape) == 4:
            assert len(nii_la.shape) == 4, f"{Path(path_sa_segm).name} - SAX has time dimension but LAX does not!"
            # If so, ED time frame is used to get the canonical transform
            # NOTE: We assume ED is the first time frame (t=0)
            segmask_sa = np.swapaxes(nii_sa.get_fdata()[..., 0], 0, 2)  # xyz --> zyx order
            segmask_la = np.swapaxes(nii_la.get_fdata()[..., 0], 0, 2)  # xyz --> zyx order
        else:
            segmask_sa = np.swapaxes(nii_sa.get_fdata(), 0, 2)  # xyz --> zyx order
            segmask_la = np.swapaxes(nii_la.get_fdata(), 0, 2)  # xyz --> zyx order
        
        # Get affine matrices (world transform)
        M_affine_sa = get_affine(nii_sa)
        M_affine_la = get_affine(nii_la)
        
        # -------------------
        # WORLD COORDINATES:
        # List of coords: [coords_lvbp, coords_lv, coords_rvbp]
        coords_sa_v, coords_sa_w, coords_la_w = [], [], []
        
        for i, label in enumerate(["LVBP", "LV", "RVBP"]):
            # Get binary segmask (zyx)
            segmask_sa_i = get_binary_segmask(segmask_sa, LabelsMap[label])
            segmask_la_i = get_binary_segmask(segmask_la, LabelsMap[label])
            
            # ----
            # Get segmask index positions (1D vector)
            idxs_sa_i = np.where(segmask_sa_i.ravel() > 0)[0]
            idxs_la_i = np.where(segmask_la_i.ravel() > 0)[0]
        
            # Get segmentation voxel coords (xyz order)
            coords_sa_i = np.unravel_index(idxs_sa_i, segmask_sa_i.shape)
            coords_sa_i = np.stack(coords_sa_i, -1)     # Stack lists in array(num points, 3)
            coords_sa_i = np.flip(coords_sa_i, -1)      # To x, y, z order
            
            coords_la_i = np.unravel_index(idxs_la_i, segmask_la_i.shape)
            coords_la_i = np.stack(coords_la_i, -1)
            coords_la_i = np.flip(coords_la_i, -1)
            # ----
            
            # Add homogeneous coordinate
            coords_sa_i = np.concatenate((coords_sa_i, np.ones((len(coords_sa_i), 1))), axis=-1)
            coords_la_i = np.concatenate((coords_la_i, np.ones((len(coords_la_i), 1))), axis=-1)
            
            # Apply world transform
            coords_sa_w_i = coords_sa_i @ M_affine_sa.T
            coords_sa_w_i = coords_sa_w_i[..., :3]      # Remove homogeneous coord
            coords_la_w_i= coords_la_i @ M_affine_la.T
            coords_la_w_i = coords_la_w_i[..., :3]
            
            coords_sa_v.append(coords_sa_i[...,:3])
            coords_sa_w.append(coords_sa_w_i)
            coords_la_w.append(coords_la_w_i)
        
        # ----------------------
        # CANONICAL ORIENTATION:
        # NOTE: the normal vector to all SAX cross-sections will be used to
        #       rotate the shapes, such that: n // z-axis
        #       https://web.ma.utexas.edu/users/m408m/Display12-5-4.shtml
        
        # Get all world coords of SAX mid z-slice of, e.g., LVBP
        coords_z = coords_sa_v[0][:,2] 
        mid_z = np.unique(coords_z)[-(len(np.unique(coords_z)) // -2) - 1]  # Ceil division to get approx z value of mid slice
        coords_mid = coords_sa_v[0][np.where(coords_z == mid_z)[0]]         # (#points, 3)
        coords_mid = np.concatenate((coords_mid, np.ones((len(coords_mid), 1))), axis=-1)
        coords_mid_w = coords_mid @ M_affine_sa.T
        coords_mid_w = coords_mid_w[..., :3]
    
        # Select 3 random points from LVBO mid z-slice
        p_idxs = np.random.choice(coords_mid_w.shape[0], 3, replace=False)
        p1, p2, p3 = coords_mid_w[p_idxs]
        
        # Make two unit vectors that belong to LVBO mid z-slice
        vec1 = (p2 - p1) / np.linalg.norm(p2 - p1)
        vec2 = (p3 - p1) / np.linalg.norm(p3 - p1)
        
        # Get normal unit vectors to LVBO mid z-slice (in both directions)
        n1 = np.cross(vec1, vec2)
        n2 = np.cross(vec2, vec1)
        n1 = n1 / np.linalg.norm(n1)
        n2 = n2 / np.linalg.norm(n2)
        
        # Choose normal vector pointing in positive z-direction
        if n1[2] > 0:
            n = n1
        elif n2[2] > 0:
            n = n2
        else:
            print("WARNING: None of the normals is pointing in the positive z-direction!!!")
            n = n1  # Default to n1
        
        # ----
        # NOTE: Matrices' multiplication is written in reverse order to which transformations 
        #       are applied, i.e., matrices written in vec1 sequence form left to right are actually 
        #       applied from right to left, because w_coords = M * v_coords, v_coords: column-vec rep
        #       -- Our coords are represented as row-vecs, hence we use the transposed matrices: 
        #          w_coords = v_coords * M.T
        
        if align_x:
            # 1 -- Rotation to align ventricles c.o.m. vector with x-axis    
            cm_lvbp_w = coords_sa_w[0].astype(np.float32).mean(0)
            cm_rvbp_w = coords_sa_w[2].astype(np.float32).mean(0)
            if flip_y:
                # LVBP-->RVBP // x-axis
                cm_vec_w = cm_rvbp_w - cm_lvbp_w
            else:
                # RVBP-->LVBP // x-axis
                cm_vec_w = cm_lvbp_w - cm_rvbp_w
            M_rot_align_x = get_rotation_matrix(cm_vec_w, np.array([1,0,0])) 
            
            # 2 -- Canonical rotation to align (newly rotated) Normal vector with z-axis
            n_align_x = (M_rot_align_x @ np.concatenate([n, [1]]))[:3]
            M_rot_align_z = get_rotation_matrix(n_align_x, np.array([0,0,1]))
            M_rot = M_rot_align_z @ M_rot_align_x
        # ----------
        elif flip_y:
            # 1 -- Canonical rotation to align Normal vector with z-axis
            M_rot_align_z = get_rotation_matrix(n, np.array([0,0,1]))
            
            # 2 -- Rotation to flip y-axis
            M_rot_flip_y = get_rotation_matrix(np.array([0,1,0]), np.array([0,-1,0]))
            M_rot = M_rot_flip_y @ M_rot_align_z
        else:
            # 1 -- Canonical rotation to align Normal vector with z-axis
            M_rot = get_rotation_matrix(n, np.array([0,0,1]))
            
        # World coords transform + Canonical Transform (rotation) + Rotation to flip y-axis (if flip_y=True)
        M_sa_cano = M_rot @ M_affine_sa
        M_la_cano = M_rot @ M_affine_la
        
        # ---------------------------------------------------
        # ALIGNMENT OF REFERENTIAL ORIGIN (0,0,0) WITH C.O.M
        # List of c.o.ms matrices: [M_lvbp, M_lv, M_rvbp, M_multi]
        M_cm_dict = {}  
        for j, label in enumerate(["LVBP", "LV", "RVBP", "multi"]):
            # Get c.o.m. in world coords
            if label == "multi":
                cm_i = np.concatenate((coords_sa_w[1], coords_sa_w[2])).astype(np.float32).mean(0)
            else:
                cm_i = coords_sa_w[j].astype(np.float32).mean(0)
            
            # Apply canonical rotation
            cm_i_cano = M_rot[:3,:3] @ cm_i 
            
            # Create translation matrix
            M_cm_cano_i = np.eye(4)
            M_cm_cano_i[:3, 3] = np.asarray(cm_i_cano).astype(np.float32)
            M_cm_dict[label] = M_cm_cano_i
        
    return M_sa_cano, M_la_cano, M_cm_dict

# =============================================