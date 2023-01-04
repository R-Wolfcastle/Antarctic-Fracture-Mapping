#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 15:40:54 2022

@author: eetss
"""

#std lib
import sys
import time

#3rd party
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from osgeo import gdal

from scipy.ndimage import convolve
from scipy.ndimage import gaussian_filter as gf
from scipy.signal import medfilt2d as mf2d
from scipy.interpolate import griddata as interp_grid

from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage import morphology
from skimage.filters import frangi

from astropy.convolution import convolve as astroconv
from astropy.convolution import convolve_fft as astroconv_fft
from astropy.convolution import Gaussian2DKernel, Box2DKernel

#local apps
sys.path.insert(1, os.environ['MODULE_HOME'])
from odd_geo_fcts import array_to_geotiff




damage_probs_filepath = str(sys.argv[1])
print("postprocessing t1b: {}".format(damage_probs_filepath))

filtered_dp_fp_dir = "/".join(damage_probs_filepath.split("/")[:-1])
filtered_dp_fp_dir = "." if (filtered_dp_fp_dir == "") else filtered_dp_fp_dir
dp_fn_stub = damage_probs_filepath.split("/")[-1].split(".tif")[0]
filtered_dp_fp = filtered_dp_fp_dir+"/{}".format(dp_fn_stub)+"_filtered.tif"


def create_pipeline(list_of_fcts):
    def pipeline(input_):
        result_ = input_
        for fct in list_of_fcts:
            result_ = fct(result_)
        return result_
    return pipeline


def gaussian_derivatives(sigma):
    if not sigma%1:
        size = 6*sigma + 1
    else:
        size = 6*sigma + 1
        size = np.ceil(size) // 2 * 2 + 1
    
    # assert size % 2 == 1, "Size must be an odd number!"
    y_coords, x_coords = np.meshgrid(-np.arange(-(size-1)/2, ((size-1)/2+1)), np.arange(-(size-1)/2, ((size-1)/2)+1), indexing='ij')
    gxx = ((-1 + (x_coords**2)/(sigma**2))/(2*np.pi*(sigma**2))) * np.exp(-(x_coords**2 + y_coords**2)/(2*(sigma**2)))
    gyy = ((-1 + (y_coords**2)/(sigma**2))/(2*np.pi*(sigma**2))) * np.exp(-(x_coords**2 + y_coords**2)/(2*(sigma**2)))
    gxy = ((y_coords*x_coords)/(2*np.pi*(sigma**6))) * np.exp(-(x_coords**2 + y_coords**2)/(2*(sigma**2)))
    gyx = -gxy
    return y_coords, x_coords, gxx, gxy, gyx, gyy
  

def vesselness_function(evals_1, evals_2):
    Rb = np.abs(evals_1)/np.abs(evals_2)
    P = np.sqrt(evals_1**2 + evals_2**2)
    beta = 0.5
    c = np.max(np.nanmax(P))
    vf_pre = np.exp(-(Rb**2)/(2*(beta**2)))
    vf_post = (1 - np.exp(-(P**2)/(2*(c**2))))
    vf = vf_pre * vf_post
    
    vf[evals_2>0]=0
    
    return vf
  
def reduce_angles_to_quadrants_1_and_4(angles):
    angles[(angles<0) & (np.abs(angles)>(np.pi/2))]=(np.pi+angles[(angles<0) & (np.abs(angles)>(np.pi/2))])
    angles[(angles>0) & (np.abs(angles)>(np.pi/2))]=(angles[(angles>0) & (np.abs(angles)>(np.pi/2))]-np.pi)
    return angles


def threshold_and_normalise_t2_damage_tiff(damage_array):
    damage_array[damage_array<=0.5]=0
    return damage_array



def create_get_hessian_fct(gxx, gxy, gyx, gyy):
    def get_hessian(image):
        hxy = convolve(image, gxy, mode="constant", cval=0)
        hyx = convolve(image, gyx, mode="constant", cval=0)
        hyy = convolve(image, gyy, mode="constant", cval=0)
        hxx = convolve(image, gxx, mode="constant", cval=0)
        
        return hxx, hxy, hyx, hyy

    return get_hessian


def create_get_hessian_eigenthings_fct():
    def get_hessian_eigenthings(hxx, hxy, hyx, hyy):
        hxy_t = np.transpose(hxy)
        hxx_t = np.transpose(hxx)
        hyx_t = np.transpose(hyx)
        hyy_t = np.transpose(hyy)

        hessians = np.array([[hxx_t, hxy_t], [hxy_t, hyy_t]])
        hess_tp = np.transpose(hessians)
        eigenthings = np.linalg.eig(hess_tp)

        return eigenthings
    return get_hessian_eigenthings


def create_sort_eigenthings_fct()
    def sort_eigenthings(eigenthings):
        eigenvalues = eigenthings[0]
        eigenvectors = eigenthings[1]

        #####Fancy indexing to get egenthings in order of absolute value of eigenvalues:
        abs_eigenvalues = np.abs(eigenvalues)
        abs_eigenvalues_sorted_indices = np.argsort(abs_eigenvalues)
        # eval_1_indices = abs_eigenvalues_sorted_indices[:,:,0]

        m, n, k = eigenvalues.shape
        idx = np.ogrid[:m, :n, :k]
        idx[2]=abs_eigenvalues_sorted_indices
        eigenvalues_sorted_by_absolute_value = eigenvalues[idx]

        m, n, k, l = eigenvectors.shape
        idx = np.ogrid[:m, :n, :k, :l]
        idx[2]=np.expand_dims(abs_eigenvalues_sorted_indices, 3)
        eigenvectorss_sorted_by_absolute_value_of_eigenvalue = eigenvectors[idx]

        evals_1 = eigenvalues_sorted_by_absolute_value[:,:,0]
        evals_2 = eigenvalues_sorted_by_absolute_value[:,:,1]

        evecs_1 = eigenvectors_sorted_by_absolute_value_of_eigenvalue[:,:,0,:]
        evecs_2 = eigenvectors_sorted_by_absolute_value_of_eigenvalue[:,:,1,:]

        return [evals_1, evals_2], [evecs_1, evecs_2]
    return sort_eigenthings


def create_get_tubular_angles_fct(tubular_structures_threshold=0.01):
    def get_tubular_angles(evals, evecs):
        evals_1, evals_2 = evals
        evecs_1, evecs_2 = evecs

        thres = tubular_structures_threshold

        ###### Working out where there are tubular structures
        tubular_likelihoods = vesselness_function(evals_1, evals_2)

        tubular_structures = tubular_likelihoods.copy()
        tubular_structures[tubular_structures<thres]=0
        tubular_structures[tubular_structures>=thres]=1

        ###### Working out angles of tubular structures
        angles = np.arctan2(-evecs_1[:,:,1], evecs_1[:,:,0])
        angles = reduce_angles_to_quadrants_1_and_4(angles)

        tubular_angles = angles * tubular_structures

        tubular_angles[tubular_angles==0]=np.nan

        return tubular_angles
    return get_tubular_angles


def local_angle_variance(angle_array, kernel_width=71):
    mean_kernel = Box2DKernel(kernel_width)
    angle_array_sq = angle_array ** 2
    angles_local_mean = astroconv_fft(angle_array, mean_kernel, allow_huge=True)
    angles_sq_local_mean = astroconv_fft(angle_array_sq, mean_kernel, allow_huge=True)
    angles_local_var = angles_sq_local_mean - (angles_local_mean ** 2)
    return angles_local_var


def create_get_local_angle_variance_mask_fct(kernel_width=71, var_threshold=0.71):
    def get_local_angle_variance_mask(tubular_angles):

        ######## Taking local sum of angles, angles^2 and calculating variance:
        ta_local_var = local_angle_variance(tubular_angles, kernel_width)
        
        tubular_angles_rot = tubular_angles + np.pi/2
        tubular_angles_rot = reduce_angles_to_quadrants_1_and_4(tubular_angles_rot)
        ta_local_var_2 = local_angle_variance(tubular_angles_rot)
        
        min_ta_local_var = np.nanmin(np.array([ta_local_var, ta_local_var_2]), axis=0)
        
        mask = min_ta_local_var.copy()
        mask[min_ta_local_var<var_threshold]=1
        mask[min_ta_local_var>=var_threshold]=0

        return mask
    return get_local_angle_variance



if __name__ == '__main__':

    large_im_tiff = gdal.Open(damage_probs_filepath, gdal.GA_ReadOnly)
    large_im = large_im_tiff.ReadAsArray()
    
    large_im[large_im<=0.7] = 0
    im1 = large_im
    
    y_coords, x_coords, gxx, gxy, gyx, gyy = gaussian_derivatives(0.5)
    
    get_hessian = create_get_hessian_fct(gxx, gxy, gyx, gyy)
    get_hessian_eigenthings = create_get_hessian_eigenthings_fct
    sort_eigenthings = create_sort_eigenthings_fct
    get_tubular_angles = create_get_tubular_angles_fct(0.01)
    get_local_angle_variance_mask = create_get_local_angle_variance_mask_fct(71, 0.71)
    
    mask_functions_list = [
                    get_hessian,
                    get_hessian_eigenthings,
                    sort_eigenthings,
                    get_tubular_angles,
                    get_local_angle_variance_mask
                ]
    mask_pipeline = create_pipeline(mask_functions_list)
    
    mask = mask_pipeline(large_im)
    
    masked_im = large_im * mask
            
    array_to_geotiff(masked_im, large_im_tiff, filtered_dp_fp, compression="DEFLATE")


