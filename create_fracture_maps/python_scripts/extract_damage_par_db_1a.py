#!/usr/bin/env python
# -*- coding: utf-8 -*-

#std lib
import os
from pathlib import Path
import sys
from multiprocessing import Pool

#3rd party
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from osgeo import gdal
from scipy import ndimage
from scipy.ndimage import binary_dilation as bd
import torch
import torchvision.transforms.functional as tf
import torch.nn.functional as fct

#local apps
sys.path.insert(1, os.environ['MODULE_HOME'])
from unet import UNet
from odd_geo_fcts import array_to_geotiff



def create_pipeline(list_of_fcts):
    def pipeline(input_):
        result_ = input_
        for fct in list_of_fcts:
            result_ = fct(result_)
        return result_
    return pipeline


def gkern(kernlen=256):
    x, y = np.meshgrid(np.linspace(-1,1,256), np.linspace(-1,1,256))
    d = np.sqrt(x*x+y*y)
    sigma, mu = 1.0, 0.0
    g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    g /= 4
    return g


def create_process_patch_fct(model, kernel, num_edge_dilations, patch_size, buffer_):
    def process_patch(patch, top_left_x, top_left_y):
        if np.sum(patch) != 0 and patch.shape[-1]==patch.shape[-2] and patch.shape[-1]==patch_size:
            sar = tf.to_tensor(patch)
            sar = tf.normalize(sar,(0.5,), (0.5,))
            sar = sar.unsqueeze(0)
    
            with torch.no_grad():
                output_activations = model(sar).squeeze(0)
                output_activations = fct.softmax(output_activations).numpy()
    
                #Fudge alert
                output_activations[buffer_:,buffer_:] = 0
                output_activations[:-buffer_,:-buffer_] = 0
                
                if kernel is not None:
                    output_activations *= kernel
    
                if np.count_nonzero(patch==0)>=10:
                    patch_mask = np.zeros_like(patch)
                    patch_mask[patch==0] = 1
                    for bdn in range(num_edge_dilations):
                        patch_mask=bd(patch_mask)
                    patch_mask = (1-patch_mask).astype("float")
                    patch_mask[patch_mask==0]=np.nan
    
                    output_activations *= patch_mask
    
        else:
            output_activations = np.zeros((2, patch.shape[0], patch.shape[1]))*np.nan
    
        return [output_activations, top_left_x, top_left_y]
    return process_patch

def create_unpack_args_proc_patch_fct(proc_patch_fct):
    def unpack_args_proc_patch(args):
        return proc_patch_fct(*args)
    return unpack_args_proc_patch


def create_tile_fct(tile_size, stride):
    def tile(sar_array):
        height, width = sar_array.shape
    
        patches = []
        tlxs = []
        tlys = []
    
        prob_map = np.zeros((2, height, width))
    
        row=1
        column=1
        patch_number=1
        while True:
    
            if ((column-1)*stride+tile_size)>width:
                top_left_x = width-tile_size
                reached_right = True
            else:
                reached_right = False
                top_left_x = (column-1)*stride
    
            if ((row-1)*stride+tile_size)>height:
                top_left_y = height-tile_size
                reached_bottom = True
            else:
                reached_bottom = False
                top_left_y = (row-1)*stride
    
            pch1_image = sar_array[top_left_y:(top_left_y+tile_size), top_left_x:(top_left_x+tile_size)]
    
            patches.append(pch1_image)
            tlxs.append(top_left_x)
            tlys.append(top_left_y)
    
            patch_number+=1
            if reached_right:
                column = 1
                row += 1
            else:
                column += 1
    
            if reached_right and reached_bottom:
                break

        return patches, tlxs, tlys
    return tile


def create_proc_tiles_fct(num_procs, unpack_args_proc_patch_fct):
    def proc_tiles(patches, tlxs, tlys):
        pool = Pool(num_procs_py)
        output_activations_and_tags = pool.map(unpack_args_proc_patch, zip(patches, tlxs, tlys))
        pool.close()
        pool.join()

        return output_activations_and_tags
    return proc_tiles 


def create_stitch_fct():
    def stitch(data_and_position_tags):
        for patch_and_tag in data_and_position_tags:
            output_activations = patch_and_tag[0]
            top_left_x = patch_and_tag[1]
            top_left_y = patch_and_tag[2]
    
            prob_map[:, (top_left_y):(top_left_y+tile_size), (top_left_x):(top_left_x+tile_size)] =\
                    prob_map[:, (top_left_y):(top_left_y+tile_size), (top_left_x):(top_left_x+tile_size)]+output_activations
    
        prob_tensor = torch.from_numpy(prob_map)
        
        #prob_map = prob_tensor.numpy()/2
        
        #prob_map = fct.softmax(prob_tensor).numpy()
        prob_map = fct.softmax(prob_tensor).numpy()
    
        #segment_data = np.argmax(prob_map, 0)

        probs_normal = probs[1,:,:]
        probs_sqrt = np.sqrt(probs_normal)
        probs_sqrt[probs_normal<0.05] = probs_normal[probs_normal<0.05]

        return prob_map, probs_sqrt

    return stitch


def open_sar_data(sar_tiff_filepath):
    date = sar_tiff_filepath.split('/')[-1].split('.')[0]

    dirpath = '/'.join(sar_tiff_filepath.split('/')[:-1])

    sar_data = gdal.Open(sar_tiff_filepath, gdal.GA_ReadOnly)

    sar_data_array = sar_data.ReadAsArray().astype(np.float32)

    return sar_data, sar_data_array, date

def save_damage_data(outpath, sar_data, date, damage_data, damage_data_sqrt):
    array_to_geotiff(damage_data[1,:,:], sar_data, outpath+"/{}.damage_probs_1a.tif".format(date), compression="DEFLATE")
    array_to_geotiff(probs_sqrt, sar_data, outpath+"/{}.damage_probs_1a_sqrt.tif".format(date), compression="DEFLATE")

if __name__ == "__main__":
    outpath = os.getcwd()
    
    #IF you want to mask the damage maps with the calving front segmentations, supply a:
    cf_seg_path=None


    
    #define some parameters
    tile_size = 256
    stride = 128
    num_edge_dilations = 10
    edge_buffer_size = 5
    num_procs = 8

    
    sar_filepath= str(sys.argv[1])
    print("processing 1a damage for: {}".format(sar_filepath))
    num_procs_py = int(sys.argv[2])
    print("number of available processes for damage mapping: {}".format(num_procs_py))
    num_omp_threads = int(sys.argv[3])
    torch.set_num_threads(num_omp_threads)
    print("number of available openMP threads for torch: {}".format(torch.get_num_threads()))
    
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    
    kernel = 1
    
    model = UNet(1,2)
    model.load_state_dict(torch.load('{}/dn_e10_bsv1_v1_e02_sh'.format(os.environ['MODELA_DICT_DIR']),
                        						  map_location=torch.device(device))['model_state_dict'])
    model.eval().to(device)



    #define sub functions for processing tiles of data with neural net in parallel:
    unpack_args_and_process_patch = create_unpack_args_proc_patch_fct(
                                    create_process_patch_fct(model, kernel, num_edge_dilations,
                                                             tile_size, edge_buffer_size))

    #define functions for tiling, processing tiles, and stitching back together:
    tile = create_tile_fct(tile_size)
    proc_tiles_par = create_proc_tiles_fct(num_procs, unpack_args_and_process_patch)
    stitch = create_stitch_fct()

    #define processing pipeline
    pipeline = create_pipeline([tile,
                                proc_tiles_par,
                                stitch])

    #I/O
    sar_data, sar_data_array, date = open_sar_data(sar_tiff_filepath)
    
    #process data
    damage_data, damage_data_sqrt = pipeline(sar_data_array)

    #I/O
    save_damage_data(outpath, sar_data, date, damage_data, damage_data_sqrt)
    
