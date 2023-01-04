#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sept 08 16:39:40 2022

@author: Trys
"""

#std libs
import sys
import os
from pathlib import Path
from itertools import product
from itertools import compress
from multiprocessing import Pool
import shutil

#3rd party
import rasterio 
from rasterio.windows import Window
from rasterio.mask import mask
import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import h5py
import shapely
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


spacing = int(sys.argv[7])
buffer_size = int(sys.argv[8])

GL_SHAPE="/resstore/b0133/eetss/MEaSUREs Antarctic Boundaries/GroundingLine_Antarctica_v2.shp"

outdir = "/nobackup/eetss/damage_mapping/sv_timeseries/multijob_version/"
#coords_fp = "/resstore/b0133/eetss/damage_mapping/arc4/all_output/daily_mosaics/ais_coords.npy"
shp_filepath = "/resstore/b0133/eetss/damage_mapping/fd_timeseries/shapefiles/bjd_julia/minimum_ice_shelf_mask_Antarctica_BJD_v03.shp"

num_procs = 12

datadir = str(sys.argv[6])
job_chunk_index = int(sys.argv[5])

tmp_dir = str(os.environ['TMPDIR'])

#probs best to start again...
#if not os.path.isfile(outdir+"/h5_files/ais_masked_{}_0.hdf5".format(job_chunk_index)):
#    shutil.copy(outdir+"/h5_files/ais_masked_{}_{}.hdf5".format(job_chunk_index, i), tmp_dir)

fps_a = sorted(list(map(lambda fp: str(fp), Path(datadir).rglob("*250_AIS_median_mosaic_masked.tiff"))))
dates_1a = [fp.split("/")[-1].split("_")[0] for fp in fps_a]

fps_bs = sorted(list(map(lambda fp: str(fp), Path(datadir).rglob("*250_AIS_median_bs_mosaic_masked.tiff"))))
dates_bs = [fp.split("/")[-1].split("_")[0] for fp in fps_bs]


def mk_ais_grid(spacing=5000, buffer_size=10000):
    ulx = int(sys.argv[1])
    uly = int(sys.argv[2])
    lrx = int(sys.argv[3])
    lry = int(sys.argv[4])

    shp_fp = "/resstore/b0133/eetss/damage_mapping/fd_timeseries/shapefiles/bjd_julia/minimum_ice_shelf_mask_Antarctica_BJD_v03.shp"

    reader = cartopy.io.shapereader.Reader(shp_fp)
    records = reader.records()
    shelves = []
    for rec in records:
        shelves.append(rec.geometry)
 
    xs = np.arange(ulx, lrx, spacing)
    ys = np.arange(lry, uly, spacing)

    all_point_coords = list(product(xs, ys))

    included_coords = []
    included_pt_objects = []

    ix = []
    iy = []


    for point_coords in all_point_coords[:1000000]:
        point = Point(point_coords[0], point_coords[1])
        #if True:
        for polygon in shelves:
            if polygon.contains(point):
                buffered_point = point.buffer(buffer_size, cap_style=3) #cap style of 1 is circle, 3 square and 2 is flat (but that doesn't make sense for a point buffer...)
                #buffer is, of course, a /radius/ not a /diameter/... remember that.
                ix.append(point_coords[0])
                iy.append(point_coords[1])

                included_coords.append(point_coords)
                included_pt_objects.append(buffered_point)
                
                break

    if len(included_coords)==0:
        raise ValueError('No points in this chunk mate!')

    make_coverage_map([ix, iy])

    return included_coords, included_pt_objects


def make_coverage_map(coords):
    xs = np.array(coords[0])
    ys = np.array(coords[1])

    minx = np.min(xs)
    maxx = np.max(xs)
    miny = np.min(ys)
    maxy = np.max(ys)

    xi = np.arange(minx-1000, maxx+1000, 1000)
    yi = np.arange(miny-1000, maxy+1000, 1000)

    xi, yi = np.meshgrid(xi, yi)

    fig = plt.figure()
    fig.set_size_inches(10, 10)

    ax = plt.subplot(111, projection=ccrs.SouthPolarStereo())
    ax.set_extent([minx, maxx, miny, maxy], crs=ccrs.SouthPolarStereo())

    plt.plot(xs, ys, 'k.')
    
    reader_gl = cartopy.io.shapereader.Reader(GL_SHAPE)
    gl_geom = next(reader_gl.geometries())

    ax.add_geometries(gl_geom, ccrs.SouthPolarStereo(), facecolor="white", edgecolor="black", alpha=0.5)
    plt.savefig('/resstore/b0133/eetss/damage_mapping/fd_timeseries/outputs/misc_space/ais_coverage.png',dpi=200)
    plt.close(fig)


def add_data(h5_file, coords, means, mean_dates, sds, sd_dates, data_index):
    group = h5_file.create_group("{}".format(data_index))
    group.create_dataset('coords', data=coords)

    fgroup = group.create_group('fracture')
    fgroup.create_dataset("dates", data=mean_dates)
    fgroup.create_dataset("fd", data=means)

    bsgroup = group.create_group('backscatter')
    bsgroup.create_dataset("dates", data=sd_dates)
    bsgroup.create_dataset("sd", data=sds)


def extract_data_from_tiff_files_by_point(point_coords, buffered_point, h5_file, data_index):

    print("extracting 1a means")
    means = []
    for fp in tqdm(fps_a):
        with rasterio.open(fp) as src:
            try:
                out_image, out_transform = rasterio.mask.mask(src, [buffered_point], crop=True)
                mean = np.mean(out_image)
            except:
                mean = 0
            means.append(mean)

    print("extracting bs sds")
    sds = []
    for fp in tqdm(fps_bs):
        with rasterio.open(fp) as src:
            try:
                out_image, out_transform = rasterio.mask.mask(src, [buffered_point], crop=True)
                sd = np.std(out_image)
            except:
                sd = 0
            sds.append(sd)
    
    add_data(h5_file, point_coords, means, dates_1a, sds, dates_bs, data_index)

    return data_index+1


def process_points(coords, points, h5_filepath):
    if os.path.isfile(h5_filepath):
        file_ = h5py.File(h5_filepath, "r")
        data_indices = [int(ind_) for ind_ in file_.keys()]

        data_index = max(data_indices) + 1
        coords = coords[data_index:]
        points = points[data_index:]

        file_.close()
    else:
        data_index = 0

    if len(coords)==0:
        raise ValueError('No points left in this chunk mate!')

    with h5py.File(h5_filepath, "a") as h5_file:
        for coords, buffered_point in list(zip(coords, points)):
            data_index = extract_data_from_tiff_files_by_point(coords, buffered_point, h5_file, data_index)
        return


def unzip_and_proc(argzip):
    process_points(*argzip)


def execute_ting():

    coords, points = mk_ais_grid(spacing, buffer_size)
#    np.save(np.array(coords), coords_fp)

    print("{} points to do! wish me luck".format(len(coords)))

    coords_subsets = np.array_split(coords, num_procs)
    pts_subsets = np.array_split(points, num_procs)

    h5_fps = [tmp_dir+"/larsens_masked_2pt5_5_{}_{}.hdf5".format(job_chunk_index, i) for i in range(num_procs)]

    argzip = zip(coords_subsets, pts_subsets, h5_fps)

    pool = Pool(num_procs)
    pool.map(unzip_and_proc, argzip)
    pool.close()
    pool.join() 
    
    if not os.path.isdir(outdir+"/hdf5_files_larsens_2pt5_5/"):
        os.mkdir(outdir+"/hdf5_files_larsens_2pt5_5/")

    for file_ in [str(p) for p in Path(tmp_dir).rglob("larsens_masked_2pt5_5*.hdf5")]:
        shutil.copy(file_, outdir+"/hdf5_files_larsens_2pt5_5/")


if __name__ == '__main__':
    execute_ting()

