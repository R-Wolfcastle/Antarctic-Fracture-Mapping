#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 16:03:34 2021

@author: eetss
"""

from osgeo import gdal
import numpy as np
from skimage.morphology import skeletonize
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pyproj import Proj, transform
# from scipy.ndimage import binary_dilation as bd
import rasterio
import rasterio.mask
#import cartopy
  
def array_to_geotiff(array, original_tif, new_tif_dir, compression=None):
    driver = gdal.GetDriverByName('GTiff')
    ny = array.shape[0]
    nx = array.shape[1]
    if compression:
        new_data = driver.Create(new_tif_dir, nx, ny, 1, gdal.GDT_Float32, options=['COMPRESS={}'.format(compression)])
    else:
        new_data = driver.Create(new_tif_dir, nx, ny, 1, gdal.GDT_Float32)
    geo_transform = original_tif.GetGeoTransform()  #get GeoTranform from existing dataset
    projection = original_tif.GetProjection() #similarly get from orignal tifD
    new_data.SetGeoTransform(geo_transform)
    new_data.SetProjection(projection)
    
    new_data.GetRasterBand(1).WriteArray(array)
    
    new_data.FlushCache() #write to disk
    return new_data

def array_to_multiband_geotiff(array, original_tif, new_tif_dir, compression=None):
    assert array.ndim==3, "This function only takes 3D arrays, if yours is 2D consider using array_to_geotiff"

    driver = gdal.GetDriverByName('GTiff')
    n_bands = array.shape[0]
    ny = array.shape[1]
    nx = array.shape[2]
    if compression:
        new_data = driver.Create(new_tif_dir, nx, ny, n_bands, gdal.GDT_Float32, options=['COMPRESS={}'.format(compression)])
    else:
        new_data = driver.Create(new_tif_dir, nx, ny, n_bands, gdal.GDT_Float32)
    geo_transform = original_tif.GetGeoTransform()  #get GeoTranform from existing dataset
    projection = original_tif.GetProjection() #similarly get from orignal tifD
    new_data.SetGeoTransform(geo_transform)
    new_data.SetProjection(projection)
	
    for band in range(n_bands):
        new_data.GetRasterBand(band+1).WriteArray(array[band,:,:])

    new_data.FlushCache() #write to disk
    return new_data
  
def get_all_dates(data_dates, date_string_format='%Y%m%d', days=6):
  from_date_time = datetime.strptime(data_dates[0], date_string_format)
  to_date_time = datetime.strptime(data_dates[-1], date_string_format)
  
  all_dates = [from_date_time.strftime(date_string_format)]
  date_time = from_date_time
  while date_time < to_date_time:
      date_time += timedelta(days=6)
      all_dates.append(date_time.strftime(date_string_format))
      
  return all_dates

def dates_between(date1, date2):
  date_string_format='%Y%m%d'
  from_date_time = datetime.strptime(date1, date_string_format)
  to_date_time = datetime.strptime(date2, date_string_format)

  all_dates = [from_date_time.strftime(date_string_format)]
  date_time = from_date_time
  while date_time < to_date_time:
      date_time += timedelta(days=1)
      all_dates.append(date_time.strftime(date_string_format))

  return all_dates

