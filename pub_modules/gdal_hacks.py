#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Feb 25 09:46:10 2022

@author: eetss
"""

#std libs
import subprocess
import os

#3rd party
import numpy as np
from osgeo import gdal

def add_blur_to_vrt(filepath: str, coefficients: np.ndarray, kernel_width: int):
    """inserts instructions for gaussian blur into vrt file
    Args:
        filepath (:obj:`string`): path to file, into which blur instructions will be inserted
        coefficients (:obj:`np.ndarray`): coefficients in the gaussian kernel
        kernel_width (:obj:`int`): width of the square kernel
    """

    precoefs = """       <Kernel normalized="1">\n       <Size>{}</Size>\n""".format(kernel_width)
    coefs_line_1 = """          <Coefs>{}\n""".format(" ".join(list(map(str, coefficients[:,0]))))
    postcoefs = """           </Coefs>\n          </Kernel>\n"""

    lines = open(filepath, 'r').readlines()
    lines[6] = """      <KernelFilteredSource>\n"""
    lines[13] = """      </KernelFilteredSource>\n"""
    lines.insert(13, precoefs)
    lines.insert(14, postcoefs)
    lines.insert(14, coefs_line_1)
    for i in range(1, coefficients.shape[1]):
        lines.insert(15, """            {}\n""".format(" ".join(list(map(str, coefficients[:,-i])))))
    open(filepath, 'w').write("".join(lines))


def reband_vrt_old(filepath:str, band_names:list):
    i = 10
    for band_name in band_names:
        print(i)
        pattern='<VRTRasterBand dataType=\"Float32\" band=\"{}\">'.format(i)
        replacement='<VRTRasterBand dataType=\"Float32\" band=\"{}\">'.format(band_name)
        command = ["sed", "0,/{}/s/{}/{}/".format(pattern, pattern, replacement), filepath]
        print(command)
        subprocess.call(command, stdout=subprocess.DEVNULL)
#        subprocess.call('sed s/\<VRTRasterBand\ dataType\=\"Float32\"\ band\=\"{}\"\>/\<VRTRasterBand\ dataType\=\"Float32\"\ band\=\"{}\"\>/g {}'.format(i, band_name, filepath))
        #subprocess.call('sed 0,/\<VRTRasterBand\ dataType\=\"Float32\"\ band\=\"{}\"\>/s/\<VRTRasterBand\ dataType\=\"Float32\"\ band\=\"{}\"\>/re\<VRTRasterBand\ dataType\=\"Float32\"\ band\=\"{}\"\>/ {}'.format(i, i, band_name, filepath))
        raise
        i += 1

def reband_vrt_new_old(filepath:str, band_names:list):
    i = 1
    infile = filepath
    outfile = filepath+"_mod"
    for band_name in band_names:
        print(i)
        #cmd = """sed 0,/\<VRTRasterBand\ dataType\=\\"Float32\\"\ band\=\\"{}\\"\>/s/\<VRTRasterBand\ dataType\=\\"Float32\\"\ band\=\\"{}\\"\>/\<VRTRasterBand\ dataType\=\\"Float32\\"\ band\=\\"{}\\"\>/g {} >{}""".format(i, i, band_name, infile, filepath+"_mod")
        cmd = """sed -i s/\<VRTRasterBand\ dataType\=\\"Float32\\"\ band\=\\"{}\\"\>/\<VRTRasterBand\ dataType\=\\"Float32\\"\ band\=\\"{}\\"\>/g {} >{}""".format(i, band_name, infile, outfile)
        os.system(cmd)
        os.system("mv {} {}".format(outfilei, infile))

        infile = outfile

        i += 1

        if i>10:
            raise


def reband_vrt(filepath:str, prior_band_ids: list, band_names:list):
    ds = gdal.Open(filepath, gdal.GA_Update)
    for band, new_name in zip(prior_band_ids, band_names):
        rb = ds.GetRasterBand(band)
        rb.SetDescription(new_name)
    del ds



#based on: https://github.com/aerospaceresearch/DirectDemod/blob/Vladyslav_dev/directdemod/merger.py
#(Copyright (c) 2018 AerospaceResearch) (changed a bit to do median mosaicing)
# only altered and tested median and max.

def get_merge_fct(name: str) -> str:
    """retrieves code for resampling method
    Args:
        name (:obj:`string`): name of resampling method
    Returns:
        method :obj:`string`: code of resample method
    """

    methods = {
        "first":
        """
import numpy as np
def first(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize,raster_ysize, buf_radius, gt, **kwargs):
    y = np.ones(in_ar[0].shape)
    for i in reversed(range(len(in_ar))):
        mask = in_ar[i] == 0
        y *= mask
        y += in_ar[i]
    np.clip(y,0,255, out=out_ar)
""",
        "last":
        """
import numpy as np
def last(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize,raster_ysize, buf_radius, gt, **kwargs):
    y = np.ones(in_ar[0].shape)
    for i in range(len(in_ar)):
        mask = in_ar[i] == 0
        y *= mask
        y += in_ar[i]
    np.clip(y,0,255, out=out_ar)
""",
        "max":
        """
import numpy as np
def max(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize,raster_ysize, buf_radius, gt, **kwargs):
    x = np.asarray(in_ar)
    y = np.nanmax(x, axis=0, out=out_ar)
""",
        "median":
        """
import numpy as np
def median(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize,raster_ysize, buf_radius, gt, **kwargs):
    x = np.asarray(in_ar)
    y = np.nanmedian(x, axis=0, out=out_ar)
""",
        "variance":
        """
import numpy as np
def variance(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize,raster_ysize, buf_radius, gt, **kwargs):
    x = np.asarray(in_ar)
    y = np.nanvar(x, axis=0, out=out_ar)
""",
        "average":
        """
import numpy as np
def average(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize,raster_ysize, buf_radius, gt, **kwargs):
    div = np.zeros(in_ar[0].shape)
    for i in range(len(in_ar)):
        div += (in_ar[i] != 0)
    div[div == 0] = 1
    
    y = np.sum(in_ar, axis = 0, dtype = 'uint16')
    y = y / div
"""}
    if name not in methods:
        raise ValueError(
            "ERROR: Unrecognized resampling method (see documentation): '{}'.".
            format(name))

    return methods[name]



def add_merge_fct_to_vrt(filename: str, resample_name: str) -> None:
    """inserts pixel-function into vrt file named 'filename'
    Args:
        filename (:obj:`string`): name of file, into which the function will be inserted
        resample_name (:obj:`string`): name of resampling method
    """

    header = """  <VRTRasterBand dataType="Float64" band="1" subClass="VRTDerivedRasterBand">"""
    contents = """
    <PixelFunctionType>{0}</PixelFunctionType>
    <PixelFunctionLanguage>Python</PixelFunctionLanguage>
    <PixelFunctionCode><![CDATA[{1}]]>
    </PixelFunctionCode>"""

    lines = open(filename, 'r').readlines()
    lines[3] = header  # FIX ME: 3 is a hand constant
    lines.insert(4, contents.format(resample_name,
                                    get_merge_fct(resample_name)))
    open(filename, 'w').write("".join(lines))





