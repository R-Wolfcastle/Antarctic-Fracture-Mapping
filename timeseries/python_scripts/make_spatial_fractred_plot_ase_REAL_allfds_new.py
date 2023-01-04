#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

#std libs
import traceback
import os
from pathlib import Path
from datetime import datetime
from itertools import compress
import sys
from datetime import datetime, timedelta

#3rd party
import rasterio
from rasterio.windows import Window
from rasterio.mask import mask
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.cm as cm
import numpy as np
from tqdm import tqdm
import h5py
import proplot as pplt
from scipy.interpolate import griddata
from osgeo import gdal, osr
from astropy.convolution import convolve as astroconv
from astropy.convolution import convolve_fft as astroconv_fft
from astropy.convolution import Gaussian2DKernel, Box2DKernel

#local apps
sys.path.insert(1, "/nfs/b0133/eetss/my_python_modules/")
from fd_calcs_and_plots import create_rfd_plot_for_point_trend_only


GL_SHAPE="/nfs/b0133/eetss/MEaSUREs Antarctic Boundaries/GroundingLine_Antarctica_v2.shp"

def geocode_array(array, bounding_crs_coords, resolution, filename, compression="DEFLATE"):
     drv = gdal.GetDriverByName("GTiff")
     if compression:
       ds = drv.Create(filename, array.shape[1], array.shape[0], 1, gdal.GDT_Float32, options=['COMPRESS={}'.format(compression)])
     else:
       ds = drv.Create(filename, array.shape[1], array.shape[0], 1, gdal.GDT_Float32)
     ds.SetGeoTransform([bounding_crs_coords[0], resolution, 0, bounding_crs_coords[3], 0, resolution])
     
     ref_ds = gdal.Open('/nfs/b0133/eetss/modis_gnd.tiff', gdal.GA_ReadOnly)
     prj = ref_ds.GetProjection()
     
     ds.SetProjection(prj)
     
     ds.GetRasterBand(1).WriteArray(array)
     ds.FlushCache()
     # return ds 


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


def create_rfd_for_point_noplot(means, dates, frac_fds, frac_dates, coords, key, step_size=0.2):

    assert (11/step_size).is_integer(), "choose a step_size that is an integer_factor of 11."

    dwbs_superlist = get_dates_for_similar_imgs(means, dates, step_size)
    frac_wbs_ts_list = get_fds_for_each_hline_ts(dwbs_superlist, frac_fds, frac_dates)

    #print(frac_wbs_ts_list)

    slope_coeffs = []
    plot_fds = np.zeros(len(frac_wbs_ts_list[0]))*np.nan
    sc_errors = []
    subplot_fits = []
    subplot_fracture_datas = []

    # prev_max = 0
    prev_min = 1e10

    #print(key)

    for fracture_data in frac_wbs_ts_list:
        num_finite_vals = np.count_nonzero(~np.isnan(np.array(fracture_data)))
        if num_finite_vals>11:
            fit_indices = [i for i in range(len(fracture_data)) if not np.isnan(fracture_data[i])]
            if ((fit_indices[-1]-fit_indices[0]) > 730):
            # if fit_indices[-1]-fit_indices[0] > -1e10:
                fit_fds = [fd for fd in fracture_data if not np.isnan(fd)]

                coeffs, vs = np.polyfit(fit_indices, fit_fds, deg=1, cov=True)
                # errors = [np.sqrt(vs[0,0]*num_finite_vals), np.sqrt(vs[1,1]*num_finite_vals)] #pretty sure below is standard error, but i'd really like something more like standard deviation...
                errors = [np.sqrt(vs[0,0]), np.sqrt(vs[1,1])]

                slope_error = errors[0]
                int_error = errors[1]

                #print(slope_error)

                coords = np.arange(len(fracture_data))

                # fit = np.zeros_like(coords)
                fit_terms = list(map(lambda i: coeffs[i]*(coords**(1-i)), list(range(2))))
                # print(fit_terms)
                fit = sum(fit_terms)


                #if slope_error<1 and coeffs[1]>0 and (fit[-1]/coeffs[1])<2 and (fit[-1]/coeffs[1])>0.5:
                if True:
                    subplot_fits.append(fit)
                    subplot_fracture_datas.append(fracture_data)

                    slope_coeffs.append(coeffs[0]/coeffs[1])


                    sc_error = np.sqrt(     (slope_error/coeffs[1])**2    +    (coeffs[0]*int_error/(coeffs[1]**2))**2    )
                    sc_errors.append(sc_error)

                    if slope_error:
                        for index in fit_indices: #should do this in fancy way really but cba
                            plot_fds[index] = (fracture_data[index]/coeffs[1]) #scaled to have the same y-intercept (see below, we add back in, the mean y-intercept)


    scs = np.array(slope_coeffs)
    dscs = np.array(sc_errors)
    # print(scs)


    ####For now, let's just combine the coeffs in an unweighted way... Could weight them, or even do an IQR of them!
    # mean_slope_coeff = np.mean(scs)
    # sd_slope_coeff = np.std(scs)


    weights = (1/dscs**2)
    # weights = np.ones_like(dscs**2)

    mean_slope_coeff = np.average(scs, weights=weights)

    ##THIS MIGHT NOT BE REALLY TRUE... I MEAN, THINK ABOUT IT: 
    sd_slope_coeff_sem = np.sqrt(1/(np.sum(1/(dscs**2)))) #actually standard error not standard deviation...
    sd_slope_coeff_sem = sd_slope_coeff_sem*np.sqrt(len(sc_errors))

    ##LET'S DO A WEIGHTED STANDARD DEVIATION INSTEAD EH?
    if not scs.size==1:
        sd_slope_coeff_wsd = np.sqrt(       np.sum(weights*(scs - mean_slope_coeff)**2) / (  ((scs.size-1)/scs.size) *  np.sum(weights)  )       )
    else:
        sd_slope_coeff_wsd = np.nan

    ##Two maybe ok ways of getting an error, so let's take a maximum over the two errors. This should work for low number of estimators and high!
    if not (np.isnan(sd_slope_coeff_sem) or np.isnan(sd_slope_coeff_wsd)):
        sd_slope_coeff = max(sd_slope_coeff_wsd, sd_slope_coeff_sem)
    elif not np.isnan(sd_slope_coeff_sem):
        sd_slope_coeff = sd_slope_coeff_sem
    elif not np.isnan(sd_slope_coeff_wsd):
        sd_slope_coeff = sd_slope_coeff_wsd
    else:
        sd_slope_coeff = np.nan




    mean_fit = 1+coords*mean_slope_coeff
    lbound = 1+coords*(mean_slope_coeff-1.96*sd_slope_coeff)
    ubound = 1+coords*(mean_slope_coeff+1.96*sd_slope_coeff)

#    print(mean_fit[-1], lbound[-1], ubound[-1])
#    print("***********")

    return lbound[-1], ubound[-1], mean_fit[-1]


def create_rfd_for_point_nostep(frac_fds, key):
    frac_fds = list(np.where(np.array(frac_fds)!=0, np.array(frac_fds), np.nan))

    fit_indices = [i for i in range(len(frac_fds)) if not np.isnan(frac_fds[i])]
    fit_fds = [fd for fd in frac_fds if not np.isnan(fd)]
    
    if (len(fit_indices)==0) or (np.sum(fit_fds)==0.):
        return np.nan, np.nan, np.nan
    
    coeffs, vs = np.polyfit(fit_indices, fit_fds, deg=1, cov=True)

    errors = [np.sqrt(vs[0,0]), np.sqrt(vs[1,1])]
    slope_error = errors[0]
    int_error = errors[1]

    coords = np.arange(len(frac_fds))

    # fit = np.zeros_like(coords)
    fit_terms = list(map(lambda i: coeffs[i]*(coords**(1-i)), list(range(2))))
    # print(fit_terms)
    fit = sum(fit_terms)

    rel_slope_coeff = coeffs[0]/coeffs[1]
    rsc_error = np.sqrt(     (slope_error/coeffs[1])**2    +    (coeffs[0]*int_error/(coeffs[1]**2))**2    )

    rel_fit = coords*rel_slope_coeff

    fit_num = 1+rel_fit[-1]
    lbound = 1+(coords*(rel_slope_coeff-rsc_error))[-1]
    ubound = 1+(coords*(rel_slope_coeff+rsc_error))[-1]

    plt.figure(figsize=(12,8))
    plt.plot(rel_fit+1)
    plt.scatter(coords, frac_fds/coeffs[1])
    plt.gca().axes.get_xaxis().set_ticks([])
    plt.savefig("/nfs/b0133/eetss/damage_mapping/fd_timeseries/outputs/ase_space/{}_nostep.png".format(key), dpi=150)
    plt.close()

    return lbound, ubound, fit_num


def get_mean_fd(frac_fds):
    frac_fds = np.where(np.array(frac_fds)!=0, np.array(frac_fds), np.nan)
    return np.nanmean(frac_fds)


def create_rfd_for_point_noplot_nostep(frac_fds):

    fit_indices = [i for i in range(len(frac_fds)) if not np.isnan(frac_fds[i])]
    fit_fds = [fd for fd in frac_fds if not np.isnan(fd)]
   
    if (len(fit_indices)==0) or (np.sum(fit_fds)==0.):
        return np.nan, np.nan, np.nan

    coeffs, vs = np.polyfit(fit_indices, fit_fds, deg=1, cov=True)

    errors = [np.sqrt(vs[0,0]), np.sqrt(vs[1,1])]
    slope_error = errors[0]
    int_error = errors[1]

    coords = np.arange(len(frac_fds))

    # fit = np.zeros_like(coords)
    fit_terms = list(map(lambda i: coeffs[i]*(coords**(1-i)), list(range(2))))
    # print(fit_terms)
    fit = sum(fit_terms)
    
    rel_slope_coeff = coeffs[0]/coeffs[1]
    rsc_error = np.sqrt(     (slope_error/coeffs[1])**2    +    (coeffs[0]*int_error/(coeffs[1]**2))**2    )

    rel_fit = coords*rel_slope_coeff

    fit_num = 1+rel_fit[-1]
    lbound = 1+(coords*(rel_slope_coeff-rsc_error))[-1]
    ubound = 1+(coords*(rel_slope_coeff+rsc_error))[-1]
    

    return lbound, ubound, fit_num


def add_data_to_bounds_h5(h5_file, key, lb, ub, mf):
    group = h5_file.create_group(key)

    group.create_dataset("lb", data=lb)
    group.create_dataset("ub", data=ub)
    group.create_dataset("mean", data=mf)


def execute_make_bounds_h5(hdf5_fp, bounds_hdf5_fp):
    mother_h5 = h5py.File(hdf5_fp, "r")

    with h5py.File(bounds_hdf5_fp, "a") as bounds_h5:

        try:
            for index in mother_h5.keys():
                frac_dates = mother_h5[index]["fracture"]["dates"][()]
                frac_dates = [f_date.decode("utf-8") for f_date in frac_dates]
                frac_fds = mother_h5[index]["fracture"]["fd"][()]
             
                bs_dates = mother_h5[index]["backscatter"]["dates"][()]
    #            bs_dates = [b_date.decode("utf-8") for b_date in bs_dates]
                bs_sds = mother_h5[index]["backscatter"]["sds"][()]
                try:
                    lb, ub, mf = create_rfd_plot_for_ice_shelf(bs_sds, bs_dates, frac_fds, frac_dates, index, step_size=0.2)
                    add_data_to_bounds_h5(bounds_h5, index, lb, ub, mf)
                except Exception as excp:
                    print(excp)
                    print(sys.exc_info()[2])
        except RuntimeError as RE:
            print(RE)

    mother_h5.close()

def make_error_map(coords_and_data, rel_change=False):
    xs = []
    ys = []
    errors = []

    xs_null_within_error = []
    ys_null_within_error = []

    for cad in coords_and_data:
        mf = cad[3]
        if not np.isnan(mf) and mf<300:
            lb = cad[1]
            ub = cad[2]
            mean_fd = cad[4]

            error = np.abs(ub-mf)
            
            #mf -= 1
            #mf = min(mf, 1) if mf>0 else max(mf, -1)

            if not rel_change:
                error *= mean_fd
            
            if not error>5:

                errors.append(error)
                xs.append(cad[0][0])
                ys.append(cad[0][1])

    print("Found data, constructing map")

    vmin, vmax = [0, 2]

    xs = np.array(xs)
    ys = np.array(ys)

    errors = np.array(errors)
    print(errors, np.mean(errors), np.std(errors))

    minx = np.min(xs)
    maxx = np.max(xs)
    miny = np.min(ys)
    maxy = np.max(ys)

    grid_spacing = 1000

    xi = np.arange(minx, maxx, grid_spacing)
    yi = np.arange(miny, maxy, grid_spacing)

    xi, yi = np.meshgrid(xi, yi)

    zi = griddata((xs,ys), errors, (xi,yi), method='linear')

    ##make mask:##
    data_cols = (list(xs) - minx)/grid_spacing - 1
    data_rows = (list(ys) - miny)/grid_spacing - 1
    z_bn = np.zeros_like(zi)
    for row, col in list(zip(data_rows, data_cols)):
        z_bn[int(row), int(col)] = 1
    mean_kernel = Box2DKernel(5000/grid_spacing)
    z_bn = astroconv_fft(z_bn, mean_kernel, allow_huge=True)
    z_bn = np.where( z_bn>0.05*(1/(5000/grid_spacing)**2), 1, 0 )
    ####

    zi = zi*z_bn

    #geocode_array(zi, [minx, maxy, maxx, miny], grid_spacing, '/nfs/b0133/eetss/damage_mapping/fd_timeseries/outputs/ase_space/ase_masked_step_02_REAL.tiff', compression="DEFLATE")


    gl_reader = cartopy.io.shapereader.Reader(GL_SHAPE)
    gl_geom = next(gl_reader.geometries())


    fig = plt.figure()
    fig.set_size_inches(20, 20)

    ax = plt.subplot(111, projection=ccrs.SouthPolarStereo())

    ax.set_extent([minx, maxx, miny, maxy], crs=ccrs.SouthPolarStereo()) ##WHOLE DOMAIN

    cmap = cm.get_cmap('cividis')

    plt.contourf(xi, yi, zi, levels=500, cmap=cmap, vmin=vmin, vmax=vmax)
    #plt.contourf(xi, yi, zi, levels=500, cmap=cmap)

    plt.plot(xs, ys, 'k,', alpha=0.5)#, markersize=0.01)


    ax.add_geometries(gl_geom, ccrs.SouthPolarStereo(), facecolor="white", edgecolor="black", alpha=1)

    gl = ax.gridlines(crs=ccrs.PlateCarree(),
                  color='gray',
                  alpha=0.5,
                  draw_labels=True,
                  dms=True,
                  x_inline=False,
                  y_inline=True,
                  linestyle='--')

    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    plt.savefig('/nfs/b0133/eetss/damage_mapping/fd_timeseries/outputs/ase_space/ase_masked_step005_errors.png', dpi=300, bbox_inches='tight')
    plt.close(fig) 



def make_map(coords_and_data, rel_change=False, trend_only=False):
    xs = []
    ys = []
    mean_fit_changes = []
   
    xs_null_within_error = []
    ys_null_within_error = []

    nanxs = []
    nanys = []

    for cad in coords_and_data:
        mf = cad[3]
        if not np.isnan(mf) and mf<300:
        #if True:
            lb = cad[1]
            ub = cad[2]
            mean_fd = cad[4]
    #        print(lb, ub)
            if ((lb-1)*(ub-1))<0:# and (mean_fd-lb)>0.2:
    #            print("null_within_error")

                #NOTE: bit of a hack as can't remember how to do properly:
#                mf = 1

                xs_null_within_error.append(cad[0][0])
                ys_null_within_error.append(cad[0][1])
            
            if not trend_only:
                mf -= 1
                mf = min(mf, 1) if mf>0 else max(mf, -1)
            

            if not rel_change and not trend_only:
                mean_fit_times_mean_fd = mf*mean_fd
            else:
                mean_fit_times_mean_fd = mf


            mean_fit_changes.append(mean_fit_times_mean_fd)
            xs.append(cad[0][0])
            ys.append(cad[0][1])
        else:
            nanxs.append(cad[0][0])
            nanys.append(cad[0][1])


    if rel_change:
        print("making fractional change map")
        vmin, vmax = [-1.5, 1.5]
    else:
        print("making `actual` change map (lol)")
        if not trend_only:
            vmin, vmax = [-0.25, 0.25]
        else:
            print("HELLLLOOOOO")
            vmin, vmax = [-0.5, 0.5]


    xs = np.array(xs)
    ys = np.array(ys)
    nanxs = np.array(nanxs)
    nanys = np.array(nanys)
    mean_fit_changes = np.array(mean_fit_changes)
    xs_null_within_error = np.array(xs_null_within_error)
    ys_null_within_error = np.array(ys_null_within_error)

    minx = np.min(xs)
    maxx = np.max(xs)
    miny = np.min(ys)
    maxy = np.max(ys)

    grid_spacing = 1000

    #xi = np.arange(minx-(50*grid_spacing), maxx+(50*grid_spacing), grid_spacing)
    #yi = np.arange(miny-(50*grid_spacing), maxy+(50*grid_spacing), grid_spacing)
    xi = np.arange(minx, maxx, grid_spacing)
    yi = np.arange(miny, maxy, grid_spacing)

    xi, yi = np.meshgrid(xi, yi)

    zi = griddata((xs,ys), mean_fit_changes, (xi,yi), method='linear')
    
    ##make mask:##
    #data_cols = (list(xs) - minx + (10*grid_spacing))/grid_spacing
    #data_rows = (list(ys) - miny + (10*grid_spacing))/grid_spacing
    data_cols = (list(xs) - minx)/grid_spacing - 1
    data_rows = (list(ys) - miny)/grid_spacing - 1
    z_bn = np.zeros_like(zi)
    for row, col in list(zip(data_rows, data_cols)):
        z_bn[int(row), int(col)] = 1
    mean_kernel = Box2DKernel(5000/grid_spacing)
    z_bn = astroconv_fft(z_bn, mean_kernel, allow_huge=True)
    #z_bn = np.where( z_bn>0.05*(1/(5000/grid_spacing)**2), 1, 0 )
    z_bn = np.where( z_bn>0.005*(1/(5000/grid_spacing)**2), 1, 0 )
    ####

    zi = zi*z_bn


    geocode_array(zi, [minx, maxy, maxx, miny], grid_spacing, outtiff_fp, compression="DEFLATE")


    gl_reader = cartopy.io.shapereader.Reader(GL_SHAPE)
    gl_geom = next(gl_reader.geometries())

#    cf_reader = cartopy.io.shapereader.Reader(GL_SHAPE)
#    cf_geom = next(cf_reader.geometries())

    
    fig = plt.figure()
    fig.set_size_inches(20, 20)

    ax = plt.subplot(111, projection=ccrs.SouthPolarStereo())
    
    ax.set_extent([minx, maxx, miny, maxy], crs=ccrs.SouthPolarStereo()) ##WHOLE DOMAIN

    #cmap = cm.get_cmap('RdBu_r')
    cmap = cm.get_cmap('seismic')
    #cmap = pplt.Colormap('Balance')

    plt.contourf(xi, yi, zi, levels=500, cmap=cmap, vmin=vmin, vmax=vmax)
    #plt.contourf(xi, yi, zi, vmin=-100, vmax=100)

    
#    plt.scatter(xs, ys, marker='.', c='black')#, markersize=0.01)
#    plt.plot(xs, ys, 'k,', alpha=0.5)#, markersize=0.01)
#    plt.plot(xs_null_within_error, ys_null_within_error, 'k.')
#    plt.plot(nanxs, nanys, 'k.')

    
    ax.add_geometries(gl_geom, ccrs.SouthPolarStereo(), facecolor="white", edgecolor="black", alpha=1)
#    ax.add_geometries(cf_geom, ccrs.SouthPolarStereo(), facecolor=None, edgecolor="black", alpha=1)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), 
                  color='gray',
                  alpha=0.5,
                  draw_labels=True, 
                  dms=True, 
                  x_inline=False, 
                  y_inline=True,
                  linestyle='--')

    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
#    gl.ylocator = mticker.FixedLocator(yticks)
#    gl.xlocator = mticker.FixedLocator(xticks)
#    gl.xlabel_style = {'rotation':0}

    plt.savefig(outmap_fp ,dpi=300, bbox_inches='tight')
    plt.close(fig)


def close_h5(h5_file):
    h5_file.close()

def make_map_from_h5(h5_filepath, rel_change=False, plot_errors=False, trend_only=False):
    things = []
    with h5py.File(point_data_h5_fp, "r") as h5_file:
        for key in h5_file:
            things.append([h5_file.get(key)["coords"][()], 
                           h5_file.get(key)["lb"][()], 
                           h5_file.get(key)["ub"][()],
                           h5_file.get(key)["mf"][()],
                           h5_file.get(key)["mean_fd"][()]])
    if plot_errors:
        make_error_map(things, rel_change, trend_only)
    else:
        make_map(things, rel_change, trend_only)


def save_things_as_hdf5_file(things, h5_filepath):
    with h5py.File(point_data_h5_fp, "a") as h5_file:
        index = 0
        for coords, lb, ub, mf, mean_fd in things:
            grp = h5_file.create_group(str(index))
            grp.create_dataset("coords", data=coords)
            grp.create_dataset("lb", data=lb)
            grp.create_dataset("ub", data=ub)
            grp.create_dataset("mf", data=mf)
            grp.create_dataset("mean_fd", data=mean_fd)

            index += 1
    return

def check_coord(bounds:list, coords:list):
    ulx, uly, lrx, lry = bounds
    x, y = coords
    if (x>ulx) and (x<lrx) and (y>lry) and (y<uly):
        return True
    return False

def execute(hdf5_fps, point_data_h5_fp, overwrite=False, display_rel_change=False, plot_errors=False, bounds=None, trend_only=False):
    step_size="variable"
    bs_lower_limit=-1
    bs_upper_limit=10
    if os.path.isfile(point_data_h5_fp) and not overwrite:
        make_map_from_h5(point_data_h5_fp, display_rel_change, plot_errors, trend_only)
    else:
        if overwrite:
            if os.path.isfile(point_data_h5_fp):
                os.remove(point_data_h5_fp)

        mother_h5s = [h5py.File(hdf5_fp, "r") for hdf5_fp in hdf5_fps]
        
        things = []
    
        dp_index = 0
        for mother_h5 in mother_h5s:
            #print(mother_h5)
            try:
                for key in mother_h5.keys():
                    if not dp_index%50:
                        print(mother_h5)
                        print(dp_index)
        
                    
                    frac_dates = mother_h5[key]["fracture"]["dates"][()]
                    frac_dates = [f_date.decode("utf-8") for f_date in frac_dates]
                    frac_fds = mother_h5[key]["fracture"]["fd"][()]
            
                    bs_dates = mother_h5[key]["backscatter"]["dates"][()]
            #        bs_dates = [b_date.decode("utf-8") for b_date in bs_dates]
                    bs_sds = mother_h5[key]["backscatter"]["sd"][()] #mistake!
            
                    point_coords = mother_h5[key]["coords"][()]
                    
                    if bounds is None or check_coord(bounds, point_coords):
        
                        try:
                            if trend_only:
                                lb, ub, mf, _, _, _ = create_rfd_plot_for_point_trend_only(bs_sds, bs_dates,
                                                                 frac_fds, frac_dates,
                                                                 step_size,
                                                                 bs_lower_limit, bs_upper_limit,
                                                                 monthly_medians=True, 
                                                                 use_smoothed_sd_timeseries=True,
                                                                 make_plots_along_the_way=False)
#                                if np.isnan(mf):
#                                    print(point_coords)
#                                print(lb, ub, mf)
                            else:
                                #NOTE
                                lb, ub, mf, _, _, _ = create_rfd_plot_for_point_trend_only(bs_sds, bs_dates,
                                                                 frac_fds, frac_dates,
                                                                 step_size,
                                                                 bs_lower_limit, bs_upper_limit,
                                                                 monthly_medians=True,
                                                                 use_smoothed_sd_timeseries=True,
                                                                 make_plots_along_the_way=False)

                            total_mean_fd = get_mean_fd(frac_fds)
         
                            things.append([point_coords, lb, ub, mf, total_mean_fd])
                        except Exception as excp:
                            print(excp)
                            print(traceback.format_exc())
                            things.append([point_coords, np.nan, np.nan, np.nan, np.nan])
                   
                    dp_index += 1
        #            if dp_index >= 10000:
        #                break
            except RuntimeError as RE:
                print(traceback.format_exc())
                print(RE)
#                pass
        
        map(close_h5, mother_h5s)

        save_things_as_hdf5_file(things, point_data_h5_fp)

        make_map_from_h5(point_data_h5_fp, display_rel_change, plot_errors, trend_only)

#h5_dir = "/nfs/b0133/eetss/damage_mapping/fd_timeseries/outputs/ase_sv_map/real_h5_files/"
#h5_dir = "/nfs/b0133/eetss/damage_mapping/fd_timeseries/outputs/ase_sv_map/hdf5_files_ase_2pt5_5/"
#h5_dir = "/nfs/b0133/eetss/damage_mapping/fd_timeseries/outputs/ais_sv_map/hdf5_files_dml_2pt5_5/"
#h5_dir = "/nfs/b0133/eetss/damage_mapping/fd_timeseries/outputs/ais_sv_map/hdf5_files_shack_2pt5_5/"
h5_dirs = ["/nfs/b0133/eetss/damage_mapping/fd_timeseries/outputs/ais_sv_map/hdf5_files_getz_2pt5_5/", "/nfs/b0133/eetss/damage_mapping/fd_timeseries/outputs/ais_sv_map/hdf5_files_larsens_2pt5_5/"]

#data_stub = "ais_masked_step01_monthly_medians_all_fds_above_error"
data_stub = "wais_2pt5_5_variable_step_monthly_medians_TRENDONLY_new"
outmap_fp = "/nfs/b0133/eetss/damage_mapping/fd_timeseries/outputs/ais_space/{}_nomult.png".format(data_stub)
outtiff_fp = "/nfs/b0133/eetss/damage_mapping/fd_timeseries/outputs/ais_space/{}_nomult.tiff".format(data_stub)


h5_filepaths = []
for h5_dir in h5_dirs:
    h5_filepaths += [str(pth) for pth in Path(h5_dir).rglob("*.hdf5")]

#point_data_h5_fp = "/nfs/b0133/eetss/damage_mapping/fd_timeseries/outputs/ase_space/point_data_ase_masked_step01_monthly_medians.hdf5"
point_data_h5_fp = "/nfs/b0133/eetss/damage_mapping/fd_timeseries/outputs/ais_space/point_data_{}.hdf5".format(data_stub)

bounds = None
#PIG:
#bounds = [-1637212, -251512, -1555614, -347729]
#TG:
#bounds = [-1617012, -405406, -1504582, -493382]
#Crosson:
#bounds = [-1588838, -525277, -1480129, -642491]
#Dotson End:
#bounds = [-1607975, -634518, -1562259, -694586]
#Brunt:
#bounds = [-816521, 1634168, -395746, 1354704]
#Larsens:
#bounds = []

execute(h5_filepaths, point_data_h5_fp, overwrite=False, display_rel_change=False, plot_errors=False, bounds=bounds, trend_only=True)



