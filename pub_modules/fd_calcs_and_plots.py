#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from datetime import datetime, timedelta


def make_all_date_arrays(bs_data, bs_dates, frac_data, frac_dates, monthly_medians):
    all_dates = return_all_dates_15()

    bs_data = np.array(bs_data)
    bs_data = np.where(bs_data!=0, bs_data, np.nan)
    #bs_dates = bs_dates[np.isfinite(bs_data)]

    frac_data = np.array(frac_data)
    frac_data = np.where(frac_data!=0, frac_data, np.nan)
    #frac_dates = frac_dates[np.isfinite(frac_data)]
    
    #assumes bs_dates and frac_dates are in order
    bs_date_all_date_indices = np.array([i for i in range(len(all_dates)) if all_dates[i] in bs_dates])
    frac_date_all_date_indices = np.array([i for i in range(len(all_dates)) if all_dates[i] in frac_dates])

    bs_data_large = np.zeros_like(all_dates)*np.nan
    bs_data_large[bs_date_all_date_indices] = np.array(bs_data)

    frac_data_large = np.zeros_like(all_dates)*np.nan
    frac_data_large[frac_date_all_date_indices] = np.array(frac_data)

    all_dates = np.array(all_dates)
    
    monthdays = all_dates%10000
    bs_data_large = np.where((monthdays>401) & (monthdays<1101), bs_data_large, np.nan)
    frac_data_large = np.where((monthdays>401) & (monthdays<1101), frac_data_large, np.nan)


    if monthly_medians:
        #median_len defines the window over which median is taken (despite the name `monthly')
        median_len = 10
        #median_len = 15
        #NOTE: change from 15 to 10 between the large map and the teis individual point!
       
        ###NOTE: HACK. I've just set median len to 15 days as I know this to be a factor of the number of dates we have!
        ##want to reshape and reduce the arrays, so have to find
        ## how to make them multiple of mendian_len in length:
        #remainder = len(all_dates)%median_len
        #if remainder>0:
        #    all_dates.append(all_dates[-1]+)

        frac_data_large = np.reshape(frac_data_large, (-1, median_len))
        frac_data_large = np.nanmedian(frac_data_large, axis=1)

        bs_data_large = np.reshape(bs_data_large, (-1, median_len))
        bs_data_large = np.nanmedian(bs_data_large, axis=1)

        all_dates = np.reshape(all_dates, (-1, median_len))
        all_dates = np.nanmin(all_dates, axis=1)


    return bs_data_large, frac_data_large, np.array(all_dates)

    

def create_rfd_plot_for_point_trend_only(bs_data, bs_dates, frac_fds, frac_dates, 
        step_size, bs_lower_limit, bs_upper_limit, monthly_medians=False, 
        use_smoothed_sd_timeseries=True, make_plots_along_the_way=False):
   

    #Generally, best to xform dates from lists of strs to lists of ints:
    #Some of the older files save the dates as byte-strings.
    #So, just in case:
    if (type(bs_dates[0])==type(b'')):
        bs_dates = [d.decode("utf-8") for d in bs_dates]
    if (type(frac_dates[0])==type(b'')):
        frac_dates = [d.decode("utf-8") for d in frac_dates]
    bs_dates = np.array([int(d) for d in bs_dates])
    frac_dates = np.array([int(d) for d in frac_dates])


    bs_data, frac_data, all_dates = make_all_date_arrays(bs_data, bs_dates, frac_fds, frac_dates, monthly_medians)

    if use_smoothed_sd_timeseries:
        smoothed_bs_data = smooth_and_interpolate_sds(bs_data)

        similar_image_index_superlist = get_indices_for_similar_imgs(smoothed_bs_data, step_size, bs_lower_limit, bs_upper_limit)
    else:
        similar_image_index_superlist = get_indices_for_similar_imgs(bs_data, step_size, bs_lower_limit, bs_upper_limit)

    
    frac_ts_and_coord_superlist = []
    for idx_array in similar_image_index_superlist:
        frac_ts = frac_data[idx_array]
        frac_ts_mod = frac_ts[np.isfinite(frac_ts)]
        idx_array_mod = idx_array[np.isfinite(frac_ts)]

        frac_ts_and_coord_superlist.append([frac_ts_mod, idx_array_mod])
    
    #print(frac_ts_and_coord_superlist)

    int_coeffs = []
    slope_coeffs = []
    #plot_fds = np.zeros(len(frac_ts_superlist[0]))*np.nan
    sc_errors = []
    subplot_fits = []
    subplot_fracture_datas = []
    subplot_fracture_coords = []
    
    # prev_max = 0
    prev_min = 1e10

    if monthly_medians:
        min_num_finite_vals = 3
        #NOTE
        min_time_datapoints_elapsed = 24
    else:
        min_num_finite_vals = 11
        min_time_datapoints_elapsed = 730

    all_coords = np.arange(len(all_dates))
    for fracture_ts, fit_coords in frac_ts_and_coord_superlist:
        num_finite_vals = np.count_nonzero(~np.isnan(fracture_ts))
        if num_finite_vals>min_num_finite_vals:
            
            #NOTE: need to fix this so that the indices are the dates from 2015 to 2022!!!

            if ((fit_coords[-1]-fit_coords[0]) > min_time_datapoints_elapsed):

                coeffs, vs = np.polyfit(fit_coords, fracture_ts, deg=1, cov=True)
                print(coeffs) 
                errors = [np.sqrt(vs[0,0]), np.sqrt(vs[1,1])]
                
                slope_error = errors[0]
                int_error = errors[1]

                fit_terms = list(map(lambda i: coeffs[i]*(all_coords**(1-i)), list(range(2))))
                fit = sum(fit_terms)
                
                subplot_fits.append(fit)

                subplot_fracture_datas.append(fracture_ts)
                subplot_fracture_coords.append(fit_coords)

                slope_coeffs.append(coeffs[0])
                int_coeffs.append(coeffs[1])    

                sc_errors.append(slope_error)

    all_frac_data_coords = np.argwhere(np.isfinite(frac_data))[:,0]
    frac_data_where_finite = frac_data[all_frac_data_coords]
    all_fit_coeffs = np.polyfit(all_frac_data_coords, frac_data_where_finite, deg=1, cov=False)
    all_fit = sum([all_fit_coeffs[i]*(all_coords**(1-i)) for i in  list(range(2))])
    print(all_fit_coeffs)
   

    if len(slope_coeffs)==0:
        print("No trends!")
        return np.nan, np.nan, np.nan, None, None, None

    if make_plots_along_the_way:
        if monthly_medians:
            fig1 = plt.figure(figsize=(10,8))
            num_subplots = len(subplot_fits)
            for l in range(num_subplots):
                rgb = plt.cm.get_cmap("ocean")(l/num_subplots)

                f_ = subplot_fits[l]
                fracs_ = subplot_fracture_datas[l]
                frac_coords = subplot_fracture_coords[l]

                plt.scatter(frac_coords, fracs_, c=tuple(rgb))
                plt.plot(f_, label="{:.2e}".format(sc_errors[l]), c=tuple(rgb))
            
            plt.plot(all_fit, '--', c='k')

            ca = plt.gca()
            ca.autoscale(enable=True, axis='x', tight=True)
            ca.axes.get_xaxis().set_ticks([])
           # ca.spines['top'].set_visible(False)
           # ca.spines['right'].set_visible(False)
            plt.ylim((0,0.35))
            plt.legend(loc=2)
            plt.title("data aliased by same sd, with fit")
            plt.savefig("/nfs/b0133/eetss/damage_mapping/fd_timeseries/outputs/teis_point_manyfits.png", dpi=200)
     
        else:
            fig1 = plt.figure(figsize=(10,8))
            for l in range(len(subplot_fits)):
                f_ = subplot_fits[l]
                fracs_ = subplot_fracture_datas[l]
                
                plt.scatter(frac_dates, fracs_)
                plt.plot(f_, label="{}".format(sc_errors[l]))
                
            ca = plt.gca()
            ca.axes.get_xaxis().set_ticks([])
            #plt.ylim((0,0.2))
            plt.legend()
            plt.title("data aliased by same sd, with fit")
       

        fig0 = plt.figure(figsize=(10,10))
        #plt.scatter(range(len(all_dates)), bs_data)
        plt.plot(range(len(all_dates)), smoothed_bs_data)
        ca = plt.gca()
        ca.axes.get_xaxis().set_ticks([])
   

    scs = np.array(slope_coeffs)
    dscs = np.array(sc_errors)
    ics = np.array(int_coeffs)

    ####For now, let's just combine the coeffs in an unweighted way... Could weight them, or even do an IQR of them!
    # mean_slope_coeff = np.mean(scs)
    # sd_slope_coeff = np.std(scs)
    
    
    weights = (1/dscs**2)
    # weights = np.ones_like(dscs**2)

    mean_slope_coeff = np.average(scs, weights=weights)
    #mean_slope_coeff = np.mean(scs)
    #print(mean_slope_coeff)
    
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
        ###-----ne-naww---ne-naww------
        #####TAKE NOTE!!! HACK HACK HACK. Orginially, we did a max
        #sd_slope_coeff = max(sd_slope_coeff_wsd, sd_slope_coeff_sem)
        sd_slope_coeff = min(sd_slope_coeff_wsd, sd_slope_coeff_sem)
    elif not np.isnan(sd_slope_coeff_sem):
        sd_slope_coeff = sd_slope_coeff_sem
    elif not np.isnan(sd_slope_coeff_wsd):
        sd_slope_coeff = sd_slope_coeff_wsd
    else:
        sd_slope_coeff = np.nan
    
    lbound = mean_slope_coeff-1.96*sd_slope_coeff
    ubound = mean_slope_coeff+1.96*sd_slope_coeff

    if make_plots_along_the_way:
        fig2 = plt.figure(figsize=(10,8))

        rgb = plt.cm.get_cmap("ocean")
        line_rgb = rgb(0.4)
        error_rgb = rgb(0.7)
        #line_rgb[:3] = line_rgb[:3]*0.5
        #line_rgb = tuple(line_rgb)

        plt.plot(np.nanmean(ics)+all_coords*mean_slope_coeff, linewidth=4, c=tuple(line_rgb))
        plt.fill_between(all_coords, np.nanmean(ics)+all_coords*lbound, np.nanmean(ics)+all_coords*ubound, alpha=0.8, color=tuple(error_rgb))

        #plt.scatter(coords, plot_fds, c="black", s=10)

        ca = plt.gca()
        ca.axes.get_xaxis().set_ticks([])
        plt.ylim((0,0.35))
        ca.autoscale(enable=True, axis='x', tight=True)
       # ca.spines['top'].set_visible(False)
       # ca.spines['right'].set_visible(False)

        plt.plot(all_fit, '--', c='k')

        plt.title("distribution of fits")
        plt.savefig("/nfs/b0133/eetss/damage_mapping/fd_timeseries/outputs/teis_point_trend_estimate.png", dpi=200)

    else:
        fig1 = None
        fig2 = None
        fig0 = None
 
    timez = len(all_coords)
    return lbound*timez, ubound*timez, mean_slope_coeff*timez, fig1, fig2, fig0


def smooth_and_interpolate_sds(sds):
    #sd_data = sds[(np.isfinite(sds))*(sds!=0)]
    sd_data = sds
    sd_coords = np.arange(len(sd_data))

    smoothed_data = lowess(sd_data, sd_coords, is_sorted=True, missing="drop", frac=0.05, it=0)

    smoothed_sds = smoothed_data[:,1]
    smoothed_sd_coords = smoothed_data[:,0]

    interpolated_smoothed_sds = np.interp(sd_coords, smoothed_sd_coords, smoothed_sds)
   
    return interpolated_smoothed_sds
    

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

def my_round(x, base):
    return base * round(x/base)


def get_indices_for_similar_imgs(bs_data, step_size=0.2, bs_lower_limit=-1, bs_upper_limit=10):
    #just make sure nan data will not contribute:
    bs_data[~np.isfinite(bs_data)]=-9999

    if step_size=="variable":
        q75, q25 = np.nanpercentile(bs_data, [90 ,10])
        iqr = q75 - q25
        #when iqr = 1, we want a step size of around 0.15. So:
        step_size = 0.15*(iqr)
        #step_size = min(0.2*(iqr/1), 0.1)
        step_size = my_round(step_size, 0.01)
        #print(iqr, "!!!!!!!!!!+++++++++++++++++++++++++!!!!!!!!!!!!!!!!!", step_size)
        #if step size is really small (<0.005), ust fit through the whole thing..
        if step_size==0.:
            step_size=22

    superlist = []
    for val_ in np.arange(bs_lower_limit, my_round(bs_upper_limit-bs_lower_limit, step_size)+bs_lower_limit, step_size):
        bs_indices_within_range = np.argwhere((bs_data>=(val_-step_size/2)) & (bs_data<=(val_+step_size/2)))
        superlist.append(bs_indices_within_range)

    return superlist


def get_dates_for_similar_imgs(bs_data, dates, step_size=0.2, bs_lower_limit=-1, bs_upper_limit=10, monthly_medians=False):
   
    fit_means, fit_indices, viable_means, plot_dates = get_fit_coords_and_data(mean_data, dates, monthly_medians=monthly_medians)
  
    if step_size=="variable":
        q75, q25 = np.nanpercentile(mean_data, [90 ,10])
        iqr = q75 - q25
        #when iqr = 1, we want a step size of around 0.1. So:
        step_size = 0.15*(iqr)
        #step_size = min(0.2*(iqr/1), 0.1)
        step_size = my_round(step_size, 0.01)
        print(iqr, "!!!!!!!!!!+++++++++++++++++++++++++!!!!!!!!!!!!!!!!!", step_size)
        #if step size is really small (<0.005), ust fit through the whole thing..
        if step_size==0.:
            step_size=22
            
            dwbs_superlist = []
            for val_ in np.arange(bs_lower_limit, bs_upper_limit, step_size):
                dwbs = get_dates_within_bounds(val_-(step_size/2), val_+(step_size/2), viable_means, plot_dates)
                dwbs_superlist.append(dwbs)
        else:
            #step_size = max(my_round(step_size, 0.01), 0.01)
            dwbs_superlist = []
            for val_ in np.arange(bs_lower_limit, my_round(bs_upper_limit-bs_lower_limit, step_size)+bs_lower_limit, step_size):
                dwbs = get_dates_within_bounds(val_-(step_size/2), val_+(step_size/2), viable_means, plot_dates)
                dwbs_superlist.append(dwbs)
    
    else:
        dwbs_superlist = []
        for val_ in np.arange(bs_lower_limit, my_round(bs_upper_limit-bs_lower_limit, step_size)+bs_lower_limit, step_size):
            dwbs = get_dates_within_bounds(val_-(step_size/2), val_+(step_size/2), viable_means, plot_dates)
            dwbs_superlist.append(dwbs)
    
    return dwbs_superlist


def get_fds_for_each_hline_ts(dwbs_superlist, frac_fds, frac_dates, monthly_medians=False):
    frac_wbs_ts_list = []

    if monthly_medians:
        ym_ints = []
        ym_fd_array_list = []
        ym_fd_list = []

        i = 0
        for date in frac_dates:
            md_int = int(date[-4:])
            y_int = int(date[:4])
            ym_int = int(date[:6])

            if ym_int not in ym_ints:
                if i>0:
                    ym_fd_array_list.append(np.array(ym_fd_list))
                ym_ints.append(ym_int)
                ym_fd_list = []

            fd = frac_fds[i]

            if ((np.isfinite(fd)) and (fd!=0) and (md_int > 301) and (md_int < 1201)):# and (y_int<=2019) and (y_int!=2015)):# and (y_int != 2019) and (y_int != 2020)):
                ym_fd_list.append(fd)

            else:
                ym_fd_list.append(np.nan)

            i+=1
        ym_fd_array_list.append(np.array(ym_fd_list))

        ym_fds = [np.nanmedian(fd_ar) for fd_ar in ym_fd_array_list]
        ym_dates = [str(ym_int)+"01" for ym_int in ym_ints]

#        print(ym_fds)
#        print(ym_dates)
#        print(dwbs_superlist)

        for dwbs_list in dwbs_superlist:
            frac_wbs_ts_list.append(get_fds_for_date_list(dwbs_list, ym_fds, ym_dates))

    else:
        for dwbs_list in dwbs_superlist:
            frac_wbs_ts_list.append(get_fds_for_date_list(dwbs_list, frac_fds, frac_dates))
        ym_dates = None

    return frac_wbs_ts_list, ym_dates



def get_fit_coords_and_data(means, dates, monthly_medians=False):
    viable_means = []
    fit_means = []
    fit_indices = []
    data_coords = []

    if monthly_medians:
        ym_ints = []
        ym_mean_array_list = []
        ym_mean_list = []

        i = 0
        if (type(dates[0])==type(b'')):
            dates = [d.decode("utf-8") for d in dates]
        for date in dates:
            md_int = int(date[-4:])
            y_int = int(date[:4])
            ym_int = int(date[:6])

            if ym_int not in ym_ints:
                if i>0:
                    ym_mean_array_list.append(np.array(ym_mean_list))
                ym_ints.append(ym_int)
                ym_mean_list = []

            mean = means[i]

            if ((np.isfinite(mean)) and (mean!=0) and (md_int > 301) and (md_int < 1201)):# and (y_int<=2019) and (y_int!=2015)):# and (y_int != 2019) and (y_int != 2020)):
                ym_mean_list.append(mean)

            else:
                ym_mean_list.append(np.nan)
                #viable_means[i] = np.nan

            i+=1
        ym_mean_array_list.append(np.array(ym_mean_list))

        # print(ym_ints, ym_mean_list_list)

        ym_means = [np.nanmedian(mean_ar) for mean_ar in ym_mean_array_list]

        j = 0
        # print(ym_ints, ym_means)
        for ym_int in ym_ints:
            # coords.append(j)
            ym_mean = ym_means[j]
            if np.isfinite(ym_mean):
                fit_indices.append(j)
                fit_means.append(ym_mean)
                viable_means.append(ym_mean)
                data_coords.append(i)
            else:
                viable_means.append(np.nan)
            j += 1

        viable_means = list(viable_means)

        plot_dates = [str(ym_int)+"01" for ym_int in ym_ints]
    
    else:
        i = 0
        if (type(dates[0])==type(b'')):
            dates = [d.decode("utf-8") for d in dates]
        for date in dates:
            md_int = int(date[-4:])
            y_int = int(date[:4])
            mean = means[i]
            if ((np.isfinite(mean)) and (mean!=0) and (md_int > 301) and (md_int < 1101)):# and (y_int != 2020)):# and (y_int != 2020)):
                viable_means.append(mean)
                fit_means.append(mean)
                fit_indices.append(i)
                data_coords.append(i)
            else:
                viable_means.append(np.nan)
            i+=1
        
        plot_dates = dates
        
    fit_means = np.array(fit_means)
    fit_indices = np.array(fit_indices)
    
#    print(fit_means)
#    print(fit_indices)
#    print(viable_means)
#    print(plot_dates)

    return fit_means, fit_indices, viable_means, plot_dates



def get_dates_within_bounds(lb, ub, viable_means, dates):
    dates_within_bounds = []
    i = 0
    for mn in viable_means:
        if ((np.isfinite(mn)) and (mn<=ub) and (mn>=lb)):
            dates_within_bounds.append(dates[i])
        i+=1

    return dates_within_bounds



def get_fds_for_date_list(date_list, frac_fds, frac_dates):
    fds_within_bs_bounds = []

    j = 0
    for frac_date in frac_dates:
        if frac_date in date_list:
            ffd = frac_fds[j]
            if np.isfinite(ffd) and (ffd>0):
                fds_within_bs_bounds.append(ffd)
            else:
                fds_within_bs_bounds.append(np.nan)
        else:
            fds_within_bs_bounds.append(np.nan)

        j+=1

    return fds_within_bs_bounds


def return_all_dates_15():
    return [20150101, 20150102, 20150103, 20150104, 20150105, 20150106, 20150107, 20150108, 20150109, 20150110, 20150111, 20150112, 20150113, 20150114, 20150115, 20150116, 20150117, 20150118, 20150119, 20150120, 20150121, 20150122, 20150123, 20150124, 20150125, 20150126, 20150127, 20150128, 20150129, 20150130, 20150131, 20150201, 20150202, 20150203, 20150204, 20150205, 20150206, 20150207, 20150208, 20150209, 20150210, 20150211, 20150212, 20150213, 20150214, 20150215, 20150216, 20150217, 20150218, 20150219, 20150220, 20150221, 20150222, 20150223, 20150224, 20150225, 20150226, 20150227, 20150228, 20150301, 20150302, 20150303, 20150304, 20150305, 20150306, 20150307, 20150308, 20150309, 20150310, 20150311, 20150312, 20150313, 20150314, 20150315, 20150316, 20150317, 20150318, 20150319, 20150320, 20150321, 20150322, 20150323, 20150324, 20150325, 20150326, 20150327, 20150328, 20150329, 20150330, 20150331, 20150401, 20150402, 20150403, 20150404, 20150405, 20150406, 20150407, 20150408, 20150409, 20150410, 20150411, 20150412, 20150413, 20150414, 20150415, 20150416, 20150417, 20150418, 20150419, 20150420, 20150421, 20150422, 20150423, 20150424, 20150425, 20150426, 20150427, 20150428, 20150429, 20150430, 20150501, 20150502, 20150503, 20150504, 20150505, 20150506, 20150507, 20150508, 20150509, 20150510, 20150511, 20150512, 20150513, 20150514, 20150515, 20150516, 20150517, 20150518, 20150519, 20150520, 20150521, 20150522, 20150523, 20150524, 20150525, 20150526, 20150527, 20150528, 20150529, 20150530, 20150531, 20150601, 20150602, 20150603, 20150604, 20150605, 20150606, 20150607, 20150608, 20150609, 20150610, 20150611, 20150612, 20150613, 20150614, 20150615, 20150616, 20150617, 20150618, 20150619, 20150620, 20150621, 20150622, 20150623, 20150624, 20150625, 20150626, 20150627, 20150628, 20150629, 20150630, 20150701, 20150702, 20150703, 20150704, 20150705, 20150706, 20150707, 20150708, 20150709, 20150710, 20150711, 20150712, 20150713, 20150714, 20150715, 20150716, 20150717, 20150718, 20150719, 20150720, 20150721, 20150722, 20150723, 20150724, 20150725, 20150726, 20150727, 20150728, 20150729, 20150730, 20150731, 20150801, 20150802, 20150803, 20150804, 20150805, 20150806, 20150807, 20150808, 20150809, 20150810, 20150811, 20150812, 20150813, 20150814, 20150815, 20150816, 20150817, 20150818, 20150819, 20150820, 20150821, 20150822, 20150823, 20150824, 20150825, 20150826, 20150827, 20150828, 20150829, 20150830, 20150831, 20150901, 20150902, 20150903, 20150904, 20150905, 20150906, 20150907, 20150908, 20150909, 20150910, 20150911, 20150912, 20150913, 20150914, 20150915, 20150916, 20150917, 20150918, 20150919, 20150920, 20150921, 20150922, 20150923, 20150924, 20150925, 20150926, 20150927, 20150928, 20150929, 20150930, 20151001, 20151002, 20151003, 20151004, 20151005, 20151006, 20151007, 20151008, 20151009, 20151010, 20151011, 20151012, 20151013, 20151014, 20151015, 20151016, 20151017, 20151018, 20151019, 20151020, 20151021, 20151022, 20151023, 20151024, 20151025, 20151026, 20151027, 20151028, 20151029, 20151030, 20151031, 20151101, 20151102, 20151103, 20151104, 20151105, 20151106, 20151107, 20151108, 20151109, 20151110, 20151111, 20151112, 20151113, 20151114, 20151115, 20151116, 20151117, 20151118, 20151119, 20151120, 20151121, 20151122, 20151123, 20151124, 20151125, 20151126, 20151127, 20151128, 20151129, 20151130, 20151201, 20151202, 20151203, 20151204, 20151205, 20151206, 20151207, 20151208, 20151209, 20151210, 20151211, 20151212, 20151213, 20151214, 20151215, 20151216, 20151217, 20151218, 20151219, 20151220, 20151221, 20151222, 20151223, 20151224, 20151225, 20151226, 20151227, 20151228, 20151229, 20151230, 20151231, 20160101, 20160102, 20160103, 20160104, 20160105, 20160106, 20160107, 20160108, 20160109, 20160110, 20160111, 20160112, 20160113, 20160114, 20160115, 20160116, 20160117, 20160118, 20160119, 20160120, 20160121, 20160122, 20160123, 20160124, 20160125, 20160126, 20160127, 20160128, 20160129, 20160130, 20160131, 20160201, 20160202, 20160203, 20160204, 20160205, 20160206, 20160207, 20160208, 20160209, 20160210, 20160211, 20160212, 20160213, 20160214, 20160215, 20160216, 20160217, 20160218, 20160219, 20160220, 20160221, 20160222, 20160223, 20160224, 20160225, 20160226, 20160227, 20160228, 20160229, 20160301, 20160302, 20160303, 20160304, 20160305, 20160306, 20160307, 20160308, 20160309, 20160310, 20160311, 20160312, 20160313, 20160314, 20160315, 20160316, 20160317, 20160318, 20160319, 20160320, 20160321, 20160322, 20160323, 20160324, 20160325, 20160326, 20160327, 20160328, 20160329, 20160330, 20160331, 20160401, 20160402, 20160403, 20160404, 20160405, 20160406, 20160407, 20160408, 20160409, 20160410, 20160411, 20160412, 20160413, 20160414, 20160415, 20160416, 20160417, 20160418, 20160419, 20160420, 20160421, 20160422, 20160423, 20160424, 20160425, 20160426, 20160427, 20160428, 20160429, 20160430, 20160501, 20160502, 20160503, 20160504, 20160505, 20160506, 20160507, 20160508, 20160509, 20160510, 20160511, 20160512, 20160513, 20160514, 20160515, 20160516, 20160517, 20160518, 20160519, 20160520, 20160521, 20160522, 20160523, 20160524, 20160525, 20160526, 20160527, 20160528, 20160529, 20160530, 20160531, 20160601, 20160602, 20160603, 20160604, 20160605, 20160606, 20160607, 20160608, 20160609, 20160610, 20160611, 20160612, 20160613, 20160614, 20160615, 20160616, 20160617, 20160618, 20160619, 20160620, 20160621, 20160622, 20160623, 20160624, 20160625, 20160626, 20160627, 20160628, 20160629, 20160630, 20160701, 20160702, 20160703, 20160704, 20160705, 20160706, 20160707, 20160708, 20160709, 20160710, 20160711, 20160712, 20160713, 20160714, 20160715, 20160716, 20160717, 20160718, 20160719, 20160720, 20160721, 20160722, 20160723, 20160724, 20160725, 20160726, 20160727, 20160728, 20160729, 20160730, 20160731, 20160801, 20160802, 20160803, 20160804, 20160805, 20160806, 20160807, 20160808, 20160809, 20160810, 20160811, 20160812, 20160813, 20160814, 20160815, 20160816, 20160817, 20160818, 20160819, 20160820, 20160821, 20160822, 20160823, 20160824, 20160825, 20160826, 20160827, 20160828, 20160829, 20160830, 20160831, 20160901, 20160902, 20160903, 20160904, 20160905, 20160906, 20160907, 20160908, 20160909, 20160910, 20160911, 20160912, 20160913, 20160914, 20160915, 20160916, 20160917, 20160918, 20160919, 20160920, 20160921, 20160922, 20160923, 20160924, 20160925, 20160926, 20160927, 20160928, 20160929, 20160930, 20161001, 20161002, 20161003, 20161004, 20161005, 20161006, 20161007, 20161008, 20161009, 20161010, 20161011, 20161012, 20161013, 20161014, 20161015, 20161016, 20161017, 20161018, 20161019, 20161020, 20161021, 20161022, 20161023, 20161024, 20161025, 20161026, 20161027, 20161028, 20161029, 20161030, 20161031, 20161101, 20161102, 20161103, 20161104, 20161105, 20161106, 20161107, 20161108, 20161109, 20161110, 20161111, 20161112, 20161113, 20161114, 20161115, 20161116, 20161117, 20161118, 20161119, 20161120, 20161121, 20161122, 20161123, 20161124, 20161125, 20161126, 20161127, 20161128, 20161129, 20161130, 20161201, 20161202, 20161203, 20161204, 20161205, 20161206, 20161207, 20161208, 20161209, 20161210, 20161211, 20161212, 20161213, 20161214, 20161215, 20161216, 20161217, 20161218, 20161219, 20161220, 20161221, 20161222, 20161223, 20161224, 20161225, 20161226, 20161227, 20161228, 20161229, 20161230, 20161231, 20170101, 20170102, 20170103, 20170104, 20170105, 20170106, 20170107, 20170108, 20170109, 20170110, 20170111, 20170112, 20170113, 20170114, 20170115, 20170116, 20170117, 20170118, 20170119, 20170120, 20170121, 20170122, 20170123, 20170124, 20170125, 20170126, 20170127, 20170128, 20170129, 20170130, 20170131, 20170201, 20170202, 20170203, 20170204, 20170205, 20170206, 20170207, 20170208, 20170209, 20170210, 20170211, 20170212, 20170213, 20170214, 20170215, 20170216, 20170217, 20170218, 20170219, 20170220, 20170221, 20170222, 20170223, 20170224, 20170225, 20170226, 20170227, 20170228, 20170301, 20170302, 20170303, 20170304, 20170305, 20170306, 20170307, 20170308, 20170309, 20170310, 20170311, 20170312, 20170313, 20170314, 20170315, 20170316, 20170317, 20170318, 20170319, 20170320, 20170321, 20170322, 20170323, 20170324, 20170325, 20170326, 20170327, 20170328, 20170329, 20170330, 20170331, 20170401, 20170402, 20170403, 20170404, 20170405, 20170406, 20170407, 20170408, 20170409, 20170410, 20170411, 20170412, 20170413, 20170414, 20170415, 20170416, 20170417, 20170418, 20170419, 20170420, 20170421, 20170422, 20170423, 20170424, 20170425, 20170426, 20170427, 20170428, 20170429, 20170430, 20170501, 20170502, 20170503, 20170504, 20170505, 20170506, 20170507, 20170508, 20170509, 20170510, 20170511, 20170512, 20170513, 20170514, 20170515, 20170516, 20170517, 20170518, 20170519, 20170520, 20170521, 20170522, 20170523, 20170524, 20170525, 20170526, 20170527, 20170528, 20170529, 20170530, 20170531, 20170601, 20170602, 20170603, 20170604, 20170605, 20170606, 20170607, 20170608, 20170609, 20170610, 20170611, 20170612, 20170613, 20170614, 20170615, 20170616, 20170617, 20170618, 20170619, 20170620, 20170621, 20170622, 20170623, 20170624, 20170625, 20170626, 20170627, 20170628, 20170629, 20170630, 20170701, 20170702, 20170703, 20170704, 20170705, 20170706, 20170707, 20170708, 20170709, 20170710, 20170711, 20170712, 20170713, 20170714, 20170715, 20170716, 20170717, 20170718, 20170719, 20170720, 20170721, 20170722, 20170723, 20170724, 20170725, 20170726, 20170727, 20170728, 20170729, 20170730, 20170731, 20170801, 20170802, 20170803, 20170804, 20170805, 20170806, 20170807, 20170808, 20170809, 20170810, 20170811, 20170812, 20170813, 20170814, 20170815, 20170816, 20170817, 20170818, 20170819, 20170820, 20170821, 20170822, 20170823, 20170824, 20170825, 20170826, 20170827, 20170828, 20170829, 20170830, 20170831, 20170901, 20170902, 20170903, 20170904, 20170905, 20170906, 20170907, 20170908, 20170909, 20170910, 20170911, 20170912, 20170913, 20170914, 20170915, 20170916, 20170917, 20170918, 20170919, 20170920, 20170921, 20170922, 20170923, 20170924, 20170925, 20170926, 20170927, 20170928, 20170929, 20170930, 20171001, 20171002, 20171003, 20171004, 20171005, 20171006, 20171007, 20171008, 20171009, 20171010, 20171011, 20171012, 20171013, 20171014, 20171015, 20171016, 20171017, 20171018, 20171019, 20171020, 20171021, 20171022, 20171023, 20171024, 20171025, 20171026, 20171027, 20171028, 20171029, 20171030, 20171031, 20171101, 20171102, 20171103, 20171104, 20171105, 20171106, 20171107, 20171108, 20171109, 20171110, 20171111, 20171112, 20171113, 20171114, 20171115, 20171116, 20171117, 20171118, 20171119, 20171120, 20171121, 20171122, 20171123, 20171124, 20171125, 20171126, 20171127, 20171128, 20171129, 20171130, 20171201, 20171202, 20171203, 20171204, 20171205, 20171206, 20171207, 20171208, 20171209, 20171210, 20171211, 20171212, 20171213, 20171214, 20171215, 20171216, 20171217, 20171218, 20171219, 20171220, 20171221, 20171222, 20171223, 20171224, 20171225, 20171226, 20171227, 20171228, 20171229, 20171230, 20171231, 20180101, 20180102, 20180103, 20180104, 20180105, 20180106, 20180107, 20180108, 20180109, 20180110, 20180111, 20180112, 20180113, 20180114, 20180115, 20180116, 20180117, 20180118, 20180119, 20180120, 20180121, 20180122, 20180123, 20180124, 20180125, 20180126, 20180127, 20180128, 20180129, 20180130, 20180131, 20180201, 20180202, 20180203, 20180204, 20180205, 20180206, 20180207, 20180208, 20180209, 20180210, 20180211, 20180212, 20180213, 20180214, 20180215, 20180216, 20180217, 20180218, 20180219, 20180220, 20180221, 20180222, 20180223, 20180224, 20180225, 20180226, 20180227, 20180228, 20180301, 20180302, 20180303, 20180304, 20180305, 20180306, 20180307, 20180308, 20180309, 20180310, 20180311, 20180312, 20180313, 20180314, 20180315, 20180316, 20180317, 20180318, 20180319, 20180320, 20180321, 20180322, 20180323, 20180324, 20180325, 20180326, 20180327, 20180328, 20180329, 20180330, 20180331, 20180401, 20180402, 20180403, 20180404, 20180405, 20180406, 20180407, 20180408, 20180409, 20180410, 20180411, 20180412, 20180413, 20180414, 20180415, 20180416, 20180417, 20180418, 20180419, 20180420, 20180421, 20180422, 20180423, 20180424, 20180425, 20180426, 20180427, 20180428, 20180429, 20180430, 20180501, 20180502, 20180503, 20180504, 20180505, 20180506, 20180507, 20180508, 20180509, 20180510, 20180511, 20180512, 20180513, 20180514, 20180515, 20180516, 20180517, 20180518, 20180519, 20180520, 20180521, 20180522, 20180523, 20180524, 20180525, 20180526, 20180527, 20180528, 20180529, 20180530, 20180531, 20180601, 20180602, 20180603, 20180604, 20180605, 20180606, 20180607, 20180608, 20180609, 20180610, 20180611, 20180612, 20180613, 20180614, 20180615, 20180616, 20180617, 20180618, 20180619, 20180620, 20180621, 20180622, 20180623, 20180624, 20180625, 20180626, 20180627, 20180628, 20180629, 20180630, 20180701, 20180702, 20180703, 20180704, 20180705, 20180706, 20180707, 20180708, 20180709, 20180710, 20180711, 20180712, 20180713, 20180714, 20180715, 20180716, 20180717, 20180718, 20180719, 20180720, 20180721, 20180722, 20180723, 20180724, 20180725, 20180726, 20180727, 20180728, 20180729, 20180730, 20180731, 20180801, 20180802, 20180803, 20180804, 20180805, 20180806, 20180807, 20180808, 20180809, 20180810, 20180811, 20180812, 20180813, 20180814, 20180815, 20180816, 20180817, 20180818, 20180819, 20180820, 20180821, 20180822, 20180823, 20180824, 20180825, 20180826, 20180827, 20180828, 20180829, 20180830, 20180831, 20180901, 20180902, 20180903, 20180904, 20180905, 20180906, 20180907, 20180908, 20180909, 20180910, 20180911, 20180912, 20180913, 20180914, 20180915, 20180916, 20180917, 20180918, 20180919, 20180920, 20180921, 20180922, 20180923, 20180924, 20180925, 20180926, 20180927, 20180928, 20180929, 20180930, 20181001, 20181002, 20181003, 20181004, 20181005, 20181006, 20181007, 20181008, 20181009, 20181010, 20181011, 20181012, 20181013, 20181014, 20181015, 20181016, 20181017, 20181018, 20181019, 20181020, 20181021, 20181022, 20181023, 20181024, 20181025, 20181026, 20181027, 20181028, 20181029, 20181030, 20181031, 20181101, 20181102, 20181103, 20181104, 20181105, 20181106, 20181107, 20181108, 20181109, 20181110, 20181111, 20181112, 20181113, 20181114, 20181115, 20181116, 20181117, 20181118, 20181119, 20181120, 20181121, 20181122, 20181123, 20181124, 20181125, 20181126, 20181127, 20181128, 20181129, 20181130, 20181201, 20181202, 20181203, 20181204, 20181205, 20181206, 20181207, 20181208, 20181209, 20181210, 20181211, 20181212, 20181213, 20181214, 20181215, 20181216, 20181217, 20181218, 20181219, 20181220, 20181221, 20181222, 20181223, 20181224, 20181225, 20181226, 20181227, 20181228, 20181229, 20181230, 20181231, 20190101, 20190102, 20190103, 20190104, 20190105, 20190106, 20190107, 20190108, 20190109, 20190110, 20190111, 20190112, 20190113, 20190114, 20190115, 20190116, 20190117, 20190118, 20190119, 20190120, 20190121, 20190122, 20190123, 20190124, 20190125, 20190126, 20190127, 20190128, 20190129, 20190130, 20190131, 20190201, 20190202, 20190203, 20190204, 20190205, 20190206, 20190207, 20190208, 20190209, 20190210, 20190211, 20190212, 20190213, 20190214, 20190215, 20190216, 20190217, 20190218, 20190219, 20190220, 20190221, 20190222, 20190223, 20190224, 20190225, 20190226, 20190227, 20190228, 20190301, 20190302, 20190303, 20190304, 20190305, 20190306, 20190307, 20190308, 20190309, 20190310, 20190311, 20190312, 20190313, 20190314, 20190315, 20190316, 20190317, 20190318, 20190319, 20190320, 20190321, 20190322, 20190323, 20190324, 20190325, 20190326, 20190327, 20190328, 20190329, 20190330, 20190331, 20190401, 20190402, 20190403, 20190404, 20190405, 20190406, 20190407, 20190408, 20190409, 20190410, 20190411, 20190412, 20190413, 20190414, 20190415, 20190416, 20190417, 20190418, 20190419, 20190420, 20190421, 20190422, 20190423, 20190424, 20190425, 20190426, 20190427, 20190428, 20190429, 20190430, 20190501, 20190502, 20190503, 20190504, 20190505, 20190506, 20190507, 20190508, 20190509, 20190510, 20190511, 20190512, 20190513, 20190514, 20190515, 20190516, 20190517, 20190518, 20190519, 20190520, 20190521, 20190522, 20190523, 20190524, 20190525, 20190526, 20190527, 20190528, 20190529, 20190530, 20190531, 20190601, 20190602, 20190603, 20190604, 20190605, 20190606, 20190607, 20190608, 20190609, 20190610, 20190611, 20190612, 20190613, 20190614, 20190615, 20190616, 20190617, 20190618, 20190619, 20190620, 20190621, 20190622, 20190623, 20190624, 20190625, 20190626, 20190627, 20190628, 20190629, 20190630, 20190701, 20190702, 20190703, 20190704, 20190705, 20190706, 20190707, 20190708, 20190709, 20190710, 20190711, 20190712, 20190713, 20190714, 20190715, 20190716, 20190717, 20190718, 20190719, 20190720, 20190721, 20190722, 20190723, 20190724, 20190725, 20190726, 20190727, 20190728, 20190729, 20190730, 20190731, 20190801, 20190802, 20190803, 20190804, 20190805, 20190806, 20190807, 20190808, 20190809, 20190810, 20190811, 20190812, 20190813, 20190814, 20190815, 20190816, 20190817, 20190818, 20190819, 20190820, 20190821, 20190822, 20190823, 20190824, 20190825, 20190826, 20190827, 20190828, 20190829, 20190830, 20190831, 20190901, 20190902, 20190903, 20190904, 20190905, 20190906, 20190907, 20190908, 20190909, 20190910, 20190911, 20190912, 20190913, 20190914, 20190915, 20190916, 20190917, 20190918, 20190919, 20190920, 20190921, 20190922, 20190923, 20190924, 20190925, 20190926, 20190927, 20190928, 20190929, 20190930, 20191001, 20191002, 20191003, 20191004, 20191005, 20191006, 20191007, 20191008, 20191009, 20191010, 20191011, 20191012, 20191013, 20191014, 20191015, 20191016, 20191017, 20191018, 20191019, 20191020, 20191021, 20191022, 20191023, 20191024, 20191025, 20191026, 20191027, 20191028, 20191029, 20191030, 20191031, 20191101, 20191102, 20191103, 20191104, 20191105, 20191106, 20191107, 20191108, 20191109, 20191110, 20191111, 20191112, 20191113, 20191114, 20191115, 20191116, 20191117, 20191118, 20191119, 20191120, 20191121, 20191122, 20191123, 20191124, 20191125, 20191126, 20191127, 20191128, 20191129, 20191130, 20191201, 20191202, 20191203, 20191204, 20191205, 20191206, 20191207, 20191208, 20191209, 20191210, 20191211, 20191212, 20191213, 20191214, 20191215, 20191216, 20191217, 20191218, 20191219, 20191220, 20191221, 20191222, 20191223, 20191224, 20191225, 20191226, 20191227, 20191228, 20191229, 20191230, 20191231, 20200101, 20200102, 20200103, 20200104, 20200105, 20200106, 20200107, 20200108, 20200109, 20200110, 20200111, 20200112, 20200113, 20200114, 20200115, 20200116, 20200117, 20200118, 20200119, 20200120, 20200121, 20200122, 20200123, 20200124, 20200125, 20200126, 20200127, 20200128, 20200129, 20200130, 20200131, 20200201, 20200202, 20200203, 20200204, 20200205, 20200206, 20200207, 20200208, 20200209, 20200210, 20200211, 20200212, 20200213, 20200214, 20200215, 20200216, 20200217, 20200218, 20200219, 20200220, 20200221, 20200222, 20200223, 20200224, 20200225, 20200226, 20200227, 20200228, 20200229, 20200301, 20200302, 20200303, 20200304, 20200305, 20200306, 20200307, 20200308, 20200309, 20200310, 20200311, 20200312, 20200313, 20200314, 20200315, 20200316, 20200317, 20200318, 20200319, 20200320, 20200321, 20200322, 20200323, 20200324, 20200325, 20200326, 20200327, 20200328, 20200329, 20200330, 20200331, 20200401, 20200402, 20200403, 20200404, 20200405, 20200406, 20200407, 20200408, 20200409, 20200410, 20200411, 20200412, 20200413, 20200414, 20200415, 20200416, 20200417, 20200418, 20200419, 20200420, 20200421, 20200422, 20200423, 20200424, 20200425, 20200426, 20200427, 20200428, 20200429, 20200430, 20200501, 20200502, 20200503, 20200504, 20200505, 20200506, 20200507, 20200508, 20200509, 20200510, 20200511, 20200512, 20200513, 20200514, 20200515, 20200516, 20200517, 20200518, 20200519, 20200520, 20200521, 20200522, 20200523, 20200524, 20200525, 20200526, 20200527, 20200528, 20200529, 20200530, 20200531, 20200601, 20200602, 20200603, 20200604, 20200605, 20200606, 20200607, 20200608, 20200609, 20200610, 20200611, 20200612, 20200613, 20200614, 20200615, 20200616, 20200617, 20200618, 20200619, 20200620, 20200621, 20200622, 20200623, 20200624, 20200625, 20200626, 20200627, 20200628, 20200629, 20200630, 20200701, 20200702, 20200703, 20200704, 20200705, 20200706, 20200707, 20200708, 20200709, 20200710, 20200711, 20200712, 20200713, 20200714, 20200715, 20200716, 20200717, 20200718, 20200719, 20200720, 20200721, 20200722, 20200723, 20200724, 20200725, 20200726, 20200727, 20200728, 20200729, 20200730, 20200731, 20200801, 20200802, 20200803, 20200804, 20200805, 20200806, 20200807, 20200808, 20200809, 20200810, 20200811, 20200812, 20200813, 20200814, 20200815, 20200816, 20200817, 20200818, 20200819, 20200820, 20200821, 20200822, 20200823, 20200824, 20200825, 20200826, 20200827, 20200828, 20200829, 20200830, 20200831, 20200901, 20200902, 20200903, 20200904, 20200905, 20200906, 20200907, 20200908, 20200909, 20200910, 20200911, 20200912, 20200913, 20200914, 20200915, 20200916, 20200917, 20200918, 20200919, 20200920, 20200921, 20200922, 20200923, 20200924, 20200925, 20200926, 20200927, 20200928, 20200929, 20200930, 20201001, 20201002, 20201003, 20201004, 20201005, 20201006, 20201007, 20201008, 20201009, 20201010, 20201011, 20201012, 20201013, 20201014, 20201015, 20201016, 20201017, 20201018, 20201019, 20201020, 20201021, 20201022, 20201023, 20201024, 20201025, 20201026, 20201027, 20201028, 20201029, 20201030, 20201031, 20201101, 20201102, 20201103, 20201104, 20201105, 20201106, 20201107, 20201108, 20201109, 20201110, 20201111, 20201112, 20201113, 20201114, 20201115, 20201116, 20201117, 20201118, 20201119, 20201120, 20201121, 20201122, 20201123, 20201124, 20201125, 20201126, 20201127, 20201128, 20201129, 20201130, 20201201, 20201202, 20201203, 20201204, 20201205, 20201206, 20201207, 20201208, 20201209, 20201210, 20201211, 20201212, 20201213, 20201214, 20201215, 20201216, 20201217, 20201218, 20201219, 20201220, 20201221, 20201222, 20201223, 20201224, 20201225, 20201226, 20201227, 20201228, 20201229, 20201230, 20201231, 20210101, 20210102, 20210103, 20210104, 20210105, 20210106, 20210107, 20210108, 20210109, 20210110, 20210111, 20210112, 20210113, 20210114, 20210115, 20210116, 20210117, 20210118, 20210119, 20210120, 20210121, 20210122, 20210123, 20210124, 20210125, 20210126, 20210127, 20210128, 20210129, 20210130, 20210131, 20210201, 20210202, 20210203, 20210204, 20210205, 20210206, 20210207, 20210208, 20210209, 20210210, 20210211, 20210212, 20210213, 20210214, 20210215, 20210216, 20210217, 20210218, 20210219, 20210220, 20210221, 20210222, 20210223, 20210224, 20210225, 20210226, 20210227, 20210228, 20210301, 20210302, 20210303, 20210304, 20210305, 20210306, 20210307, 20210308, 20210309, 20210310, 20210311, 20210312, 20210313, 20210314, 20210315, 20210316, 20210317, 20210318, 20210319, 20210320, 20210321, 20210322, 20210323, 20210324, 20210325, 20210326, 20210327, 20210328, 20210329, 20210330, 20210331, 20210401, 20210402, 20210403, 20210404, 20210405, 20210406, 20210407, 20210408, 20210409, 20210410, 20210411, 20210412, 20210413, 20210414, 20210415, 20210416, 20210417, 20210418, 20210419, 20210420, 20210421, 20210422, 20210423, 20210424, 20210425, 20210426, 20210427, 20210428, 20210429, 20210430, 20210501, 20210502, 20210503, 20210504, 20210505, 20210506, 20210507, 20210508, 20210509, 20210510, 20210511, 20210512, 20210513, 20210514, 20210515, 20210516, 20210517, 20210518, 20210519, 20210520, 20210521, 20210522, 20210523, 20210524, 20210525, 20210526, 20210527, 20210528, 20210529, 20210530, 20210531, 20210601, 20210602, 20210603, 20210604, 20210605, 20210606, 20210607, 20210608, 20210609, 20210610, 20210611, 20210612, 20210613, 20210614, 20210615, 20210616, 20210617, 20210618, 20210619, 20210620, 20210621, 20210622, 20210623, 20210624, 20210625, 20210626, 20210627, 20210628, 20210629, 20210630, 20210701, 20210702, 20210703, 20210704, 20210705, 20210706, 20210707, 20210708, 20210709, 20210710, 20210711, 20210712, 20210713, 20210714, 20210715, 20210716, 20210717, 20210718, 20210719, 20210720, 20210721, 20210722, 20210723, 20210724, 20210725, 20210726, 20210727, 20210728, 20210729, 20210730, 20210731, 20210801, 20210802, 20210803, 20210804, 20210805, 20210806, 20210807, 20210808, 20210809, 20210810, 20210811, 20210812, 20210813, 20210814, 20210815, 20210816, 20210817, 20210818, 20210819, 20210820, 20210821, 20210822, 20210823, 20210824, 20210825, 20210826, 20210827, 20210828, 20210829, 20210830, 20210831, 20210901, 20210902, 20210903, 20210904, 20210905, 20210906, 20210907, 20210908, 20210909, 20210910, 20210911, 20210912, 20210913, 20210914, 20210915, 20210916, 20210917, 20210918, 20210919, 20210920, 20210921, 20210922, 20210923, 20210924, 20210925, 20210926, 20210927, 20210928, 20210929, 20210930, 20211001, 20211002, 20211003, 20211004, 20211005, 20211006, 20211007, 20211008, 20211009, 20211010, 20211011, 20211012, 20211013, 20211014, 20211015, 20211016, 20211017, 20211018, 20211019, 20211020, 20211021, 20211022, 20211023, 20211024, 20211025, 20211026, 20211027, 20211028, 20211029, 20211030, 20211031, 20211101, 20211102, 20211103, 20211104, 20211105, 20211106, 20211107, 20211108, 20211109, 20211110, 20211111, 20211112, 20211113, 20211114, 20211115, 20211116, 20211117, 20211118, 20211119, 20211120, 20211121, 20211122, 20211123, 20211124, 20211125, 20211126, 20211127, 20211128, 20211129, 20211130, 20211201, 20211202, 20211203, 20211204, 20211205, 20211206, 20211207, 20211208, 20211209, 20211210, 20211211, 20211212, 20211213, 20211214, 20211215, 20211216, 20211217, 20211218, 20211219, 20211220, 20211221, 20211222, 20211223, 20211224, 20211225, 20211226, 20211227, 20211228, 20211229, 20211230, 20211231, 20220101, 20220102, 20220103, 20220104, 20220105, 20220106, 20220107, 20220108, 20220109, 20220110, 20220111, 20220112, 20220113, 20220114, 20220115, 20220116, 20220117, 20220118, 20220119, 20220120, 20220121, 20220122, 20220123, 20220124, 20220125, 20220126, 20220127, 20220128, 20220129, 20220130, 20220131, 20220201, 20220202, 20220203, 20220204, 20220205, 20220206, 20220207, 20220208, 20220209, 20220210, 20220211, 20220212, 20220213, 20220214, 20220215, 20220216, 20220217, 20220218, 20220219, 20220220, 20220221, 20220222, 20220223, 20220224, 20220225, 20220226, 20220227, 20220228, 20220301, 20220302, 20220303, 20220304, 20220305, 20220306, 20220307, 20220308, 20220309, 20220310, 20220311, 20220312, 20220313, 20220314, 20220315, 20220316, 20220317, 20220318, 20220319, 20220320, 20220321, 20220322, 20220323, 20220324, 20220325, 20220326, 20220327, 20220328, 20220329, 20220330, 20220331, 20220401, 20220402, 20220403, 20220404, 20220405, 20220406, 20220407, 20220408, 20220409, 20220410, 20220411, 20220412, 20220413, 20220414, 20220415, 20220416, 20220417, 20220418, 20220419, 20220420, 20220421, 20220422, 20220423, 20220424, 20220425, 20220426, 20220427, 20220428, 20220429, 20220430, 20220501, 20220502, 20220503, 20220504, 20220505, 20220506, 20220507, 20220508, 20220509, 20220510, 20220511, 20220512, 20220513, 20220514, 20220515, 20220516, 20220517, 20220518, 20220519, 20220520, 20220521, 20220522, 20220523, 20220524, 20220525, 20220526, 20220527, 20220528, 20220529, 20220530, 20220531, 20220601, 20220602, 20220603, 20220604, 20220605, 20220606, 20220607, 20220608, 20220609, 20220610, 20220611, 20220612, 20220613, 20220614, 20220615, 20220616, 20220617, 20220618, 20220619, 20220620, 20220621, 20220622, 20220623, 20220624, 20220625, 20220626, 20220627, 20220628, 20220629, 20220630, 20220701, 20220702, 20220703, 20220704, 20220705, 20220706, 20220707, 20220708, 20220709, 20220710, 20220711, 20220712, 20220713, 20220714, 20220715, 20220716, 20220717, 20220718, 20220719, 20220720, 20220721, 20220722, 20220723, 20220724, 20220725, 20220726, 20220727, 20220728, 20220729, 20220730, 20220731, 20220801]



def return_all_dates():
    return [20140101, 20140102, 20140103, 20140104, 20140105, 20140106, 20140107, 20140108, 20140109, 20140110, 20140111, 20140112, 20140113, 20140114, 20140115, 20140116, 20140117, 20140118, 20140119, 20140120, 20140121, 20140122, 20140123, 20140124, 20140125, 20140126, 20140127, 20140128, 20140129, 20140130, 20140131, 20140201, 20140202, 20140203, 20140204, 20140205, 20140206, 20140207, 20140208, 20140209, 20140210, 20140211, 20140212, 20140213, 20140214, 20140215, 20140216, 20140217, 20140218, 20140219, 20140220, 20140221, 20140222, 20140223, 20140224, 20140225, 20140226, 20140227, 20140228, 20140301, 20140302, 20140303, 20140304, 20140305, 20140306, 20140307, 20140308, 20140309, 20140310, 20140311, 20140312, 20140313, 20140314, 20140315, 20140316, 20140317, 20140318, 20140319, 20140320, 20140321, 20140322, 20140323, 20140324, 20140325, 20140326, 20140327, 20140328, 20140329, 20140330, 20140331, 20140401, 20140402, 20140403, 20140404, 20140405, 20140406, 20140407, 20140408, 20140409, 20140410, 20140411, 20140412, 20140413, 20140414, 20140415, 20140416, 20140417, 20140418, 20140419, 20140420, 20140421, 20140422, 20140423, 20140424, 20140425, 20140426, 20140427, 20140428, 20140429, 20140430, 20140501, 20140502, 20140503, 20140504, 20140505, 20140506, 20140507, 20140508, 20140509, 20140510, 20140511, 20140512, 20140513, 20140514, 20140515, 20140516, 20140517, 20140518, 20140519, 20140520, 20140521, 20140522, 20140523, 20140524, 20140525, 20140526, 20140527, 20140528, 20140529, 20140530, 20140531, 20140601, 20140602, 20140603, 20140604, 20140605, 20140606, 20140607, 20140608, 20140609, 20140610, 20140611, 20140612, 20140613, 20140614, 20140615, 20140616, 20140617, 20140618, 20140619, 20140620, 20140621, 20140622, 20140623, 20140624, 20140625, 20140626, 20140627, 20140628, 20140629, 20140630, 20140701, 20140702, 20140703, 20140704, 20140705, 20140706, 20140707, 20140708, 20140709, 20140710, 20140711, 20140712, 20140713, 20140714, 20140715, 20140716, 20140717, 20140718, 20140719, 20140720, 20140721, 20140722, 20140723, 20140724, 20140725, 20140726, 20140727, 20140728, 20140729, 20140730, 20140731, 20140801, 20140802, 20140803, 20140804, 20140805, 20140806, 20140807, 20140808, 20140809, 20140810, 20140811, 20140812, 20140813, 20140814, 20140815, 20140816, 20140817, 20140818, 20140819, 20140820, 20140821, 20140822, 20140823, 20140824, 20140825, 20140826, 20140827, 20140828, 20140829, 20140830, 20140831, 20140901, 20140902, 20140903, 20140904, 20140905, 20140906, 20140907, 20140908, 20140909, 20140910, 20140911, 20140912, 20140913, 20140914, 20140915, 20140916, 20140917, 20140918, 20140919, 20140920, 20140921, 20140922, 20140923, 20140924, 20140925, 20140926, 20140927, 20140928, 20140929, 20140930, 20141001, 20141002, 20141003, 20141004, 20141005, 20141006, 20141007, 20141008, 20141009, 20141010, 20141011, 20141012, 20141013, 20141014, 20141015, 20141016, 20141017, 20141018, 20141019, 20141020, 20141021, 20141022, 20141023, 20141024, 20141025, 20141026, 20141027, 20141028, 20141029, 20141030, 20141031, 20141101, 20141102, 20141103, 20141104, 20141105, 20141106, 20141107, 20141108, 20141109, 20141110, 20141111, 20141112, 20141113, 20141114, 20141115, 20141116, 20141117, 20141118, 20141119, 20141120, 20141121, 20141122, 20141123, 20141124, 20141125, 20141126, 20141127, 20141128, 20141129, 20141130, 20141201, 20141202, 20141203, 20141204, 20141205, 20141206, 20141207, 20141208, 20141209, 20141210, 20141211, 20141212, 20141213, 20141214, 20141215, 20141216, 20141217, 20141218, 20141219, 20141220, 20141221, 20141222, 20141223, 20141224, 20141225, 20141226, 20141227, 20141228, 20141229, 20141230, 20141231, 20150101, 20150102, 20150103, 20150104, 20150105, 20150106, 20150107, 20150108, 20150109, 20150110, 20150111, 20150112, 20150113, 20150114, 20150115, 20150116, 20150117, 20150118, 20150119, 20150120, 20150121, 20150122, 20150123, 20150124, 20150125, 20150126, 20150127, 20150128, 20150129, 20150130, 20150131, 20150201, 20150202, 20150203, 20150204, 20150205, 20150206, 20150207, 20150208, 20150209, 20150210, 20150211, 20150212, 20150213, 20150214, 20150215, 20150216, 20150217, 20150218, 20150219, 20150220, 20150221, 20150222, 20150223, 20150224, 20150225, 20150226, 20150227, 20150228, 20150301, 20150302, 20150303, 20150304, 20150305, 20150306, 20150307, 20150308, 20150309, 20150310, 20150311, 20150312, 20150313, 20150314, 20150315, 20150316, 20150317, 20150318, 20150319, 20150320, 20150321, 20150322, 20150323, 20150324, 20150325, 20150326, 20150327, 20150328, 20150329, 20150330, 20150331, 20150401, 20150402, 20150403, 20150404, 20150405, 20150406, 20150407, 20150408, 20150409, 20150410, 20150411, 20150412, 20150413, 20150414, 20150415, 20150416, 20150417, 20150418, 20150419, 20150420, 20150421, 20150422, 20150423, 20150424, 20150425, 20150426, 20150427, 20150428, 20150429, 20150430, 20150501, 20150502, 20150503, 20150504, 20150505, 20150506, 20150507, 20150508, 20150509, 20150510, 20150511, 20150512, 20150513, 20150514, 20150515, 20150516, 20150517, 20150518, 20150519, 20150520, 20150521, 20150522, 20150523, 20150524, 20150525, 20150526, 20150527, 20150528, 20150529, 20150530, 20150531, 20150601, 20150602, 20150603, 20150604, 20150605, 20150606, 20150607, 20150608, 20150609, 20150610, 20150611, 20150612, 20150613, 20150614, 20150615, 20150616, 20150617, 20150618, 20150619, 20150620, 20150621, 20150622, 20150623, 20150624, 20150625, 20150626, 20150627, 20150628, 20150629, 20150630, 20150701, 20150702, 20150703, 20150704, 20150705, 20150706, 20150707, 20150708, 20150709, 20150710, 20150711, 20150712, 20150713, 20150714, 20150715, 20150716, 20150717, 20150718, 20150719, 20150720, 20150721, 20150722, 20150723, 20150724, 20150725, 20150726, 20150727, 20150728, 20150729, 20150730, 20150731, 20150801, 20150802, 20150803, 20150804, 20150805, 20150806, 20150807, 20150808, 20150809, 20150810, 20150811, 20150812, 20150813, 20150814, 20150815, 20150816, 20150817, 20150818, 20150819, 20150820, 20150821, 20150822, 20150823, 20150824, 20150825, 20150826, 20150827, 20150828, 20150829, 20150830, 20150831, 20150901, 20150902, 20150903, 20150904, 20150905, 20150906, 20150907, 20150908, 20150909, 20150910, 20150911, 20150912, 20150913, 20150914, 20150915, 20150916, 20150917, 20150918, 20150919, 20150920, 20150921, 20150922, 20150923, 20150924, 20150925, 20150926, 20150927, 20150928, 20150929, 20150930, 20151001, 20151002, 20151003, 20151004, 20151005, 20151006, 20151007, 20151008, 20151009, 20151010, 20151011, 20151012, 20151013, 20151014, 20151015, 20151016, 20151017, 20151018, 20151019, 20151020, 20151021, 20151022, 20151023, 20151024, 20151025, 20151026, 20151027, 20151028, 20151029, 20151030, 20151031, 20151101, 20151102, 20151103, 20151104, 20151105, 20151106, 20151107, 20151108, 20151109, 20151110, 20151111, 20151112, 20151113, 20151114, 20151115, 20151116, 20151117, 20151118, 20151119, 20151120, 20151121, 20151122, 20151123, 20151124, 20151125, 20151126, 20151127, 20151128, 20151129, 20151130, 20151201, 20151202, 20151203, 20151204, 20151205, 20151206, 20151207, 20151208, 20151209, 20151210, 20151211, 20151212, 20151213, 20151214, 20151215, 20151216, 20151217, 20151218, 20151219, 20151220, 20151221, 20151222, 20151223, 20151224, 20151225, 20151226, 20151227, 20151228, 20151229, 20151230, 20151231, 20160101, 20160102, 20160103, 20160104, 20160105, 20160106, 20160107, 20160108, 20160109, 20160110, 20160111, 20160112, 20160113, 20160114, 20160115, 20160116, 20160117, 20160118, 20160119, 20160120, 20160121, 20160122, 20160123, 20160124, 20160125, 20160126, 20160127, 20160128, 20160129, 20160130, 20160131, 20160201, 20160202, 20160203, 20160204, 20160205, 20160206, 20160207, 20160208, 20160209, 20160210, 20160211, 20160212, 20160213, 20160214, 20160215, 20160216, 20160217, 20160218, 20160219, 20160220, 20160221, 20160222, 20160223, 20160224, 20160225, 20160226, 20160227, 20160228, 20160229, 20160301, 20160302, 20160303, 20160304, 20160305, 20160306, 20160307, 20160308, 20160309, 20160310, 20160311, 20160312, 20160313, 20160314, 20160315, 20160316, 20160317, 20160318, 20160319, 20160320, 20160321, 20160322, 20160323, 20160324, 20160325, 20160326, 20160327, 20160328, 20160329, 20160330, 20160331, 20160401, 20160402, 20160403, 20160404, 20160405, 20160406, 20160407, 20160408, 20160409, 20160410, 20160411, 20160412, 20160413, 20160414, 20160415, 20160416, 20160417, 20160418, 20160419, 20160420, 20160421, 20160422, 20160423, 20160424, 20160425, 20160426, 20160427, 20160428, 20160429, 20160430, 20160501, 20160502, 20160503, 20160504, 20160505, 20160506, 20160507, 20160508, 20160509, 20160510, 20160511, 20160512, 20160513, 20160514, 20160515, 20160516, 20160517, 20160518, 20160519, 20160520, 20160521, 20160522, 20160523, 20160524, 20160525, 20160526, 20160527, 20160528, 20160529, 20160530, 20160531, 20160601, 20160602, 20160603, 20160604, 20160605, 20160606, 20160607, 20160608, 20160609, 20160610, 20160611, 20160612, 20160613, 20160614, 20160615, 20160616, 20160617, 20160618, 20160619, 20160620, 20160621, 20160622, 20160623, 20160624, 20160625, 20160626, 20160627, 20160628, 20160629, 20160630, 20160701, 20160702, 20160703, 20160704, 20160705, 20160706, 20160707, 20160708, 20160709, 20160710, 20160711, 20160712, 20160713, 20160714, 20160715, 20160716, 20160717, 20160718, 20160719, 20160720, 20160721, 20160722, 20160723, 20160724, 20160725, 20160726, 20160727, 20160728, 20160729, 20160730, 20160731, 20160801, 20160802, 20160803, 20160804, 20160805, 20160806, 20160807, 20160808, 20160809, 20160810, 20160811, 20160812, 20160813, 20160814, 20160815, 20160816, 20160817, 20160818, 20160819, 20160820, 20160821, 20160822, 20160823, 20160824, 20160825, 20160826, 20160827, 20160828, 20160829, 20160830, 20160831, 20160901, 20160902, 20160903, 20160904, 20160905, 20160906, 20160907, 20160908, 20160909, 20160910, 20160911, 20160912, 20160913, 20160914, 20160915, 20160916, 20160917, 20160918, 20160919, 20160920, 20160921, 20160922, 20160923, 20160924, 20160925, 20160926, 20160927, 20160928, 20160929, 20160930, 20161001, 20161002, 20161003, 20161004, 20161005, 20161006, 20161007, 20161008, 20161009, 20161010, 20161011, 20161012, 20161013, 20161014, 20161015, 20161016, 20161017, 20161018, 20161019, 20161020, 20161021, 20161022, 20161023, 20161024, 20161025, 20161026, 20161027, 20161028, 20161029, 20161030, 20161031, 20161101, 20161102, 20161103, 20161104, 20161105, 20161106, 20161107, 20161108, 20161109, 20161110, 20161111, 20161112, 20161113, 20161114, 20161115, 20161116, 20161117, 20161118, 20161119, 20161120, 20161121, 20161122, 20161123, 20161124, 20161125, 20161126, 20161127, 20161128, 20161129, 20161130, 20161201, 20161202, 20161203, 20161204, 20161205, 20161206, 20161207, 20161208, 20161209, 20161210, 20161211, 20161212, 20161213, 20161214, 20161215, 20161216, 20161217, 20161218, 20161219, 20161220, 20161221, 20161222, 20161223, 20161224, 20161225, 20161226, 20161227, 20161228, 20161229, 20161230, 20161231, 20170101, 20170102, 20170103, 20170104, 20170105, 20170106, 20170107, 20170108, 20170109, 20170110, 20170111, 20170112, 20170113, 20170114, 20170115, 20170116, 20170117, 20170118, 20170119, 20170120, 20170121, 20170122, 20170123, 20170124, 20170125, 20170126, 20170127, 20170128, 20170129, 20170130, 20170131, 20170201, 20170202, 20170203, 20170204, 20170205, 20170206, 20170207, 20170208, 20170209, 20170210, 20170211, 20170212, 20170213, 20170214, 20170215, 20170216, 20170217, 20170218, 20170219, 20170220, 20170221, 20170222, 20170223, 20170224, 20170225, 20170226, 20170227, 20170228, 20170301, 20170302, 20170303, 20170304, 20170305, 20170306, 20170307, 20170308, 20170309, 20170310, 20170311, 20170312, 20170313, 20170314, 20170315, 20170316, 20170317, 20170318, 20170319, 20170320, 20170321, 20170322, 20170323, 20170324, 20170325, 20170326, 20170327, 20170328, 20170329, 20170330, 20170331, 20170401, 20170402, 20170403, 20170404, 20170405, 20170406, 20170407, 20170408, 20170409, 20170410, 20170411, 20170412, 20170413, 20170414, 20170415, 20170416, 20170417, 20170418, 20170419, 20170420, 20170421, 20170422, 20170423, 20170424, 20170425, 20170426, 20170427, 20170428, 20170429, 20170430, 20170501, 20170502, 20170503, 20170504, 20170505, 20170506, 20170507, 20170508, 20170509, 20170510, 20170511, 20170512, 20170513, 20170514, 20170515, 20170516, 20170517, 20170518, 20170519, 20170520, 20170521, 20170522, 20170523, 20170524, 20170525, 20170526, 20170527, 20170528, 20170529, 20170530, 20170531, 20170601, 20170602, 20170603, 20170604, 20170605, 20170606, 20170607, 20170608, 20170609, 20170610, 20170611, 20170612, 20170613, 20170614, 20170615, 20170616, 20170617, 20170618, 20170619, 20170620, 20170621, 20170622, 20170623, 20170624, 20170625, 20170626, 20170627, 20170628, 20170629, 20170630, 20170701, 20170702, 20170703, 20170704, 20170705, 20170706, 20170707, 20170708, 20170709, 20170710, 20170711, 20170712, 20170713, 20170714, 20170715, 20170716, 20170717, 20170718, 20170719, 20170720, 20170721, 20170722, 20170723, 20170724, 20170725, 20170726, 20170727, 20170728, 20170729, 20170730, 20170731, 20170801, 20170802, 20170803, 20170804, 20170805, 20170806, 20170807, 20170808, 20170809, 20170810, 20170811, 20170812, 20170813, 20170814, 20170815, 20170816, 20170817, 20170818, 20170819, 20170820, 20170821, 20170822, 20170823, 20170824, 20170825, 20170826, 20170827, 20170828, 20170829, 20170830, 20170831, 20170901, 20170902, 20170903, 20170904, 20170905, 20170906, 20170907, 20170908, 20170909, 20170910, 20170911, 20170912, 20170913, 20170914, 20170915, 20170916, 20170917, 20170918, 20170919, 20170920, 20170921, 20170922, 20170923, 20170924, 20170925, 20170926, 20170927, 20170928, 20170929, 20170930, 20171001, 20171002, 20171003, 20171004, 20171005, 20171006, 20171007, 20171008, 20171009, 20171010, 20171011, 20171012, 20171013, 20171014, 20171015, 20171016, 20171017, 20171018, 20171019, 20171020, 20171021, 20171022, 20171023, 20171024, 20171025, 20171026, 20171027, 20171028, 20171029, 20171030, 20171031, 20171101, 20171102, 20171103, 20171104, 20171105, 20171106, 20171107, 20171108, 20171109, 20171110, 20171111, 20171112, 20171113, 20171114, 20171115, 20171116, 20171117, 20171118, 20171119, 20171120, 20171121, 20171122, 20171123, 20171124, 20171125, 20171126, 20171127, 20171128, 20171129, 20171130, 20171201, 20171202, 20171203, 20171204, 20171205, 20171206, 20171207, 20171208, 20171209, 20171210, 20171211, 20171212, 20171213, 20171214, 20171215, 20171216, 20171217, 20171218, 20171219, 20171220, 20171221, 20171222, 20171223, 20171224, 20171225, 20171226, 20171227, 20171228, 20171229, 20171230, 20171231, 20180101, 20180102, 20180103, 20180104, 20180105, 20180106, 20180107, 20180108, 20180109, 20180110, 20180111, 20180112, 20180113, 20180114, 20180115, 20180116, 20180117, 20180118, 20180119, 20180120, 20180121, 20180122, 20180123, 20180124, 20180125, 20180126, 20180127, 20180128, 20180129, 20180130, 20180131, 20180201, 20180202, 20180203, 20180204, 20180205, 20180206, 20180207, 20180208, 20180209, 20180210, 20180211, 20180212, 20180213, 20180214, 20180215, 20180216, 20180217, 20180218, 20180219, 20180220, 20180221, 20180222, 20180223, 20180224, 20180225, 20180226, 20180227, 20180228, 20180301, 20180302, 20180303, 20180304, 20180305, 20180306, 20180307, 20180308, 20180309, 20180310, 20180311, 20180312, 20180313, 20180314, 20180315, 20180316, 20180317, 20180318, 20180319, 20180320, 20180321, 20180322, 20180323, 20180324, 20180325, 20180326, 20180327, 20180328, 20180329, 20180330, 20180331, 20180401, 20180402, 20180403, 20180404, 20180405, 20180406, 20180407, 20180408, 20180409, 20180410, 20180411, 20180412, 20180413, 20180414, 20180415, 20180416, 20180417, 20180418, 20180419, 20180420, 20180421, 20180422, 20180423, 20180424, 20180425, 20180426, 20180427, 20180428, 20180429, 20180430, 20180501, 20180502, 20180503, 20180504, 20180505, 20180506, 20180507, 20180508, 20180509, 20180510, 20180511, 20180512, 20180513, 20180514, 20180515, 20180516, 20180517, 20180518, 20180519, 20180520, 20180521, 20180522, 20180523, 20180524, 20180525, 20180526, 20180527, 20180528, 20180529, 20180530, 20180531, 20180601, 20180602, 20180603, 20180604, 20180605, 20180606, 20180607, 20180608, 20180609, 20180610, 20180611, 20180612, 20180613, 20180614, 20180615, 20180616, 20180617, 20180618, 20180619, 20180620, 20180621, 20180622, 20180623, 20180624, 20180625, 20180626, 20180627, 20180628, 20180629, 20180630, 20180701, 20180702, 20180703, 20180704, 20180705, 20180706, 20180707, 20180708, 20180709, 20180710, 20180711, 20180712, 20180713, 20180714, 20180715, 20180716, 20180717, 20180718, 20180719, 20180720, 20180721, 20180722, 20180723, 20180724, 20180725, 20180726, 20180727, 20180728, 20180729, 20180730, 20180731, 20180801, 20180802, 20180803, 20180804, 20180805, 20180806, 20180807, 20180808, 20180809, 20180810, 20180811, 20180812, 20180813, 20180814, 20180815, 20180816, 20180817, 20180818, 20180819, 20180820, 20180821, 20180822, 20180823, 20180824, 20180825, 20180826, 20180827, 20180828, 20180829, 20180830, 20180831, 20180901, 20180902, 20180903, 20180904, 20180905, 20180906, 20180907, 20180908, 20180909, 20180910, 20180911, 20180912, 20180913, 20180914, 20180915, 20180916, 20180917, 20180918, 20180919, 20180920, 20180921, 20180922, 20180923, 20180924, 20180925, 20180926, 20180927, 20180928, 20180929, 20180930, 20181001, 20181002, 20181003, 20181004, 20181005, 20181006, 20181007, 20181008, 20181009, 20181010, 20181011, 20181012, 20181013, 20181014, 20181015, 20181016, 20181017, 20181018, 20181019, 20181020, 20181021, 20181022, 20181023, 20181024, 20181025, 20181026, 20181027, 20181028, 20181029, 20181030, 20181031, 20181101, 20181102, 20181103, 20181104, 20181105, 20181106, 20181107, 20181108, 20181109, 20181110, 20181111, 20181112, 20181113, 20181114, 20181115, 20181116, 20181117, 20181118, 20181119, 20181120, 20181121, 20181122, 20181123, 20181124, 20181125, 20181126, 20181127, 20181128, 20181129, 20181130, 20181201, 20181202, 20181203, 20181204, 20181205, 20181206, 20181207, 20181208, 20181209, 20181210, 20181211, 20181212, 20181213, 20181214, 20181215, 20181216, 20181217, 20181218, 20181219, 20181220, 20181221, 20181222, 20181223, 20181224, 20181225, 20181226, 20181227, 20181228, 20181229, 20181230, 20181231, 20190101, 20190102, 20190103, 20190104, 20190105, 20190106, 20190107, 20190108, 20190109, 20190110, 20190111, 20190112, 20190113, 20190114, 20190115, 20190116, 20190117, 20190118, 20190119, 20190120, 20190121, 20190122, 20190123, 20190124, 20190125, 20190126, 20190127, 20190128, 20190129, 20190130, 20190131, 20190201, 20190202, 20190203, 20190204, 20190205, 20190206, 20190207, 20190208, 20190209, 20190210, 20190211, 20190212, 20190213, 20190214, 20190215, 20190216, 20190217, 20190218, 20190219, 20190220, 20190221, 20190222, 20190223, 20190224, 20190225, 20190226, 20190227, 20190228, 20190301, 20190302, 20190303, 20190304, 20190305, 20190306, 20190307, 20190308, 20190309, 20190310, 20190311, 20190312, 20190313, 20190314, 20190315, 20190316, 20190317, 20190318, 20190319, 20190320, 20190321, 20190322, 20190323, 20190324, 20190325, 20190326, 20190327, 20190328, 20190329, 20190330, 20190331, 20190401, 20190402, 20190403, 20190404, 20190405, 20190406, 20190407, 20190408, 20190409, 20190410, 20190411, 20190412, 20190413, 20190414, 20190415, 20190416, 20190417, 20190418, 20190419, 20190420, 20190421, 20190422, 20190423, 20190424, 20190425, 20190426, 20190427, 20190428, 20190429, 20190430, 20190501, 20190502, 20190503, 20190504, 20190505, 20190506, 20190507, 20190508, 20190509, 20190510, 20190511, 20190512, 20190513, 20190514, 20190515, 20190516, 20190517, 20190518, 20190519, 20190520, 20190521, 20190522, 20190523, 20190524, 20190525, 20190526, 20190527, 20190528, 20190529, 20190530, 20190531, 20190601, 20190602, 20190603, 20190604, 20190605, 20190606, 20190607, 20190608, 20190609, 20190610, 20190611, 20190612, 20190613, 20190614, 20190615, 20190616, 20190617, 20190618, 20190619, 20190620, 20190621, 20190622, 20190623, 20190624, 20190625, 20190626, 20190627, 20190628, 20190629, 20190630, 20190701, 20190702, 20190703, 20190704, 20190705, 20190706, 20190707, 20190708, 20190709, 20190710, 20190711, 20190712, 20190713, 20190714, 20190715, 20190716, 20190717, 20190718, 20190719, 20190720, 20190721, 20190722, 20190723, 20190724, 20190725, 20190726, 20190727, 20190728, 20190729, 20190730, 20190731, 20190801, 20190802, 20190803, 20190804, 20190805, 20190806, 20190807, 20190808, 20190809, 20190810, 20190811, 20190812, 20190813, 20190814, 20190815, 20190816, 20190817, 20190818, 20190819, 20190820, 20190821, 20190822, 20190823, 20190824, 20190825, 20190826, 20190827, 20190828, 20190829, 20190830, 20190831, 20190901, 20190902, 20190903, 20190904, 20190905, 20190906, 20190907, 20190908, 20190909, 20190910, 20190911, 20190912, 20190913, 20190914, 20190915, 20190916, 20190917, 20190918, 20190919, 20190920, 20190921, 20190922, 20190923, 20190924, 20190925, 20190926, 20190927, 20190928, 20190929, 20190930, 20191001, 20191002, 20191003, 20191004, 20191005, 20191006, 20191007, 20191008, 20191009, 20191010, 20191011, 20191012, 20191013, 20191014, 20191015, 20191016, 20191017, 20191018, 20191019, 20191020, 20191021, 20191022, 20191023, 20191024, 20191025, 20191026, 20191027, 20191028, 20191029, 20191030, 20191031, 20191101, 20191102, 20191103, 20191104, 20191105, 20191106, 20191107, 20191108, 20191109, 20191110, 20191111, 20191112, 20191113, 20191114, 20191115, 20191116, 20191117, 20191118, 20191119, 20191120, 20191121, 20191122, 20191123, 20191124, 20191125, 20191126, 20191127, 20191128, 20191129, 20191130, 20191201, 20191202, 20191203, 20191204, 20191205, 20191206, 20191207, 20191208, 20191209, 20191210, 20191211, 20191212, 20191213, 20191214, 20191215, 20191216, 20191217, 20191218, 20191219, 20191220, 20191221, 20191222, 20191223, 20191224, 20191225, 20191226, 20191227, 20191228, 20191229, 20191230, 20191231, 20200101, 20200102, 20200103, 20200104, 20200105, 20200106, 20200107, 20200108, 20200109, 20200110, 20200111, 20200112, 20200113, 20200114, 20200115, 20200116, 20200117, 20200118, 20200119, 20200120, 20200121, 20200122, 20200123, 20200124, 20200125, 20200126, 20200127, 20200128, 20200129, 20200130, 20200131, 20200201, 20200202, 20200203, 20200204, 20200205, 20200206, 20200207, 20200208, 20200209, 20200210, 20200211, 20200212, 20200213, 20200214, 20200215, 20200216, 20200217, 20200218, 20200219, 20200220, 20200221, 20200222, 20200223, 20200224, 20200225, 20200226, 20200227, 20200228, 20200229, 20200301, 20200302, 20200303, 20200304, 20200305, 20200306, 20200307, 20200308, 20200309, 20200310, 20200311, 20200312, 20200313, 20200314, 20200315, 20200316, 20200317, 20200318, 20200319, 20200320, 20200321, 20200322, 20200323, 20200324, 20200325, 20200326, 20200327, 20200328, 20200329, 20200330, 20200331, 20200401, 20200402, 20200403, 20200404, 20200405, 20200406, 20200407, 20200408, 20200409, 20200410, 20200411, 20200412, 20200413, 20200414, 20200415, 20200416, 20200417, 20200418, 20200419, 20200420, 20200421, 20200422, 20200423, 20200424, 20200425, 20200426, 20200427, 20200428, 20200429, 20200430, 20200501, 20200502, 20200503, 20200504, 20200505, 20200506, 20200507, 20200508, 20200509, 20200510, 20200511, 20200512, 20200513, 20200514, 20200515, 20200516, 20200517, 20200518, 20200519, 20200520, 20200521, 20200522, 20200523, 20200524, 20200525, 20200526, 20200527, 20200528, 20200529, 20200530, 20200531, 20200601, 20200602, 20200603, 20200604, 20200605, 20200606, 20200607, 20200608, 20200609, 20200610, 20200611, 20200612, 20200613, 20200614, 20200615, 20200616, 20200617, 20200618, 20200619, 20200620, 20200621, 20200622, 20200623, 20200624, 20200625, 20200626, 20200627, 20200628, 20200629, 20200630, 20200701, 20200702, 20200703, 20200704, 20200705, 20200706, 20200707, 20200708, 20200709, 20200710, 20200711, 20200712, 20200713, 20200714, 20200715, 20200716, 20200717, 20200718, 20200719, 20200720, 20200721, 20200722, 20200723, 20200724, 20200725, 20200726, 20200727, 20200728, 20200729, 20200730, 20200731, 20200801, 20200802, 20200803, 20200804, 20200805, 20200806, 20200807, 20200808, 20200809, 20200810, 20200811, 20200812, 20200813, 20200814, 20200815, 20200816, 20200817, 20200818, 20200819, 20200820, 20200821, 20200822, 20200823, 20200824, 20200825, 20200826, 20200827, 20200828, 20200829, 20200830, 20200831, 20200901, 20200902, 20200903, 20200904, 20200905, 20200906, 20200907, 20200908, 20200909, 20200910, 20200911, 20200912, 20200913, 20200914, 20200915, 20200916, 20200917, 20200918, 20200919, 20200920, 20200921, 20200922, 20200923, 20200924, 20200925, 20200926, 20200927, 20200928, 20200929, 20200930, 20201001, 20201002, 20201003, 20201004, 20201005, 20201006, 20201007, 20201008, 20201009, 20201010, 20201011, 20201012, 20201013, 20201014, 20201015, 20201016, 20201017, 20201018, 20201019, 20201020, 20201021, 20201022, 20201023, 20201024, 20201025, 20201026, 20201027, 20201028, 20201029, 20201030, 20201031, 20201101, 20201102, 20201103, 20201104, 20201105, 20201106, 20201107, 20201108, 20201109, 20201110, 20201111, 20201112, 20201113, 20201114, 20201115, 20201116, 20201117, 20201118, 20201119, 20201120, 20201121, 20201122, 20201123, 20201124, 20201125, 20201126, 20201127, 20201128, 20201129, 20201130, 20201201, 20201202, 20201203, 20201204, 20201205, 20201206, 20201207, 20201208, 20201209, 20201210, 20201211, 20201212, 20201213, 20201214, 20201215, 20201216, 20201217, 20201218, 20201219, 20201220, 20201221, 20201222, 20201223, 20201224, 20201225, 20201226, 20201227, 20201228, 20201229, 20201230, 20201231, 20210101, 20210102, 20210103, 20210104, 20210105, 20210106, 20210107, 20210108, 20210109, 20210110, 20210111, 20210112, 20210113, 20210114, 20210115, 20210116, 20210117, 20210118, 20210119, 20210120, 20210121, 20210122, 20210123, 20210124, 20210125, 20210126, 20210127, 20210128, 20210129, 20210130, 20210131, 20210201, 20210202, 20210203, 20210204, 20210205, 20210206, 20210207, 20210208, 20210209, 20210210, 20210211, 20210212, 20210213, 20210214, 20210215, 20210216, 20210217, 20210218, 20210219, 20210220, 20210221, 20210222, 20210223, 20210224, 20210225, 20210226, 20210227, 20210228, 20210301, 20210302, 20210303, 20210304, 20210305, 20210306, 20210307, 20210308, 20210309, 20210310, 20210311, 20210312, 20210313, 20210314, 20210315, 20210316, 20210317, 20210318, 20210319, 20210320, 20210321, 20210322, 20210323, 20210324, 20210325, 20210326, 20210327, 20210328, 20210329, 20210330, 20210331, 20210401, 20210402, 20210403, 20210404, 20210405, 20210406, 20210407, 20210408, 20210409, 20210410, 20210411, 20210412, 20210413, 20210414, 20210415, 20210416, 20210417, 20210418, 20210419, 20210420, 20210421, 20210422, 20210423, 20210424, 20210425, 20210426, 20210427, 20210428, 20210429, 20210430, 20210501, 20210502, 20210503, 20210504, 20210505, 20210506, 20210507, 20210508, 20210509, 20210510, 20210511, 20210512, 20210513, 20210514, 20210515, 20210516, 20210517, 20210518, 20210519, 20210520, 20210521, 20210522, 20210523, 20210524, 20210525, 20210526, 20210527, 20210528, 20210529, 20210530, 20210531, 20210601, 20210602, 20210603, 20210604, 20210605, 20210606, 20210607, 20210608, 20210609, 20210610, 20210611, 20210612, 20210613, 20210614, 20210615, 20210616, 20210617, 20210618, 20210619, 20210620, 20210621, 20210622, 20210623, 20210624, 20210625, 20210626, 20210627, 20210628, 20210629, 20210630, 20210701, 20210702, 20210703, 20210704, 20210705, 20210706, 20210707, 20210708, 20210709, 20210710, 20210711, 20210712, 20210713, 20210714, 20210715, 20210716, 20210717, 20210718, 20210719, 20210720, 20210721, 20210722, 20210723, 20210724, 20210725, 20210726, 20210727, 20210728, 20210729, 20210730, 20210731, 20210801, 20210802, 20210803, 20210804, 20210805, 20210806, 20210807, 20210808, 20210809, 20210810, 20210811, 20210812, 20210813, 20210814, 20210815, 20210816, 20210817, 20210818, 20210819, 20210820, 20210821, 20210822, 20210823, 20210824, 20210825, 20210826, 20210827, 20210828, 20210829, 20210830, 20210831, 20210901, 20210902, 20210903, 20210904, 20210905, 20210906, 20210907, 20210908, 20210909, 20210910, 20210911, 20210912, 20210913, 20210914, 20210915, 20210916, 20210917, 20210918, 20210919, 20210920, 20210921, 20210922, 20210923, 20210924, 20210925, 20210926, 20210927, 20210928, 20210929, 20210930, 20211001, 20211002, 20211003, 20211004, 20211005, 20211006, 20211007, 20211008, 20211009, 20211010, 20211011, 20211012, 20211013, 20211014, 20211015, 20211016, 20211017, 20211018, 20211019, 20211020, 20211021, 20211022, 20211023, 20211024, 20211025, 20211026, 20211027, 20211028, 20211029, 20211030, 20211031, 20211101, 20211102, 20211103, 20211104, 20211105, 20211106, 20211107, 20211108, 20211109, 20211110, 20211111, 20211112, 20211113, 20211114, 20211115, 20211116, 20211117, 20211118, 20211119, 20211120, 20211121, 20211122, 20211123, 20211124, 20211125, 20211126, 20211127, 20211128, 20211129, 20211130, 20211201, 20211202, 20211203, 20211204, 20211205, 20211206, 20211207, 20211208, 20211209, 20211210, 20211211, 20211212, 20211213, 20211214, 20211215, 20211216, 20211217, 20211218, 20211219, 20211220, 20211221, 20211222, 20211223, 20211224, 20211225, 20211226, 20211227, 20211228, 20211229, 20211230, 20211231, 20220101, 20220102, 20220103, 20220104, 20220105, 20220106, 20220107, 20220108, 20220109, 20220110, 20220111, 20220112, 20220113, 20220114, 20220115, 20220116, 20220117, 20220118, 20220119, 20220120, 20220121, 20220122, 20220123, 20220124, 20220125, 20220126, 20220127, 20220128, 20220129, 20220130, 20220131, 20220201, 20220202, 20220203, 20220204, 20220205, 20220206, 20220207, 20220208, 20220209, 20220210, 20220211, 20220212, 20220213, 20220214, 20220215, 20220216, 20220217, 20220218, 20220219, 20220220, 20220221, 20220222, 20220223, 20220224, 20220225, 20220226, 20220227, 20220228, 20220301, 20220302, 20220303, 20220304, 20220305, 20220306, 20220307, 20220308, 20220309, 20220310, 20220311, 20220312, 20220313, 20220314, 20220315, 20220316, 20220317, 20220318, 20220319, 20220320, 20220321, 20220322, 20220323, 20220324, 20220325, 20220326, 20220327, 20220328, 20220329, 20220330, 20220331, 20220401, 20220402, 20220403, 20220404, 20220405, 20220406, 20220407, 20220408, 20220409, 20220410, 20220411, 20220412, 20220413, 20220414, 20220415, 20220416, 20220417, 20220418, 20220419, 20220420, 20220421, 20220422, 20220423, 20220424, 20220425, 20220426, 20220427, 20220428, 20220429, 20220430, 20220501, 20220502, 20220503, 20220504, 20220505, 20220506, 20220507, 20220508, 20220509, 20220510, 20220511, 20220512, 20220513, 20220514, 20220515, 20220516, 20220517, 20220518, 20220519, 20220520, 20220521, 20220522, 20220523, 20220524, 20220525, 20220526, 20220527, 20220528, 20220529, 20220530, 20220531, 20220601, 20220602, 20220603, 20220604, 20220605, 20220606, 20220607, 20220608, 20220609, 20220610, 20220611, 20220612, 20220613, 20220614, 20220615, 20220616, 20220617, 20220618, 20220619, 20220620, 20220621, 20220622, 20220623, 20220624, 20220625, 20220626, 20220627, 20220628, 20220629, 20220630, 20220701, 20220702, 20220703, 20220704, 20220705, 20220706, 20220707, 20220708, 20220709, 20220710, 20220711, 20220712, 20220713, 20220714, 20220715, 20220716, 20220717, 20220718, 20220719, 20220720, 20220721, 20220722, 20220723, 20220724, 20220725, 20220726, 20220727, 20220728, 20220729, 20220730, 20220731, 20220801]

