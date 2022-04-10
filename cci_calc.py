import os
import re
import glob
import numpy as np
import pandas as pd
import pickle
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.toolkit as st
import spikeinterface.widgets as sw
import matplotlib.pyplot as plt
from scipy.stats import ranksums, zscore, binom_test
from pathlib import Path

def get_correlograms(target, recalc_we=True, sortid='KKS', labelid='good',
                     window_ms = 1000.0, bin_ms=5.0, twin = np.array([[0, 50],
                                                                      [50, 100]]),
                     cci_thresh=0, dist_thresh_upper=8000, dist_thresh_lower=-1,
                     num_shuf=10000, percentile_thresh_upper=0.95, percentile_thresh_lower=0.05):
    #Target: str; NPXX_BXX ie NP03_B02
    #recalc_we: Bool, recaculate waveform extractor
    #sortid: str; column from .tsv or .csv from which inclusion/exclusion is decided
    #labelid: str; label from .tsv or .csv from which inclusion/exclusion is decided
    #window_ms: float; window for correlogram calc in ms
    #bin_ms: float; bin size for correlogram calc in ms
    #twin: 2 x 2 arr: time window for CCI calculation in ms
    twin_bin = twin/bin_ms
    twin_bin = twin_bin.astype(int)
    recording_path = '/userdata/ksellers/Neuropixels/' + target + '_g0/' + target + '_g0_imec0/'
    sorting_path = recording_path + 'KilosortOutput/'
    path = sorting_path + 'QualityMetrics/KKS_good/run0/'
    tsv_path = sorting_path + 'cluster_info.tsv'
    sorting = se.KiloSortSortingExtractor(sorting_path, keep_good_only=False)
    curated_sorting = filter_units(sorting, sortid, labelid, tsv_path)
    depth = curated_sorting.__dict__['_properties']['depth']
    distances = calc_um_distances_between_clusters(depth)
    distances = np.ravel(distances)
    correlograms = st.postprocessing.compute_correlograms(curated_sorting, window_ms = window_ms, bin_ms=bin_ms)
    correlograms = correlograms[0]
    shapelist = list(np.shape(correlograms))
    shapelist.append(num_shuf)
    shuf_correlograms = np.zeros(tuple(shapelist))
    shuf_cci_pairs = np.zeros(tuple([shapelist[0],shapelist[1],num_shuf]))
    for ii in range(num_shuf):
        shuf_correlogram = st.postprocessing.compute_correlograms(curated_sorting, 
                                                                  window_ms = window_ms, 
                                                                  bin_ms=bin_ms, shuffle=True)
        shuf_correlograms[:,:,:,ii] = shuf_correlogram[0]
        [shuf_cci_self, shuf_cci_pair, shuf_autocorr] = calc_cci_correlograms(shuf_correlogram[0], twin_bin)
        shuf_cci_pairs[:,:,ii] = shuf_cci_pair
    #SHUFFLER IN correlograms.py - based upon spikeinterface correlograms.py
    #For every pair, generate a distribution of cci's
    #CCI distribution cutoff for shuffled correlograms
    [cci_selfs, cci_pairs, autocorrs] = calc_cci_correlograms(correlograms, twin_bin)
    cci_pairs = np.reshape(cci_pairs, np.shape(cci_pairs)[0]*np.shape(cci_pairs)[1])
    shuf_cci_pairs = np.reshape(shuf_cci_pairs, tuple([np.shape(shuf_cci_pairs)[0]*np.shape(shuf_cci_pairs)[1], np.shape(shuf_cci_pairs)[-1]]))
    cci_pair_percentiles = np.ones(np.shape(cci_pairs))*np.nan
    for pair in range(np.shape(shuf_cci_pairs)[0]):
        #percentile = number of shuffled cci's that are less than the actual divided by number of shuffles
        #return a p value here too
        nan_shuf_mask = ~np.isnan(shuf_cci_pairs[pair,:])
        eligible_shufs = np.sum(nan_shuf_mask)
        if ~np.isnan(cci_pairs[pair]):            
            cci_pair_percentiles[pair] = np.sum(cci_pairs[pair]>shuf_cci_pairs[pair, nan_shuf_mask])/eligible_shufs
    #Reshape correlograms
    correlograms = np.reshape(correlograms, tuple([np.shape(correlograms)[0]*np.shape(correlograms)[1],np.shape(correlograms)[-1]]))
    percentile_mask = np.logical_or(cci_pair_percentiles>percentile_thresh_upper, cci_pair_percentiles<percentile_thresh_lower)
    #Apply percentile mask later as follows:
    #distances = distances[percentile_mask]
    #cci_pairs = cci_pairs[percentile_mask]
    #correlograms = correlograms[percentile_mask,:]
    print(target)
    return autocorrs, cci_pairs, cci_pair_percentiles, percentile_mask, distances, correlograms, shuf_cci_pairs

def calc_cci_correlograms(correlograms, twin_bin, min_events=50):
    cci_selfs = np.ones([correlograms.shape[0]])*np.nan
    cci_pairs = np.ones([correlograms.shape[0],correlograms.shape[1]])*np.nan
    norm_correlograms = np.ones(np.shape(correlograms))*np.nan
    autocorrs = np.ones([correlograms.shape[0],correlograms.shape[2]])*np.nan
    for ii in np.arange(correlograms.shape[0]):
        for ij in np.arange(correlograms.shape[0]):
            if ii != ij:
                correlogram = correlograms[ii,ij,:]
                if np.sum(correlogram)>min_events:
                    cci_pairs[ii,ij] = calc_cci_pair(correlogram, twin_bin)
                    norm_correlogram = correlogram/np.max(correlogram)
                    #cci_pairs[ii,ij] = calc_cci_pair(norm_correlogram, twin_bin)
                    norm_correlograms[ii, ij] = norm_correlogram
            elif ii == ij:
                #Autocorrelogram
                correlogram = correlograms[ii,ij,:]
                cci_pairs[ii,ij] = np.nan
                if np.sum(correlogram)>min_events:
                    cci_selfs[ii] = calc_cci_self(correlogram, twin_bin)
                    norm_correlogram = correlogram/np.max(correlogram)
                    autocorrs[ii,:] = norm_correlogram           
    norm_correlograms = norm_correlograms.reshape([(correlograms.shape[0]*correlograms.shape[1]),correlograms.shape[2]])
    return cci_selfs, cci_pairs, autocorrs

def calc_cci_pair(correlogram, twin_bin, min_events=50):
    if np.sum(correlogram)>min_events:
        p0 = np.sum(correlogram[twin_bin[0,0]:twin_bin[0,1]])
        p1 = np.sum(correlogram[twin_bin[1,0]:twin_bin[1,1]])
        cci_pair = (p0-p1)/np.max([p0,p1])
        return cci_pair
    else:
        return np.nan

def calc_cci_self(correlogram, twin_bin):
    p0 = np.max(correlogram[twin_bin[0,0]:twin_bin[0,1]])
    p1 = np.max(correlogram[twin_bin[1,0]:twin_bin[1,1]])
    cci_self = (p0-p1)/np.max([p0,p1])
    return cci_self  

def calc_um_distances_between_clusters(depths):
    distances = np.zeros([np.size(depths),np.size(depths)])
    for ii, depth1 in enumerate(depths):
        for ij, depth2 in enumerate(depths):
            distances[ii,ij]=np.abs(depth1-depth2)
            if ii==ij:
                distances[ii,ij] = np.nan
    return distances

def filter_units(sorting, column, value, tsv_path, query=None):
    """
    Return the `unit_ids` from the table at `tsv_path` given
    a `column` and `value`.
    
    Parameters
    ----------
    
    sorting : KilosortSortingExtractor, UnitsSelectionSorting
        The sorting extractor to filter.
    column : str
        The column used to filter units given the cluster_info
        at `tsv_path`
    value : str
        The value used to filter units given the cluster_info
        at `tsv_path`
    tsv_path : str
        The path to the file containing the cluster_info. The 
        path should end in '*/cluster_info.tsv'
        
    Returns
    -------
    filtered_units : UnitsSelectionSorting
        The filtered units given `column` and `value` in the
        table at `tsv_path`.
    
    """
    # Override the `column` and `value` parameters if a query is provided
    if not query:
        query = "{} == '{}'".format(column, value)
        
    # Load the tsv cluster info table
    cluster_info = pd.read_csv(tsv_path, delimiter='\t')
    
    # Filter the cluster list
    filtered_cluster_info = cluster_info.query(query)
    
    # Extract the filtered index values
    keep_unit_indexes = filtered_cluster_info.index.values
    # Remove units not in `sorting`
    keep_unit_ids = sorting._main_ids[keep_unit_indexes]
    # Select the desired units 
    filtered_units = sorting.select_units(keep_unit_ids)
    return filtered_units