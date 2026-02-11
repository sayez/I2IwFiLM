import os
from typing import Union
import zarr

import gcsfs
import s3fs
import sunpy.map

import dask.array as da
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from pandas.tseries.offsets import DateOffset

import sunpy.visualization.colormaps as cm

from astropy.time import Time
from sunpy.visualization import axis_labels_from_ctype, wcsaxes_compat

from matplotlib import animation
from IPython.display import HTML
from pathlib import Path

from tqdm import tqdm

from multiprocessing import Pool
from itertools import repeat
import time

from datetime import datetime, timedelta


import argparse


def s3_connection(path_to_zarr: os.path) -> s3fs.S3Map:
    """
    Instantiate connection to aws for a given path `path_to_zarr`
    """
    return s3fs.S3Map(
        root=path_to_zarr,
        s3=s3fs.S3FileSystem(anon=True),
        # anonymous access requires no credentials
        check=False,
    )


def load_single_aws_zarr(
    path_to_zarr: os.path,
    cache_max_single_size: int = None,
) -> Union[zarr.Array, zarr.Group]:
    """
    load zarr from s3 using LRU cache
    """
    return zarr.open(
        zarr.LRUStoreCache(
            store=s3_connection(path_to_zarr),
            max_size=cache_max_single_size,
        ),
        mode="r",
    )

def remove_timezone(dt):   
    # HERE `dt` is a python datetime 
    # object that used .replace() method
    return dt.replace(tzinfo=None)


def main(year, output_folder, redo_all=False):
    show=False
    save=True
    

    y = year
    output_folder_root = Path(output_folder)

    output_aia_folder = output_folder_root / "AIA_4h"
    output_hmi_folder = output_folder_root / "HMI_4h"

    aws_zarr_roots = {
    "aia": f"s3://gov-nasa-hdrl-data1/contrib/fdl-sdoml/fdl-sdoml-v2/sdomlv2.zarr/{y}/304A",
    "hmi": f"s3://gov-nasa-hdrl-data1/contrib/fdl-sdoml/fdl-sdoml-v2/sdomlv2_hmi.zarr/{y}/Bz",
    }

    selected_times = pd.date_range(
        start=f"{y}-01-01 00:00:00", end=f"{y}-12-31 23:59:59",  tz="UTC",
        # freq="D",
        freq="4h",
    )
    # # add offset to noon
    selected_times = selected_times # + DateOffset(hours=12)


    #####################################################################################
    # GET the indices of closest EUV times to the selected times
    root_hmi = load_single_aws_zarr(
        path_to_zarr=aws_zarr_roots["hmi"],
    )
    data_hmi = root_hmi
    print(f"Connected to HMI data for year {y} ...")

    hmi_obs_times = data_hmi.attrs["T_OBS"]
    hmi_obs_times = pd.to_datetime(hmi_obs_times, format='%Y.%m.%d_%H:%M:%S_TAI').to_pydatetime()
    
    hmi_arr = da.from_array(data_hmi)
    # #####################################################################################
    # GET the indices of closest AIA times to the selected times
    root_aia = load_single_aws_zarr(
        path_to_zarr=aws_zarr_roots["aia"],
    )
    data_aia = root_aia
    print(f"Connected to AIA data for year {y} ...")

    tmp = data_aia.attrs['T_OBS']
    tmp = [item[:-1]+'Z' if item[-1] != 'Z' else item for item in tmp]
    aia_obs_times = pd.to_datetime(tmp, format='%Y-%m-%dT%H:%M:%S.%fZ').to_pydatetime()
    aia_arr = da.from_array(data_aia)

    time_data_mapping = {t: [] for t in selected_times}
    selected_aia_indices = {t: [] for t in selected_times}
    selected_hmi_indices = {t: [] for t in selected_times}
    selected_hmi_times = {t: [] for t in selected_times}
    selected_aia_times = {t: [] for t in selected_times}
    
    for t in tqdm(selected_times):
        if np.min(np.abs(hmi_obs_times - t.replace(tzinfo=None).to_pydatetime())) > timedelta(minutes=30):
            continue
        if np.min(np.abs(aia_obs_times - t.replace(tzinfo=None).to_pydatetime())) > timedelta(minutes=30):
            continue

        idx_hmi = np.argmin(np.abs(hmi_obs_times - t.replace(tzinfo=None).to_pydatetime()))
        idx_aia = np.argmin(np.abs(aia_obs_times - t.replace(tzinfo=None).to_pydatetime()))

        cur_aia = aia_arr[idx_aia]
        cur_hmi = hmi_arr[idx_hmi]

        time_data_mapping[t] += [(cur_aia, cur_hmi)]
        selected_aia_indices[t] += [idx_aia]
        selected_hmi_indices[t] += [idx_hmi]

        selected_aia_times[t] += [aia_obs_times[idx_aia]]
        selected_hmi_times[t] += [hmi_obs_times[idx_hmi]]

    processed=0
    max_process = len(time_data_mapping)
    missing_date= []

    for i, (d, cur) in tqdm(enumerate(time_data_mapping.items()), desc='Dumping data of selected times', total=len(selected_times)):
        if len(cur) == 0:
            continue
        cur_aia, cur_hmi = cur[0][0], cur[0][1]

        if len(cur_aia) == 0:
            missing_date.append(d)
            continue

        if processed >= max_process:
            break


        cur_selected_time_str = d.strftime("%Y%m%dT%H%M%S")
        # save in FTS files
        aia_out_bn = f"AIA_{cur_selected_time_str}.fits"
        hmi_out_bn = f"HMI_{cur_selected_time_str}.fits"

        if not redo_all:
            if (output_aia_folder / aia_out_bn).exists() and (output_hmi_folder / hmi_out_bn).exists():
                processed += 1
                continue

        cur_hmi_index = None
        cur_aia_index = None
        try:
            cur_hmi_index = selected_hmi_indices[d][0]
            cur_aia_index = selected_aia_indices[d][0]

            cur_hmi_data = cur_hmi.compute()
            cur_hmi_header = {}
            for k,v in data_hmi.attrs.items():
                try:
                    cur_hmi_header[k] = v[cur_hmi_index]
                except IndexError as e:
                    cur_hmi_header[k] = v[-1]

            cur_aia_data = np.squeeze(da.stack(cur_aia).compute())
            cur_aia_header = {}
            for k,v in data_aia.attrs.items():
                try:
                    cur_aia_header[k] = v[cur_aia_index]
                except IndexError as e:
                    cur_aia_header[k] = v[-1]
        
        except Exception as e:
            if cur_aia_index is not None:
                print(f"Error: {e}-> cur_aia: {cur_aia_index}")
            elif cur_hmi_index is not None:
                print(f"Error: {e}-> cur_hmi: {cur_hmi_index}")
            else:
                print(f"Error: {e}-> cur_aia: {cur_aia_index}, cur_hmi: {cur_hmi_index} ...")
            print(f"Number of entries in data_aia.attrs['T_OBS']: {len(data_aia.attrs['T_OBS'])}")
            print(f"Number of entries in data_aia: {len(data_aia)}")
            print(f"Number of entries in data_hmi.attrs['T_OBS']: {len(data_hmi.attrs['T_OBS'])}")
            print(f"Number of entries in data_hmi: {len(data_hmi)}")
            print(selected_hmi_indices)

            raise e
    
        hmi_map = sunpy.map.Map((cur_hmi_data, cur_hmi_header))
        aia_map = sunpy.map.Map((cur_aia_data, cur_aia_header))

        if save:

            if (not (output_aia_folder / aia_out_bn).exists()) or redo_all:
                aia_map.save(output_aia_folder / aia_out_bn, overwrite=True)
            if (not (output_hmi_folder / hmi_out_bn).exists()) or redo_all:
                hmi_map.save(output_hmi_folder / hmi_out_bn, overwrite=True)


        if show:
            fig = plt.figure(figsize=(10, 5))
            ax1 = fig.add_subplot(1, 2, 1, projection=aia_map)
            ax2 = fig.add_subplot(1, 2, 2, projection=hmi_map)
            aia_map.plot(axes=ax1, title=f"AIA {aia_map.meta['wavelnth']} {aia_map.meta['date-obs']}")
            hmi_map.plot(axes=ax2, title=f"HMI {hmi_map.meta['date-obs']}")
            plt.show()

        processed += 1




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--year", type=int, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("-r", "--redo_all", type=bool, default=False)

    args = parser.parse_args()

    main(args.year, args.output_dir, args.redo_all)
