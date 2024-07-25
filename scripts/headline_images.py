import marimo

__generated_with = "0.7.9"
app = marimo.App(width="medium")


@app.cell
def __():
    import h5py
    import os
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import scipy
    import torch
    import massdynamics
    import json
    import pandas as pd
    from massdynamics import create_model
    from massdynamics.data_generation import data_generation
    from scipy.interpolate import interp1d
    import brokenaxes as ba
    import corner
    return (
        GridSpec,
        ba,
        corner,
        create_model,
        data_generation,
        h5py,
        interp1d,
        json,
        massdynamics,
        matplotlib,
        np,
        os,
        pd,
        plt,
        scipy,
        torch,
    )


@app.cell
def __(data_generation, np):
    rand_model = data_generation.generate_data(
        2, 
        128, 
                2, 
        128, 
        3, 
        detectors=["H1", "L1", "V1"], 
        window="none", 
        window_acceleration="hann", 
        basis_type="timeseries",
        data_type = "random-uniform",
        fourier_weight=0.4,
        coordinate_type="cartesian",
        prior_args={
            "cycles_min": 2,
            "cycles_max": 2,
            "sky_position":(np.pi, np.pi/2)
        })
    return rand_model,


@app.cell
def __(plt, rand_model):
    fig, ax = plt.subplots()
    ax.plot(rand_model[1][0,0,0],rand_model[1][0,0,1], marker=".", ls="-")
    ax.plot(rand_model[1][0,1,0],rand_model[1][0,1,1], marker=".", ls="-")
    return ax, fig


@app.cell
def __(json, os):
    root_dir = '/Users/joebayley/projects/massdynamics_project/results/random_noise/test_2mass_fourier32_2d_3det_windowcoeffsstrain_sr32_transformer_3_masstriangle_snr20_rt2/'
    with open(os.path.join(root_dir, 'config.json'), 'r') as _f:
        config = json.load(_f)
    return config, root_dir


@app.cell
def __(os, root_dir):
    data_dir = os.path.join(root_dir, 'testout_2', 'data_output')
    data_index = 35
    data_files = os.listdir(data_dir)
    fname = os.path.join(data_dir, data_files[data_index])
    return data_dir, data_files, data_index, fname


@app.cell
def __(fname, pd):
    df = pd.read_hdf(fname)
    return df,


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
