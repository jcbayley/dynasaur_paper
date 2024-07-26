import marimo

__generated_with = "0.7.12"
app = marimo.App(width="medium")


@app.cell
def __():
    import h5py
    import os
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.mplot3d import Axes3D

    import scipy
    import torch
    import massdynamics
    import json
    import pandas as pd
    from massdynamics import create_model
    from massdynamics.data_generation import data_generation
    from scipy.interpolate import interp1d
    from scipy.stats import gaussian_kde
    import brokenaxes as ba
    import corner
    return (
        Axes3D,
        GridSpec,
        ba,
        corner,
        create_model,
        data_generation,
        gaussian_kde,
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
    fig_m, ax_m = plt.subplots()
    ax_m.plot(rand_model[1][0,0,0],rand_model[1][0,0,1], marker=".", ls="-")
    ax_m.plot(rand_model[1][0,1,0],rand_model[1][0,1,1], marker=".", ls="-")
    return ax_m, fig_m


@app.cell
def __(json, os):
    root_dir="/Users/joebayley/projects/massdynamics_project/results/circular/test_2mass_timeseries32_2d_3det_nowindow_sr32_period1_2_transformer_1_masstriangle_unhanded"
    with open(os.path.join(root_dir, 'config.json'), 'r') as _f:
        config = json.load(_f)
    return config, root_dir


@app.cell
def __(os, root_dir):
    data_dir = os.path.join(root_dir, 'testout_2', 'data_output')
    data_index = 5
    data_files = os.listdir(data_dir)
    fname = os.path.join(data_dir, data_files[data_index])
    return data_dir, data_files, data_index, fname


@app.cell
def __(fname, h5py, np):
    data = {}
    with h5py.File(fname, "r") as f:
        print(f.keys())
        for key in f.keys():
            data[key] = np.array(f[key])

    data["recon_velocities"] = np.gradient(data["recon_timeseries"], axis=-1)
    data["source_velocities"] = np.gradient(data["source_timeseries"], axis=-1)
    return data, f, key


@app.cell
def __(data, np):
    # Assuming you have your data in the array `data`
    Nsamples, Nobjects, Ndimensions, Ntimesamples = data["recon_timeseries"].shape
    _, Ndetectors, _ = data["recon_strain"].shape
    # Let's assume you want to plot the second time sample
    time_index = 17  # Change this to the time step you want to plot
    times = np.linspace(0,1,np.shape(data["source_strain"])[-1])
    return (
        Ndetectors,
        Ndimensions,
        Nobjects,
        Nsamples,
        Ntimesamples,
        time_index,
        times,
    )


@app.cell
def __(data, np):
    strain_quantiles = np.quantile(data["recon_strain"][:], [0.1, 0.5, 0.9], axis=0)
    return strain_quantiles,


@app.cell
def __(Ndetectors, config, data, plt):
    # Plot the data
    fig_ts, ax_ts = plt.subplots()
    for i1 in range(Ndetectors):
        ax_ts.plot(data["source_strain"][i1].T, label=config["detectors"][i1])
    ax_ts.set_xlabel('Time sample')
    ax_ts.set_ylabel('Strain')
    ax_ts.legend()
    plt.show()
    return ax_ts, fig_ts, i1


@app.cell
def __(Nobjects, data, plt, time_index):
    # Plot the data
    fig_sa, ax_sa = plt.subplots()
    colors=["C0", "C1"]
    for i in range(Nobjects):
        ax_sa.scatter(
            data["recon_timeseries"][:,i,0,time_index], 
            data["recon_timeseries"][:,i,1,time_index],
            color=colors[i], label=f'Mass {i+1}', alpha=0.01)
    ax_sa.set_xlabel('X-axis')
    ax_sa.set_ylabel('Y-axis')
    ax_sa.legend()
    plt.show()
    return ax_sa, colors, fig_sa, i


@app.cell
def __(data, gaussian_kde, np, time_index):
    max_pos_val = 0.2#np.max(np.abs(data["recon_timeseries"][:,:,:,time_index]))
    xvals,yvals = np.meshgrid(np.linspace(
                                -max_pos_val,
                                max_pos_val,
                                100),
                              np.linspace(
                                -max_pos_val,
                                max_pos_val,
                                100))
    positions = np.vstack([xvals.ravel(), yvals.ravel()])
    n_kde_samples = -1
    kde_m1 = gaussian_kde(np.array([
        data["recon_timeseries"][:n_kde_samples,0,0,time_index], 
        data["recon_timeseries"][:n_kde_samples,0,1,time_index]]))
    eval_kde_m1 = kde_m1.evaluate(positions)
    kde_m2 = gaussian_kde(np.array([
        data["recon_timeseries"][:n_kde_samples,1,0,time_index], 
        data["recon_timeseries"][:n_kde_samples,1,1,time_index]]))
    eval_kde_m2 = kde_m2.evaluate(positions)
    return (
        eval_kde_m1,
        eval_kde_m2,
        kde_m1,
        kde_m2,
        max_pos_val,
        n_kde_samples,
        positions,
        xvals,
        yvals,
    )


@app.cell
def __(eval_kde_m1, eval_kde_m2, plt, xvals, yvals):
    fig_su = plt.figure(figsize=(10, 10))
    ax_su = fig_su.add_subplot(111, projection='3d')
    ax_su.plot_surface(xvals, yvals, eval_kde_m1.reshape(len(xvals), len(yvals)), cmap='Blues', alpha=0.5)
    ax_su.plot_surface(xvals, yvals, eval_kde_m2.reshape(len(xvals), len(yvals)), cmap='Reds', alpha=0.5)

    ax_su.view_init(elev=40, azim=10)
    fig_su
    return ax_su, fig_su


@app.cell
def __(
    config,
    data,
    eval_kde_m1,
    eval_kde_m2,
    matplotlib,
    plt,
    strain_quantiles,
    time_index,
    times,
    xvals,
    yvals,
):
    fig_co = plt.figure(figsize=(10, 10))
    ax_co = fig_co.add_subplot(211)
    ax_st = fig_co.add_subplot(212)

    contour1 = ax_co.contour(xvals, yvals, eval_kde_m1.reshape(len(xvals), len(yvals)), levels=4, cmap='Blues', label="Mass 1", zorder=0)
    contour2 = ax_co.contour(xvals, yvals, eval_kde_m2.reshape(len(xvals), len(yvals)), levels=4, cmap='Reds', label="Mass 2", zorder=0)
    ax_co.autoscale(False) # To avoid that the scatter changes limits

    sc1 = ax_co.scatter(data["source_timeseries"][0,0,time_index], data["source_timeseries"][0,1,time_index], color="k", label="True mass 1", marker="o", s=60,zorder=1)

    sc2 = ax_co.scatter(data["source_timeseries"][1,0,time_index], data["source_timeseries"][1,1,time_index], color="k", label="True mass 2", marker="*", s=60, zorder=1)

    ax_co.set_xlabel("X position")
    ax_co.set_ylabel("Y position")

    legend_elements = [matplotlib.lines.Line2D([0], [0], color=color, lw=2, label=f'Mass {mind+1}') for mind, color in enumerate(["C3", "C0"])]
    legend_elements += [sc1, sc2]

    ax_co.set_aspect('equal', 'box')

    ax_co.legend(handles=legend_elements,  loc='upper left')

    for dind, det in enumerate(config["detectors"]):
        ax_st.plot(times, data["source_strain"][dind], color="k")
        ax_st.plot(times, strain_quantiles[1,dind], color=f"C{dind}", label=det)
        ax_st.fill_between(times, strain_quantiles[0,dind], strain_quantiles[2,dind], alpha=0.5, color=f"C{dind}")

    ax_st.set_xlabel("Time [s]")
    ax_st.set_ylabel("Strain")
    ax_st.legend()
    #ax_co.legend()
    fig_co
    return (
        ax_co,
        ax_st,
        contour1,
        contour2,
        det,
        dind,
        fig_co,
        legend_elements,
        sc1,
        sc2,
    )


@app.cell
def __(data, plt, time_index):
    fig_v, ax_v = plt.subplots()
    n_v_samples = 20
    ar_scale=0.5
    head_width=4
    ax_v.scatter(data["recon_timeseries"][:n_v_samples, 0,0,time_index],
                 data["recon_timeseries"][:n_v_samples, 0,1,time_index],
                 color='C0', alpha=0.3, s=30, label='recovered (m1)')
    ax_v.scatter(data["recon_timeseries"][:n_v_samples, 1,0,time_index],
                 data["recon_timeseries"][:n_v_samples, 1,1,time_index],
                 color='C1', alpha=0.3, s=30, label='recovered (m2)')

        # plot reconstruted sample vector arrows
    ax_v.quiver(data["recon_timeseries"][:n_v_samples, 0,0,time_index],
                data["recon_timeseries"][:n_v_samples, 0,1,time_index],
                data["recon_velocities"][:n_v_samples, 0,0,time_index],
                data["recon_velocities"][:n_v_samples, 0,1,time_index],
                color='C0', alpha=0.7, scale=ar_scale, headwidth=head_width)

    ax_v.quiver(data["recon_timeseries"][:n_v_samples, 1,0,time_index],
                data["recon_timeseries"][:n_v_samples, 1,1,time_index],
                data["recon_velocities"][:n_v_samples, 1,0,time_index],
                data["recon_velocities"][:n_v_samples, 1,1,time_index],
                color='C1', alpha=0.7, scale=ar_scale, headwidth=head_width)

    sc1_v = ax_v.scatter(data["source_timeseries"][0,0,time_index], data["source_timeseries"][0,1,time_index], color="k", label="True mass 1", marker="o", s=60,zorder=1)

    sc2_v = ax_v.scatter(data["source_timeseries"][1,0,time_index], data["source_timeseries"][1,1,time_index], color="k", label="True mass 2", marker="*", s=60, zorder=1)

    ax_v.quiver(data["source_timeseries"][0,0,time_index],
                data["source_timeseries"][0,1,time_index],
                data["source_velocities"][0,0,time_index],
                data["source_velocities"][0,1,time_index],
                color='k', alpha=0.7, scale=ar_scale, headwidth=head_width)

    ax_v.quiver(data["source_timeseries"][1,0,time_index],
                data["source_timeseries"][1,1,time_index],
                data["source_velocities"][1,0,time_index],
                data["source_velocities"][ 1,1,time_index],
                color='k', alpha=0.7, scale=ar_scale, headwidth=head_width)

    ax_v.set_xlabel("X position")
    ax_v.set_ylabel("Y position")

    ax_v.set_aspect('equal', 'box')

    ax_v.legend( loc='upper left')

    ax_v.set_xlim([-0.2,0.2])
    ax_v.set_ylim([-0.2,0.2])

    fig_v
    return ar_scale, ax_v, fig_v, head_width, n_v_samples, sc1_v, sc2_v


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
