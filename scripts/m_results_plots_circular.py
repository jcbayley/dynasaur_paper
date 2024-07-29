import marimo

__generated_with = "0.7.12"
app = marimo.App(width="full")


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
    from massdynamics import create_model
    from scipy.interpolate import interp1d
    import brokenaxes as ba
    import corner
    from plotting import plot_motions_and_strain
    #plt.rcParams['text.usetex'] = True
    return (
        GridSpec,
        ba,
        corner,
        create_model,
        h5py,
        interp1d,
        json,
        massdynamics,
        matplotlib,
        np,
        os,
        plot_motions_and_strain,
        plt,
        scipy,
        torch,
    )


@app.cell
def __():
    save_plots=False
    return save_plots,


@app.cell
def __():
    #root_dir = "/Users/joebayley/projects/massdynamics_project/results/circular/test_2mass_timeseries32_2d_3det_nowindow_sr32_period1_2_transformer_1_masstriangle_unhanded"
    root_dir = "/home/jf/projects/massdynamics/results/circular/test_2mass_timeseries32_2d_3det_nowindow_sr32_period1_2_transformer_1_masstriangle_unhanded"
    return root_dir,


@app.cell
def __(json, os, root_dir):
    with open(os.path.join(root_dir, 'config.json'), 'r') as _f:
        config = json.load(_f)
    return config,


@app.cell
def __(os, root_dir, torch):
    weights = torch.load(os.path.join(root_dir, 'test_model.pt'), map_location='cpu')
    return weights,


@app.cell
def __(config, create_model):
    (model, pre_model) = create_model.create_models(config, device='cpu')
    return model, pre_model


@app.cell
def __(os, root_dir):
    data_dir = os.path.join(root_dir, 'testout_2', 'data_output')
    data_index = 2
    return data_dir, data_index


@app.cell
def __(data_dir, h5py, interp1d, np, os):
    data_files = os.listdir(data_dir)
    r_recon_strain = []
    r_source_strain = []
    r_recon_timeseries = []
    r_source_timeseries = []
    recon_masses = []
    source_masses = []
    for k, fname in enumerate(data_files[:50]):
        fpath = os.path.join(data_dir, fname)
        with h5py.File(fpath, 'r') as _f:
            if k == 0:
                print(_f.keys())
            r_recon_strain.append(np.array(_f['recon_strain']))
            r_source_strain.append(np.array(_f['source_strain']))
            r_recon_timeseries.append(np.array(_f['recon_timeseries']))
            r_source_timeseries.append(np.array(_f['source_timeseries']))
            recon_masses.append(np.array(_f['recon_masses']))
            source_masses.append(np.array(_f['source_masses']))


    r_recon_strain = np.array(r_recon_strain)
    r_source_strain = np.array(r_source_strain)
    r_recon_timeseries = np.array(r_recon_timeseries)
    r_source_timeseries = np.array(r_source_timeseries)
    recon_masses = np.array(recon_masses)
    source_masses = np.array(source_masses)
    r_recon_velocities = np.gradient(r_recon_timeseries, axis=-1)
    r_source_velocities = np.gradient(r_source_timeseries, axis=-1)

    r_times = np.linspace(0, 1, np.shape(r_source_strain)[-1])
    interp_times = np.linspace(0, 1, 128)
    strain_fn = interp1d(r_times, r_source_strain, kind='cubic')
    source_strain = strain_fn(interp_times)
    dyn_fn = interp1d(r_times, r_source_timeseries, kind='cubic')
    source_timeseries = dyn_fn(interp_times)
    vel_fn = interp1d(r_times, r_source_velocities, kind='cubic')
    source_velocities = vel_fn(interp_times)
    strain_fn = interp1d(r_times, r_recon_strain, kind='cubic')
    recon_strain = strain_fn(interp_times)
    dyn_fn = interp1d(r_times, r_recon_timeseries, kind='cubic')
    recon_timeseries = dyn_fn(interp_times)
    vel_fn = interp1d(r_times, r_recon_velocities, kind='cubic')
    recon_velocities = vel_fn(interp_times)
    return (
        data_files,
        dyn_fn,
        fname,
        fpath,
        interp_times,
        k,
        r_recon_strain,
        r_recon_timeseries,
        r_recon_velocities,
        r_source_strain,
        r_source_timeseries,
        r_source_velocities,
        r_times,
        recon_masses,
        recon_strain,
        recon_timeseries,
        recon_velocities,
        source_masses,
        source_strain,
        source_timeseries,
        source_velocities,
        strain_fn,
        vel_fn,
    )


@app.cell
def __(source_masses):
    source_masses.shape
    return


@app.cell
def __(np):
    def cartesian_to_polar(xy_positions):
        """
        Convert Cartesian coordinates to polar coordinates.

        Parameters:
        - xy_positions: numpy array of shape (Ndata, N_masses, N_dimensions, Nsamples)
          where N_dimensions is expected to be 2 (x and y positions).

        Returns:
        - radii: numpy array of shape (Ndata, N_masses, Nsamples)
          representing the radial distances.
        - angles: numpy array of shape (Ndata, N_masses, Nsamples)
          representing the angles in radians.
        """
        x = xy_positions[..., 0, :]
        y = xy_positions[..., 1, :]
        radii = np.sqrt(x ** 2 + y ** 2)
        angles = np.arctan2(y, x)
        return (radii, angles)

    def compute_xy_rmse(recon_timeseries, source_timeseries):
        difference = np.array(recon_timeseries) - np.array(source_timeseries)[:, np.newaxis, :, :]
        max_source_strain = np.max(np.abs(source_timeseries), axis=2)[:, np.newaxis, :, np.newaxis]
        rmse = np.sqrt((difference / max_source_strain) ** 2)
        median_rmse = np.nanmedian(rmse, axis=-1)
        return median_rmse

    def compute_strain_rmse(recon_strain, source_strain):
        difference = np.array(recon_strain) - np.array(source_strain)[:, np.newaxis, :, :]
        max_source_strain = np.max(np.abs(source_strain), axis=2)[:, np.newaxis, :, np.newaxis]
        rmse = np.sqrt((difference / max_source_strain) ** 2)
        median_rmse = np.nanmedian(rmse, axis=-1)
        return median_rmse
    return cartesian_to_polar, compute_strain_rmse, compute_xy_rmse


@app.cell
def __(cartesian_to_polar, recon_timeseries, source_timeseries):
    (recon_radii, recon_angles) = cartesian_to_polar(recon_timeseries)
    (source_radii, source_angles) = cartesian_to_polar(source_timeseries)
    return recon_angles, recon_radii, source_angles, source_radii


@app.cell
def __(np, recon_angles, recon_radii, source_angles, source_radii):
    diff_radii = recon_radii - source_radii[:, np.newaxis]
    diff_angles = np.mod(recon_angles - source_angles[:, np.newaxis] + np.pi, 2*np.pi) - np.pi
    return diff_angles, diff_radii


@app.cell
def __(
    compute_xy_rmse,
    recon_angles,
    recon_radii,
    source_angles,
    source_radii,
):
    rmse_radii = compute_xy_rmse(recon_radii, source_radii)
    rmse_angles = compute_xy_rmse(recon_angles, source_angles)
    return rmse_angles, rmse_radii


@app.cell
def __(data_index, np, rmse_radii):
    mean_radii = np.mean(rmse_radii)
    dind_radii = np.mean(rmse_radii[data_index])
    print(f"mean rmse radii :{mean_radii}, data rmse radii: {dind_radii}")
    return dind_radii, mean_radii


@app.cell
def __(data_index, np, rmse_angles):
    mean_angles = np.mean(rmse_angles)
    dind_angles = np.mean(rmse_angles[data_index])
    print(f"mean rmse angles :{mean_angles}, data rmse angles: {dind_angles}")
    return dind_angles, mean_angles


@app.cell
def __(GridSpec, ba, diff_angles, diff_radii, np, plt):
    fig_rarmse = plt.figure()
    gs_rarmse = GridSpec(2, 2, hspace=0.5)

    l_fontsize = 20
    ax_s1_rarmse = fig_rarmse.add_subplot(gs_rarmse[0, 0])
    ax_s2_rarmse = fig_rarmse.add_subplot(gs_rarmse[1, 0])

    plot_ind = 0
    rbins = np.linspace(-0.05, 0.05, 50)
    ax_s1_rarmse.hist(np.ravel(diff_radii[plot_ind,:,0]), bins=rbins, density=True, alpha=0.8)
    #ax_s1.axvline(np.mean(source_radii[plot_ind,0]), color="r")
    ax_s1_rarmse.axvline(0, color="C3")
    #ax[0,0].set_xlim([0.,0.5])
    ax_s2_rarmse.hist(np.ravel(diff_radii[plot_ind,:,1]), bins=rbins , density=True, alpha=0.8)
    #ax_s2.axvline(np.mean(source_radii[plot_ind,1]), color="r")
    ax_s2_rarmse.axvline(0, color="C3")
    #ax[0,1].set_xlim([0.,0.7])
    ax_s1_rarmse.set_xlabel("Radius difference")
    ax_s2_rarmse.set_xlabel("Radius difference")
    ax_s1_rarmse.set_xlim([-0.03,0.01])
    ax_s2_rarmse.set_xlim([-0.01,0.03])


    diff_angles[diff_angles < -1] += 2*np.pi 
    bax1 = ba.BrokenAxes(xlims=((-0.03, 0.05), (3.1, 3.18)), hspace=0.1, subplot_spec=gs_rarmse[0,1])
    bax1.hist(np.ravel(diff_angles[plot_ind,:,0] ), bins=800 , density=True, alpha=0.8)
    #ax[0].axvline(np.mean(source_angles[plot_ind,0]), color="r")
    bax1.axvline(0, color="C3")
    #ax[0].set_xlim([0.,0.5])
    bax2 = ba.BrokenAxes(xlims=((-0.03, 0.05), (3.1, 3.18)), hspace=0.1, subplot_spec=gs_rarmse[1,1])
    bax2.hist(np.ravel(diff_angles[plot_ind,:,1] ), bins=800 , density=True, alpha=0.8)
    #ax[1].axvline(np.mean(source_angles[plot_ind,1]), color="r")
    bax2.axvline(0, color="C3")#
    #ax[1,1].set_xlim([-0.05,0.05])
    bax1.set_xlabel("Angle difference [radians]", labelpad=30.5)
    bax2.set_xlabel("Angle difference [radians]", labelpad=30)
    bax1.tick_params(axis='x', labelrotation=45)
    bax2.tick_params(axis='x', labelrotation=45)
    bax1.set_yticklabels([])
    bax2.set_yticklabels([])
    ax_s2_rarmse.set_yticklabels([])
    ax_s1_rarmse.set_yticklabels([])
    ax_s1_rarmse.set_ylabel("Mass 1")
    ax_s2_rarmse.set_ylabel("Mass 2")
    #ax_s1.spines[['right', 'top']].set_visible(False)
    bax1.spines['top'][0].set_visible(True)
    bax1.spines['top'][1].set_visible(True)
    bax1.spines['right'][0].set_visible(True)
    bax2.spines['top'][0].set_visible(True)
    bax2.spines['top'][1].set_visible(True)
    bax2.spines['right'][0].set_visible(True)
    #fig.savefig("../paper/circular_radius_anglediff.pdf", bbox_inches="tight")
    fig_rarmse
    return (
        ax_s1_rarmse,
        ax_s2_rarmse,
        bax1,
        bax2,
        fig_rarmse,
        gs_rarmse,
        l_fontsize,
        plot_ind,
        rbins,
    )


@app.cell
def __(fig_rarmse, save_plots):
    if save_plots:
        fig_rarmse.savefig("./scripts/figures/circular_radius_anglediff.pdf", bbox_inches="tight")
    return


@app.cell
def __(compute_strain_rmse, recon_strain, source_strain):
    rmse = compute_strain_rmse(recon_strain, source_strain)
    return rmse,


@app.cell
def __(np, rmse):
    np.median(rmse)
    return


@app.cell
def __(data_index, np, rmse):
    np.mean(rmse[data_index])
    return


@app.cell
def __(np, plt, rmse):
    (fig_rmse, rmse_ax) = plt.subplots(nrows=2, figsize=(7, 13), gridspec_kw={'height_ratios': [4, 1]})
    (pmin, pmax) = (-3, 0.5)
    rmse_vp = rmse_ax[0].boxplot(np.log10(np.mean(rmse[:, :, :], axis=-1).T), patch_artist=True, notch=True, vert=False, widths=0.8, boxprops=dict(facecolor='C0', color='C0'), medianprops=dict(color='red'), whiskerprops=dict(color='C0'), capprops=dict(color='C0'))
    for flier in rmse_vp['fliers']:
        flier.set(marker='.', ms=1, color='#e7298a', alpha=0.5)
    det = ['H1', 'L1', 'V1']
    for _i in range(3):
        rmse_ax[1].hist(np.concatenate(np.log10(rmse), 0)[:, _i], bins=50, range=(pmin, pmax), label=det[_i], histtype='step', lw=2, density=True)
    rmse_ax[1].set_xlabel('Log10[RMSE of strain]', fontsize=17)
    rmse_ax[0].set_ylabel('Data index', fontsize=17)
    rmse_ax[0].set_xlim([pmin, pmax])
    rmse_ax[1].set_xlim([pmin, pmax])
    rmse_ax[1].legend(fontsize=15)
    rmse_ax[0].set_xticklabels([])
    rmse_ax[0].tick_params(axis='both', labelsize=12)
    rmse_ax[1].tick_params(axis='both', labelsize=12)
    plt.subplots_adjust(hspace=0)
    fig_rmse
    return det, fig_rmse, flier, pmax, pmin, rmse_ax, rmse_vp


@app.cell
def __(
    GridSpec,
    data_index,
    matplotlib,
    np,
    plt,
    r_recon_strain,
    r_source_strain,
    r_times,
    recon_strain,
    recon_timeseries,
    recon_velocities,
    source_strain,
    source_timeseries,
    source_velocities,
):
    motion_fig = plt.figure(figsize=(10, 15))
    ########
    # Set parameters
    #######
    motion_detector = 0
    motion_fontsize = 20
    (_tstart, _tend) = (1, -1)
    axlim = 0.2
    #########
    # setup the grid
    #############
    motion_gs = GridSpec(4, 6, height_ratios=[2.5, 2, 1, 2.0], hspace=0.5)
    motion_ax_s1 = motion_fig.add_subplot(motion_gs[0, 0:3])
    motion_ax_s2 = motion_fig.add_subplot(motion_gs[0, 3:6])
    motion_axs = [motion_ax_s1, motion_ax_s2]
    motion_ax_l = motion_fig.add_subplot(motion_gs[1, :])
    motion_ax_ld = motion_fig.add_subplot(motion_gs[2, :])
    motion_ax_a1 = motion_fig.add_subplot(motion_gs[3, 0:2])
    motion_ax_a2 = motion_fig.add_subplot(motion_gs[3, 2:4])
    motion_ax_a3 = motion_fig.add_subplot(motion_gs[3, 4:6])
    motion_axa = [motion_ax_a1, motion_ax_a2, motion_ax_a3]

    ############
    # Set axis limits and lavels for motion and strain
    ##############
    # motion
    for i in range(3):
        motion_axa[i].set_xlabel('$x$ position', fontsize=motion_fontsize)
        if i == 0:
            motion_axa[i].set_ylabel('$y$ position', fontsize=motion_fontsize)
        slim = axlim
        motion_axa[i].set_xlim([-slim, slim])
        motion_axa[i].set_ylim([-slim, slim])
    for i in range(2):
        motion_axs[i].tick_params(axis='both', labelsize=12)
        tlim = axlim
        motion_axs[i].set_xlim([-tlim, tlim])
        motion_axs[i].set_ylim([-tlim, tlim])
        motion_axs[i].set_xlabel('$x$ position', fontsize=motion_fontsize)
        motion_axs[i].set_ylabel('$y$ position', fontsize=motion_fontsize)

    # strain
    motion_ax_l.tick_params(axis='both', labelsize=12)
    motion_ax_ld.set_xlabel('Time [s]', fontsize=motion_fontsize)
    motion_ax_l.set_ylabel('$h^{\\rm{H}}_{\\rm{true}}$', fontsize=motion_fontsize)
    motion_ax_ld.set_ylabel('$h^{\\rm{H}}_{\\rm{recon}} - h^{\\rm{H}}_{\\rm{true}}$', fontsize=motion_fontsize)

    ############
    # Compute strain quantiles and plot strain
    ##########
    time = np.linspace(0, 1, len(source_strain[data_index][motion_detector]))
    motion_ax_l.plot(time, source_strain[data_index][motion_detector], color='k', label='true')

    # find and plot quantiles
    motion_qnts = np.quantile(np.array(recon_strain)[data_index, :,motion_detector], [0.1, 0.5, 0.9], axis=0)
    r_motion_qnts = np.quantile(np.array(r_recon_strain)[data_index, :,motion_detector], [0.1, 0.5, 0.9], axis=0)

    motion_ax_l.plot(time, motion_qnts[1], color='C2', label='reconstructed 90% confidence')
    motion_ax_l.fill_between(time, motion_qnts[0], motion_qnts[2], alpha=0.5, color='C2')

    # plot data points
    motion_ax_l.plot(r_times, r_source_strain[data_index][motion_detector], color="k", marker="o", ms=3, ls="none", label="True datapoints")
    yerr_up_l = r_motion_qnts[2] - r_motion_qnts[1]
    yerr_low_l = r_motion_qnts[1] - r_motion_qnts[0]
    #motion_ax_l.errorbar(r_times, r_motion_qnts[1], yerr=[yerr_low_l, yerr_up_l], color="C2", marker="o", ms=3, ls="none", label="Recon datapoints", capsize=3)
    motion_ax_l.plot(r_times, r_motion_qnts[1], color="C2", marker="o", ms=3, ls="none", label="Recon datapoints")

    # residual plot
    motion_ax_ld.plot(time, motion_qnts[1] - source_strain[data_index][motion_detector], color='C2', label='recovered 90% confidence')
    motion_ax_ld.fill_between(time, motion_qnts[0] - source_strain[data_index][motion_detector], motion_qnts[2] - source_strain[data_index][motion_detector], alpha=0.5, color='C2')
    motion_ax_ld.plot(time, source_strain[data_index][motion_detector] - source_strain[data_index][motion_detector], color='k', label='true')
    motion_ax_l.legend()

    # plot residual data points
    motion_ax_ld.plot(r_times, r_source_strain[data_index][motion_detector] - r_source_strain[data_index][motion_detector], color="k", marker="o", ms=3, ls="none", label="True datapoints")
    #motion_ax_ld.errorbar(r_times, r_motion_qnts[1] - r_source_strain[data_index][motion_detector], yerr=[yerr_low_l, yerr_up_l], color="C2", marker="o", ms=3, ls="none", label="Recon datapoints", capsize=3)
    motion_ax_ld.plot(r_times, r_motion_qnts[1] - r_source_strain[data_index][motion_detector], color="C2", marker="o", ms=3, ls="none", label="Recon datapoints")


    ###############
    # Plot the motion at all times
    ################
    motion_sinds = np.arange(recon_timeseries.shape[1])
    #motion_tsteps = np.random.choice(_sinds, 3)

    motion_tsteps = np.array([87, 698])
    for _i in range(2):
        motion_tstep_time = motion_tsteps[_i] / len(source_strain[data_index][motion_detector])
        _width = 3 / 120

        motion_axs[_i].plot(recon_timeseries[data_index, motion_tsteps[_i], 0, 0, _tstart:_tend], recon_timeseries[data_index, motion_tsteps[_i], 0, 1, _tstart:_tend], color='C0', label='recovered (m1)')
        motion_axs[_i].plot(recon_timeseries[data_index, motion_tsteps[_i], 1, 0, _tstart:_tend], recon_timeseries[data_index, motion_tsteps[_i], 1, 1, _tstart:_tend], color='C1', ls='--', label='recovered (m2)')
        motion_axs[_i].plot(source_timeseries[data_index, 1, 0, _tstart:_tend], source_timeseries[data_index, 1, 1, _tstart:_tend], color='k', label='true (m2)')
        motion_axs[_i].plot(source_timeseries[data_index, 0, 0, _tstart:_tend], source_timeseries[data_index, 0, 1, _tstart:_tend], color='k', ls='--', label='true (m1)')
        motion_axs[_i].scatter(recon_timeseries[data_index, motion_tsteps[_i], 0, 0, -1], recon_timeseries[data_index, motion_tsteps[_i], 0, 1, -1], color='C0', s=20)
        motion_axs[_i].scatter(recon_timeseries[data_index, motion_tsteps[_i], 1, 0, -1], recon_timeseries[data_index, motion_tsteps[_i], 1, 1, -1], color='C1', s=20)
        motion_axs[_i].scatter(source_timeseries[data_index, 1, 0, -1], source_timeseries[data_index, 1, 1, -1], color='k', s=20)
        motion_axs[_i].scatter(source_timeseries[data_index, 0, 0, -1], source_timeseries[data_index, 0, 1, -1], color='k', s=20)

        motion_axs[_i].set_aspect('equal', 'box')
        motion_axs[_i].set_aspect('equal', 'box')


    ##############
    # Plot the motion at a single point in time
    ################
    motion_tsteps = np.array(np.array([0.15, 0.4, 0.6]) * np.shape(recon_timeseries)[-1]).astype(int)
    nsamples = 20
    ar_scale = 0.5
    for _i in range(3):
        _tstep_time = motion_tsteps[_i] / len(source_strain[data_index][motion_detector])
        motion_ax_l.axvline(_tstep_time, color='r', lw=2)
        motion_ax_ld.axvline(_tstep_time, color='r', lw=2)
        _width = 3 / 120

        # plot reconstructed sample potition
        motion_axa[_i].scatter(recon_timeseries[data_index, :nsamples, 0, 0, motion_tsteps[_i]], recon_timeseries[data_index, :nsamples, 0, 1, motion_tsteps[_i]], color='C0', alpha=0.3, s=30, label='recovered (m1)')
        motion_axa[_i].scatter(recon_timeseries[data_index, :nsamples, 1, 0, motion_tsteps[_i]], recon_timeseries[data_index, :nsamples, 1, 1, motion_tsteps[_i]], color='C1', alpha=0.3, s=30, marker='*', label='recovered (m2)')
        # plot reconstruted sample vector arrows
        motion_axa[_i].quiver(recon_timeseries[data_index, :nsamples, 0, 0, motion_tsteps[_i]], recon_timeseries[data_index, :nsamples, 0, 1, motion_tsteps[_i]], recon_velocities[data_index, :nsamples, 0, 0, motion_tsteps[_i]], recon_velocities[data_index, :nsamples, 0, 1, motion_tsteps[_i]], color='C0', alpha=0.7, scale=ar_scale, headwidth=5)
        motion_axa[_i].quiver(recon_timeseries[data_index, :nsamples, 1, 0, motion_tsteps[_i]], recon_timeseries[data_index, :nsamples, 1, 1, motion_tsteps[_i]], recon_velocities[data_index, :nsamples, 1, 0, motion_tsteps[_i]], recon_velocities[data_index, :nsamples, 1, 1, motion_tsteps[_i]], color='C1', alpha=0.7, scale=ar_scale, headwidth=5)

        # plot true positions
        motion_axa[_i].plot(source_timeseries[data_index, 1, 0, motion_tsteps[_i]], source_timeseries[data_index, 1, 1, motion_tsteps[_i]], color='k', marker='*', label='true (m2)')
        motion_axa[_i].plot(source_timeseries[data_index, 0, 0, motion_tsteps[_i]], source_timeseries[data_index, 0, 1, motion_tsteps[_i]], color='k', marker='o', label='true (m1)')
        # plot the vector arrows on samples for truth
        motion_axa[_i].quiver(source_timeseries[data_index, 0, 0, motion_tsteps[_i]], source_timeseries[data_index, 0, 1, motion_tsteps[_i]], source_velocities[data_index, 0, 0, motion_tsteps[_i]], source_velocities[data_index, 0, 1, motion_tsteps[_i]], color='k', alpha=0.8, scale=ar_scale, headwidth=5)
        motion_axa[_i].quiver(source_timeseries[data_index, 1, 0, motion_tsteps[_i]], source_timeseries[data_index, 1, 1, motion_tsteps[_i]], source_velocities[data_index, 1, 0, motion_tsteps[_i]], source_velocities[data_index, 1, 1, motion_tsteps[_i]], color='k', alpha=0.8, scale=ar_scale, headwidth=5)

        # make plots square
        motion_axa[_i].set_aspect('equal', 'box')

        # Add arrows between sub plots
        figtr = motion_fig.transFigure.inverted()
        print(_tstep_time)
        ptB = figtr.transform(motion_ax_ld.transData.transform((_tstep_time * 1.0 - 0.0, -0.02)))
        ptE = figtr.transform(motion_axa[_i].transData.transform((0.0, axlim)))
        arrow = matplotlib.patches.FancyArrowPatch(ptB, ptE, transform=motion_fig.transFigure, fc='r', arrowstyle='simple', alpha=0.5, mutation_scale=20.0)
        motion_fig.patches.append(arrow)


    ############
    # Define plotting parameters
    ##########
    motion_axa[1].legend(bbox_to_anchor=(0.78,0.05), fancybox=True, ncol=4, bbox_transform=motion_fig.transFigure)
    motion_fig.tight_layout()
    pos10 = motion_axs[0].get_position()
    pos11 = motion_axs[1].get_position()
    pos2 = motion_ax_l.get_position()
    pos3 = motion_ax_ld.get_position()
    motion_ax_l.set_position([pos2.x0, pos3.y1, pos2.width, pos2.height])
    motion_ax_ld.set_position([pos3.x0, pos3.y0, pos3.width, pos3.height])
    motion_axs[0].set_position([pos10.x0, pos10.y0 - 0.05, pos10.width, pos10.height])
    motion_axs[1].set_position([pos11.x0, pos11.y0 - 0.05, pos11.width, pos11.height])
    motion_fig
    return (
        ar_scale,
        arrow,
        axlim,
        figtr,
        i,
        motion_ax_a1,
        motion_ax_a2,
        motion_ax_a3,
        motion_ax_l,
        motion_ax_ld,
        motion_ax_s1,
        motion_ax_s2,
        motion_axa,
        motion_axs,
        motion_detector,
        motion_fig,
        motion_fontsize,
        motion_gs,
        motion_qnts,
        motion_sinds,
        motion_tstep_time,
        motion_tsteps,
        nsamples,
        pos10,
        pos11,
        pos2,
        pos3,
        ptB,
        ptE,
        r_motion_qnts,
        slim,
        time,
        tlim,
        yerr_low_l,
        yerr_up_l,
    )


@app.cell
def __(motion_fig, save_plots):
    if save_plots:
        motion_fig.savefig("./scripts/figures/circular_reconstruct.pdf", bbox_inches="tight")
    return


@app.cell
def __(interp1d, np):
    def interpolate_positions(old_times, new_times, positions):
        """Interpolate between points over dimension 1 """
        interp_dict = np.zeros((positions.shape[0], positions.shape[1], len(new_times)))
        for object_ind in range(positions.shape[0]):
            interp = interp1d(old_times, positions[object_ind], kind='cubic')
            interp_dict[object_ind] = interp(new_times)
        return interp_dict
    return interpolate_positions,


@app.cell
def __(
    data_index,
    interpolate_positions,
    np,
    recon_strain,
    recon_timeseries,
    source_strain,
    source_timeseries,
):
    _times = np.linspace(0, 1, np.shape(recon_timeseries)[-1])
    new_times = np.linspace(0, 1, 128)
    recon_interp = np.array([interpolate_positions(_times, new_times, recon_timeseries[data_index][sind]) for sind in range(np.shape(recon_timeseries)[1])])
    source_interp = interpolate_positions(_times, new_times, source_timeseries[data_index])
    '\nn=128\nfreconwave = np.fft.rfft(recon_strain)\n#freconwave[...,-1:] = 0 + 0j\nrecon_wave_interp = np.fft.irfft(freconwave, n=n) * n/np.shape(recon_strain)[-1]\nfsourcewave = np.fft.rfft(source_strain)\n#fsourcewave[...,-1:] = 0 + 0j\nsource_wave_interp = np.fft.irfft(fsourcewave, n=n) * n/np.shape(recon_strain)[-1]\n'
    recon_wave_interp = interpolate_positions(_times, new_times, recon_strain[data_index])
    source_wave_interp = interpolate_positions(_times, new_times, source_strain)
    return (
        new_times,
        recon_interp,
        recon_wave_interp,
        source_interp,
        source_wave_interp,
    )


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __(np):
    def xyz_to_rthetaphi(x, y, z):
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        return (r, theta, np.unwrap(phi))
    return xyz_to_rthetaphi,


@app.cell
def __(data_index, source_timeseries):
    source_timeseries[data_index].shape
    return


@app.cell
def __(data_index, recon_timeseries, source_timeseries, xyz_to_rthetaphi):
    (r1, theta1, phi1) = xyz_to_rthetaphi(source_timeseries[data_index, :, 0], source_timeseries[data_index, :, 1], source_timeseries[data_index, :, 2])
    (rr, thetar, phir) = xyz_to_rthetaphi(recon_timeseries[data_index, :, :, 0], recon_timeseries[data_index, :, :, 1], recon_timeseries[data_index, :, :, 2])
    return phi1, phir, r1, rr, theta1, thetar


@app.cell
def __(data_index, recon_timeseries):
    recon_timeseries[data_index, :, :, 0].shape
    return


@app.cell
def __(phi1, phir, plt, r1, rr, theta1, thetar):
    (_fig, _ax) = plt.subplots(nrows=3, figsize=(10, 7))
    _ax[0].plot(r1[0], color='k')
    _ax[1].plot(theta1[0], color='k')
    _ax[2].plot(phi1[0], color='k')
    _ax[0].plot(rr[:, 0].T, color='C0', alpha=0.01)
    _ax[1].plot(thetar[:, 0].T, color='C0', alpha=0.01)
    _ax[2].plot(phir[:, 0].T, color='C0', alpha=0.01)
    _ax[2].set_ylim([-20, 20])
    _ax[0].set_ylabel('R')
    _ax[1].set_ylabel('theta')
    _ax[2].set_ylabel('phi')
    _a = 2
    return


@app.cell
def __(phi1, phir, plt, r1, rr, theta1, thetar):
    (_fig, _ax) = plt.subplots(nrows=3, figsize=(10, 7))
    _ax[0].plot(r1[1], color='k')
    _ax[1].plot(theta1[1], color='k')
    _ax[2].plot(phi1[1], color='k')
    _ax[0].plot(rr[:, 1].T, color='C0', alpha=0.01)
    _ax[1].plot(thetar[:, 1].T, color='C0', alpha=0.01)
    _ax[2].plot(phir[:, 1].T, color='C0', alpha=0.01)
    _ax[2].set_ylim([-20, 20])
    _ax[0].set_ylabel('R')
    _ax[1].set_ylabel('theta')
    _ax[2].set_ylabel('phi')
    _a = 2
    return


@app.cell
def __(
    np,
    plt,
    recon_strain,
    recon_timeseries,
    source_strain,
    source_timeseries,
):
    (_fig, _ax) = plt.subplots(nrows=2)
    _ax[0].plot(source_strain[4][0], color='C0')
    qnts = np.quantile(np.array(recon_strain)[4, :, 0], [0.1, 0.5, 0.9], axis=0)
    print(qnts[0].shape)
    _ax[0].plot(np.arange(len(qnts[1])), qnts[1], color='C1')
    _ax[0].fill_between(np.arange(len(qnts[1])), qnts[0], qnts[2], alpha=0.5, color='C1')
    _ax[1].plot(source_timeseries[4, 0, 0].T)
    _ax[1].plot(recon_timeseries[4, :, 0, 0].T, color='C1', alpha=0.01)
    _a = 2
    return qnts,


@app.cell
def __(data_index, recon_timeseries):
    recon_timeseries[data_index, :, 0, 0].shape
    return


@app.cell
def __(data_index, recon_timeseries, scipy):
    rt = recon_timeseries[data_index, :, :, :, 40:80]
    (centroids, mean) = scipy.cluster.vq.kmeans(rt.reshape(rt.shape[0], -1), 4)
    (clusters, distances) = scipy.cluster.vq.vq(rt.reshape(rt.shape[0], -1), centroids)
    return centroids, clusters, distances, mean, rt


@app.cell
def __(centroids, clusters, mean):
    (centroids.shape, mean.shape, clusters.shape)
    return


@app.cell
def __(clusters, data_index, recon_strain, recon_timeseries):
    dind = 4
    cl1 = recon_timeseries[data_index, clusters == 0]
    cl2 = recon_timeseries[data_index, clusters == 1]
    cl3 = recon_timeseries[data_index, clusters == 2]
    cl4 = recon_timeseries[data_index, clusters == 3]
    st1 = recon_strain[data_index, clusters == 0]
    st2 = recon_strain[data_index, clusters == 1]
    st3 = recon_strain[data_index, clusters == 2]
    st4 = recon_strain[data_index, clusters == 3]
    return cl1, cl2, cl3, cl4, dind, st1, st2, st3, st4


@app.cell
def __(cl1, cl2, cl3, cl4, data_index, plt, source_timeseries):
    (_fig, _ax) = plt.subplots()
    _ax.plot(cl1[:, 0, 0].T, color='C0', alpha=0.03)
    _ax.plot(cl2[:, 0, 0].T, color='C1', alpha=0.03)
    _ax.plot(cl3[:, 0, 0].T, color='C2', alpha=0.03)
    _ax.plot(cl4[:, 0, 0].T, color='C3', alpha=0.03)
    _ax.plot(source_timeseries[data_index, 0, 0].T, color='k')
    _a = 2
    return


@app.cell
def __(cl1, cl2, cl4, data_index, plt, source_timeseries):
    (_fig, _ax) = plt.subplots()
    _ax.plot(cl1[:, 0, 1].T, color='C0', alpha=0.03)
    _ax.plot(cl2[:, 0, 1].T, color='C1', alpha=0.03)
    _ax.plot(cl4[:, 0, 1].T, color='C3', alpha=0.03)
    _ax.plot(source_timeseries[data_index, 0, 1].T, color='k')
    _a = 2
    return


@app.cell
def __(data_index, np, source_strain, st1, st2, st3, st4):
    print('st1', np.mean((st1[:, 0] - source_strain[data_index, 0]) ** 2))
    print('st2', np.mean((st2[:, 0] - source_strain[data_index, 0]) ** 2))
    print('st3', np.mean((st3[:, 0] - source_strain[data_index, 0]) ** 2))
    print('st4', np.mean((st4[:, 0] - source_strain[data_index, 0]) ** 2))
    return


@app.cell
def __(cl1, cl2, cl3, cl4, data_index, np, source_timeseries):
    print('clmin1', np.argmin(np.mean((cl1 - source_timeseries[data_index]) ** 2, axis=(1, 2, 3))))
    print('clmin2', np.argmin(np.mean((cl2 - source_timeseries[data_index]) ** 2, axis=(1, 2, 3))))
    print('clmin3', np.argmin(np.mean((cl3 - source_timeseries[data_index]) ** 2, axis=(1, 2, 3))))
    print('clmin4', np.argmin(np.mean((cl4 - source_timeseries[data_index]) ** 2, axis=(1, 2, 3))))
    return


@app.cell
def __(cl1, cl2, cl3, cl4, data_index, np, source_timeseries):
    print('cl1', np.mean((cl1 - source_timeseries[data_index]) ** 2))
    print('cl2', np.mean((cl2 - source_timeseries[data_index]) ** 2))
    print('cl3', np.mean((cl3 - source_timeseries[data_index]) ** 2))
    print('cl4', np.mean((cl4 - source_timeseries[data_index]) ** 2))
    return


@app.cell
def __(
    cl1,
    cl2,
    data_index,
    plt,
    source_strain,
    source_timeseries,
    st1,
    st2,
):
    (_fig, _ax) = plt.subplots(nrows=2, ncols=4, figsize=(15, 8))
    sst = 187
    sed = sst + 1
    alpha = 1.0
    _ax[0, 0].plot(st1[sst:sed, 0].T, color='C0', alpha=alpha)
    _ax[1, 0].plot(st2[sst:sed, 0].T, color='C1', alpha=alpha)
    _ax[0, 0].plot(source_strain[data_index, 0].T, color='k')
    _ax[1, 0].plot(source_strain[data_index, 0].T, color='k')
    _ax[0, 0].set_ylim([-0.2, 0.2])
    _ax[1, 0].set_ylim([-0.2, 0.2])
    _ax[0, 1].plot(cl1[sst:sed, 0, 0].T, color='C0', alpha=alpha)
    _ax[1, 1].plot(cl2[sst:sed, 0, 0].T, color='C1', alpha=alpha)
    _ax[0, 1].plot(source_timeseries[data_index, 0, 0].T, color='k')
    _ax[1, 1].plot(source_timeseries[data_index, 0, 0].T, color='k')
    _ylim = 0.3
    _ax[0, 1].set_ylim([-_ylim, _ylim])
    _ax[1, 1].set_ylim([-_ylim, _ylim])
    _ax[0, 2].plot(cl1[sst:sed, 0, 1].T, color='C0', alpha=alpha)
    _ax[1, 2].plot(cl2[sst:sed, 0, 1].T, color='C1', alpha=alpha)
    _ax[0, 2].plot(source_timeseries[data_index, 0, 1].T, color='k')
    _ax[1, 2].plot(source_timeseries[data_index, 0, 1].T, color='k')
    _ylim = 0.3
    _ax[0, 2].set_ylim([-_ylim, _ylim])
    _ax[1, 2].set_ylim([-_ylim, _ylim])
    _ax[0, 3].plot(cl1[sst:sed, 0, 2].T, color='C0', alpha=alpha)
    _ax[1, 3].plot(cl2[sst:sed, 0, 2].T, color='C1', alpha=alpha)
    _ylim = 0.3
    _ax[0, 3].set_ylim([-_ylim, _ylim])
    _ax[1, 3].set_ylim([-_ylim, _ylim])
    _ax[0, 3].plot(cl1[sst:sed, 1, 2].T, color='C0', ls='--', alpha=alpha)
    _ax[1, 3].plot(cl2[sst:sed, 1, 2].T, color='C1', ls='--', alpha=alpha)
    _ylim = 0.3
    _ax[0, 3].set_ylim([-_ylim, _ylim])
    _ax[1, 3].set_ylim([-_ylim, _ylim])
    return alpha, sed, sst


@app.cell
def __(cl1, cl3, plt, source_timeseries):
    (_fig, _ax) = plt.subplots(nrows=2)
    _ax[0].plot(cl1[:, 0, 0].T, color='C1', alpha=0.03)
    _ax[1].plot(cl3[:, 0, 0].T, color='C2', alpha=0.03)
    _ax[0].plot(source_timeseries[4, 0, 0].T, color='k')
    _ax[1].plot(source_timeseries[4, 0, 0].T, color='k')
    _ylim = 0.3
    _ax[0].set_ylim([-_ylim, _ylim])
    _ax[1].set_ylim([-_ylim, _ylim])
    return


@app.cell
def __(np, plt, qnts, source_strain):
    (_fig, _ax) = plt.subplots()
    _ax.plot(np.arange(len(qnts[1])), source_strain[4][0] - qnts[1], color='C0')
    _ax.fill_between(np.arange(len(qnts[1])), source_strain[4][0] - qnts[0], source_strain[4][0] - qnts[2], alpha=0.5, color='C1')
    return


@app.cell
def __(recon_timeseries):
    recon_timeseries.shape
    return


@app.cell
def __(source_timeseries):
    source_timeseries.shape
    return


@app.cell
def __(np, recon_timeseries, source_timeseries):
    ts_sqerr = np.sum(np.abs((recon_timeseries - source_timeseries[:, np.newaxis, :, :, :]) / source_timeseries[:, np.newaxis, :, :, :]) ** 2, 3)
    ts_mse = np.mean(ts_sqerr, axis=(-1, -2))
    ts_mean_sep = np.mean(ts_sqerr, axis=1)
    return ts_mean_sep, ts_mse, ts_sqerr


@app.cell
def __(np, source_timeseries):
    separation = np.sqrt(np.sum((source_timeseries[:, 0, :, :] - source_timeseries[:, 1, :, :]) ** 2, axis=1))
    return separation,


@app.cell
def __(ts_mean_sep, ts_mse, ts_sqerr):
    print(ts_sqerr.shape)
    print(ts_mse.shape)
    print(ts_mean_sep.shape)
    return


@app.cell
def __(plt, source_timeseries, ts_mean_sep):
    (_fig, _ax) = plt.subplots(nrows=4)
    _ax[0].plot(source_timeseries[0, 0, 0])
    _ax[1].plot(source_timeseries[0, 0, 1])
    _ax[2].plot(source_timeseries[0, 0, 2])
    _ax[0].plot(source_timeseries[0, 1, 0])
    _ax[1].plot(source_timeseries[0, 1, 1])
    _ax[2].plot(source_timeseries[0, 1, 2])
    _ax[3].plot(ts_mean_sep[0, 0])
    return


@app.cell
def __(separation):
    separation.shape
    return


@app.cell
def __(np, plt, separation, ts_mean_sep):
    (_fig, _ax) = plt.subplots()
    _ax.plot(separation[0])
    _ax.plot(np.sqrt(ts_mean_sep[0, 0]))
    return


@app.cell
def __(np, plt, ts_mse):
    (_fig, _ax) = plt.subplots()
    _vp = _ax.violinplot(np.log10(ts_mse[:, :].T), showextrema=True, widths=2, showmedians=True)
    _ax.set_ylabel('Log10 MSE of mass separation/true separation')
    _ax.set_xlabel('Data index')
    _fig.savefig('./paper/random_position_separation_mse_dist.png')
    return


@app.cell
def __(np, os, root_dir):
    with open(os.path.join(root_dir, 'train_losses.txt'), 'r') as _f:
        losses = np.loadtxt(_f)
    return losses,


@app.cell
def __(losses):
    losses.shape
    return


@app.cell
def __(np, os, plt, root_dir):
    with open(os.path.join(root_dir, 'train_losses.txt'), 'r') as _f:
        losses = np.loadtxt(_f)
    (_fig, _ax) = plt.subplots(figsize=(6, 3))
    fontsize = 15
    _ax.plot(losses[0], label='Training loss', lw=2)
    _ax.plot(losses[1], alpha=0.5, label='Validation loss', lw=2)
    _ax.set_xlabel('Training epoch', fontsize=fontsize)
    _ax.set_ylabel('Loss', fontsize=fontsize)
    _ax.legend(fontsize=fontsize)
    _fig.tight_layout()
    _fig.savefig('../paper/training_losses.pdf', format='pdf')
    return fontsize, losses


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
