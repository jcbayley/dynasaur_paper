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
    from massdynamics.data_generation import data_generation, data_processing, compute_waveform
    from scipy.interpolate import interp1d
    import brokenaxes as ba
    import corner
    from plotting import plot_motions_and_strain
    #plt.rcParams['text.usetex'] = True
    return (
        GridSpec,
        ba,
        compute_waveform,
        corner,
        create_model,
        data_generation,
        data_processing,
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
    root_dir = "/Users/joebayley/projects/massdynamics_project/results/random/test_2mass_fourier16_2d_3det_windowcoeffsstrain_sr16_transformer_5_masstriangle"
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
def __(config, np):
    config["prior_args"].setdefault("sky_position", (np.pi, np.pi/2))
    return


@app.cell
def __(config, create_model, root_dir):
    config["root_dir"] = root_dir
    pre_model, model, _ = create_model.load_models(config, device="cpu")
    return model, pre_model


@app.cell
def __(config, data_generation, np):
    prior_args = {
            "cycles_min": 5.,
            "cycles_max": 5.,
            "mass_min": 10,
            "mass_max": 10,
            "handed_orbit": True,
            "sky_position": [np.pi, np.pi/2]
        }

    data_arr = data_generation.generate_data(
            n_data=1, 
            basis_order=config["basis_order"], 
            n_masses=config["n_masses"], 
            sample_rate=config["sample_rate"], 
            n_dimensions=3, 
            detectors=config["detectors"], 
            window=config["window"], 
            window_acceleration="none", 
            basis_type=config["basis_type"],
            data_type = "circular",
            fourier_weight=0.0,
            coordinate_type="cartesian",
            noise_variance = False,
            prior_args=prior_args)

    data = {
        "times": data_arr[0],
        "basis_dynamics": data_arr[1],
        "source_masses": data_arr[2],
        "strain_timeseries": data_arr[3],
        "feature_shape": data_arr[4],
        "all_dynamics": data_arr[5],
        "all_basis_dynamics": data_arr[6],
    }
    return data, data_arr, prior_args


@app.cell
def __(config, data, data_processing, pre_model):
    _, _, processed_strain = data_processing.preprocess_data(
                pre_model, 
                data["basis_dynamics"],
                data["source_masses"], 
                data["strain_timeseries"], 
                window_strain=config["window_strain"], 
                spherical_coords=config["spherical_coords"], 
                initial_run=False,
                n_masses=config["n_masses"],
                device=config["device"],
                basis_type=config["basis_type"],
                n_dimensions=3)
    return processed_strain,


@app.cell
def __(data, np):
    upsample_rate = 128
    interp_times = np.linspace(np.min(data["times"]),np.max(data["times"]), upsample_rate)
    return interp_times, upsample_rate


@app.cell
def __(data, np, processed_strain):
    basis_dynamics = data["basis_dynamics"][0]/np.max(np.abs(data["basis_dynamics"][0]))
    source_masses = data["source_masses"][0]/np.sum(data["source_masses"][0])
    strain_timeseries = processed_strain[0]*np.hanning(np.shape(processed_strain)[-1])
    return basis_dynamics, source_masses, strain_timeseries


@app.cell
def __(
    basis_dynamics,
    compute_waveform,
    config,
    data,
    data_processing,
    interp1d,
    interp_times,
    pre_model,
    source_masses,
):
    t_source_tseries = compute_waveform.get_time_dynamics(
            basis_dynamics, 
            data["times"], 
            basis_type=config["basis_type"]
            )

    t_source_strain, source_energy = compute_waveform.get_waveform(
        data["times"], 
        source_masses, 
        basis_dynamics, 
        config["detectors"], 
        basis_type=config["basis_type"],
        compute_energy=True)

    t2_source_strain, _ = data_processing.normalise_data(t_source_strain, pre_model.norm_factor)
    t3_source_strain = data_processing.get_window_strain(t2_source_strain, window_type=config["window_strain"])

    s_strain_fn = interp1d(data["times"], t3_source_strain, kind="cubic")
    source_strain = s_strain_fn(interp_times)
    s_dyn_fn = interp1d(data["times"], t_source_tseries, kind="cubic")
    source_tseries = s_dyn_fn(interp_times)

    t_data_strain, _ = compute_waveform.get_waveform(
        data["times"], 
        source_masses, 
        basis_dynamics, 
        config["detectors"], 
        basis_type=config["basis_type"],
        compute_energy=True)

    t2_data_strain, _ = data_processing.normalise_data(t_data_strain, pre_model.norm_factor)
    data_strain = data_processing.get_window_strain(t2_data_strain, window_type=config["window_strain"])

    return (
        data_strain,
        s_dyn_fn,
        s_strain_fn,
        source_energy,
        source_strain,
        source_tseries,
        t2_data_strain,
        t2_source_strain,
        t3_source_strain,
        t_data_strain,
        t_source_strain,
        t_source_tseries,
    )


@app.cell
def __(data_strain, model, np, pre_model, torch):
    n_flow_samples = 800
    n_animate_samples = 100
    input_data = pre_model(torch.from_numpy(np.array([data_strain])).to(torch.float32))
    multi_coeffmass_samples = model(input_data).sample((n_flow_samples, )).cpu().squeeze(1)
    return (
        input_data,
        multi_coeffmass_samples,
        n_animate_samples,
        n_flow_samples,
    )


@app.cell
def __(
    config,
    data_processing,
    multi_coeffmass_samples,
    pre_model,
    strain_timeseries,
):
    _, multi_mass_samples, multi_coeff_samples, _ = data_processing.unpreprocess_data(
                pre_model, 
                multi_coeffmass_samples,
                strain_timeseries, 
                window_strain=config["window_strain"], 
                spherical_coords=config["spherical_coords"], 
                initial_run=False,
                n_masses=config["n_masses"],
                device=config["device"],
                basis_type=config["basis_type"],
                basis_order=config["basis_order"],
                n_dimensions=config["n_dimensions"])
    return multi_coeff_samples, multi_mass_samples


@app.cell
def __(
    compute_waveform,
    config,
    data,
    data_processing,
    interp1d,
    interp_times,
    multi_coeff_samples,
    multi_mass_samples,
    n_flow_samples,
    np,
    pre_model,
    source_tseries,
):
    n_masses = 2 
    n_dimensions = 3
    m_recon_tseries, m_recon_masses = np.zeros((n_flow_samples, n_masses, n_dimensions, len(interp_times))), np.zeros((n_flow_samples, n_masses))
    m_recon_strain = np.zeros((n_flow_samples, 3, len(interp_times)))

    for i_f in range(n_flow_samples):
        t_co = multi_coeff_samples[i_f]
        t_mass = multi_mass_samples[i_f]
        t_time = compute_waveform.get_time_dynamics(
            multi_coeff_samples[i_f],
            data["times"],  
            basis_type=config["basis_type"])


        temp_recon_strain, temp_recon_energy, temp_m_recon_coeffs = data_processing.get_strain_from_samples(
            data["times"], 
            t_mass,
            np.array(t_co), 
            detectors=["H1","L1","V1"],
            window_acceleration=config["window_acceleration"], 
            window=config["window"], 
            basis_type=config["basis_type"],
            basis_order=config["basis_order"])

        temp_recon_strain, _ = data_processing.normalise_data(temp_recon_strain, pre_model.norm_factor)


        #print(np.shape(data["times"]), np.shape(temp_recon_strain))
        strain_fn = interp1d(data["times"], temp_recon_strain, kind="cubic")
        recon_strain_interp = strain_fn(interp_times)
        dyn_fn = interp1d(data["times"], t_time, kind="cubic")
        recon_dyn_interp = dyn_fn(interp_times)

        m_recon_tseries[i_f] = recon_dyn_interp
        m_recon_masses[i_f] = t_mass
        m_recon_strain[i_f] = recon_strain_interp

    recon_masses = m_recon_masses
    recon_timeseries = m_recon_tseries
    recon_strain = m_recon_strain
    #source_masses = source_masses
    source_timeseries = source_tseries
    #source_strain = source_strain
    recon_velocities = np.gradient(recon_timeseries, axis=-1)
    source_velocities = np.gradient(source_timeseries, axis=-1)

    return (
        dyn_fn,
        i_f,
        m_recon_masses,
        m_recon_strain,
        m_recon_tseries,
        n_dimensions,
        n_masses,
        recon_dyn_interp,
        recon_masses,
        recon_strain,
        recon_strain_interp,
        recon_timeseries,
        recon_velocities,
        source_timeseries,
        source_velocities,
        strain_fn,
        t_co,
        t_mass,
        t_time,
        temp_m_recon_coeffs,
        temp_recon_energy,
        temp_recon_strain,
    )


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
        difference = np.array(recon_timeseries) - np.array(source_timeseries)
        max_source_strain = np.max(np.abs(source_timeseries), axis=2)[:, np.newaxis, :, np.newaxis]
        rmse = np.sqrt((difference / max_source_strain) ** 2)
        median_rmse = np.nanmedian(rmse, axis=-1)
        return median_rmse

    def compute_strain_rmse(recon_strain, source_strain):
        difference = np.array(recon_strain) - np.array(source_strain)
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
    diff_radii = recon_radii - source_radii[np.newaxis, ]
    diff_angles = np.mod(recon_angles - source_angles[np.newaxis, :] + np.pi, 2*np.pi) - np.pi
    return diff_angles, diff_radii


@app.cell
def __(
    compute_xy_rmse,
    np,
    recon_angles,
    recon_radii,
    source_angles,
    source_radii,
):
    rmse_radii = compute_xy_rmse(recon_radii, source_radii[np.newaxis, ])
    rmse_angles = compute_xy_rmse(recon_angles, source_angles[np.newaxis,])
    return rmse_angles, rmse_radii


@app.cell
def __():
    data_index = 0
    return data_index,


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
def __(GridSpec, data_index, diff_angles, diff_radii, np, plt):
    fig_ra_rmse = plt.figure()
    gs_ra_rmse = GridSpec(2, 2, hspace=0.5)
    ra_rmse_fontsize = 20
    ra_ax1 = fig_ra_rmse.add_subplot(gs_ra_rmse[0, 0])
    ra_ax2 = fig_ra_rmse.add_subplot(gs_ra_rmse[1, 0])
    ra_ax3 = fig_ra_rmse.add_subplot(gs_ra_rmse[0, 1])
    ra_ax4 = fig_ra_rmse.add_subplot(gs_ra_rmse[1, 1])
    ra_ax1.hist(np.ravel(diff_radii[data_index, :, 0]), bins=100, density=True, alpha=0.8)
    ra_ax1.axvline(0, color='C3')
    ra_ax2.hist(np.ravel(diff_radii[data_index, :, 1]), bins=100, density=True, alpha=0.8)
    ra_ax2.axvline(0, color='C3')
    ra_ax1.set_xlabel('Radius difference')
    ra_ax2.set_xlabel('Radius difference')
    #ra_ax1.set_xlim([-0.13, 0.2])
    #ra_ax2.set_xlim([-0.13, 0.2])
    ra_ax3.hist(np.ravel(diff_angles[data_index, :, 0]), bins=100, density=True, alpha=0.8)
    ra_ax3.axvline(0, color='C3')
    ra_ax4.hist(np.ravel(diff_angles[data_index, :, 1]), bins=100, density=True, alpha=0.8)
    ra_ax4.axvline(0, color='C3')
    ra_ax3.set_xlabel('Angle difference [Radians]')
    ra_ax4.set_xlabel('Angle difference [Radians]')
    ra_ax1.set_yticklabels([])
    ra_ax2.set_yticklabels([])
    ra_ax4.set_yticklabels([])
    ra_ax3.set_yticklabels([])
    ra_ax1.set_ylabel('Mass 1')
    ra_ax2.set_ylabel('Mass 2')
    plt.show()
    return (
        fig_ra_rmse,
        gs_ra_rmse,
        ra_ax1,
        ra_ax2,
        ra_ax3,
        ra_ax4,
        ra_rmse_fontsize,
    )


@app.cell
def __(compute_strain_rmse, np, recon_strain, source_strain):
    rmse = compute_strain_rmse(recon_strain, source_strain[np.newaxis, ])
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
    (pmin, pmax) = (-1.5, -0.2)
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
def __(source_strain):
    source_strain.shape
    return


@app.cell
def __(
    GridSpec,
    matplotlib,
    np,
    plt,
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
    time = np.linspace(0, 1, len(source_strain[motion_detector]))
    (_tstart, _tend) = int(0.25*len(time)),int(0.75*len(time))
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
    motion_ax_l.plot(time, source_strain[motion_detector], color='k', label='true')

    # find and plot quantiles
    motion_qnts = np.quantile(np.array(recon_strain)[:,motion_detector], [0.1, 0.5, 0.9], axis=0)

    motion_ax_l.axvspan(time[0], time[_tstart], color='#d3d3d3',  lw=0)
    motion_ax_l.axvspan(time[_tend], time[-1], color='#d3d3d3',  lw=0)
    motion_ax_ld.axvspan(time[0], time[_tstart], color='#d3d3d3',  lw=0)
    motion_ax_ld.axvspan(time[_tend], time[-1], color='#d3d3d3', lw=0)

    motion_ax_l.plot(time, motion_qnts[1], color='C2', label='reconstructed 90% confidence')
    motion_ax_l.fill_between(time, motion_qnts[0], motion_qnts[2], alpha=0.5, color='C2')

    # residual plot
    motion_ax_ld.plot(time, motion_qnts[1] - source_strain[motion_detector], color='C2', label='recovered 90% confidence')
    motion_ax_ld.fill_between(time, motion_qnts[0] - source_strain[motion_detector], motion_qnts[2] - source_strain[motion_detector], alpha=0.5, color='C2')
    motion_ax_ld.plot(time, source_strain[motion_detector] - source_strain[motion_detector], color='k', label='true')
    motion_ax_l.legend()



    motion_ax_l.set_xlim([time[0], time[-1]])
    motion_ax_ld.set_xlim([time[0], time[-1]])

    ###############
    # Plot the motion at all times
    ################
    motion_sinds = np.arange(recon_timeseries.shape[1])
    #motion_tsteps = np.random.choice(_sinds, 3)

    motion_tsteps = np.array([88, 699])
    for _i in range(2):
        motion_tstep_time = motion_tsteps[_i] / len(source_strain[motion_detector])
        _width = 3 / 120

        motion_axs[_i].plot(recon_timeseries[motion_tsteps[_i], 0, 0, _tstart:_tend], recon_timeseries[motion_tsteps[_i], 0, 1, _tstart:_tend], color='C0', label='recovered (m1)')
        motion_axs[_i].plot(recon_timeseries[motion_tsteps[_i], 1, 0, _tstart:_tend], recon_timeseries[motion_tsteps[_i], 1, 1, _tstart:_tend], color='C1', ls='--', label='recovered (m2)')
        motion_axs[_i].plot(source_timeseries[1, 0, _tstart:_tend], source_timeseries[ 1, 1, _tstart:_tend], color='k', label='true (m2)')
        motion_axs[_i].plot(source_timeseries[0, 0, _tstart:_tend], source_timeseries[0, 1, _tstart:_tend], color='k', ls='--', label='true (m1)')
        motion_axs[_i].scatter(recon_timeseries[motion_tsteps[_i], 0, 0, -1], recon_timeseries[motion_tsteps[_i], 0, 1, -1], color='C0', s=20)
        motion_axs[_i].scatter(recon_timeseries[motion_tsteps[_i], 1, 0, -1], recon_timeseries[motion_tsteps[_i], 1, 1, -1], color='C1', s=20)
        motion_axs[_i].scatter(source_timeseries[1, 0, -1], source_timeseries[1, 1, -1], color='k', s=20)
        motion_axs[_i].scatter(source_timeseries[0, 0, -1], source_timeseries[0, 1, -1], color='k', s=20)

        motion_axs[_i].set_aspect('equal', 'box')
        motion_axs[_i].set_aspect('equal', 'box')


    ##############
    # Plot the motion at a single point in time
    ################
    motion_tsteps = np.array(np.array([0.15, 0.4, 0.6]) * np.shape(recon_timeseries)[-1]).astype(int)
    nsamples = 30
    ar_scale = 2
    for _i in range(3):
        _tstep_time = motion_tsteps[_i] / len(source_strain[motion_detector])
        motion_ax_l.axvline(_tstep_time, color='r', lw=2)
        motion_ax_ld.axvline(_tstep_time, color='r', lw=2)
        _width = 3 / 120

        if motion_tsteps[_i] > _tend or motion_tsteps[_i] < _tstart:
            motion_axa[_i].set_facecolor("#d3d3d3")

        # plot reconstructed sample potition
        motion_axa[_i].scatter(recon_timeseries[:nsamples, 0, 0, motion_tsteps[_i]], recon_timeseries[:nsamples, 0, 1, motion_tsteps[_i]], color='C0', alpha=0.3, s=30, label='recovered (m1)')
        motion_axa[_i].scatter(recon_timeseries[:nsamples, 1, 0, motion_tsteps[_i]], recon_timeseries[:nsamples, 1, 1, motion_tsteps[_i]], color='C1', alpha=0.3, s=30, marker='*', label='recovered (m2)')
        # plot reconstruted sample vector arrows
        motion_axa[_i].quiver(recon_timeseries[:nsamples, 0, 0, motion_tsteps[_i]], recon_timeseries[:nsamples, 0, 1, motion_tsteps[_i]], recon_velocities[:nsamples, 0, 0, motion_tsteps[_i]], recon_velocities[:nsamples, 0, 1, motion_tsteps[_i]], color='C0', alpha=0.7, scale=ar_scale, headwidth=5)
        motion_axa[_i].quiver(recon_timeseries[:nsamples, 1, 0, motion_tsteps[_i]], recon_timeseries[:nsamples, 1, 1, motion_tsteps[_i]], recon_velocities[:nsamples, 1, 0, motion_tsteps[_i]], recon_velocities[:nsamples, 1, 1, motion_tsteps[_i]], color='C1', alpha=0.7, scale=ar_scale, headwidth=5)

        # plot true positions
        motion_axa[_i].plot(source_timeseries[1, 0, motion_tsteps[_i]], source_timeseries[1, 1, motion_tsteps[_i]], color='k', marker='*', label='true (m2)')
        motion_axa[_i].plot(source_timeseries[0, 0, motion_tsteps[_i]], source_timeseries[0, 1, motion_tsteps[_i]], color='k', marker='o', label='true (m1)')
        # plot the vector arrows on samples for truth
        motion_axa[_i].quiver(source_timeseries[0, 0, motion_tsteps[_i]], source_timeseries[0, 1, motion_tsteps[_i]], source_velocities[0, 0, motion_tsteps[_i]], source_velocities[0, 1, motion_tsteps[_i]], color='k', alpha=0.8, scale=ar_scale, headwidth=5)
        motion_axa[_i].quiver(source_timeseries[1, 0, motion_tsteps[_i]], source_timeseries[1, 1, motion_tsteps[_i]], source_velocities[1, 0, motion_tsteps[_i]], source_velocities[1, 1, motion_tsteps[_i]], color='k', alpha=0.8, scale=ar_scale, headwidth=5)

        # make plots square
        motion_axa[_i].set_aspect('equal', 'box')

        # Add arrows between sub plots
        figtr = motion_fig.transFigure.inverted()
        print(_tstep_time)
        ptB = figtr.transform(motion_ax_ld.transData.transform((_tstep_time * 1.0 - 0.0, -0.009)))
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
        slim,
        time,
        tlim,
    )


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
