import matplotlib.pyplot as plt
import numpy as np


def plot_motions_and_strain(
        source_timeseries,
        source_strain,
        recon_timeseries,
        recon_strain,
        axlim = 0.5,
        motion_detector=0):

    motion_fig = plt.figure(figsize=(10, 15))
    motion_gs = GridSpec(4, 6, height_ratios=[2.5, 2, 1, 2.0], hspace=0.5)
    motion_detector = 0
    motion_fontsize = 20
    motion_ax_s1 = motion_fig.add_subplot(motion_gs[0, 0:3])
    motion_ax_s2 = motion_fig.add_subplot(motion_gs[0, 3:6])
    motion_axs = [motion_ax_s1, motion_ax_s2]
    motion_ax_l = motion_fig.add_subplot(motion_gs[1, :])
    motion_ax_ld = motion_fig.add_subplot(motion_gs[2, :])
    motion_ax_a1 = motion_fig.add_subplot(motion_gs[3, 0:2])
    motion_ax_a2 = motion_fig.add_subplot(motion_gs[3, 2:4])
    motion_ax_a3 = motion_fig.add_subplot(motion_gs[3, 4:6])
    motion_axa = [motion_ax_a1, motion_ax_a2, motion_ax_a3]
    for i in range(3):
        motion_axa[i].set_xlabel('$x$ position', fontsize=motion_fontsize)
        if i == 0:
            motion_axa[i].set_ylabel('$y$ position', fontsize=motion_fontsize)
            slim = 0.5
            motion_axa[i].set_xlim([-slim, slim])
            motion_axa[i].set_ylim([-slim, slim])
    for i in range(2):
        motion_axs[i].tick_params(axis='both', labelsize=12)
        tlim = 0.5
        motion_axs[i].set_xlim([-tlim, tlim])
        motion_axs[i].set_ylim([-tlim, tlim])
    motion_axs[i].set_xlabel('$x$ position', fontsize=motion_fontsize)
    motion_axs[i].set_ylabel('$y$ position', fontsize=motion_fontsize)
    motion_ax_l.tick_params(axis='both', labelsize=12)
    (_tstart, _tend) = (1, -1)

    motion_ax_ld.set_xlabel('Time [s]', fontsize=motion_fontsize)
    motion_ax_l.set_ylabel('$h^{\\rm{H}}_{\\rm{true}}$', fontsize=motion_fontsize)
    motion_ax_ld.set_ylabel('$h^{\\rm{H}}_{\\rm{recon}} - h^{\\rm{H}}_{\\rm{true}}$', fontsize=motion_fontsize)
    time = np.linspace(0, 1, len(source_strain[motion_detector]))
    
    motion_ax_l.plot(time, source_strain[motion_detector], color='k', label='true')
    motion_qnts = np.quantile(np.array(recon_strain)[:,motion_detector], [0.1, 0.5, 0.9], axis=0)
    
    motion_ax_l.plot(time, motion_qnts[1], color='C2', label='reconstructed 90% confidence')
    motion_ax_l.fill_between(time, motion_qnts[0], motion_qnts[2], alpha=0.5, color='C2')
    
    motion_ax_ld.plot(time, motion_qnts[1] - source_strain[motion_detector], color='C2', label='recovered 90% confidence')
    motion_ax_ld.fill_between(time, motion_qnts[0] - source_strain[motion_detector], motion_qnts[2] - source_strain[motion_detector], alpha=0.5, color='C2')
    
    motion_ax_ld.plot(time, source_strain[motion_detector] - source_strain[motion_detector], color='k', label='true')
    motion_ax_l.legend()

    motion_sinds = np.arange(recon_timeseries.shape[1])
    #motion_tsteps = np.random.choice(_sinds, 3)
    
    motion_tsteps = np.array([88, 699])
    for _i in range(2):
        motion_tstep_time = motion_tsteps[_i] / len(source_strain[motion_detector])
        _width = 3 / 120
        
        motion_axs[_i].plot(recon_timeseries[motion_tsteps[_i], 0, 0, _tstart:_tend], recon_timeseries[motion_tsteps[_i], 0, 1, _tstart:_tend], color='C0', label='recovered (m1)')
        motion_axs[_i].plot(recon_timeseries[motion_tsteps[_i], 1, 0, _tstart:_tend], recon_timeseries[motion_tsteps[_i], 1, 1, _tstart:_tend], color='C1', ls='--', label='recovered (m2)')
        motion_axs[_i].plot(source_timeseries[1, 0, _tstart:_tend], source_timeseries[1, 1, _tstart:_tend], color='k', label='true (m2)')
        motion_axs[_i].plot(source_timeseries[0, 0, _tstart:_tend], source_timeseries[0, 1, _tstart:_tend], color='k', ls='--', label='true (m1)')
        motion_axs[_i].scatter(recon_timeseries[motion_tsteps[_i], 0, 0, -1], recon_timeseries[motion_tsteps[_i], 0, 1, -1], color='C0', s=20)
        motion_axs[_i].scatter(recon_timeseries[motion_tsteps[_i], 1, 0, -1], recon_timeseries[motion_tsteps[_i], 1, 1, -1], color='C1', s=20)
        motion_axs[_i].scatter(source_timeseries[1, 0, -1], source_timeseries[1, 1, -1], color='k', s=20)
        motion_axs[_i].scatter(source_timeseries[0, 0, -1], source_timeseries[0, 1, -1], color='k', s=20)
        
        motion_axs[_i].set_aspect('equal', 'box')
        motion_axs[_i].set_aspect('equal', 'box')
    motion_tsteps = np.array(np.array([0.15, 0.4, 0.6]) * np.shape(recon_timeseries)[-1]).astype(int)
    nsamples = 30
    ar_scale = 2
    for _i in range(3):
        _tstep_time = motion_tsteps[_i] / len(source_strain[motion_detector])
        motion_ax_l.axvline(_tstep_time, color='r', lw=2)
        motion_ax_ld.axvline(_tstep_time, color='r', lw=2)
        _width = 3 / 120

        motion_axa[_i].scatter(recon_timeseries[:nsamples, 0, 0, motion_tsteps[_i]], recon_timeseries[:nsamples, 0, 1, motion_tsteps[_i]], color='C0', alpha=0.3, s=30, label='recovered (m1)')
        motion_axa[_i].scatter(recon_timeseries[:nsamples, 1, 0, motion_tsteps[_i]], recon_timeseries[:nsamples, 1, 1, motion_tsteps[_i]], color='C1', alpha=0.3, s=30, marker='*', label='recovered (m2)')
        motion_axa[_i].quiver(recon_timeseries[:nsamples, 0, 0, motion_tsteps[_i]], recon_timeseries[:nsamples, 0, 1, motion_tsteps[_i]], recon_velocities[:nsamples, 0, 0, motion_tsteps[_i]], recon_velocities[:nsamples, 0, 1, motion_tsteps[_i]], color='C0', alpha=0.7, scale=ar_scale, headwidth=5)
        motion_axa[_i].quiver(recon_timeseries[:nsamples, 1, 0, motion_tsteps[_i]], recon_timeseries[:nsamples, 1, 1, motion_tsteps[_i]], recon_velocities[:nsamples, 1, 0, motion_tsteps[_i]], recon_velocities[:nsamples, 1, 1, motion_tsteps[_i]], color='C1', alpha=0.7, scale=ar_scale, headwidth=5)
        motion_axa[_i].plot(source_timeseries[data_index, 1, 0, motion_tsteps[_i]], source_timeseries[1, 1, motion_tsteps[_i]], color='k', marker='*', label='true (m2)')
        motion_axa[_i].plot(source_timeseries[data_index, 0, 0, motion_tsteps[_i]], source_timeseries[0, 1, motion_tsteps[_i]], color='k', marker='o', label='true (m1)')
        motion_axa[_i].quiver(source_timeseries[0, 0, motion_tsteps[_i]], source_timeseries[0, 1, motion_tsteps[_i]], source_velocities[0, 0, motion_tsteps[_i]], source_velocities[0, 1, motion_tsteps[_i]], color='k', alpha=0.8, scale=ar_scale, headwidth=5)
        motion_axa[_i].quiver(source_timeseries[1, 0, motion_tsteps[_i]], source_timeseries[1, 1, motion_tsteps[_i]], source_velocities[1, 0, motion_tsteps[_i]], source_velocities[1, 1, motion_tsteps[_i]], color='k', alpha=0.8, scale=ar_scale, headwidth=5)
        motion_axa[_i].set_aspect('equal', 'box')
        figtr = motion_fig.transFigure.inverted()
        print(_tstep_time)
        ptB = figtr.transform(motion_ax_ld.transData.transform((_tstep_time * 1.0 - 0.0, -0.005)))
        ptE = figtr.transform(motion_axa[_i].transData.transform((0.0, 0.5)))
        arrow = matplotlib.patches.FancyArrowPatch(ptB, ptE, transform=motion_fig.transFigure, fc='r', arrowstyle='simple', alpha=0.5, mutation_scale=20.0)
        motion_fig.patches.append(arrow)

        
    motion_axa[0].legend()
    motion_fig.tight_layout()
    pos10 = motion_axs[0].get_position()
    pos11 = motion_axs[1].get_position()
    pos2 = motion_ax_l.get_position()
    pos3 = motion_ax_ld.get_position()
    motion_ax_l.set_position([pos2.x0, pos3.y1, pos2.width, pos2.height])
    motion_ax_ld.set_position([pos3.x0, pos3.y0, pos3.width, pos3.height])
    motion_axs[0].set_position([pos10.x0, pos10.y0 - 0.05, pos10.width, pos10.height])
    motion_axs[1].set_position([pos11.x0, pos11.y0 - 0.05, pos11.width, pos11.height])

    return fig
