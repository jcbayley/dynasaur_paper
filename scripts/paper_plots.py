import marimo

__generated_with = "0.7.9"
app = marimo.App(width="medium")


@app.cell
def __():
    #from pycbc.waveform import get_td_waveform
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np
    from massdynamics.basis_functions import basis
    from massdynamics.data_generation import compute_waveform, data_processing, data_generation, models
    import scipy
    from scipy.interpolate import interp1d
    from scipy.integrate import solve_ivp
    from massdynamics.plotting import make_animations
    import marimo as mo
    return (
        basis,
        compute_waveform,
        data_generation,
        data_processing,
        interp1d,
        make_animations,
        matplotlib,
        mo,
        models,
        np,
        plt,
        scipy,
        solve_ivp,
    )


@app.cell
def __(basis, np, scipy):
    basis_type = "fourier"
    n_basis = 128
    n_use_basis = n_basis if basis_type == "fourier" else 2*n_basis

    times = np.linspace(-1,1,2*n_basis)
    #window = np.hanning(len(times))
    window = scipy.signal.windows.tukey(len(times), alpha=0.9)
    window_fit = basis[basis_type]["fit"](times, window[np.newaxis, :], n_use_basis)
    return basis_type, n_basis, n_use_basis, times, window, window_fit


@app.cell
def __():
    #hp2, hc2 = get_td_waveform(approximant="IMRPhenomPv2",
    #                        mass1 = 300,
    #                        mass2 = 300,
    #                        delta_t = 1./256,
    #                        f_lower=5
    #                        )

    #fig_w, ax_w = plt.subplots()
    #ax_w.plot(hp2[2400:2900], lw=3)
    return


@app.cell
def __(np, plt, scipy):
    fig_window, ax_window = plt.subplots()
    x = np.linspace(0,1, 1000)
    y = scipy.stats.norm(loc=0.5,scale=0.1).pdf(x)
    ax_window.plot(x,y, lw=3)
    return ax_window, fig_window, x, y


@app.cell
def __(np, plt, scipy):
    fig_f, ax_f = plt.subplots()
    xf = np.linspace(0,1, 1000)
    yf1 = scipy.stats.norm(loc=0.6,scale=0.15).pdf(xf)
    yf2 = scipy.stats.norm(loc=0.35,scale=0.12).pdf(xf)
    ax_f.plot(xf,yf1+yf2, lw=8)
    return ax_f, fig_f, xf, yf1, yf2


@app.cell
def __(np, plt, scipy):
    fig_f2, ax_f2 = plt.subplots()
    xf2 = np.linspace(0,1, 1000)
    yf21 = scipy.stats.norm(loc=0.8,scale=0.13).pdf(xf2)
    yf22 = scipy.stats.norm(loc=0.6,scale=0.13).pdf(xf2)
    yf23 = scipy.stats.norm(loc=0.3,scale=0.08).pdf(xf2)
    ax_f2.plot(xf2,yf21+yf22+yf23, lw=8)
    return ax_f2, fig_f2, xf2, yf21, yf22, yf23


@app.cell
def __(np, plt):
    fig_s, ax_s = plt.subplots()
    nsamps = 500
    x_s = np.random.normal(0,0.5, size=nsamps)
    y_s = np.random.normal(0,0.5, size=nsamps)
    x1_s = np.random.normal(0,0.5, size=nsamps)
    y1_s = np.random.normal(0,0.5, size=nsamps)
    ax_s.scatter(x_s, y_s)
    ax_s.scatter(x1_s, y1_s)
    return ax_s, fig_s, nsamps, x1_s, x_s, y1_s, y_s


@app.cell
def __(np, nsamps, plt):
    nsamps_b = 500
    std_b = 0.05
    times_b = np.linspace(0,1,100)
    x_b = np.random.normal(0,std_b, size=nsamps)
    y_b = np.random.normal(0,std_b, size=nsamps)
    x1_b = np.random.normal(0,std_b, size=nsamps)
    y1_b = np.random.normal(0,std_b, size=nsamps)
    f_b = 5
    r_b = 1./(times_b+1)*1
    posx_b = r_b*np.sin(times_b*f_b)
    posy_b = r_b*np.cos(times_b*f_b)

    fig_b, ax_b = plt.subplots()
    ax_b.set_xlim([-1.1,1.1])
    ax_b.set_ylim([-1.1,1.1])
    ax_b.scatter(posx_b[-1]+x_b, posy_b[-1]+y_b, color="C0", alpha=0.1, s=4)
    ax_b.scatter(-posx_b[-1]+x1_b, -posy_b[-1]+y1_b, color="C1", alpha=0.1, s=4)
    ax_b.plot(posx_b, posy_b, color="C0")
    ax_b.plot(posx_b[-1], posy_b[-1], marker="o", color="C0")
    ax_b.plot(-posx_b, -posy_b,color="C1")
    ax_b.plot(-posx_b[-1], -posy_b[-1], marker="o", color="C1")

    #fig_b.show()
    #plt.show()
    return (
        ax_b,
        f_b,
        fig_b,
        nsamps_b,
        posx_b,
        posy_b,
        r_b,
        std_b,
        times_b,
        x1_b,
        x_b,
        y1_b,
        y_b,
    )


@app.cell
def __(mo):
    mo.md(r"# Model Comparison")
    return


@app.cell
def __():
    dataindex = 0
    return dataindex,


@app.cell
def __(data_generation, np):
    rand_model = data_generation.generate_data(
        2, 
        32, 
        2, 
        32, 
        3, 
        detectors=["H1", "L1", "V1"], 
        window="none", 
        window_acceleration="hann", 
        basis_type="fourier",
        data_type = "random-uniform",
        fourier_weight=0.3,
        coordinate_type="cartesian",
        prior_args={
            "cycles_min": 2,
            "cycles_max": 2,
            "sky_position":(np.pi, np.pi/2)
        })
    return rand_model,


@app.cell
def __(data_generation, np):
    randnoise_model = data_generation.generate_data(
        2, 
        32, 
        2, 
        32, 
        3, 
        detectors=["H1", "L1", "V1"], 
        window="none", 
        window_acceleration="hann", 
        basis_type="fourier",
        data_type = "random-uniform",
        fourier_weight=0.0,
        coordinate_type="cartesian",
        noise_variance=0.0,
        snr=10,
        prior_args={
            "sky_position":(np.pi, np.pi/2)
        })
    return randnoise_model,


@app.cell
def __(data_generation, np):
    circ_model = data_generation.generate_data(
        2, 
        64, 
        2, 
        64, 
        3, 
        detectors=["H1", "L1", "V1"], 
        window="none", 
        window_acceleration=False,
        basis_type="timeseries",
        data_type = "circular",
        fourier_weight=0.0,
        coordinate_type="cartesian",
        prior_args={
            "cycles_min": 1,
            "cycles_max": 1,
            "mass_min": 5,
            "mass_max": 10,
            "sky_position":(np.pi, np.pi/2)
        })
    return circ_model,


@app.cell
def __(interp1d, np):
    def interpolate_positions(old_times, new_times, positions):
        """Interpolate between points over dimension 1 """
        curshape = list(positions.shape)
        curshape[-1] = len(new_times)
        interp_dict = np.zeros(tuple(curshape))
        for object_ind in range(positions.shape[0]):
            #print(np.shape(positions[object_ind]))
            interp = interp1d(old_times, positions[object_ind], kind="cubic")
            interp_dict[object_ind] = interp(new_times)
        return interp_dict

    def interpolate_positions2(old_times, new_times, positions):
        """Interpolate between points over dimension 1 """
        interp = interp1d(old_times, positions, kind="cubic")
        interp = interp(new_times)
        return interp
    return interpolate_positions, interpolate_positions2


@app.cell
def __(
    circ_model,
    dataindex,
    interpolate_positions,
    np,
    rand_model,
    randnoise_model,
):
    n_interp_points = 1000
    circ_times = np.linspace(np.min(circ_model[0]), np.max(circ_model[0]), n_interp_points)
    circ_interp = interpolate_positions(circ_model[0], circ_times, circ_model[5][dataindex])
    circ_interp = circ_interp/np.max(circ_interp)

    #circdecay_times = np.linspace(np.min(circdecay_model[0]), np.max(circdecay_model[0]), npoints)
    #circdecay_interp = interpolate_positions(circdecay_model[0], circdecay_times, circdecay_model[5][dataindex])

    rand_times = np.linspace(np.min(rand_model[0]), np.max(rand_model[0]), n_interp_points)
    rand_interp = interpolate_positions(rand_model[0], rand_times, rand_model[5][dataindex])
    rand_interp = rand_interp/np.max(rand_interp)

    randnoise_times = np.linspace(np.min(randnoise_model[0]), np.max(randnoise_model[0]), n_interp_points)
    randnoise_interp = interpolate_positions(randnoise_model[0], randnoise_times, randnoise_model[5][dataindex])
    randnoise_interp = randnoise_interp/np.max(randnoise_interp)

    print(np.shape(circ_model[3][0]), np.shape(circ_model[5]))

    circ_strain_interp = interpolate_positions(circ_model[0], circ_times, circ_model[3][dataindex])
    circ_strain_interp = circ_strain_interp/np.max(circ_strain_interp)

    rand_strain_interp = interpolate_positions(rand_model[0], rand_times, rand_model[3][dataindex])
    rand_strain_interp = rand_strain_interp/np.max(rand_strain_interp)

    randnoise_strain_interp = interpolate_positions(randnoise_model[0], randnoise_times, randnoise_model[3][dataindex])
    randnoise_strain_interp = randnoise_strain_interp/np.max(randnoise_strain_interp)
    return (
        circ_interp,
        circ_strain_interp,
        circ_times,
        n_interp_points,
        rand_interp,
        rand_strain_interp,
        rand_times,
        randnoise_interp,
        randnoise_strain_interp,
        randnoise_times,
    )


@app.cell
def __(
    circ_interp,
    circ_strain_interp,
    circ_times,
    matplotlib,
    np,
    plt,
    rand_interp,
    rand_strain_interp,
    rand_times,
    randnoise_interp,
    randnoise_strain_interp,
    randnoise_times,
):
    fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(8,10))
    ind=1
    r,g,b = matplotlib.colors.to_rgb("C0")
    r1,g1,b1 = matplotlib.colors.to_rgb("C1")
    alphas = np.logspace(1,4, len(circ_times) - 2*ind)
    alphas = (alphas -np.min(alphas))/np.max(alphas)
    cvs = np.linspace(0,1,len(circ_times)-2*ind)
    colours = plt.cm.Blues(cvs)#[(r, g, b, a) for a in alphas]
    colours1 = plt.cm.Reds(cvs)#[(r1, g1, b1, a) for a in alphas]

    gc = 6.6e-11/((3e8)**2)
    fontsize = 20

    ax[0,0].scatter(circ_interp[0][0][ind:-ind], circ_interp[0][1][ind:-ind], s=1, c=colours)
    ax[0,0].scatter(circ_interp[1][0][ind:-ind], circ_interp[1][1][ind:-ind], s=1, c=colours1)
    ax[0, 0].axis('equal')
    ax[0,0].plot(0,0, color="C2", marker="o")

    alphas = np.logspace(1,4, len(rand_times)-2*ind)
    alphas = (alphas -np.min(alphas))/np.max(alphas)
    cvs = np.linspace(0,1,len(rand_times)-2*ind)
    colours = plt.cm.Blues(cvs)#[(r, g, b, a) for a in alphas]
    colours1 = plt.cm.Reds(cvs)#[(r1, g1, b1, a) for a in alphas]
    ax[1,0].scatter(rand_interp[0][0][ind:-ind], rand_interp[0][1][ind:-ind], s=1, c=colours)
    ax[1,0].scatter(rand_interp[1][0][ind:-ind], rand_interp[1][1][ind:-ind], s=1, c=colours1)
    ax[1, 0].axis('equal')
    ax[1,0].plot(0,0, color="C2", marker="o")

    alphas = np.logspace(1,4, len(rand_times)-2*ind)
    alphas = (alphas -np.min(alphas))/np.max(alphas)
    cvs = np.linspace(0,1,len(rand_times)-2*ind)
    colours = plt.cm.Blues(cvs)#[(r, g, b, a) for a in alphas]
    colours1 = plt.cm.Reds(cvs)#[(r1, g1, b1, a) for a in alphas]
    ax[2,0].scatter(randnoise_interp[0][0][ind:-ind], randnoise_interp[0][1][ind:-ind], s=1, c=colours, label=r"$m_1$")
    ax[2,0].scatter(randnoise_interp[1][0][ind:-ind], randnoise_interp[1][1][ind:-ind], s=1, c=colours1, label=r"$m_2$")
    ax[2, 0].axis('equal')
    ax[2,0].plot(0,0, color="C2", label="COM", marker="o")

    lw=2
    """
    ax[0,1].plot(circ_model[0], circ_model[3][dataindex][0], lw=lw, color="C0", label="")
    ax[1,1].plot(rand_model[0], rand_model[3][dataindex][0], lw=lw, color="C0")
    ax[2,1].plot(randnoise_model[0], randnoise_model[3][dataindex][0], lw=lw, color="C0")

    ax[0,1].plot(circ_model[0], circ_model[3][dataindex][1], lw=lw, color="C1", ls="--")
    ax[1,1].plot(rand_model[0], rand_model[3][dataindex][1], lw=lw, color="C1", ls="--")
    ax[2,1].plot(randnoise_model[0], randnoise_model[3][dataindex][1], lw=lw, color="C1", ls="--")

    ax[0,1].plot(circ_model[0], circ_model[3][dataindex][2], lw=lw, color="C2", ls="-.")
    ax[1,1].plot(rand_model[0], rand_model[3][dataindex][2], lw=lw, color="C2", ls="-.")
    ax[2,1].plot(randnoise_model[0], randnoise_model[3][dataindex][2], lw=lw, color="C2", ls="-.")
    """
    ax[0,1].plot(circ_times, circ_strain_interp[0], lw=lw, color="C0")
    ax[1,1].plot(rand_times, rand_strain_interp[0], lw=lw, color="C0")
    ax[2,1].plot(randnoise_times, randnoise_strain_interp[0], lw=lw, color="C0", label="H1")

    ax[0,1].plot(circ_times, circ_strain_interp[1], lw=lw, color="C1", ls="--")
    ax[1,1].plot(rand_times, rand_strain_interp[1], lw=lw, color="C1", ls="--")
    ax[2,1].plot(randnoise_times, randnoise_strain_interp[1], lw=lw, color="C1", ls="--", label="L1")

    ax[0,1].plot(circ_times, circ_strain_interp[2], lw=lw, color="C2", ls="-.")
    ax[1,1].plot(rand_times, rand_strain_interp[2], lw=lw, color="C2", ls="-.")
    ax[2,1].plot(randnoise_times, randnoise_strain_interp[2], lw=lw, color="C2", ls="-.", label="V1")

    for i in range(3):
        ax[i,0].set_ylabel("Y position", fontsize=fontsize)
        ax[i,0].set_xlabel("X position", fontsize=fontsize)

        ax[i,1].set_xlabel("Time [s]", fontsize=fontsize)
        ax[i,1].set_ylabel("Strain", fontsize=fontsize)

    ax[2,1].legend()
    ax[2,0].legend()
    leg = ax[2,0].get_legend()
    leg.legendHandles[0].set_color('red')
    leg.legendHandles[1].set_color('blue')

    fig.tight_layout()

    #plt.show()
    return (
        alphas,
        ax,
        b,
        b1,
        colours,
        colours1,
        cvs,
        fig,
        fontsize,
        g,
        g1,
        gc,
        i,
        ind,
        leg,
        lw,
        r,
        r1,
    )


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
