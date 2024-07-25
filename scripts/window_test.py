import marimo

__generated_with = "0.7.9"
app = marimo.App(width="medium")


@app.cell
def __():
    import matplotlib.pyplot as plt
    import numpy as np
    from massdynamics.data_generation import compute_waveform, data_processing, data_generation, models
    return (
        compute_waveform,
        data_generation,
        data_processing,
        models,
        np,
        plt,
    )


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
        window_acceleration="none", 
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
def __(mo):
    mo.md(r"Windowing here happens due to the multiplication of the fourier coefficients requires an inverse FFT to multiply in the time domain")
    return


@app.cell
def __(plt, rand_model):
    fig, ax = plt.subplots()
    ax.plot(rand_model[3][1,0])
    return ax, fig


@app.cell
def __():
    return


@app.cell
def __():
    import marimo as mo
    return mo,


if __name__ == "__main__":
    app.run()
