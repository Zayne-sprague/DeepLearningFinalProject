from src.models.imp.IGR.code.reconstruction.run import ReconstructionRunner

from src.activations.activation import KernelActivation, batch_excitator_v2
from functools import partial

if __name__ == "__main__":
    custom_activation = partial(KernelActivation, partial(batch_excitator_v2, influence=0.1), is_batch_activation=True, kernel_size=4, is_linear=True)
    plot_frequency = None
    plot_on_epochs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80,
    90, 100, 500, 1000, 2500, 5000]

    expname = f'excitator_v2_k_4_inf_0p1'
    trainRunner = ReconstructionRunner(
        conf="igr_recon_test1.conf",
        points_batch=16384,
        nepochs=5001,
        expname=expname,
        gpu_index="0",
        is_continue=False,
        timestamp='latest',
        checkpoint='latest',
        eval=False,
        custom_activation=custom_activation,
        plot_frequency=plot_frequency,
        plot_on_epochs=plot_on_epochs
    )

    trainRunner.run()