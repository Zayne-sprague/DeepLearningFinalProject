from src.models.imp.IGR.code.reconstruction.run import ReconstructionRunner

from src.activations.activation import KernelActivation, batch_excitator_v2
from functools import partial

if __name__ == "__main__":
    custom_activation = partial(KernelActivation, partial(batch_excitator_v2, influence=0.1), is_batch_activation=True, kernel_size=4, is_linear=True)

    expname = f'test'
    trainRunner = ReconstructionRunner(
        conf="igr_recon_test1.conf",
        points_batch=1000,
        nepochs=5000,
        expname=expname,
        gpu_index="0",
        is_continue=False,
        timestamp='latest',
        checkpoint='latest',
        eval=False,
        custom_activation=custom_activation
    )

    trainRunner.run()