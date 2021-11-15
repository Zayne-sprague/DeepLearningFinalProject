import src
import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple

from functools import partial

from src.paths import CHECKPOINTS_DIR, DATA_DIR
from src.lifecycles import train, test, save_model, load_modal, save_stats, load_stats
from src.viz_helper import compare_training_stats, save_plt
from src.activations.KNL import Inhibitor
from src.backbone import Backbone
from src.dataloader import get_dataloder

from src.activations.activation import KernelActivation
torch.backends.cudnn.benchmark = True
def run(
        epochs: int = 40,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        optimizer=torch.optim.Adam,
        normalization=torch.nn.BatchNorm2d,
        activation=torch.nn.ReLU,

        train_models: bool = True,
        test_models: bool = True,

        loss_fig_title: str = None,
        acc_fig_title: str = None,

        test_loss_fig_title: str = None,
        test_acc_fig_title: str = None,

        dataset: str = "CIFAR-100",

        checkpoint: int = -1,
        save_on_checkpoint: bool = True,

    ):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    EPOCHS = epochs
    BATCH_SIZE = batch_size
    optim_name = optimizer.__name__

    if loss_fig_title is None:
        loss_fig_title = f'WNT_training_loss'
    if acc_fig_title is None:
        acc_fig_title = f'WNT_training_acc'
    if test_loss_fig_title is None:
        test_loss_fig_title = f'WNT_test_loss'
    if test_acc_fig_title is None:
        test_acc_fig_title = f'WNT_test_acc'

    if checkpoint == -1:
        checkpoint = EPOCHS

    loss_fig_title += f'_{learning_rate}|{optim_name}|{dataset}'
    acc_fig_title += f'_{learning_rate}|{optim_name}{dataset}'
    test_loss_fig_title += f'_{learning_rate}|{optim_name}|{dataset}'
    test_acc_fig_title += f'_{learning_rate}|{optim_name}|{dataset}'

    NUM_CLASSES = 100
    if dataset == 'STL10':
        NUM_CLASSES = 10
    if dataset == 'TINY':
        NUM_CLASSES = 200

    net_bb_torch_norm = Backbone(NUM_CLASSES, normalization, activation)
    net_bb_torch_norm.to(device)

    # Different models with some parameters I want to compare against
    configs = []

    configs.append({'name': f'Backbone with Torch Weight Norm LR {learning_rate} USING {optim_name} ON {dataset}', 'label': 'BackBone Torch', 'model': net_bb_torch_norm, 'save_model': f'WN_bb_torch_norm@{learning_rate}|{optim_name}{dataset}', 'save_stats': f'WN_bb_torch_norm_training@{learning_rate}|{optim_name}|{dataset}', 'LR': learning_rate})

    # Train each model
    if train_models:
        for config in configs:
            test_stats = {}
            train_stats = {}

            net = config['model']

            # Grab the CIFAR-100 dataset, with a batch size of 10, and store it in the Data Directory (src/data)
            train_dataloader, test_dataloader = get_dataloder(dataset, BATCH_SIZE, DATA_DIR)


            # Set up a learning rate and optimizer
            opt = optimizer(net.parameters(), lr=config['LR'])

            for i in range(0, EPOCHS, checkpoint):
                # Train the network on the optimizer, using the training data loader, for EPOCHS epochs.
                stats = train(net, opt, train_dataloader, starting_epoch=i, epochs=i+checkpoint, loader_description=config['name'])
                train_stats.update(stats)

                if save_on_checkpoint:
                    # Save the model for testing later
                    save_model(net, f"EPOCH_{i}_checkpoint_{config['save_model']}")
                    # Save the stats from the training loop for later
                    save_stats(train_stats, f"EPOCH_{i}_checkpoint_{config['save_stats']}")

                if test_models:
                    test_stats[f'epoch_{i+1}'] = test(net, test_dataloader, loader_description=f'TESTING @ epoch {i}: {config["name"]}')
                    if save_on_checkpoint:
                        save_stats(train_stats, f"test_EPOCH_{i}_checkpoint_{config['save_stats']}")

            save_model(net, f"{config['save_model']}")
            # Save the stats from the training loop for later
            save_stats(train_stats, f"{config['save_stats']}")

            if test_models:
                test_stats[f'epoch_{EPOCHS + 1}'] = test(net, test_dataloader, loader_description=f'TESTING: {config["name"]}')

                save_stats(test_stats, f'test_{config["save_stats"]}')

    # Models have run, lets plot the stats
    train_stats = []
    test_stats = []
    labels = []
    for config in configs:
        train_stats.append(load_stats(config['save_stats']))
        test_stats.append(load_stats(f'test_{config["save_stats"]}'))
        labels.append(config['label'])


    if len(train_stats) and train_models:
        # For every config, plot the loss across number of epochs
        plt = compare_training_stats(train_stats, labels)
        save_plt(plt, loss_fig_title)
        # plt.show WILL WIPE THE PLT, so make sure you save the plot before you show it
        plt.show()

        # For every config, plot the accuracy across the number of epochs
        plt = compare_training_stats(train_stats, labels, metric_to_compare='accuracy', y_label='accuracy',
                                     title='Accuracy vs Epoch', legend_loc='lower right')
        save_plt(plt, acc_fig_title)
        # plt.show WILL WIPE THE PLT, so make sure you save the plot before you show it
        plt.show()


    if len(test_stats) > 0 and test_models:
        plt = compare_training_stats(test_stats, labels, title="Test Loss vs Checkpoint")
        save_plt(plt, test_loss_fig_title)
        plt.show()

        plt = compare_training_stats(test_stats, labels, metric_to_compare='accuracy', y_label='accuracy',
                                     title='Test Accuracy vs Checkpoint', legend_loc='lower right')
        save_plt(plt, test_acc_fig_title)
        plt.show()


if __name__ == "__main__":
    lr = 0.0001
    dataset = "CIFAR-100"
    from functools import partial
    from src.activations.activation import nelu, batch_nelu

    ACTIVATION = batch_nelu
    IS_BATCH_ACTIVATION = True
    KERNEL_SIZE = 8

    act = partial(KernelActivation, partial(ACTIVATION, influence=0.1), is_batch_activation=IS_BATCH_ACTIVATION, kernel_size=KERNEL_SIZE)
    # act = nn.ReLU
    print("NO DROPOUT")
    print("--------------------")
    run(
        epochs=25,
        batch_size=64,
        learning_rate=0.0001,

        optimizer=torch.optim.Adam,
        normalization=torch.nn.Identity,
        # activation=nn.ReLU,
        activation=act,

        train_models=True,
        test_models=True,

        loss_fig_title=f'training_loss_{dataset}@{lr}',
        acc_fig_title=f'training_acc_on_{dataset}@{lr}',
        test_loss_fig_title=f'test_loss_on_{dataset}@{lr}',
        test_acc_fig_title=f'test_acc_on_{dataset}@{lr}',

        dataset=dataset,
        checkpoint=1,
        save_on_checkpoint=False,
    )
