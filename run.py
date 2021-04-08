from argparse import ArgumentParser
from pprint import pprint
import pytorch_lightning as pl
from model import LitMNIST
from data import MNISTDataModule
from ray_lightning import RayPlugin
import ray


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--ray_accelerator_num_workers", type=int, default=4)
    parser.add_argument("--ray_accelerator_cpus_per_worker", type=int, default=4)
    parser.add_argument("--use_gpu", type=bool, default=False)
    parser = LitMNIST.add_model_specific_args(parser)
    parser = MNISTDataModule.add_argparse_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dm = MNISTDataModule.from_argparse_args(args)

    # ------------
    # model
    # ------------
    model = LitMNIST(args.hidden_dim, args.learning_rate)

    # ------------
    # training
    # ------------
    # ray.init()
    plugin = RayPlugin(num_workers=args.ray_accelerator_num_workers, cpus_per_worker=args.ray_accelerator_cpus_per_worker,
                       use_gpu=args.use_gpu)
    trainer = pl.Trainer(
        gpus=args.gpus, precision=args.precision, plugins=[plugin],
        max_epochs=args.max_epochs
    )
    trainer.fit(model, datamodule=dm)

    # ------------
    # testing
    # ------------
    result = trainer.test(model, datamodule=dm)
    pprint(result)


if __name__ == '__main__':
    cli_main()
