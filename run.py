from argparse import ArgumentParser
from pprint import pprint
import pytorch_lightning as pl
from model import LitMNIST
from data import MNISTDataModule
import pytorch_lightning as ptl
import pytorch_lightning as ptl
from ray_lightning import RayPlugin



def cli_main():
  pl.seed_everything(1234)

  # ------------
  # args
  # ------------
  parser = ArgumentParser()
  parser = pl.Trainer.add_argparse_args(parser)
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
  plugin = RayPlugin(num_workers=4, cpus_per_worker=1, use_gpu=True)
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