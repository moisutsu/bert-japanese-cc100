import sys, os
import argparse
import pytorch_lightning as pl

from src.experiment import Experiment


def main(args):
    pl.seed_everything(42)

    trainer = pl.Trainer(
        max_epochs          =   args.epochs,
        gpus                =   args.gpus,
        distributed_backend =   args.distributed_backend,
        reload_dataloaders_every_epoch  =   True,
        checkpoint_callback =   False,
    )

    exp = Experiment(
        data_path=args.data_path,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    exp.fit(trainer)
    exp.save(trainer, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Definition generation benchmark")
    parser.add_argument('--data_path', type=str, help='data path')
    parser.add_argument('--save_path', type=str, default="./model_weights.pt", help='model save path')
    parser.add_argument('--gpus', type=str, default="0", help='')
    parser.add_argument('--distributed_backend', type=str, default="ddp", help='')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--epochs', type=int, default=3, help='')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='')

    args = parser.parse_args()

    main(args)
