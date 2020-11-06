import sys, os
import argparse
import pytorch_lightning as pl

from src.experiment import Experiment


def main(args):
    pl.seed_everything(42)

    version = 'test'

    # default logger used by trainer
    logger = pl.loggers.TensorBoardLogger(
        save_dir    =   os.getcwd(),
        version     =   version,
        name        =   'lightning_logs',
    )

    # early_stop_callback = pl.callbacks.EarlyStopping(
    #     monitor     =   'val_loss',
    #     min_delta   =   0.00,
    #     patience    =   1,
    #     mode        =   'min'
    # )

    trainer = pl.Trainer(
        # profiler            =   args.profile and pl.profiler.AdvancedProfiler(output_filename = args.profiler_output_filepath),
        max_epochs          =   3,
        gpus                =   args.gpus,
        distributed_backend =   'dp',
        reload_dataloaders_every_epoch  =   True,
        # weights_summary     =   'full',
        logger              =   logger,
        # early_stop_callback =   args.early_stopping and early_stop_callback,
        checkpoint_callback =   False,
        # deterministic     =   True,
    )

    exp = Experiment()
    exp.fit(trainer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Definition generation benchmark")
    parser.add_argument('--data_path', type=str, help='data path')
    parser.add_argument('--save_path', type=str, default="saved_model", help='model save path')
    parser.add_argument('--gpus', type=str, default="0", help='')

    args = parser.parse_args()

    main(args)
