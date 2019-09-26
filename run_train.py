from comet_ml import Experiment
import os
import torch
import argparse
from act.train import train
from act.data import resolve_data_manager
from act.utils import load_checkpoint, get_path
from act.models import BaseACT
from act.config import cfg, config_to_comet


def main(cfg, experiment):
    parser = argparse.ArgumentParser(description="Train Loop for ACT")
    parser.add_argument(
        "--config-file",
        required=True,
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    device = torch.device(cfg.MODEL.DEVICE if torch.cuda.is_available() else "cpu")
    print("using {}".format(device))

    scheduler = None
    if cfg.LOAD:
        # TODO: save and load scheduler (and maybe configuration?)
        load_path, epochs, save_path, load_best = cfg.LOAD_PATH, cfg.SOLVER.EPOCHS, cfg.SAVE_PATH, cfg.LOAD_BEST

        model = BaseACT(cfg).to(device)
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.SOLVER.LR)
        load_checkpoint(load_path, model, optimizer, cfg, device)
    else:
        model = BaseACT(cfg).to(device)
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.SOLVER.LR)
        if cfg.SOLVER.USE_SCHEDULER:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=10, threshold=0.0, threshold_mode='rel', cooldown=10, min_lr=1e-5)

    cfg.freeze()

    # log configs to comet.ml
    experiment.log_parameters(config_to_comet(cfg))

    train(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        data_manager=resolve_data_manager(cfg),
        device=device,
        experiment=experiment
    )


if __name__ == "__main__":
    # Create an experiment
    experiment = Experiment(        
        api_key = os.environ["COMET_KEY"], 
        project_name="general", 
        workspace = os.environ["COMET_USER"],
        disabled=True,
        auto_metric_logging=False)
    main(cfg, experiment)
