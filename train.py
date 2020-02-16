import argparse
import os
import pathlib
import shutil
from typing import Any
from typing import Dict

import torch
from torch.utils.tensorboard import SummaryWriter

import datasets.registry as datasets_registry
import models.registry as models_registry
import utils.metrics as metrics
import utils.utils as utils


def train_one_epoch(
    epoch: int,
    model: torch.nn.Model,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    args: Dict[str, Any],
    writer: SummaryWriter,
) -> None:
    model.train()

    loss_m = metrics.AvgMetric(metrics.loss_fn, name="loss")
    selective_acc_m = metrics.AvgMetric(metrics.selective_acc_fn, name="selective_acc")
    rejected_acc_m = metrics.AvgMetric(metrics.rejected_acc_fn, name="rejected_acc")
    acc_m = metrics.AvgMetric(metrics.acc_fn, name="acc")
    coverage_m = metrics.AvgMetric(metrics.coverage_fn, name="coverage")
    metrics_list = [loss_m, selective_acc_m, rejected_acc_m, acc_m, coverage_m]

    alpha, target_coverage = args["alpha"], args["target_coverage"]
    for i, (data, targets) in enumerate(train_loader):
        data, targets = data.cuda(), targets.cuda()

        optimizer.zero_grad()

        f_out, g_out, h_out = model(data)
        loss = loss_m.accumulate(targets, f_out, g_out, h_out, alpha, target_coverage)
        selective_acc_m.accumulate(targets, f_out, g_out)
        rejected_acc_m.accumulate(targets, f_out, g_out)
        acc_m.accumulate(targets, f_out)
        coverage_m.accumulate(targets, f_out, g_out)
        loss.backward()
        optimizer.step()

        freq = 10
        if i % freq == freq - 1:
            step = epoch * len(train_loader) + i
            for m in metrics_list:
                writer.add_scalar(m.name, m.compute_and_reset(), step)


def evaluate(global_step, model, test_loader, args, writer):
    model.eval()
    with torch.no_grad():
        loss_m = metrics.AvgMetric(metrics.loss_fn, name="loss")
        selective_acc_m = metrics.AvgMetric(
            metrics.selective_acc_fn, name="selective_acc"
        )
        rejected_acc_m = metrics.AvgMetric(metrics.rejected_acc_fn, name="rejected_acc")
        acc_m = metrics.AvgMetric(metrics.acc_fn, name="acc")
        coverage_m = metrics.AvgMetric(metrics.coverage_fn, name="coverage")
        metrics_list = [loss_m, selective_acc_m, rejected_acc_m, acc_m, coverage_m]

        alpha, target_coverage = args["alpha"], args["target_coverage"]
        for i, (data, targets) in enumerate(test_loader):
            data, targets = data.cuda(), targets.cuda()
            f_out, g_out, h_out = model(data)
            loss_m.accumulate(targets, f_out, g_out, h_out, alpha, target_coverage)
            selective_acc_m.accumulate(targets, f_out, g_out)
            rejected_acc_m.accumulate(targets, f_out, g_out)
            acc_m.accumulate(targets, f_out)
            coverage_m.accumulate(targets, f_out, g_out)

        for m in metrics_list:
            writer.add_scalar(m.name, m.compute(), global_step)

    return loss_m.compute()


def train(args: Dict[str, Any]) -> None:
    """Trains the model."""

    # Init datasets and data loaders.
    train_data, test_data = datasets_registry.load_dataset(
        args["dataset"], params={"data_dir": pathlib.Path("./.data_dir")}
    )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=len(test_data),
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = models_registry.load_model(
        args["backbone"],
        {"num_classes": 10, "use_auxiliary_head": args["no_aux_head"] is False},
    ).cuda()
    # TODO: enable me.
    # model = torch.nn.parallel.DistributedDataParallel(model)

    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=args["lr"],
        momentum=0.9,
        dampening=0,
        weight_decay=5e-4,
        nesterov=True,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[args["epochs"] // 2, 3 * args["epochs"] // 4],
        gamma=0.1,
        last_epoch=-1,
    )

    train_writer = SummaryWriter(args["log_dir"] / "train")
    eval_writer = SummaryWriter(args["log_dir"] / "eval")

    min_loss = float("inf")
    for epoch in range(0, args["epochs"]):
        train_one_epoch(epoch, model, optimizer, train_loader, args, train_writer)
        test_loss = evaluate(
            epoch * len(train_loader), model, test_loader, args, eval_writer
        )
        scheduler.step()

        # save checkpoint
        is_best = test_loss < min_loss
        min_loss = min(test_loss, min_loss)
        utils.save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "min_loss": min_loss,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            path=args["checkpoint_dir"],
        )


def main(args: Dict[str, Any]):

    # Build model output dir.
    base_dir = pathlib.Path(
        os.path.join(args["output_dir"], utils.build_experiment_name(args))
    )
    if args["force"] and os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    if os.path.exists(base_dir):
        raise ValueError("output_dir already exists: {}.".format(base_dir))
    checkpoint_dir = base_dir / "checkpoints"
    logs_dir = base_dir / "logs"
    os.makedirs(base_dir)
    os.mkdir(checkpoint_dir)
    os.mkdir(logs_dir)

    # Write args to output dir.
    utils.write_dict(args, base_dir / "train_flags.json")

    # TODO: set debugging output log levels here.
    # TODO: set random seed here.

    train(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains a SelectiveNet on the given dataset with a VGG backbone.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # meta args
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="top level directory in which results, checkpoints, etc. are written.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="removes existing experiment dir in output_dir if it already exists.",
    )

    # general training args
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["cifar10", "svhn"],
        help="target dataset.",
    )
    parser.add_argument(
        "--epochs", type=int, default=300, help="number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="training batch size."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="initial learning rate (decreased during training by scheduler).",
    )

    # selectivenet args
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="convex combination weight."
    )
    parser.add_argument(
        "--target_coverage", type=float, default=0.8, help="target coverage."
    )
    parser.add_argument(
        "--backbone", type=str, default="vgg", choices=["vgg"], help="model backbone."
    )
    parser.add_argument(
        "--no_aux_head",
        action="store_true",
        default=False,
        help=(
            "whether to calculate auxilary loss with a separate auxilary head or to "
            "use f()."
        ),
    )

    args = vars(parser.parse_args())

    main(args)
