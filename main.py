from pyexpat import model

from data import datamodule
import torch
import os

from lightning import Trainer, seed_everything
from data.datamodule import ImageNet10Datamodule, DistechDataDatamodule, CIFAR100Datamodule
from argparse import ArgumentParser
from utils.flops import calculate_flops, calculate_inference_time
from utils.models import models
from torchinfo import summary
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from lightning.pytorch.loggers import WandbLogger
import wandb

from utils.models import models
from external.RANet.adaptive_inference import Tester

torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_float32_matmul_precision("medium")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="train_baseline",
        help="Mode: train_baseline, test, flops or inference_time",
    )
    parser.add_argument("--model_path", type=str, help="Path to the model checkpoint")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="imagenet10",
        help="Dataset to use (imagenet10, distech, or cifar100)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./image-net/classes/",
        help="Path to the ImageNet-10 dataset",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--img_size", type=int, default=224, help="Image size (64, 128 or 224)"
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--resnet-size", type=int, default=18, help="ResNet size (18, 34 or 50)"
    )
    parser.add_argument(
        "--model-family",
        type=str,
        default="resnet",
        help="Model family: resnet18, mobilenet-large, efficientnet, shufflenet, convnext, mnasnet and etc",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Patience for early stopping (default: 15)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Decide if to use deterministic algorithms for reproducibility (default: True)",
    )
    parser.add_argument(
        "--use-aug", action="store_true", default=False, help="Use data augmentation"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="default-project",
        help="WandB project name",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default="default-ensemble-group",
        help="WandB group name for ensemble runs",
    )
    parser.add_argument(
        "--ensemble-resolutions",
        type=int,
        nargs="+",
        default=None,
        help="List of resolutions for ensemble training (e.g., --ensemble-resolutions 16 32 64 128 224)",
    )
    parser.add_argument(
        "--ensemble-models",
        type=str,
        nargs="+",
        default=None,
        help="List of model families for ensemble training (e.g., --ensemble-models mlp resnet mobilenet)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Number of classes in the dataset (default: 10)",
    )
    return parser.parse_args()

def get_model_cls(model_name):
    if models.get(model_name) is None:
        raise ValueError(f"Invalid model name: {model_name}")
    return models[model_name]

def calc_logit(model, dataloader, num_exits=2):
    model.eval()
    device = next(model.parameters()).device
    logits = None
    targets = []
    n_stage = None
    for i, (input, target) in enumerate(dataloader):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        targets.append(target.detach().cpu())
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            output = model(input_var)
            if not isinstance(output, list):
                output = [output]
            if logits is None:
                n_stage = len(output)
                logits = [[] for _ in range(n_stage)]

            for b in range(n_stage):
                _t = torch.nn.functional.softmax(output[b], dim=1)
                logits[b].append(_t.detach().cpu())

    for b in range(n_stage):
        logits[b] = torch.cat(logits[b], dim=0)

    size = (n_stage, logits[0].size(0), logits[0].size(1))
    ts_logits = torch.Tensor().resize_(size).zero_()
    for b in range(n_stage):
        ts_logits[b].copy_(logits[b])

    targets = torch.cat(targets, dim=0)
    ts_targets = torch.Tensor().resize_(size[1]).copy_(targets)

    return ts_logits, ts_targets


def train_ensemble(args, list_models, ensemble_resolutions):
    flops = []
    all_models = []
    all_train_loaders = []
    all_val_loaders = []
    all_test_loaders = []
    all_accuracies = []

    ensemble_logger = WandbLogger(
        project=args.wandb_project,
        name=f"ensemble_summary_{args.model_family}_{args.seed}",
        group=args.wandb_group,
        mode="offline"
    )

    ensemble_logger.log_hyperparams(vars(args))
    

    for model, res in zip(list_models, ensemble_resolutions):
        if args.data == "imagenet10":
            datamodule = ImageNet10Datamodule(
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                img_size=res,
                aug=args.use_aug,
            )
            eval_datamodule = datamodule
        elif args.data == "cifar100":
            datamodule = CIFAR100Datamodule(
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                img_size=res,
                resize=True,
                transform=None,
                use_valid=False
            )

            eval_datamodule = CIFAR100Datamodule(
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                img_size=res,
                resize=True,
                transform=None,
                use_valid=True,
                val_size=5000
            )

        datamodule.setup()
        eval_datamodule.setup()

        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints",
            filename=f"best_model_res{res}_{model.__class__.__name__}_seed{args.seed}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )

        early_stopping = EarlyStopping(monitor="val_loss", patience=args.patience)

        trainer = Trainer(
            max_epochs=args.epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            callbacks=[checkpoint_callback, early_stopping],
            deterministic=args.deterministic,
            logger=ensemble_logger,
        )

        all_train_loaders.append(datamodule.train_dataloader())

        if args.data == "cifar100":
            #all_val_loaders.append(datamodule.val_dataloader())
            #all_test_loaders.append(datamodule.test_dataloader())
            all_val_loaders.append(eval_datamodule.val_dataloader())
            all_test_loaders.append(eval_datamodule.test_dataloader())
        else:
            #print(f"Train: {len(datamodule.train_dataset)}, Val: {len(datamodule.val_dataset)}, Test: {len(datamodule.test_dataset)}")
            # Verificar overlap
            #train_indices = set(datamodule.train_dataset.subset.indices.tolist() if hasattr(datamodule.train_dataset.subset.indices, 'tolist') else datamodule.train_dataset.subset.indices)
            #test_indices = set(datamodule.test_dataset.subset.indices.tolist() if hasattr(...) else ...)
            #print(f"Overlap: {len(train_indices & test_indices)}")
            all_val_loaders.append(datamodule.val_dataloader())
            all_test_loaders.append(datamodule.test_dataloader())

        trainer.fit(model, datamodule=datamodule)
        all_models.append(model)

        results = trainer.test(model, datamodule=datamodule)
        acc = results[0]["test_acc"]
        all_accuracies.append(acc)

        dummy_input = torch.randn(1, 3, res, res)
        fwd_flops = calculate_flops(model, dummy_input)

        print(fwd_flops / 1e9)
        flops.append(fwd_flops / 1e9)  # GFLOPs

    all_acc = []
    all_flops = []
    all_T = []

    for p in range(1, 40):
        print("Evaluating ensemble with p =", p * 1.0 / 20)
        _p = torch.FloatTensor(1).fill_(p * 1.0 / 20)
        probs = torch.exp(torch.log(_p) * torch.arange(1, len(all_models) + 1))
        probs /= probs.sum()

        val_pred, val_target = [], []
        test_pred, test_target = [], []

        for i, model in enumerate(all_models):
            model.eval()
            val_pred_i, val_target_i = calc_logit(model, all_val_loaders[i])
            test_pred_i, test_target_i = calc_logit(model, all_test_loaders[i])

            val_pred.append(val_pred_i)
            val_target.append(val_target_i)
            test_pred.append(test_pred_i)
            test_target.append(test_target_i)

        tester = Tester(None)

        logits_val = (
            torch.cat(val_pred, dim=0) if isinstance(val_pred, list) else val_pred
        )
        targets_val = val_target[0] if isinstance(val_target, list) else val_target

        acc_val, _, T = tester.dynamic_eval_find_threshold(
            logits_val, targets_val, probs, flops
        )

        logits_test = (
            torch.cat(test_pred, dim=0) if isinstance(test_pred, list) else test_pred
        )
        targets_test = test_target[0] if isinstance(test_target, list) else test_target

        acc_test, exp_flops, exit_buckets = tester.dynamic_eval_with_threshold(
            logits_test, targets_test, flops, T
        )

        all_acc.append(acc_test)
        all_flops.append(exp_flops.item())
        all_T.append(T)

    print("Ensemble metrics: ")
    print("Accuracies:", all_acc)
    print("FLOPs:", all_flops)
    print("Thresholds:", all_T)

    for i, res in enumerate(ensemble_resolutions):
        ensemble_logger.log_hyperparams(
            {
                f"model_{i}_resolution": res,
                f"model_{i}_flops": flops[i],
            }
        )

    for p, acc, exp_flops, T in zip(range(1, 10), all_acc, all_flops, all_T):
        ensemble_logger.log_metrics(
            {
                "p": p * 1.0 / 20,
                "ensemble/accuracy": acc,
                "ensemble/expected_flops": exp_flops,
                "ensemble/threshold": T,
            }
        )

    ensemble_logger.log_hyperparams(
        {
            "ensemble_accuracies": all_acc,
            "ensemble_flops": all_flops,
            "ensemble_thresholds": all_T,
            "all_accuracies": all_accuracies,
        }
    )

    ensemble_logger.experiment.finish()

def train_baseline(args, ensemble_resolutions=None):
    if args.data == "distech":
        data_module = DistechDataDatamodule(
            batch_size=args.batch_size,
            img_size=args.img_size,
            resize=True,
            transform=None,
        )
    elif args.data == "imagenet10":
        data_module = ImageNet10Datamodule(
            data_dir=args.data_dir, batch_size=args.batch_size, img_size=args.img_size
        )
    elif args.data == "cifar100":
        data_module = CIFAR100Datamodule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            img_size=args.img_size,
            resize=True,
            transform=None,
            use_valid=True,
            val_size=5000,
        )


    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        mode="min",
        verbose=False,
    )

    list_models = []
    checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
    if args.ensemble_resolutions is not None:
        all_models = args.ensemble_models[0].split(" ")
        for model_name, res in zip(all_models, args.ensemble_resolutions):
            model = get_model_cls(model_name)
            if model_name == "mlp":
                instance_model = model(3* res* res, num_classes=args.num_classes)
            else:
                instance_model = model(num_classes=args.num_classes)
            list_models.append(instance_model)
    else:
        model = get_model_cls(args.model_family)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best_model_img{img_size}_{family}_seed{seed}".format(
            img_size=args.img_size,
            family=args.model_family,
            seed=args.seed,
        ),
    )

    if args.ensemble_resolutions is not None:
        train_ensemble(args, list_models, args.ensemble_resolutions)
        return

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=f"baseline_{args.model_family}_{args.img_size}_{args.seed}",
        group=args.wandb_group,
    )

    wandb_logger.log_hyperparams(vars(args))

    dummy_input = torch.randn(1, 3, args.img_size, args.img_size)
    fwd_flops = calculate_flops(model, dummy_input)
    wandb_logger.log_hyperparams(
        {
            "flops": fwd_flops,
            "gflops": fwd_flops / 1e9,
            "model_family": args.model_family
        }
    )

    summary(model, input_size=(args.batch_size, 3, args.img_size, args.img_size))
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback, early_stopping_callback],
        deterministic=args.deterministic,
        logger=wandb_logger,
    )

    trainer.fit(model, datamodule=data_module)
    best_model_path = checkpoint_callback.best_model_path

    best_model = get_model_cls(args.model_family).load_from_checkpoint(best_model_path)
    results = trainer.test(best_model, datamodule=data_module)

    wandb_logger.log_metrics(results[0])
    wandb_logger.experiment.finish()


def flops(args):
    model_cls, size = get_model_cls(args)
    model = model_cls(**get_model_kwargs(args.model_family, size))

    print(args.img_size)

    dummy_input = torch.randn(1, 3, args.img_size, args.img_size)
    fwd_flops = calculate_flops(model, dummy_input)
    print(f"GFLOPs: {fwd_flops / (1e9)} GFLOPs")


def inference_time(args):
    model_cls, _ = get_model_cls(args)
    model = model_cls.load_from_checkpoint(args.model_path)
    dummy_input = torch.randn(args.batch_size, 3, args.img_size, args.img_size)

    avg_time_cpu = calculate_inference_time(model, dummy_input, device="cpu")
    print(
        f"[CPU] Avg inference time (batch={args.batch_size}): {avg_time_cpu * 1000:.2f} ms"
    )

    if torch.cuda.is_available():
        avg_time_gpu = calculate_inference_time(model, dummy_input, device="cuda")
        print(
            f"[GPU] Avg inference time (batch={args.batch_size}): {avg_time_gpu * 1000:.2f} ms"
        )


def evaluate(args):
    data_module = ImageNet10Datamodule(
        data_dir=args.data_dir, batch_size=args.batch_size, img_size=args.img_size
    )
    model_cls, _ = get_model_cls(args)
    model = model_cls.load_from_checkpoint(args.model_path)
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        deterministic=args.deterministic,
    )
    results = trainer.test(model, datamodule=data_module)
    print(f"Metrics: {results}")


def main():
    args = parse_args()
    seed_everything(args.seed, workers=True)

    if args.data == "cifar100" and args.num_classes == 10:
        args.num_classes = 100

    match args.mode:

        case "train_baseline":
            train_baseline(args)
        case "flops":
            flops(args)
        case "inference_time":
            inference_time(args)
        case "evaluate":
            evaluate(args)
        case "ensemble":
            train_baseline(args, ensemble_resolutions=args.ensemble_resolutions)
        case _:
            print("Invalid mode")


if __name__ == "__main__":
    main()
