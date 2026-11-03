import os
import glob
import time
import argparse

model_names = ["RANet"]

# --resume --save /export/livia/home/vision/Jaraujo/projects/ETS/external/RANet_modified/save/default-1776286848.608474/
# --use-gelu --use-se cam --invert-attention --use-mixup --use-distillation --use-ema --ema-eval --exit-weights 4,4,4,4,1,1,1,1 --use-random-erasing --use-random-augmentation
"""
ranet-cifar100-cosine-gelu | --lr-type cosine --use-gelu
ranet-cifar100-kd4-cosine-gelu | --lr-type cosine --use-distillation --distill-alpha 4 --use-gelu
ranet-cifar100-cam-cosine-gelu | --use-se cam --lr-type cosine --use-gelu
ranet-cifar100-mixup-cosine-gelu | --use-mixup --lr-type cosine --use-gelu
ranet-cifar100-cosine | --lr-type cosine
ranet-cifar100-kd4-cosine | --lr-type cosine --use-distillation --distill-alpha 4
ranet-cifar100-cam-cosine | --use-se cam --lr-type cosine
ranet-cifar100-mixup-cosine | --use-mixup --lr-type cosine
"""

arg_parser = argparse.ArgumentParser(description="RANet Image classification")

exp_group = arg_parser.add_argument_group("exp", "experiment setting")
exp_group.add_argument(
    "--save",
    default="save/default-{}".format(time.time()),
    type=str,
    metavar="SAVE",
    help="path to the experiment logging directory" "(default: save/debug)",
)
exp_group.add_argument(
    "--resume",
    action="store_true",
    default=None,
    help="path to latest checkpoint (default: none)",
)
exp_group.add_argument(
    "--evalmode",
    default=None,
    choices=["anytime", "dynamic", "exit_stats", "both", "temperature_scaling", "gap", "dead_neurons", "cka", "cross_model_cka", "reliability", "grad_conflict"],
    help="which mode to evaluate",
)
exp_group.add_argument("--dead-neuron-threshold", type=float, default=1e-3,
                     help="Limiar de |ativação| abaixo do qual um neurônio é considerado morto.")
exp_group.add_argument("--dead-neuron-max-batches", type=int, default=None,
                     help="Limita o número de batches avaliados (None = dataset inteiro).")

exp_group.add_argument(
    "--evaluate-from",
    default="",
    type=str,
    metavar="PATH",
    help="path to saved checkpoint (default: none)",
)
exp_group.add_argument(
    "--print-freq",
    "-p",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 100)",
)
exp_group.add_argument("--seed", default=0, type=int, help="random seed")
exp_group.add_argument("--gpu", default=None, type=str, help="GPU available.")
exp_group.add_argument(
    "--wandb-project",
    default="RANet",
    type=str,
    help="WandB project name (default: None)",
)
exp_group.add_argument(
    "--wandb-group",
    default="ImageNet",
    type=str,
    help="WandB group name (default: None)",
)

exp_group.add_argument(
    "--use-distillation",
    action="store_true",
    default=False,
    help="Use distillation loss from teacher model (default: False)",
)

exp_group.add_argument(
    "--use-color-jitter",
    action="store_true",
    default=False,
    help="Use color jitter data augmentation (default: False)",
)

exp_group.add_argument(
    "--use-random-erasing",
    action="store_true",
    default=False,
    help="Use random erasing data augmentation (default: False)",
)

exp_group.add_argument(
    "--use-random-augmentation",
    action="store_true",
    default=False,
    help="Use random augmentation data augmentation (default: False)"
)

exp_group.add_argument(
    "--use-ema",
    action="store_true",
    default=False,
    help="Use Exponential Moving Average (EMA) of model weights (default: False)"
)

exp_group.add_argument(
    "--ema-decay",
    default=0.9999,
    type=float,
    help="Decay rate for the EMA of model weights (default: 0.9999). "
         "Closer to 1.0 = smoother/slower-updating average.",
)

exp_group.add_argument(
    "--ema-force-cpu",
    action="store_true",
    default=False,
    help="Keep the EMA weights on CPU instead of GPU (saves GPU memory, "
         "slightly slower updates). Default: False",
)

exp_group.add_argument(
    "--ema-eval",
    action="store_true",
    default=False,
    help="Use EMA weights (instead of raw weights) for final calibration, "
         "test evaluation and dynamic_evaluate. Default: False",
)

exp_group.add_argument(
    "--distilation-type",
    default="cascaded",
    choices=["cascaded", "last_exit", "all_subsequent_exits", "naive", "mate", "hdkd", "mlcsd"],
    help="Type of distillation loss to use (default: cascaded)",
)

exp_group.add_argument('--mate-alpha', type=float, default=3.0,
                     help='Peso do professor primário no loss do aluno (Eq. 6 do MATE, default do paper = 3)')
exp_group.add_argument('--mate-alpha-teacher', type=float, default=None,
                     help='Peso do aluno primário no loss do professor (Eq. 7). Default = --mate-alpha')
exp_group.add_argument('--mate-proj-dim', type=int, default=None,
                     help='Dimensão de projeção Q/K da weight network do MATE. Default = num_classes')

exp_group.add_argument('--evaluate_from_b', type=str, default=None,
                     help='Second checkpoint path for cross-model comparisons')
exp_group.add_argument('--cka_label_a', type=str, default='modelA')
exp_group.add_argument('--cka_label_b', type=str, default='modelB')

exp_group.add_argument(
    "--distill-annealing",
    action="store_true",
    default=False,
    help="Anneal distillation loss weight over epochs (default: False)",
)

exp_group.add_argument(
    "--distill-temp",
    default=3.0,
    type=float,
    help="Temperature for distillation loss (default: 3.0)",
)

exp_group.add_argument(
    "--normalize-loss",
    action="store_true",
    default=False,
    help="Normalize loss by number of exits",
)

exp_group.add_argument(
    "--compare-name",
    type=str,
    default="",
    help="Name of another experiment to compare with (for plotting) "   
)

exp_group.add_argument(
    "--distill-alpha",
    default=5,
    type=float,
    help="Weight for distillation loss (default: 5)",
)

exp_group.add_argument(
    "--exit-weights",
    default="",
    type=str,
    help="Comma-separated weights per exit (e.g., 1,1,1,0.5)",
)

# dataset related
data_group = arg_parser.add_argument_group("data", "dataset setting")
data_group.add_argument(
    "--data",
    metavar="D",
    default="cifar10",
    choices=["cifar10", "cifar100", "ImageNet", "imagenet100", "distech", "ImageNet_LR", "tinyimagenet"],
    help="data to work on",
)

data_group.add_argument(
    "--data-root",
    metavar="DIR",
    default="/data/cx/data",
    help="path to dataset (default: data)",
)

data_group.add_argument(
    "--use-valid", action="store_true", default=False, help="use validation set or not"
)

data_group.add_argument(
    "-j",
    "--workers",
    default=8,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)

data_group.add_argument(
    "--resize", action="store_true", default=False, help="Resize input images"
)

data_group.add_argument("--erde-topology", type=str, default="last_exit",
                     choices=["last_exit", "cascaded", "all_subsequent_exits"],
                     help="Which teacher topology ERDE's entropy-conditional loss is applied over.")
data_group.add_argument("--erde-entropy-weight", type=float, default=0.02,
                     help="Weight of the entropy term for teacher-wrong samples (paper ratio ~0.005/0.25).")

# model arch related
arch_group = arg_parser.add_argument_group("arch", "model architecture setting")
arch_group.add_argument("--arch", type=str, default="RANet")
arch_group.add_argument(
    "--reduction",
    default=0.5,
    type=float,
    metavar="C",
    help="compression ratio of DenseNet"
    " (1 means dot't use compression) (default: 0.5)",
)

# msdnet config
# arch_group.add_argument("--nBlocks", type=int, default=2)
# arch_group.add_argument("--nChannels", type=int, default=3)
# arch_group.add_argument("--growthRate", type=int, default=6)
# arch_group.add_argument("--grFactor", default="4-2-1", type=str)
# arch_group.add_argument("--bnFactor", default="4-2-1", type=str)
# arch_group.add_argument("--block-step", type=int, default=2)
# arch_group.add_argument("--scale-list", default="1-2-3", type=str)
# arch_group.add_argument("--compress-factor", default=0.25, type=float)
# arch_group.add_argument("--step", type=int, default=4)
# arch_group.add_argument("--stepmode", type=str, default="even", choices=["even", "lg"])
# arch_group.add_argument("--bnAfter", action="store_true", default=True)

# DISTECH config
# arch_group.add_argument("--nBlocks", type=int, default=2)
# arch_group.add_argument("--nChannels", type=int, default=3)
# arch_group.add_argument("--growthRate", type=int, default=6)
# arch_group.add_argument("--grFactor", default="4-2-1", type=str)
# arch_group.add_argument("--bnFactor", default="4-2-1", type=str)
# arch_group.add_argument("--block-step", type=int, default=2)
# arch_group.add_argument("--scale-list", default="1-2-3", type=str)
# arch_group.add_argument("--compress-factor", default=0.25, type=float)
# arch_group.add_argument("--step", type=int, default=4)
# arch_group.add_argument("--stepmode", type=str, default="even", choices=["even", "lg"])
# arch_group.add_argument("--bnAfter", action="store_true", default=True)

# CIFAR 10 config
# arch_group.add_argument("--nBlocks", type=int, default=3)
# arch_group.add_argument("--nChannels", type=int, default=16)
# arch_group.add_argument("--growthRate", type=int, default=6)
# arch_group.add_argument("--grFactor", default="4-2-1", type=str)
# arch_group.add_argument("--bnFactor", default="4-2-1", type=str)
# arch_group.add_argument("--block-step", type=int, default=2)
# arch_group.add_argument("--scale-list", default="1-2-3", type=str)
# arch_group.add_argument("--compress-factor", default=0.25, type=float)
# arch_group.add_argument("--step", type=int, default=4)
# arch_group.add_argument("--stepmode", type=str, default="even", choices=["even", "lg"])
# arch_group.add_argument("--bnAfter", action="store_true", default=True)

# Model-C-3 for CIFAR
arch_group.add_argument("--nBlocks",        type=int, default=2)
arch_group.add_argument("--nChannels",      type=int, default=16)
arch_group.add_argument("--growthRate",     type=int, default=6)
arch_group.add_argument("--grFactor",       default="4-2-1-1", type=str)
arch_group.add_argument("--bnFactor",       default="4-2-1-1", type=str)
arch_group.add_argument("--block-step",     type=int, default=2)
arch_group.add_argument("--scale-list",     default="1-2-3-3", type=str)
arch_group.add_argument("--compress-factor",default=0.25, type=float)
arch_group.add_argument("--step",           type=int, default=4)
arch_group.add_argument("--stepmode",       type=str, default="even", choices=["even", "lg"])
arch_group.add_argument("--bnAfter",        action="store_true", default=True)

arch_group.add_argument(
    "--use-se", type=str, default=None, choices=["cam", "cbam", "eca", "ema", "noresidual"],
)
arch_group.add_argument("--se-reduction", default=2, type=int)
arch_group.add_argument("--use-cmsa", action="store_true", default=False)

arch_group.add_argument(
    "--invert-attention",
    action="store_true",
    default=False,
    help="Use attention before transition layers",
)

arch_group.add_argument("--active-scales", type=int, default=0,
                        help="0 = all scales; N = only first N scales")

arch_group.add_argument("--detach-scales", action="store_true", default=False)
arch_group.add_argument("--skip-fusion", action="store_true", default=False)
arch_group.add_argument("--no-fusion", action="store_true", default=False)
arch_group.add_argument("--partial-fusion", default=1.0, type=float, help="Partial fusion factor (0=no fusion, 1=full fusion)")
arch_group.add_argument("--detach-cross-scale", action="store_true", default=False, help="Detach cross-scale features during backpropagation")

arch_group.add_argument("--use-dfs", action="store_true", default=False, help="Enable DFS feature partitioning/rerouting")
arch_group.add_argument("--dfs-beta", default=0.5, type=float, help="DFS split ratio beta (default: 0.5)")
arch_group.add_argument(
    "--dfs-use-dns",
    dest="dfs_use_dns",
    action="store_true",
    help="Enable DFS feature partitioning logic; defaults to --use-dfs when omitted",
)
arch_group.add_argument(
    "--no-dfs-use-dns",
    dest="dfs_use_dns",
    action="store_false",
    help="Disable DFS feature partitioning logic",
)
arch_group.add_argument(
    "--dfs-use-fr",
    dest="dfs_use_fr",
    action="store_true",
    help="Enable DFS feature referencing logic; defaults to --use-dfs when omitted",
)
arch_group.add_argument(
    "--no-dfs-use-fr",
    dest="dfs_use_fr",
    action="store_false",
    help="Disable DFS feature referencing logic",
)
arch_group.add_argument(
    "--dfs-on-fusion",
    dest="dfs_on_fusion",
    action="store_true",
    help="Apply DFS reroute on cross-scale fusion inputs; defaults to --use-dfs when omitted",
)
arch_group.add_argument(
    "--no-dfs-on-fusion",
    dest="dfs_on_fusion",
    action="store_false",
    help="Disable DFS reroute on cross-scale fusion inputs",
)
arch_group.add_argument(
    "--dfs-on-classifier",
    dest="dfs_on_classifier",
    action="store_true",
    help="Apply DFS reroute at exit classifiers; defaults to --use-dfs when omitted",
)
arch_group.add_argument(
    "--no-dfs-on-classifier",
    dest="dfs_on_classifier",
    action="store_false",
    help="Disable DFS reroute at exit classifiers",
)
arch_group.set_defaults(
    dfs_use_dns=None,
    dfs_use_fr=None,
    dfs_on_fusion=None,
    dfs_on_classifier=None,
)
arch_group.add_argument("--use-gelu", action="store_true", default=False, help="Use GELU activation instead of ReLU")
arch_group.add_argument("--use-leaky-relu", action="store_true", default=False, help="Use LeakyReLU activation instead of ReLU")

arch_group.add_argument("--use-partial-rcam", action="store_true", default=False, help="Use partial RCAM (Residual Channel Attention Module) instead of full RCAM")

# Model I-1 ImageNet
# arch_group.add_argument("--nBlocks",       type=int, default=2)
# arch_group.add_argument("--nChannels",     type=int, default=64)
# arch_group.add_argument("--growthRate",    type=int, default=16)
# arch_group.add_argument("--grFactor",      default="4-2-2-1", type=str)
# arch_group.add_argument("--bnFactor",      default="4-2-2-1", type=str)
# arch_group.add_argument("--block-step",    type=int, default=2)
# arch_group.add_argument("--scale-list",    default="1-2-3-4", type=str)
# arch_group.add_argument("--compress-factor", default=0.25, type=float)
# arch_group.add_argument("--step",          type=int, default=8)
# arch_group.add_argument("--stepmode",      type=str, default="even")

# Model-I-2 (só muda nChannels — dobra tudo)
# --nChannels 64  (64×4=256, 64×2=128, 64×2=128, 64×1=64)

arch_group.add_argument("--enhanced-classifier", action="store_true", default=False,
                        help="Use larger classifiers with dropout on final exits")
arch_group.add_argument("--enhanced-last-n", type=int, default=3,
                        help="Number of final exits to enhance (default: 3)")
arch_group.add_argument("--enhanced-channels", type=int, default=256,
                        help="Hidden channels for enhanced classifiers (default: 256)")
arch_group.add_argument("--classifier-dropout", type=float, default=0.2,
                        help="Dropout rate for enhanced classifiers (default: 0.2)")

calib_group = arg_parser.add_argument_group("calibration", "calibration settings")
calib_group.add_argument(
    "--label-smoothing",
    default=0.0,
    type=float,
    help="Label smoothing alpha (0=off, 0.1=typical). "
         "Soft targets, reduced overconfidence (Szegedy 2016).",
)
 
calib_group.add_argument(
    "--use-focal-loss",
    action="store_true",
    default=False,
    help="Use Focal Loss instead of CrossEntropy. "
         "Focuses on hard samples",
)
calib_group.add_argument(
    "--focal-gamma",
    default=2.0,
    type=float,
    help="Gamma do Focal Loss (0=CE, 2=clássico). Default: 2.0",
)
calib_group.add_argument(
    "--focal-flsd",
    action="store_true",
    default=False,
    help="Use FLSD (Focal Loss Sample-Dependent). "
         "Gamma varies by sample based on confidence.",
)
 
calib_group.add_argument(
    "--use-mixup",
    action="store_true",
    default=False,
    help="Activate Mixup data augmentation. "
         "Generate soft labels.",
)

calib_group.add_argument(
    "--use-cutmix",
    action="store_true",
    default=False,
    help="Activate CutMix data augmentation. "
)

calib_group.add_argument(
    "--cutmix-alpha",
    default=1.0,
    type=float,
    help="CutMix alpha (Beta distribution). "
         "1.0 is typical. Default: 1.0",
)

calib_group.add_argument(
    "--mixup-alpha",
    default=0.2,
    type=float,
    help="MixUp alpha (Beta distribution). "
         "0.2=soft, 1.0=strong. Default: 0.2",
)

calib_group.add_argument(
    "--use-manifold-mixup", action="store_true", default=False,
    help="Manifold Mixup",
)
 
calib_group.add_argument(
    "--calibration",
    default=None,
    type=str,
    choices=["temp_scaling", "vector_scaling"],
    help="Post-hoc calibration method "
         "Applied after training in calibration set",
)
# --use-dfs --dfs-use-dns --dfs-use-fr --dfs-on-classifier --no-dfs-on-fusion --dfs-beta 0.3
# training related
optim_group = arg_parser.add_argument_group("optimization", "optimization setting")
optim_group.add_argument(
    "--epochs",
    default=300,
    type=int,
    metavar="N",
    help="number of total epochs to run (default: 300)",
)
optim_group.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
optim_group.add_argument(
    "-b",
    "--batch-size",
    default=64,
    type=int,
    metavar="N",
    help="mini-batch size (default: 64)",
)
optim_group.add_argument(
    "--optimizer",
    default="sgd",
    choices=["sgd", "rmsprop", "adam"],
    metavar="N",
    help="optimizer (default=sgd)",
)
optim_group.add_argument(
    "--lr",
    "--learning-rate",
    default=0.1,
    type=float,
    metavar="LR",
    help="initial learning rate (default: 0.1)",
)
optim_group.add_argument(
    "--lr-type",
    default="multistep",
    type=str,
    metavar="T",
    help="learning rate strategy (default: multistep)",
    choices=["cosine", "multistep", "mlcsd"],
)
optim_group.add_argument(
    "--decay-rate",
    default=0.1,
    type=float,
    metavar="N",
    help="decay rate of learning rate (default: 0.1)",
)
optim_group.add_argument(
    "--momentum", default=0.9, type=float, metavar="M", help="momentum (default=0.9)"
)
optim_group.add_argument(
    "--weight-decay",
    "--wd",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
)

optim_group.add_argument(
    "--img_size", default=224, type=int, help="image net input size"
)

optim_group.add_argument(
    "--input_sizes",
    default="64-128-224",
    type=str,
    help="input sizes for each scale (e.g., 64-128-224)",
)

router_group = arg_parser.add_argument_group("router", "exit router settings")
router_group.add_argument(
    "--use-exit-router",
    action="store_true",
    default=False,
    help="Enable differentiable exit router (learns thresholds during training)",
)
router_group.add_argument(
    "--lambda-flops",
    default=1e-2,
    type=float,
    help="Weight for FLOPs penalty in router loss (default: 1e-2)",
)
router_group.add_argument(
    "--router-warmup",
    default=10,
    type=int,
    help="Epochs before router starts annealing tau (default: 10)",
)
router_group.add_argument(
    "--router-tau-min",
    default=0.1,
    type=float,
    help="Minimum Gumbel-Softmax temperature after annealing (default: 0.1)",
)
router_group.add_argument(
    "--router-hard",
    action="store_true",
    default=False,
    help="Use STE (hard forward, soft backward) in Gumbel-Softmax",
)

# Post-training threshold search
router_group.add_argument(
    "--calibrate",
    action="store_true",
    default=False,
    help="Enable per-exit temperature scaling calibration on val set",
)
router_group.add_argument(
    "--confidence-metric",
    default="max_softmax",
    type=str,
    choices=["max_softmax", "margin", "hybrid"],
    help="Confidence metric: max_softmax, margin, or hybrid (softmax on early exits, margin on later)",
)
router_group.add_argument(
    "--sweep-steps",
    default=200,
    type=int,
    help="Number of sweep points in Phase A (default: 200)",
)
router_group.add_argument(
    "--sweep-div",
    default=100,
    type=int,
    help="Divisor for sweep probability (default: 100)",
)

# Optuna refinement (post-sweep)
router_group.add_argument(
    "--optuna-refine",
    action="store_true",
    default=False,
    help="Run Optuna multi-objective refinement after sweep",
)
router_group.add_argument(
    "--optuna-trials",
    default=200,
    type=int,
    help="Number of Optuna trials for refinement (default: 200)",
)
router_group.add_argument(
    "--optuna-folds",
    default=5,
    type=int,
    help="Number of CV folds for Optuna objective (default: 5)",
)
router_group.add_argument(
    "--optuna-sampler",
    default="tpe",
    type=str,
    choices=["tpe", "nsgaii", "nsgaiii", "qmc", "random", "botorch"],
    help="Optuna sampler: tpe, nsgaii, nsgaiii, qmc, random, botorch (default: tpe)",
)

# FLOPs budget search
router_group.add_argument(
    "--flops-budget",
    default=None,
    type=float,
    help="FLOPs budget in millions (e.g. 15.0). Finds best accuracy under this budget.",
)
router_group.add_argument(
    "--budget-trials",
    default=200,
    type=int,
    help="Number of Optuna trials for budget-constrained search (default: 200)",
)


args = arg_parser.parse_args()
# args = arg_parser.parse_known_args()

if args.exit_weights:
    args.exit_weights = [float(w.strip()) for w in args.exit_weights.split(",") if w.strip()]
else:
    args.exit_weights = None

args.grFactor = list(map(int, args.grFactor.split("-")))
args.bnFactor = list(map(int, args.bnFactor.split("-")))
args.scale_list = list(map(int, args.scale_list.split("-")))
args.input_sizes = list(map(int, args.input_sizes.split("-")))
args.nScales = len(args.grFactor)

if args.use_valid:
    args.splits = ["train", "val", "test"]
else:
    args.splits = ["train", "val"]

if args.data == "cifar10":
    args.num_classes = 10
elif args.data == "cifar100":
    args.num_classes = 100
elif args.data == "ImageNet":
    args.num_classes = 1000
elif args.data == "imagenet100":
    args.num_classes = 100
else:
    args.num_classes = 3