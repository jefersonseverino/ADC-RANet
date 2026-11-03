import pdb
import os
import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .SE import SEBlock, CAM, CBAM, SAM, EMA, NORESIDUAL

# Module-level activation constructor. Use set_activation() to change.
ACTIVATION = nn.GELU



def set_activation(use_gelu: bool, use_leaky_relu: bool):
    """Set the module-level activation constructor.

    If use_gelu is True, ACTIVATION will be nn.GELU, otherwise nn.ReLU.
    """
    global ACTIVATION
    # ACTIVATION = nn.GELU if use_gelu else nn.ReLU
    if use_gelu:
        ACTIVATION = nn.GELU
    elif use_leaky_relu:
        ACTIVATION = nn.LeakyReLU
    else:
        ACTIVATION = nn.ReLU

class ConvBasic(nn.Module):
    def __init__(self, nIn, nOut, kernel=3, stride=1, padding=1):
        super(ConvBasic, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                nIn,
                nOut,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(nOut),
            ACTIVATION(),
        )

    def forward(self, x):
        return self.net(x)


class ConvBN(nn.Module):
    def __init__(self, nIn, nOut, type: str, bnAfter, bnWidth):
        """
        a basic conv in RANet, two type
        :param nIn:
        :param nOut:
        :param type: normal or down
        :param bnAfter: the location of batch Norm
        :param bnWidth: bottleneck factor
        """
        super(ConvBN, self).__init__()
        layer = []
        nInner = nIn
        if bnAfter is True:
            nInner = min(nInner, bnWidth * nOut)
            layer.append(
                nn.Conv2d(nIn, nInner, kernel_size=1, stride=1, padding=0, bias=False)
            )
            layer.append(nn.BatchNorm2d(nInner))
            layer.append(ACTIVATION())
            if type == "normal":
                layer.append(
                    nn.Conv2d(
                        nInner, nOut, kernel_size=3, stride=1, padding=1, bias=False
                    )
                )
            elif type == "down":
                layer.append(
                    nn.Conv2d(
                        nInner, nOut, kernel_size=3, stride=2, padding=1, bias=False
                    )
                )
            else:
                raise ValueError
            layer.append(nn.BatchNorm2d(nOut))
            layer.append(ACTIVATION())

        else:
            nInner = min(nInner, bnWidth * nOut)
            layer.append(nn.BatchNorm2d(nIn))
            layer.append(ACTIVATION())
            layer.append(
                nn.Conv2d(nIn, nInner, kernel_size=1, stride=1, padding=0, bias=False)
            )
            layer.append(nn.BatchNorm2d(nInner))
            layer.append(ACTIVATION())
            if type == "normal":
                layer.append(
                    nn.Conv2d(
                        nInner, nOut, kernel_size=3, stride=1, padding=1, bias=False
                    )
                )
            elif type == "down":
                layer.append(
                    nn.Conv2d(
                        nInner, nOut, kernel_size=3, stride=2, padding=1, bias=False
                    )
                )
            else:
                raise ValueError

        self.net = nn.Sequential(*layer)

    def forward(self, x):
        return self.net(x)


class ConvUpNormal(nn.Module):
    def __init__(
        self,
        nIn1,
        nIn2,
        nOut,
        bottleneck,
        bnWidth1,
        bnWidth2,
        compress_factor,
        down_sample,
    ):
        """
        The convolution with normal and up-sampling connection.
        """
        super(ConvUpNormal, self).__init__()
        self.conv_up = ConvBN(
            nIn2, math.floor(nOut * compress_factor), "normal", bottleneck, bnWidth2
        )
        if down_sample:
            self.conv_normal = ConvBN(
                nIn1,
                nOut - math.floor(nOut * compress_factor),
                "down",
                bottleneck,
                bnWidth1,
            )
        else:
            self.conv_normal = ConvBN(
                nIn1,
                nOut - math.floor(nOut * compress_factor),
                "normal",
                bottleneck,
                bnWidth1,
            )

        #self.cam_up = SEBlock(nIn2, reduction=2)
        #self.cam_normal = SEBlock(nIn1, reduction=2)

    def forward(self, x):
        #x0 = self.cam_up(x[0])
        #x1 = self.cam_normal(x[1])
        #x0 = x[0]
        #x1 = x[1]

        res = self.conv_normal(x[1])
        _, _, h, w = res.size() 

        res = [
            F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=False),
            F.interpolate(
                self.conv_up(x[0]), size=(h, w), mode="bilinear", align_corners=False
            ),
            res,
        ]
        return torch.cat(res, dim=1)
 

class ConvNormal(nn.Module):
    def __init__(self, nIn, nOut, bottleneck, bnWidth):
        """
        The convolution with normal connection.
        """
        super(ConvNormal, self).__init__()
        self.conv_normal = ConvBN(nIn, nOut, "normal", bottleneck, bnWidth)

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]
        res = [x[0], self.conv_normal(x[0])]
        return torch.cat(res, dim=1)


class _BlockNormal(nn.Module):
    def __init__(self, num_layers, nIn, growth_rate, reduction_rate, trans, bnFactor, use_se=None, se_reduction=16):
        """
        The basic computational block in RANet with num_layers layers.
        trans: If True, the block will add a transition layer at the end of the block
                with reduction_rate.
        """
        super(_BlockNormal, self).__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        for i in range(num_layers):
            self.layers.append(
                ConvNormal(nIn + i * growth_rate, growth_rate, True, bnFactor)
            )
        nOut = nIn + num_layers * growth_rate
        self.trans_flag = trans
        if trans:
            self.trans = ConvBasic(
                nOut,
                math.floor(1.0 * reduction_rate * nOut),
                kernel=1,
                stride=1,
                padding=0,
            )
            nOut = math.floor(1.0 * reduction_rate * nOut)

        # self.use_se = use_se
        # if use_se == "cam":
        #     self.se = SEBlock(nOut, reduction=se_reduction)
        # elif use_se == "cbam":
        #     self.se = CBAM(nOut, r=se_reduction)

    def forward(self, x):
        output = [x]
        for i in range(self.num_layers):
            x = self.layers[i](x)
            output.append(x)
        x = output[-1]
        if self.trans_flag:
            x = self.trans(x)
        #if self.use_se:
        #   x = self.se(x)
        return x, output

    def _blockType(self):
        return "norm"

class _BlockUpNormal(nn.Module):
    def __init__(
        self,
        num_layers,
        nIn,
        nIn_lowFtrs,
        growth_rate,
        reduction_rate,
        trans,
        down,
        compress_factor,
        bnFactor1,
        bnFactor2,
        use_se=None,
        se_reduction=16,
        skip_fusion=False,
        partial_fusion=1.0,
        detach_cross_scale=False,
        invert_attention=False,
        partial_rcam=False
    ):
        """
        The basic fusion block in RANet with num_layers layers.
        trans: If True, the block will add a transition layer at the end of the block
                with reduction_rate.
        compress_factor: There will be compress_factor*100% information from the previous
                sub-network.
        """
        super(_BlockUpNormal, self).__init__()

        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        for i in range(num_layers - 1):
            self.layers.append(
                ConvUpNormal(
                    nIn + i * growth_rate,
                    nIn_lowFtrs[i],
                    growth_rate,
                    True,
                    bnFactor1,
                    bnFactor2,
                    compress_factor,
                    False,
                )
            )

        self.layers.append(
            ConvUpNormal(
                nIn + (i + 1) * growth_rate,
                nIn_lowFtrs[i + 1],
                growth_rate,
                True,
                bnFactor1,
                bnFactor2,
                compress_factor,
                down,
            )
        )

        nOut = nIn + num_layers * growth_rate
        branch_b_channels = math.floor(nOut * compress_factor)
        self.conv_last = ConvBasic(
            nIn_lowFtrs[num_layers],
            math.floor(nOut * compress_factor),
            kernel=1,
            stride=1,
            padding=0,
        )
        nOut = nOut + math.floor(nOut * compress_factor)
        self.trans_flag = trans
        self.invert_attention = invert_attention
        self.use_partial_rcam = partial_rcam

        nOut_pre_trans = nOut

        if trans:
            self.trans = ConvBasic(
                nOut,
                math.floor(1.0 * reduction_rate * nOut),
                kernel=1,
                stride=1,
                padding=0,
            )
            nOut = math.floor(1.0 * reduction_rate * nOut)

        if not invert_attention:
            nOut_pre_trans = nOut

        self.skip_fusion = skip_fusion
        self.detach_cross_scale = detach_cross_scale
        self.down = down
        self.use_se = use_se
        self.se = None
        
        if use_se is not None and use_se == "cam":
            #nOut = nIn + self.num_layers * growth_rate
            #self.se = SEBlock(nOut, reduction=se_reduction)
            self.se = SEBlock(nOut_pre_trans, reduction=se_reduction)
        elif use_se is not None and use_se == "cbam":
            #nOut = nIn + self.num_layers * growth_rate
            #self.se = CBAM(nOut, r=se_reduction)
            self.se = CBAM(nOut_pre_trans, r=se_reduction)
        elif use_se is not None and use_se == "eca":
            nOut = nIn + self.num_layers * growth_rate
            self.se = ECABlock(nOut_pre_trans, k=3)
            # self.se = ECABlock(nOut, k=3)
        elif use_se is not None and use_se == "ema":
            #nOut = nIn + self.num_layers * growth_rate
            #self.se = EMA(nOut, factor=se_reduction)
            self.se = EMA(nOut_pre_trans, factor=se_reduction)
        elif use_se is not None and use_se == "noresidual":
            self.se = NORESIDUAL(nOut_pre_trans, reduction=se_reduction)

        if self.use_partial_rcam:
            # self.se_partial = SEBlock(nIn_lowFtrs[num_layers], reduction=2)
            # self.se_partial = SEBlock(self.conv_last.out_channels, reduction=2)
            self.se_partial = SEBlock(
                branch_b_channels, reduction=2
            )

    def _apply_scale_aware_fusion(self, x, low_feat):
        filtered = []
        _, _, h_curr, w_curr = x.size()
        for f in low_feat:
            _, _, h_low, w_low = f.size()
            if h_low == h_curr and w_low == w_curr:
                filtered.append(f)
            else:
                filtered.append(f.detach())
        return filtered


    def forward(self, x, low_feat):
        if self.skip_fusion:
            low_feat = [torch.zeros_like(f) for f in low_feat]
        elif self.detach_cross_scale and self.down:
            # low_feat = [f.detach() for f in low_feat]
            low_feat = self._apply_scale_aware_fusion(x, low_feat)

        output = [x]
        for i in range(self.num_layers):
            inp = [low_feat[i]]
            inp.append(x)
            x = self.layers[i](inp)
            output.append(x)
        x = output[-1]
        _, _, h, w = x.size()
        # x = [x]
        # x.append(
        #     F.interpolate(
        #         self.conv_last(low_feat[self.num_layers]),
        #         size=(h, w),
        #         mode="bilinear", align_corners=False,
        #     )
        # )
        # x = torch.cat(x, dim=1)

        branch_a = x
        branch_b = F.interpolate(
            self.conv_last(low_feat[self.num_layers]),
            size=(h, w),
            mode="bilinear", align_corners=False,
        )

        if self.use_partial_rcam:
            # self.se_partial = CAM(branch_b.size(1), reduction=2)
            branch_b = self.se_partial(branch_b)
            # branch_b = self.se(branch_b)

        x = torch.cat([branch_a, branch_b], dim=1)

        if self.invert_attention and not self.use_partial_rcam:
            if self.use_se == "cross_ema":
                x = self.se(x, low_feat[-1])
            if self.use_se is not None:
                x = self.se(x)

        if self.trans_flag:
            x = self.trans(x)

        if not self.invert_attention and not self.use_partial_rcam:
            if self.use_se == "cross_ema":
                x = self.se(x, low_feat[-1])
            if self.use_se is not None:
                x = self.se(x)
    
        return x, output

    def _blockType(self):
        return "up"

class RAFirstLayer(nn.Module):
    def __init__(self, nIn, nOut, args):
        """
        RAFirstLayer gennerates the base features for RANet.
        The scale 1 means the lowest resoultion in the network.
        """
        super(RAFirstLayer, self).__init__()
        _grFactor = args.grFactor[::-1]  # 1-2-4
        _scale_list = args.scale_list[::-1]  # 3-2-1
        # The resolution decreases and the number of feature maps increases...
        self.layers = nn.ModuleList()

        if args.data.startswith("cifar"):
            self.layers.append(
                ConvBasic(nIn, nOut * _grFactor[0], kernel=3, stride=1, padding=1)
            )
        
        elif args.data == "ImageNet" or args.data == "imagenet100":
            # reduces the spatial size from 224x224 to 56x56 for imagenet
            conv = nn.Sequential(
                nn.Conv2d(nIn, nOut * _grFactor[0], 7, 2, 3),
                nn.BatchNorm2d(nOut * _grFactor[0]),
                ACTIVATION(),
                nn.MaxPool2d(3, 2, 1),
            )
            self.layers.append(conv)
        elif args.data == "tinyimagenet":
            # reduces the spatial size from 64x64 to 32x32 for tinyimagenet
            conv = nn.Sequential(
                nn.Conv2d(nIn, nOut * _grFactor[0], kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(nOut * _grFactor[0]),
                ACTIVATION(),
            )
            self.layers.append(conv)
        elif args.data == "distech":
            # reduces the spatial size from 256x192 to 64x48 for distech
            conv = nn.Sequential(
                nn.Conv2d(nIn, nOut * _grFactor[0], 7, 2, 3),
                nn.BatchNorm2d(nOut * _grFactor[0]),
                ACTIVATION(),
                nn.MaxPool2d(3, 2, 1),
            )
            self.layers.append(conv)

        # instead of creating an initial layer, create a dimension
        # reduction block, increase the number of feature maps
        # also do it by "demand", i.e., when the current scale
        # is not capable of classifying the input, then add
        # a higher resolution through to the reduction block
        # use higher dimension feature map for the subnetwork right above it

        # Number of input channels for the next layer
        nIn = nOut * _grFactor[0]

        # grFactor -> Defines the channel multiplier for each scale
        # scale_list -> Defines the scale levels, higher values mean high resolutions

        s = _scale_list[0]
        for i in range(0, args.nScales):
            if s == _scale_list[i]:
                # same scale, stride=1
                self.layers.append(
                    ConvBasic(nIn, nOut * _grFactor[i], kernel=3, stride=1, padding=1)
                )
            else:
            # down-sampling, stride=2, obtain lower resolution features
                self.layers.append(
                    ConvBasic(nIn, nOut * _grFactor[i], kernel=3, stride=2, padding=1)
                )
            s = _scale_list[i]
            nIn = nOut * _grFactor[i]

    def forward(self, x):
        # res[0] with the smallest resolution
        res = []
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            res.append(x)
        return res[::-1]


class RANet(nn.Module):
    def __init__(self, args):
        super(RANet, self).__init__()

        # Set activation function globally for this module (GELU or ReLU)
        use_gelu = getattr(args, "use_gelu", True)
        use_leaky_relu = getattr(args, "use_leaky_relu", False)
        
        set_activation(bool(use_gelu), bool(use_leaky_relu))
        self.use_gelu = bool(use_gelu)
        self.use_leaky_relu = bool(use_leaky_relu)

        self.scale_flows = nn.ModuleList()
        self.classifier = nn.ModuleList()

        self.compress_factor = args.compress_factor
        self.bnFactor = copy.copy(args.bnFactor)
        self.use_se = args.use_se
        self.se_reduction = getattr(args, 'se_reduction', 16) # args.se_reduction

        scale_list = args.scale_list
        self.nScales = len(args.scale_list)
        self.active_scales = args.active_scales if args.active_scales > 0 else self.nScales
        self.skip_fusion = args.skip_fusion
        self.partial_fusion = args.partial_fusion
        self.detach_cross_scale = args.detach_cross_scale
        self.invert_attention = args.invert_attention
        self.partial_rcam = args.use_partial_rcam

        self.nBlocks = [0]
        for i in range(self.nScales):
            self.nBlocks.append(args.block_step * i + args.nBlocks)  

        self.steps = args.step

        self.FirstLayer = RAFirstLayer(3, args.nChannels, args)

        steps = [args.step]
        #for ii in range(self.nScales):
        for ii in range(self.active_scales):
            scale_flow = nn.ModuleList()

            n_block_curr = 1
            nIn = args.nChannels * args.grFactor[ii]  # grFactor = [4,2,1]
            _nIn_lowFtrs = []

            for i in range(self.nBlocks[ii + 1]):
                growth_rate = args.growthRate * args.grFactor[ii]

                # If translation 
                trans = self._trans_flag(
                    n_block_curr,
                    n_block_all=self.nBlocks[ii + 1],
                    inScale=scale_list[ii],
                )

                if args.no_fusion:
                    m, nOuts = self._build_norm_block(
                        nIn,
                        steps[n_block_curr - 1],
                        growth_rate,
                        args.reduction,
                        trans,
                        bnFactor=self.bnFactor[ii],
                    )
                    if n_block_curr > self.nBlocks[ii]:
                        if args.stepmode == "even":
                            steps.append(args.step)
                        elif args.stepmode == "lg":
                            steps.append(steps[-1] + args.step)
                else:
                    if n_block_curr > self.nBlocks[ii]:
                        m, nOuts = self._build_norm_block(
                            nIn,
                            steps[n_block_curr - 1],
                            growth_rate,
                            args.reduction,
                            trans,
                            bnFactor=self.bnFactor[ii],
                        )
                        if args.stepmode == "even":
                            steps.append(args.step)
                        elif args.stepmode == "lg":
                            steps.append(steps[-1] + args.step)
                        else:
                            raise NotImplementedError
                    else:
                        if n_block_curr in self.nBlocks[: ii + 1][-(scale_list[ii] - 1) :]:
                            m, nOuts = self._build_upNorm_block(
                                nIn,
                                nIn_lowFtrs[i],
                                steps[n_block_curr - 1],
                                growth_rate,
                                args.reduction,
                                trans,
                                down=True,
                                bnFactor1=self.bnFactor[ii],
                                bnFactor2=self.bnFactor[ii - 1],
                            )
                        else:
                            m, nOuts = self._build_upNorm_block(
                                nIn,
                                nIn_lowFtrs[i],
                                steps[n_block_curr - 1],
                                growth_rate,
                                args.reduction,
                                trans,
                                down=False,
                                bnFactor1=self.bnFactor[ii],
                                bnFactor2=self.bnFactor[ii - 1],
                            )

                nIn = nOuts[-1]
                scale_flow.append(m)

                if n_block_curr > self.nBlocks[ii]:
                    if args.data.startswith("cifar100"):
                        self.classifier.append(self._build_classifier_cifar(nIn, 100))
                    elif args.data.startswith("cifar10"):
                        self.classifier.append(self._build_classifier_cifar(nIn, 10))
                    elif args.data == "imagenet100":
                        self.classifier.append(
                            self._build_classifier_imagenet(nIn, 100)
                        )
                    elif args.data == "ImageNet":
                        self.classifier.append(
                            self._build_classifier_imagenet(nIn, 1000)
                        )
                    elif args.data == "distech":
                        self.classifier.append(
                            self._build_classifier_imagenet(nIn, 3)
                        )
                    elif args.data == "tinyimagenet":
                        self.classifier.append(
                            self._build_classifier_imagenet(nIn, 200)
                        )
                    else:
                        raise NotImplementedError

                _nIn_lowFtrs.append(nOuts[:-1])
                n_block_curr += 1

            nIn_lowFtrs = _nIn_lowFtrs
            self.scale_flows.append(scale_flow)

        args.num_exits = len(self.classifier)
        self.detach_scales = args.detach_scales
        self.use_dfs = bool(args.use_dfs)
        self.dfs_beta = max(min(float(args.dfs_beta), 1.0), 0.0)
        self.dfs_use_dns = (
            self.use_dfs if args.dfs_use_dns is None else bool(args.dfs_use_dns)
        )
        self.dfs_use_fr = (
            self.use_dfs if args.dfs_use_fr is None else bool(args.dfs_use_fr)
        )
        self.dfs_on_fusion = (
            self.use_dfs if args.dfs_on_fusion is None else bool(args.dfs_on_fusion)
        )
        self.dfs_on_classifier = (
            self.use_dfs
            if args.dfs_on_classifier is None
            else bool(args.dfs_on_classifier)
        )
        if not self.dfs_use_dns:
            self.dfs_use_fr = False

        for m in self.scale_flows:
            for _m in m.modules():
                self._init_weights(_m)

        for m in self.classifier:
            for _m in m.modules():
                self._init_weights(_m)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            # MODIFIED
            if m.bias is not None:
                m.bias.data.zero_()

    def _build_norm_block(
        self, nIn, step, growth_rate, reduction_rate, trans, bnFactor=2
    ):

        block = _BlockNormal(
            step, nIn, growth_rate, reduction_rate, trans, bnFactor=bnFactor,
            use_se=self.use_se, se_reduction=self.se_reduction
        )
        nOuts = []
        for i in range(step + 1):
            nOut = nIn + i * growth_rate
            nOuts.append(nOut)
        if trans:
            nOut = math.floor(1.0 * reduction_rate * nOut)
        nOuts.append(nOut)

        return block, nOuts

    def _build_upNorm_block(
        self,
        nIn,
        nIn_lowFtr,
        step,
        growth_rate,
        reduction_rate,
        trans,
        down,
        bnFactor1=1,
        bnFactor2=2,
    ):
        compress_factor = self.compress_factor

        block = _BlockUpNormal(
            step,
            nIn,
            nIn_lowFtr,
            growth_rate,
            reduction_rate,
            trans,
            down,
            compress_factor,
            bnFactor1=bnFactor1,
            bnFactor2=bnFactor2,
            use_se=self.use_se,
            se_reduction=self.se_reduction,
            skip_fusion=self.skip_fusion,
            partial_fusion=self.partial_fusion,
            detach_cross_scale=self.detach_cross_scale,
            invert_attention=self.invert_attention,
            partial_rcam=self.partial_rcam
        )
        nOuts = []
        for i in range(step + 1):
            nOut = nIn + i * growth_rate
            nOuts.append(nOut)
        nOut = nOut + math.floor(nOut * compress_factor)
        if trans:
            nOut = math.floor(1.0 * reduction_rate * nOut)
        nOuts.append(nOut)

        return block, nOuts

    def _build_regularFusion_block(
        self, nIn, nIn_lowFtr, step, growth_rate, reduction_rate, trans, bnFactor
    ):
        block = _BlockRegularFusion(
            step, nIn, nIn_lowFtr, growth_rate, reduction_rate, trans,
            bnFactor=bnFactor,
            use_se=self.use_se,
            se_reduction=self.se_reduction,
            
        )
        nOuts = []
        for i in range(step + 1):
            nOut = nIn + i * growth_rate
            nOuts.append(nOut)
        if trans:
            nOut = math.floor(1.0 * reduction_rate * nOut)
        nOuts.append(nOut)
        return block, nOuts

    def _trans_flag(self, n_block_curr, n_block_all, inScale):
        flag = False
        for i in range(inScale - 1):
            if n_block_curr == math.floor((i + 1) * n_block_all / inScale):
                flag = True
        return flag

    def _dfs_feature_reroute(self, x):
        if (not self.use_dfs) or (not self.dfs_use_dns):
            return x, x

        channels = x.shape[1]
        split = int(round(channels * self.dfs_beta))
        split = max(0, min(channels, split))

        if split >= channels:
            if self.dfs_use_fr:
                return x, x.detach()
            return x, torch.zeros_like(x, requires_grad=False)

        if split <= 0:
            if self.dfs_use_fr:
                return x.detach(), x
            return torch.zeros_like(x, requires_grad=False), x

        x_shared, x_exclusive = torch.split(x, [split, channels - split], dim=1)

        if self.dfs_use_fr:
            x_plus = torch.cat([x_shared, x_exclusive.detach()], dim=1)
            # x_sub = x
            x_sub = torch.cat([x_shared.detach(), x_exclusive], dim=1)
        else:
            x_plus = torch.cat(
                [x_shared, torch.zeros_like(x_exclusive, requires_grad=False)], dim=1
            )
            x_sub = torch.cat(
                [torch.zeros_like(x_shared, requires_grad=False), x_exclusive], dim=1
            )

        return x_plus, x_sub

    # def forward(self, x):
    #     inp = self.FirstLayer(x) 
    #     res, low_ftrs = [], []
    #     classifier_idx = 0
    #     for ii in range(self.active_scales):
    #         _x = inp[ii]
    #         if self.detach_scales:
    #             _x = inp[ii].detach() if ii > 0 else inp[ii]
    #         else:
    #             _x = inp[ii]
    #         _low_ftrs = []
    #         n_block_curr = 0
    #         for i in range(self.nBlocks[ii + 1]):
    #             if self.scale_flows[ii][i]._blockType() == "norm":
    #                 _x, _low_ftr = self.scale_flows[ii][i](_x)
    #                 _low_ftrs.append(_low_ftr)
    #             else:
    #                 if self.detach_scales:
    #                     detached_low = [f.detach() for f in low_ftrs[i]]
    #                     _x, _low_ftr = self.scale_flows[ii][i](_x, detached_low)
    #                 elif self.use_dfs and self.dfs_on_fusion:
    #                     surgered_low = []
    #                     for f in low_ftrs[i]:
    #                         f_plus, _ = self._dfs_feature_reroute(f)
    #                         surgered_low.append(f_plus)
    #                     _x, _low_ftr = self.scale_flows[ii][i](_x, surgered_low)
    #                 else:
    #                     _x, _low_ftr = self.scale_flows[ii][i](_x, low_ftrs[i])
    #                 _low_ftrs.append(_low_ftr)
    #             n_block_curr += 1

    #             if n_block_curr > self.nBlocks[ii]:
    #                 is_last_exit = classifier_idx == (len(self.classifier) - 1)
    #                 if self.use_dfs and self.dfs_on_classifier and not is_last_exit:
    #                     _x_plus, _x_sub = self._dfs_feature_reroute(_x)
    #                     res.append(self.classifier[classifier_idx](_x_sub))
    #                     _x = _x_plus
    #                 else:
    #                     res.append(self.classifier[classifier_idx](_x))
    #                 classifier_idx += 1

    #         low_ftrs = _low_ftrs
    #     return res

    def forward(self, x):
        inp = self.FirstLayer(x)
        res, feats, low_ftrs = [], [], []
        classifier_idx = 0
        for ii in range(self.active_scales):
            _x = inp[ii]
            if self.detach_scales:
                _x = inp[ii].detach() if ii > 0 else inp[ii]
            else:
                _x = inp[ii]
            _low_ftrs = []
            n_block_curr = 0
            for i in range(self.nBlocks[ii + 1]):
                if self.scale_flows[ii][i]._blockType() == "norm":
                    _x, _low_ftr = self.scale_flows[ii][i](_x)
                    _low_ftrs.append(_low_ftr)
                else:
                    if self.detach_scales:
                        detached_low = [f.detach() for f in low_ftrs[i]]
                        _x, _low_ftr = self.scale_flows[ii][i](_x, detached_low)
                    elif self.use_dfs and self.dfs_on_fusion:
                        surgered_low = []
                        for f in low_ftrs[i]:
                            f_plus, _ = self._dfs_feature_reroute(f)
                            surgered_low.append(f_plus)
                        _x, _low_ftr = self.scale_flows[ii][i](_x, surgered_low)
                    else:
                        _x, _low_ftr = self.scale_flows[ii][i](_x, low_ftrs[i])
                    _low_ftrs.append(_low_ftr)
                n_block_curr += 1

                if n_block_curr > self.nBlocks[ii]:
                    is_last_exit = classifier_idx == (len(self.classifier) - 1)
                    if self.use_dfs and self.dfs_on_classifier and not is_last_exit:
                        _x_plus, _x_sub = self._dfs_feature_reroute(_x)
                        logits_i, feat_i = self.classifier[classifier_idx](_x_sub)
                        _x = _x_plus
                    else:
                        logits_i, feat_i = self.classifier[classifier_idx](_x)
                    res.append(logits_i)
                    feats.append(feat_i)
                    classifier_idx += 1

            low_ftrs = _low_ftrs
        return res, feats

    def _build_classifier_cifar(self, nIn, num_classes):
        interChannels1, interChannels2 = 128, 128
        conv = nn.Sequential(
            ConvBasic(nIn, interChannels1, kernel=3, stride=2, padding=1),
            ConvBasic(interChannels1, interChannels2, kernel=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        return ClassifierModule(conv, interChannels2, num_classes)

    def _build_classifier_imagenet(self, nIn, num_classes):
        conv = nn.Sequential(
            ConvBasic(nIn, nIn, kernel=3, stride=2, padding=1),
            ConvBasic(nIn, nIn, kernel=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        return ClassifierModule(conv, nIn, num_classes)


# class ClassifierModule(nn.Module):
#     def __init__(self, m, channel, num_classes):
#         super(ClassifierModule, self).__init__()
#         self.m = m
#         self.linear = nn.Linear(channel, num_classes)

#     def forward(self, x):
#         res = self.m(x)
#         res = res.view(
#             res.size(0), -1
#         )  # tensor de forma (B, N) onde N = C * H * W (se res era um tensor 4D [B, C, H, W])
#         return self.linear(res)

class ClassifierModule(nn.Module):
    def __init__(self, m, channel, num_classes):
        super(ClassifierModule, self).__init__()
        self.m = m
        self.linear = nn.Linear(channel, num_classes)

    def forward(self, x):
        res = self.m(x)
        feat = res.view(res.size(0), -1)  # F^i: feature resized/flattened, pre-classificador
        logits = self.linear(feat)
        return logits, feat


if __name__ == "__main__":
    from args_v5 import arg_parser
    from op_counter import measure_model

    args = arg_parser.parse_args()
    # if args.gpu:
    #   os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args.nBlocks = 2
    args.Block_base = 2
    args.step = 8
    args.stepmode = "even"
    args.compress_factor = 0.25
    args.nChannels = 64
    args.data = "ImageNet"
    args.growthRate = 16

    args.grFactor = "4-2-2-1"
    args.bnFactor = "4-2-2-1"
    args.scale_list = "1-2-3-4"

    args.reduction = 0.5

    args.grFactor = list(map(int, args.grFactor.split("-")))
    args.bnFactor = list(map(int, args.bnFactor.split("-")))
    args.scale_list = list(map(int, args.scale_list.split("-")))
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
    elif args.data == "distech":
        args.num_classes = 3
    elif args.data == "imagenet100":
        args.num_classes = 100
    else:
        args.num_classes = 3

    inp_c = torch.rand(16, 3, 224, 224)

    model = MSDNet(args)

    n_flops, n_params = measure_model(model, 224, 224)
    # net = _BlockNormal(num_layers = 4, nIn = 64, growth_rate = 24, reduction_rate = 0.5, trans_down = True)