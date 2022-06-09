# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detr.models.detr import build


def build_model(args, num_classes, device):
    return build(args, num_classes, device)
