"""Export DM-Count (Wang et al., NeurIPS 2020, MIT licence) to ONNX for the OpenCV 5 DNN engine.

Weights: the lwcc release assets (tersekmatija/lwcc_weights v0.1), loaded as a plain state
dict with torch.load(weights_only=True). The network is VGG-19 features, a 2x bilinear
upsample and a small regression head; the output density map has 1/8 of the input size and
sums to the people count.

    python tools/export_dmcount.py models/src/DM-Count_SHA.pth models/dmcount_sha.onnx
"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512]


def make_features():
    layers, c = [], 3
    for v in CFG:
        if v == "M":
            layers.append(nn.MaxPool2d(2, 2))
        else:
            layers += [nn.Conv2d(c, v, 3, padding=1), nn.ReLU(inplace=True)]
            c = v
    return nn.Sequential(*layers)


class DMCount(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = make_features()
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.density_layer = nn.Sequential(nn.Conv2d(128, 1, 1), nn.ReLU())

    def forward(self, x):
        x = self.features(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return self.density_layer(self.reg_layer(x))


def main(src: str, dst: str) -> None:
    sd = torch.load(src, map_location="cpu", weights_only=True)
    sd = sd.get("model", sd)
    net = DMCount().eval()
    net.load_state_dict(sd, strict=True)
    x = torch.randn(1, 3, 512, 640)
    torch.onnx.export(
        net, x, dst, opset_version=17, input_names=["image"], output_names=["density"],
        dynamic_axes={"image": {2: "h", 3: "w"}, "density": {2: "h8", 3: "w8"}},
        dynamo=False,
    )
    print("wrote", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
