import torch.nn as nn

def vgg_layers(in_channel, cfg, batch_norm=False):
    layers = []
    in_channels = in_channel
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)

            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self, class_num=10):
        super(VGG, self).__init__()
        self.features1 = vgg_layers(3, [64, 64], batch_norm=True)
        self.features2 = vgg_layers(64, ['M', 128, 128], batch_norm=True)
        self.features3 = vgg_layers(128, ['M', 256, 256, 256, 256], batch_norm=True)
        self.features4 = vgg_layers(256, ['M', 512, 512, 512, 512], batch_norm=True)
        self.features5 = vgg_layers(512, ['M', 512, 512, 512, 512, 'M'], batch_norm=True)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, class_num),
        )

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        x = self.features4(x)
        x = self.features5(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)

        return out


def vgg(num_classes):
    net = VGG(num_classes)
    return net