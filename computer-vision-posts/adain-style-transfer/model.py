import torch
from torch import nn

# vgg encoder model
class VGGEncoder(nn.Module):

    def __init__(self, weight_path=None) -> None:
        super(VGGEncoder, self).__init__()
        # layers to extract features from
        self.feature_layers = [3, 10, 17, 30]

        # creating model and adding first layer
        self.model = nn.ModuleList()
        self.model.append(nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(1, 1)))
        self.model.append(nn.ReflectionPad2d((1, 1, 1, 1)))

        # parameters for remaining layers
        # (in_channels, out_channels, kernel_size)
        params = [
            (3, 64, (3, 3)),
            (64, 64, (3, 3)),
            (64, 128, (3, 3)),
            (128, 128, (3, 3)),
            (128, 256, (3, 3)),
            (256, 256, (3, 3)),
            (256, 256, (3, 3)),
            (256, 256, (3, 3)),
            (256, 512, (3, 3)),
            (512, 512, (3, 3)),
            (512, 512, (3, 3)),
            (512, 512, (3, 3)),
            (512, 512, (3, 3)),
            (512, 512, (3, 3)),
            (512, 512, (3, 3)),
            (512, 512, (3, 3))
        ]

        # adding layers to model
        # also adding a maxpool layer whenever number of out_channels changes
        for i, param in enumerate(params[:-1]):
            self.model.append(nn.Conv2d(in_channels=param[0], out_channels=param[1], kernel_size=param[2]))
            self.model.append(nn.ReLU())
            if params[i + 1][0] != params[i + 1][1]:
                self.model.append(nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=(0,0), ceil_mode=True))
            self.model.append(nn.ReflectionPad2d((1, 1, 1, 1)))

        # inserting extra maxpool layer based on vgg architecture
        self.model.insert(40, nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=(0,0), ceil_mode=True))

        # adding last layer
        self.model.append(nn.Conv2d(in_channels=params[-1][0], out_channels=params[-1][1], kernel_size=params[-1][2]))
        self.model.append(nn.ReLU())

        # loading pretrained model weights if path is provided
        if weight_path:
            self.model.load_state_dict(torch.load(weight_path, map_location="cpu"))
            
        # encoder is fully pretrained, so no weights need to be adjusted
        # gradients need not be accumulated
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        # list to store activations from layers in self.feature_layers
        activations = []
        for i, layer in enumerate(self.model[:31]):
            x = layer(x)
            if i in self.feature_layers:
                activations.append(x)
        return activations


class VGGDecoder(nn.Module):

    def __init__(self, encoder, weight_path=None) -> None:
        super(VGGDecoder, self).__init__()
        # decoder model
        self.decoder = nn.ModuleList()

        # iterate through encoder layer in reverse order, not including the last 2
        # interchange reflectionpad and relu layers
        # replace maxpool with upsample
        # interchange in_channels and out_channels in cover layers
        for layer in encoder.model[:31][::-1][:-2]:
            if isinstance(layer, nn.ReflectionPad2d):
                self.decoder.append(nn.ReLU())
            elif isinstance(layer, nn.ReLU):
                self.decoder.append(nn.ReflectionPad2d((1, 1, 1, 1)))
            elif isinstance(layer, nn.MaxPool2d):
                self.decoder.append(nn.Upsample(scale_factor=2, mode='nearest'))
            elif isinstance(layer, nn.Conv2d):
                layer = nn.Conv2d(
                    in_channels=layer.out_channels,
                    out_channels=layer.in_channels,
                    kernel_size=layer.kernel_size
                )
                self.decoder.append(layer)

        # load weights if weight path provided
        if weight_path:
            self.decoder.load_state_dict(torch.load(weight_path, map_location="cpu"))
            
    def forward(self, x):
        for layer in self.decoder:
            x = layer(x)
        return x
        


def AdaIN_realign(style, content):
    # flatten images while retaining batchs and channels, compute mean and std
    B, C = content.shape[0], content.shape[1]
    content_mean = content.view(B, C, -1).mean(dim=2)
    content_std = content.view(B, C, -1).std(dim=2)
    style_mean = style.view(B, C, -1).mean(dim=2)
    style_std = style.view(B, C, -1).std(dim=2)

    # reshape mean and std to perform normalization and realignment
    content_mean, content_std = content_mean.view(B, C, 1, 1), content_std.view(B, C, 1, 1)
    style_mean, style_std = style_mean.view(B, C, 1, 1), style_std.view(B, C, 1, 1)

    content_mean, content_std = content_mean.expand(content.size()), content_std.expand(content.size())
    style_mean, style_std = style_mean.expand(style.size()), style_std.expand(style.size())

    # normalize content features. Small constant added to avoid zero division.
    normalized = (content - content_mean) / (content_std + 0.00001)
    # realign normalized content features with style mean and std
    realigned = (normalized * style_std) + style_mean
    return realigned
        

def test_models():
    enc = VGGEncoder()
    dec = VGGDecoder()

    content = torch.randn((8, 3, 256, 256))
    style = torch.randn((8, 3, 256, 256))

    content_enc = enc(content)[-1]
    style_enc = enc(style)[-1]
    content_realigned = AdaIN_realign(style_enc, content_enc)
    decoded = dec(content_realigned)

    print(content.shape)
    print(style.shape)
    print(content_enc.shape)
    print(style_enc.shape)
    print(content_realigned.shape)
    print(decoded.shape)


if __name__ == "__main__":
    test_models()