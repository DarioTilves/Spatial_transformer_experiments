import torch
from models.coord_conv import CoordConv2d


class ConvLocalisation(torch.nn.Sequential):
    def __init__(self):
        super(ConvLocalisation, self).__init__(
            torch.nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 7),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            torch.nn.ReLU(inplace = True),
            torch.nn.Conv2d(in_channels = 8, out_channels = 10, kernel_size = 5),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            torch.nn.ReLU(inplace = True))


class ConvCoordLocalisation(torch.nn.Sequential):
    def __init__(self):
        super(ConvCoordLocalisation, self).__init__(
            CoordConv2d(in_channels = 1, out_channels = 8, kernel_size = 7),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            torch.nn.ReLU(inplace = True),
            CoordConv2d(in_channels = 8, out_channels = 10, kernel_size = 5),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            torch.nn.ReLU(inplace = True))


class LocalisationRegressor(torch.nn.Sequential):
    def __init__(self):
        super(LocalisationRegressor, self).__init__(
            torch.nn.Linear(in_features = 10 * 3 * 3, out_features = 32),
            torch.nn.ReLU(inplace = True),
            torch.nn.Linear(in_features = 32, out_features = 3 * 2))


class LocalisationNet(torch.nn.Module):
    def __init__(self, conv_localisation: bool = True):
        super(LocalisationNet, self).__init__()
        if conv_localisation:
            self.localisation = ConvLocalisation()
        else:
            self.localisation = ConvCoordLocalisation()
        self.localisation_regressor = LocalisationRegressor()
        self.localisation_regressor[2].weight.data.zero_()
        self.localisation_regressor[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, input):
        x = self.localisation(input)
        x = x.view(-1, 10 * 3 * 3)
        theta = self.localisation_regressor(x)
        theta = theta.view(-1, 2, 3)
        return theta


class ConvClassification(torch.nn.Sequential):
    def __init__(self):
        super(ConvClassification, self).__init__(
            torch.nn.Conv2d(in_channels = 1, out_channels = 10, kernel_size=5),
            torch.nn.MaxPool2d(kernel_size = 2),
            torch.nn.ReLU(inplace = True),
            torch.nn.Conv2d(in_channels = 10, out_channels = 20, kernel_size=5),
            torch.nn.Dropout2d(),
            torch.nn.MaxPool2d(kernel_size = 2),
            torch.nn.ReLU(inplace = True),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features = 320, out_features = 50),
            torch.nn.ReLU(inplace = True),
            torch.nn.Linear(in_features = 50, out_features = 10))


class ConvCoordClassification(torch.nn.Sequential):
    def __init__(self):
        super(ConvCoordClassification, self).__init__(
            CoordConv2d(in_channels = 1, out_channels = 10, kernel_size=5),
            torch.nn.MaxPool2d(kernel_size = 2),
            torch.nn.ReLU(inplace = True),
            CoordConv2d(in_channels = 10, out_channels = 20, kernel_size=5),
            torch.nn.Dropout2d(),
            torch.nn.MaxPool2d(kernel_size = 2),
            torch.nn.ReLU(inplace = True),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features = 320, out_features = 50),
            torch.nn.ReLU(inplace = True),
            torch.nn.Linear(in_features = 50, out_features = 10))


class SpatialTransformerClassificationNet(torch.nn.Module):
    def __init__(self, conv_localisation: bool = True, conv_classification: bool = True):
        super(SpatialTransformerClassificationNet, self).__init__()
        self.localization = LocalisationNet(conv_localisation = conv_localisation)
        if conv_classification:
            self.classification = ConvClassification()
        else:
            self.classification = ConvCoordClassification()

    def spatial_transformer_net(self, input):
        theta = self.localization(input)
        grid = torch.nn.functional.affine_grid(theta = theta, size = input.size(), align_corners = False)
        v = torch.nn.functional.grid_sample(input = input, grid = grid, align_corners = False)
        return v

    def forward(self, input):
        v = self.spatial_transformer_net(input)
        x = self.classification(v)
        x = torch.nn.functional.log_softmax(input = x, dim = 1)
        return x
