import torch
from models.spatial_transformer import LocalisationRegressor 
from models.spatial_transformer import ConvLocalisation, ConvCoordLocalisation 
from models.spatial_transformer import ConvClassification, ConvCoordClassification


class SequentialSpatialTransformerClassificationNet(torch.nn.Module):
    def __init__(self, conv_localisation: bool = True, conv_classification: bool = True, 
                 iterations: int = 20):
        super(SequentialSpatialTransformerClassificationNet, self).__init__()
        if conv_localisation:
            self.localisation = ConvLocalisation()
        else:
            self.localisation = ConvCoordLocalisation()
        self.localisation_regressor = LocalisationRegressor()
        self.localisation_regressor[2].weight.data.zero_()
        self.localisation_regressor[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.localisation_lstm = torch.nn.LSTMCell(input_size = 12, hidden_size = 6)
        self.iterations = iterations if iterations >= 1 else 1
        if conv_classification:
            self.classification = ConvClassification()
        else:
            self.classification = ConvCoordClassification()

    def sequential_spatial_transformer_net(self, input):
        v = input
        theta_featured = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float, device = input.device).repeat(input.shape[0], 1)
        for _ in range(self.iterations):
            x = self.localisation(input)
            x = x.view(-1, 10 * 3 * 3)
            x = self.localisation_regressor(x)
            x = torch.cat([x, theta_featured], dim = -1)
            theta_featured, _ = self.localisation_lstm(x)
            theta = theta_featured.view(-1, 2, 3)
            grid = torch.nn.functional.affine_grid(theta = theta, size = input.size(), align_corners = False)
            v = torch.nn.functional.grid_sample(input = input, grid = grid, align_corners = False)
        return v

    def forward(self, input):
        v = self.sequential_spatial_transformer_net(input)
        x = self.classification(v)
        x = torch.nn.functional.log_softmax(input = x, dim = 1)
        return x  