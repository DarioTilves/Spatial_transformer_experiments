import torch
from models.coord_conv import CoordConv2d
from models.spatial_transformer import ConvClassification, ConvCoordClassification


class Critic(torch.nn.Sequential):
    def __init__(self):
        super(Critic, self).__init__(
            torch.nn.Linear(in_features = 137, out_features = 64),
            torch.nn.ReLU(inplace = True),
            torch.nn.Linear(in_features = 64, out_features = 1))


class ConvLenet(torch.nn.Sequential):
    def __init__(self):
        super(ConvLenet, self).__init__(
            torch.nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5),
            torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            torch.nn.ReLU(inplace = True),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features = 6400, out_features = 2048),
            torch.nn.ReLU(inplace = True),
            torch.nn.Linear(in_features = 2048, out_features = 128))


class ConvCoordLenet(torch.nn.Sequential):
    def __init__(self):
        super(ConvCoordLenet, self).__init__(
            CoordConv2d(in_channels = 1, out_channels = 32, kernel_size = 5),
            CoordConv2d(in_channels = 32, out_channels = 64, kernel_size = 5),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2),
            torch.nn.ReLU(inplace = True),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features = 6400, out_features = 2048),
            torch.nn.ReLU(inplace = True),
            torch.nn.Linear(in_features = 2048, out_features = 128))


class ReinforcedSpatialTransformerClassificationNet(torch.nn.Module):
    def __init__(self, conv_localisation: bool = True, conv_classification: bool = True, 
                 iterations: int = 20):
        super(ReinforcedSpatialTransformerClassificationNet, self).__init__()
        if conv_localisation:
            self.lenet = ConvLenet()
        else:
            self.lenet = ConvCoordLenet()
        self.localisation_lstm = torch.nn.LSTMCell(input_size = 128+9, hidden_size = 9)
        self.localisation_softmax = torch.nn.Softmax(dim = -1)
        self.iterations = iterations if iterations >= 2 else 2
        if conv_classification:
            self.classification = ConvClassification()
        else:
            self.classification = ConvCoordClassification()
        self.critic = Critic()
        self.affine_transforms = self.affine_transformations()

    def affine_transformations(self):
        identity = torch.tensor([1, 0, 0, 0, 1, 0])
        translation1 = torch.tensor([1, 0, 0.1, 0, 1, 0.1])
        translation2 = torch.tensor([1, 0, -0.15, 0, 1, -0.05])
        scale1 = torch.tensor([1.15, 0, 0, 0, 0.95, 0])
        scale2 = torch.tensor([0.8, 0, 0, 0, 1.1, 0])
        angle1 = torch.tensor(6)
        rotate1 = torch.tensor([torch.cos(angle1), -torch.sin(angle1), 0, torch.sin(angle1), torch.cos(angle1), 0])
        angle2 = torch.tensor(-8)
        rotate2 = torch.tensor([torch.cos(angle2), -torch.sin(angle2), 0, torch.sin(angle2), torch.cos(angle2), 0])
        shear1 = torch.tensor([1, 0.2, 0, 0, -0.1, 0])
        shear2 = torch.tensor([1, -0.15, 0, 0.15, 1, 0])
        affine_transforms = torch.stack((identity, translation1, translation2, scale1, scale2, rotate1, rotate2, shear1, shear2))
        return affine_transforms

    def reinforced_spatial_transformer_net(self, input):
        v = input
        images, values = ([] for i in range(2))
        action_one_hot = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0], device = input.device).repeat(input.shape[0], 1)
        for _ in range(self.iterations):
            x = self.lenet(v)
            state = torch.cat([x, action_one_hot], dim = -1)
            policy_lstm, _ = self.localisation_lstm(state)
            policy = self.localisation_softmax(policy_lstm)
            action_dist = torch.distributions.categorical.Categorical(policy)
            action_sampled = action_dist.sample()
            action_one_hot = torch.nn.functional.one_hot(action_sampled)
            if action_one_hot.size()[1] < 9:
                action_one_hot = torch.nn.functional.pad(action_one_hot, (0, 9 - action_one_hot.size()[1]), "constant", 0)
            action = self.affine_transforms[action_sampled].to(device = input.device)
            grid = torch.nn.functional.affine_grid(theta = action.view(-1, 2, 3), size = input.size(), align_corners = False)
            v = torch.nn.functional.grid_sample(input = v, grid = grid, align_corners = False)
            images.append(v)
            values.append(self.critic(state))
        return v, images, values, action_dist

    def forward(self, input):
        classifications = []
        _, images, values, distribution = self.reinforced_spatial_transformer_net(input)
        for image in images:
            x = self.classification(image)
            classifications.append(torch.nn.functional.log_softmax(input = x, dim = 1))
        return {'output': classifications[-1], 'classifications': classifications, 'values': values, 'distribution': distribution}