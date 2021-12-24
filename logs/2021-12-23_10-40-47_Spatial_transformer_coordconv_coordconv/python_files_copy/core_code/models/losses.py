import torch


class ReinforcedLosses(torch.nn.Module):
    def __init__(self, iterations: int = 20, gamma: float = 0.98):
        super(ReinforcedLosses, self).__init__()
        self.iterations = iterations if iterations >= 2 else 2
        self.gamma = torch.tensor(gamma)
        self.mse = torch.nn.MSELoss()
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def determine_rewards(self, input_dict):
        rewards = [torch.tensor(0)]
        reward_old = self.cross_entropy(input_dict['classifications'][0], input_dict['labels'])
        for timestamp in range(1, self.iterations):
            reward = self.cross_entropy(input_dict['classifications'][timestamp], input_dict['labels'])
            rewards.append(torch.pow(self.gamma, self.iterations - timestamp)*(reward_old - reward))
            reward_old = reward
        return rewards

    def determine_mse(self, values, rewards):
        timesteps = len(values)
        mse_loss = 0
        for timestamp in range(timesteps):
            mse_loss += self.mse(values[timestamp], rewards[timestamp])
        mse_loss /= timesteps
        return mse_loss

    def determine_log_prob(self, values, rewards, distribution):
        prob_loss = 0
        for timestamp in range(len(values)):
            prob_loss += rewards[timestamp] - values[timestamp]
        prob_loss[prob_loss < 0] = 0
        prob_loss[prob_loss > 8] = 8
        log_prob_loss = -torch.mean(distribution.log_prob(prob_loss.long()))
        return log_prob_loss

    def forward(self, input_dict):
        values = [torch.mean(value_timestamp) for value_timestamp in input_dict['values']]
        timestamp = torch.randint(low = 1, high = self.iterations, size = (1,))
        rewards = self.determine_rewards(input_dict)
        mse_loss = self.determine_mse(values[timestamp:], rewards[timestamp:])
        log_prob_loss = self.determine_log_prob(values[timestamp:], rewards[timestamp:], 
                                                distribution = input_dict['distribution'])
        cross_entropy_loss = self.cross_entropy(input_dict['output'], input_dict['labels'])
        loss = mse_loss + log_prob_loss + cross_entropy_loss
        return loss
