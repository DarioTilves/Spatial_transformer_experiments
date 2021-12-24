import torch 
import random
import torchvision
import numpy as np
import pandas as pd
import seaborn as sn
from six.moves import urllib
import matplotlib.pyplot as plt
from utils import get_configuration_file_and_get_project_path, log_training_files_and_get_log_path, visualize_stn
from models.spatial_transformer import SpatialTransformerClassificationNet
from models.sequential_spatial_transformer import SequentialSpatialTransformerClassificationNet
from models.reinforced_spatial_transformer import ReinforcedSpatialTransformerClassificationNet
from models.losses import ReinforcedLosses
from metrics.metric import accumulate_confusion_matrix, calculate_metrics, save_metrics 


def main():
    gen = torch.Generator()
    gen.manual_seed(0)
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    conf_dict, python_path = get_configuration_file_and_get_project_path()
    log_path = log_training_files_and_get_log_path(configuration_dict = conf_dict, python_path = python_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_transform = torchvision.transforms.Compose([ torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(
                    torchvision.datasets.MNIST(root = '.', train = True, 
                        download = True, transform = torch_transform), 
                    generator = gen,
                    **conf_dict['dataloader_parameters'])
    test_loader = torch.utils.data.DataLoader(
                    torchvision.datasets.MNIST(root = '.', train = False, transform = torch_transform), 
                    generator = gen,
                    **conf_dict['dataloader_parameters'])
    if conf_dict['model_selection']['Spatial_transformer'] == 0:
        model = SpatialTransformerClassificationNet(**conf_dict['conv_coord_selection']).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr = conf_dict['model_hyperparameters']['learning_rate'])

    elif conf_dict['model_selection']['Spatial_transformer'] == 1:
        model = SequentialSpatialTransformerClassificationNet(iterations = conf_dict['model_selection']['iterations'],
                                                              **conf_dict['conv_coord_selection']).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr = conf_dict['model_hyperparameters']['learning_rate'])
    else:
        model = ReinforcedSpatialTransformerClassificationNet(iterations = conf_dict['model_selection']['iterations'],
                                                              **conf_dict['conv_coord_selection']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = conf_dict['model_hyperparameters']['learning_rate'])
        loss_class = ReinforcedLosses(iterations = conf_dict['model_selection']['iterations'],
                                      gamma = conf_dict['model_selection']['loss_gamma'])
    for epoch in range(conf_dict['model_hyperparameters']['epoch_end']):
        model.train()
        confusion_matrix = np.zeros([len(classes), len(classes)]).astype(int)
        train_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            for p in model.parameters(): p.grad = None            
            output = model(data)
            if conf_dict['model_selection']['Spatial_transformer'] == 2:
                output['labels'] = target
                loss = loss_class.forward(output)
                output = output['output']
            else:
                loss = torch.nn.functional.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            confusion_matrix = accumulate_confusion_matrix(confusion_matrix = confusion_matrix, 
                                                           targets = target.cpu().detach().numpy(), 
                                                           predictions = output.max(1, keepdim=True)[1].cpu().detach().numpy()[:,0])
        train_loss /= len(train_loader.dataset)
        metrics_dict = calculate_metrics(confusion_matrix)
        if conf_dict['visualizations']['print_metrics']:
            print('\nTrain epoch {} loss {:.4f}, mean accuracy {:.4f}, mean precision {:.4f},'.format(
                  epoch, loss, metrics_dict['accuracy'], metrics_dict['precision']))
            print('mean sensitivity {:.4f}, mean specifity {:.4f}, mean f1 score {:.4f}'.format(
                  metrics_dict['sensitivity'], metrics_dict['specifity'], metrics_dict['f1score']))
        with torch.no_grad():
            model.eval()
            confusion_matrix_test = np.zeros([len(classes), len(classes)]).astype(int)
            test_loss = 0
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                if conf_dict['model_selection']['Spatial_transformer'] == 2:
                    output['labels'] = target
                    loss = loss_class.forward(output)
                    output = output['output']
                    test_loss += loss.item()
                else:
                    test_loss += torch.nn.functional.nll_loss(output, target, size_average=False).item()
                confusion_matrix_test = accumulate_confusion_matrix(confusion_matrix = confusion_matrix_test, 
                                                                    targets = target.cpu().detach().numpy(), 
                                                                    predictions = output.max(1, keepdim=True)[1].cpu().detach().numpy()[:,0])
            test_loss /= len(test_loader.dataset)
            metrics_test = calculate_metrics(confusion_matrix_test)
            if conf_dict['visualizations']['print_metrics']:
                print('\nTest epoch {} loss {:.4f}, mean accuracy {:.4f}, mean precision {:.4f},'.format(
                  epoch, loss, metrics_test['accuracy'], metrics_test['precision']))
                print('mean sensitivity {:.4f}, mean specifity {:.4f}, mean f1 score {:.4f}'.format(
                  metrics_test['sensitivity'], metrics_test['specifity'], metrics_test['f1score']))          
            if conf_dict['visualizations']['save_metrics']:
                save_metrics(epoch = epoch, save_dict = metrics_test, save_path = log_path + '/summary.csv')
    cf_matrix_test = confusion_matrix_test / np.sum(confusion_matrix_test, axis = 1)
    df_cm = pd.DataFrame(cf_matrix_test, index = [i for i in classes], columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    if conf_dict['visualizations']['save_cf_matrix']:
        plt.savefig(log_path + '/confusion_matrix_test.png')
    if conf_dict['visualizations']['show_cf_matrix']:
        plt.show()
    plt.close()
    visualize_stn(loader = test_loader, model = model, device = device, 
                  mode = conf_dict['model_selection']['Spatial_transformer'],
                  iterations = conf_dict['model_selection']['iterations'])
    if conf_dict['visualizations']['save_stn']:
        plt.savefig(log_path + '/transformation.png')
    if conf_dict['visualizations']['show_stn']:
        plt.show()
    plt.close()


if __name__ == '__main__':
    torch.manual_seed(100)
    random.seed(100)
    np.random.seed(100)
    main()