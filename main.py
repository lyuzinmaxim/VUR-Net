from VURNet import *
from dataset_generation import *
from train import *
import csv
import os
import hydra
from omegaconf import DictConfig

n = 50
dataset = np.empty([n, 256, 256])
for i in range(n):
    if i % 2 == 0:
        size = np.random.permutation(np.arange(2, 15, 1))[0]
        dataset[i] = create_dataset_element(size, 256, 4, 20)
    else:
        num_gauss = np.random.permutation(np.arange(1, 7, 1))[0]
        dataset[i] = make_gaussian(
            num_gauss,
            sigma_min=1,
            sigma_max=4,
            shift_max=4,
            magnitude_min=2,
            magnitude_max=20)

dataset_torch = torch.from_numpy(dataset)
dataset_unsqueezed = dataset_torch.unsqueeze(1).float()
X = wraptopi(dataset_unsqueezed);

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X[:, :, :, :],
    dataset_unsqueezed[:, :, :, :],
    test_size=0.3,
    shuffle=True)

print(X_train.shape, 'Размерность тренировочных картинок "wrapped phase"')
print(X_test.shape, 'Размерность тестовых картинок "wrapped phase"')
print(Y_train.shape, 'Размерность тренировочных картинок ground truth')
print(Y_test.shape, 'Размерность тестовых картинок ground truth')

print(X_test.shape)

model_VURNet = VURnet()


def model_train(
        model,
        name,
        batch_size,
        total_epochs,
        learning_rate,
        loss_freq,
        metric_freq,
        lr_freq,
        save_freq):
    """
    That function makes train process easier, only optimizer hyperparameters
    should be defined in function manually

    function returns:
    1. trained model
    2. list of metric history for every "metric_freq" epoch
    3. list of losses history for every "loss_freq" epoch
    4. list of train losses history for every "loss_freq" epoch

    args:
    model - torch.nn.Module object - defined model
    name - string, model checkpoints will be saved with this name
    batch size - integer, defines number of images in one batch
    total epoch - integer, defines number of epochs for learning
    learning rate - float, learning rate of an optimizer
    loss_freq - integer, loss function will be computed every "loss_freq" epochs
    metric_freq - integer, metric (AU) -||-
    lr_freq - integer, learning rate will be decreased -||-
    save_freq - integer, model checkpoints for train and validation will
                be saved  -||-

    *time computing supports only GPU calculations
    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

    model = model.to(device)
    print('[INFO] Model will be learned on {}'.format(device))

    metric_history = []
    test_loss_history = []
    train_loss_history = []
    train_loss_epoch = 0

    loss = torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')
    # loss = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    if device.type == 'cuda':
        start.record()

    for epoch in np.arange(0, total_epochs, 1):

        print('>> Epoch: {}/{} Learning rate: {}'.format(epoch, total_epochs, learning_rate))

        order = np.random.permutation(len(X_train))

        for start_index in range(0, len(X_train), batch_size):
            optimizer.zero_grad()
            model.train()
            batch_indexes = order[start_index:start_index + batch_size]

            X_batch = X_train[batch_indexes].to(device)
            Y_batch = Y_train[batch_indexes].to(device)

            preds = model.forward(X_batch)

            loss_value = loss(preds, Y_batch)
            loss_value.backward()

            train_loss_epoch += loss_value.item()

            optimizer.step()
            ##### memory optimization start #####
            # GPUtil.showUtilization()

            del X_batch, Y_batch
            torch.cuda.empty_cache()

            # GPUtil.showUtilization()
            ##### memory optimization end #####

        train_loss_history.append(train_loss_epoch)
        print('[LOSS TRAIN] mean value of MSE {:.4f} on train set at epoch number {}'.format(train_loss_epoch, epoch))
        train_loss_epoch = 0

        if epoch % loss_freq == 0:
            test_per_batch = []
            print('[INFO] beginning to calculate loss')
            model.eval()
            order_test = np.random.permutation(len(X_test))

            for start_index_test in range(0, len(X_test), batch_size):
                test_per_batch = []

                batch_indexes_test = order_test[start_index_test:start_index_test + batch_size]

                with torch.no_grad():
                    X_batch_test = X_test[batch_indexes_test].to(device)
                    Y_batch_test = Y_train[batch_indexes_test].to(device)

                    test_preds = model.forward(X_batch_test)
                    metric_loss = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')

                    test_loss = metric_loss(test_preds, Y_batch_test)
                    test_per_batch.append(test_loss.data.cpu())

                    ##### memory optimization start #####
                    del X_batch_test, Y_batch_test
                    torch.cuda.empty_cache()
                    ##### memory optimization end #####

            test_loss_epoch = sum(test_per_batch) / len(test_per_batch)
            test_loss_history.append(test_loss_epoch.tolist())

            print('[LOSS TEST] mean value of MSE {:.4f} on test set at epoch number {}'.format(test_loss_epoch, epoch))

        if epoch % metric_freq == 0:
            model.eval()

            order_metric = np.random.permutation(len(X_test))

            for start_index_metric in range(0, len(X_test), batch_size):
                metric_per_batch = []

                batch_indexes_metric = order_metric[start_index_metric:start_index_metric + batch_size]

                with torch.no_grad():
                    X_batch_metric = X_test[batch_indexes_metric].to(device)

                    Y_batch_metric = Y_test[batch_indexes_metric]

                    metric_preds = model.forward(X_batch_metric)

                    # mean_au,_ = au_and_bem_torch(Y_batch_metric,metric_preds.detach().to('cpu'),calc_bem=False)
                    mean_au_batch, _ = au_and_bem_torch(metric_preds.detach().to('cpu'), Y_batch_metric, calc_bem=False)

                    metric_per_batch.append(mean_au_batch)
                    # metric_per_batch.append(mean_au_batch.data.cpu())

                    ##### memory optimization start #####
                    # GPUtil.showUtilization()
                    del X_batch_metric, Y_batch_metric, metric_preds
                    torch.cuda.empty_cache()
                    # GPUtil.showUtilization()
                    ##### memory optimization end #####

            test_metric_epoch = sum(metric_per_batch) / len(metric_per_batch)
            metric_history.append(test_metric_epoch)
            print('[METRIC] Accuracy of unwrapping on test images is {:.4f} %,'.format(test_metric_epoch * 100))

        if epoch % save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, '{}/{}_checkpoint_{}'.format(path, name, epoch))
            print('[SAVE] {}/{}_checkpoint_{} was saved'.format(path, name, epoch), )

        if (epoch + 1) % lr_freq == 0:
            learning_rate /= 2
            update_lr(optimizer, learning_rate)
            print('[lr]New learning rate: {}'.format(learning_rate))

    print('[END]Learning is done')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
        # ,'lr': learning_rate
    }, '{}/{}_checkpoint_end'.format(path, name))
    print('[END]{}/{}_checkpoint_end was saved'.format(path, name))

    if device.type == 'cuda':
        end.record()
        torch.cuda.synchronize()
        print('Learning time is {:.1f} min'.format(start.elapsed_time(end) / (1000 * 60)))

    with open('{}/metric_{}.csv'.format(path, name), 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
        wr.writerow(metric_history)
        print('Metric was saved')

    with open('{}/test_loss_{}.csv'.format(path, name), 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
        wr.writerow(test_loss_history)
        print('Test loss was saved')

    with open('{}/train_loss_{}.csv'.format(path, name), 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
        wr.writerow(train_loss_history)
        print('Train loss was saved')

    return model, metric_history, test_loss_history, train_loss_history


working_dir = os.getcwd()
path = os.path.join(working_dir, "model")

try:
    os.mkdir(path)
except OSError as error:
    print('directory exists')

print(f"The current base directory is {working_dir}")


@hydra.main(config_path=os.path.join(working_dir, "config.yml"))
def train(cfg: DictConfig):
    working_dir = os.getcwd()
    print(f"The current working directory is {working_dir}")

    # To access elements of the config
    print(f"The batch size is {cfg.batch_size}")
    print(f"The learning rate is {cfg.lr}")
    print(f"Total epochs: {cfg['total_epochs']}")

    trained_model, list_metric, list_test_loss, list_train_loss = model_train(
        model=model_VURNet,
        name=cfg.name,
        batch_size=cfg.batch_size,
        total_epochs=cfg.total_epochs,
        learning_rate=cfg.lr,
        loss_freq=cfg.loss_freq,
        metric_freq=cfg.metric_freq,
        lr_freq=cfg.lr_freq,
        save_freq=cfg.save_freq)


if __name__ == "__main__":
    train()
