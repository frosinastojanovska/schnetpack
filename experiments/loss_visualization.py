import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_loss(data, data_labels_to_plot, save_file, x_label='', y_label='', title=''):
    assert len(data_labels_to_plot) > 0
    plt.style.use('classic')
    sns.set()
    sns.set_style("ticks")
    sns.set_palette("Paired")
    x = np.array(range(list(data.values())[0].shape[0]))
    for label in data_labels_to_plot:
        plt.plot(x, data[label], 'o-', label=label)
    plt.title(title, fontsize=18)
    plt.legend(ncol=1, loc='upper right')
    plt.xlabel(x_label, fontsize=15)
    plt.ylabel(y_label, fontsize=15)
    plt.ylim(0, 0.15)
    sns.despine()
    plt.savefig(save_file)


def read_log_file(file):
    data_vals = np.loadtxt(file, delimiter=',', skiprows=1)
    with open(file, 'r') as f:
        line = f.readline().strip()
    header = line.split(',')

    data = {}
    for i in range(len(header)):
        data[header[i]] = data_vals[:, i]

    return data


def plot_train_progress():
    labels = ['uniform', 'normal', 'lecun', 'lecun_normal', 'xavier', 'xavier_normal', 'kaiming', 'kaiming_normal']
    all_data = {}
    for label in labels:
        all_data[label] = read_log_file(f'data/schnet_{label}/log/log.csv')['Validation loss']
    plot_loss(all_data, labels, 'validation_loss.png', x_label='Epoch number', y_label='Loss value',
              title='Validation loss')


def plot_testing_results():
    labels = ['uniform', 'normal', 'lecun', 'lecun_normal', 'xavier', 'xavier_normal', 'kaiming', 'kaiming_normal']
    data = {'method': [], 'Dataset and metric': [], 'Value': []}
    for label in labels:
        data['method'].extend([label] * 6)
        with open(f'data/schnet_{label}/test_eval.txt', 'r') as f:
            _ = f.readline()
            mae, rmse = list(map(float, f.readline().strip().split(',')))
            data['Dataset and metric'].extend(['MAE test', 'RMSE test'])
            data['Value'].extend([mae, rmse])
        with open(f'data/schnet_{label}/validation_eval.txt', 'r') as f:
            _ = f.readline()
            mae, rmse = list(map(float, f.readline().strip().split(',')))
            data['Dataset and metric'].extend(['MAE validation', 'RMSE validation'])
            data['Value'].extend([mae, rmse])
        with open(f'data/schnet_{label}/train_eval.txt', 'r') as f:
            _ = f.readline()
            mae, rmse = list(map(float, f.readline().strip().split(',')))
            data['Dataset and metric'].extend(['MAE train', 'RMSE train'])
            data['Value'].extend([mae, rmse])

    pd.DataFrame.from_dict(data).to_csv('evaluation.csv')

    sns.set_style("whitegrid")
    sns.stripplot(x="Dataset and metric", y="Value", hue="method", data=pd.DataFrame.from_dict(data), size=10,
                  palette='Paired', alpha=0.7)
    sns.despine()
    plt.xticks(rotation=30)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('Method evaluation', fontsize=18)
    plt.xlabel('Dataset and metric', fontsize=15)
    plt.ylabel('Value', fontsize=15)
    plt.savefig('evaluation.png', bbox_inches='tight')


if __name__ == '__main__':
    # plot_train_progress()
    plot_testing_results()
