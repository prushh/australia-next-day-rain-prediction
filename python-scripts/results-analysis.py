import os
from typing import Dict

import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def _dark_subplots(nrows: int = 1, ncols: int = 1) -> tuple:
    plt.style.use('dark_background')
    fig, axes = plt.subplots()
    fig.patch.set_facecolor('#252526')
    axes.set_facecolor('#3c3c3c')

    return (fig, axes)


def mu_confidence_interval(data: np.ndarray, metric: str) -> Dict:
    t = 2.13
    mu = np.mean(data)
    standard_deviation = np.std(data)
    M = data.shape[0]
    t_student = t * standard_deviation / np.sqrt(M)
    first_interval = mu - t_student
    second_interval = mu + t_student
    return {
        f'mu_{metric}': mu,
        f't_student_{metric}': t_student,
        f'first_interval_{metric}': first_interval,
        f'second_interval_{metric}': second_interval
    }


def main():
    str_error = "Error: copy results inside strong/weak directory with pattern experiment_procN_workN_samples.csv (e.g. strong_processors4_worker1_10000.csv)"

    cwd = os.getcwd()
    results_dir = os.path.join(cwd, 'results')

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    # Specify strong or weak based on experiment
    test_type = 'strong'
    test_dir = os.path.join(results_dir, test_type)
    images_dir = os.path.join(test_dir, 'images')
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
        if not os.path.exists(images_dir):
            os.mkdir(images_dir)
            print(str_error)
            return 1

    df_results = pd.DataFrame()

    filenames = os.listdir(test_dir)
    files = [filename for filename in filenames if filename.endswith('.csv')]

    if not files:
        print(str_error)
        return 1

    for filename in files:
        filepath = os.path.join(test_dir, filename)
        df = pd.read_csv(filepath)
        df['Time'] = df['Time'].div(1000)
        classifiers = df['Algorithm'].unique()

        for classifier in classifiers:
            run_clf = df[df['Algorithm'] == classifier]
            accuracy = run_clf['Accuracy'].mean()
            time = run_clf['Time'].mean()

            (core, worker, _) = re.findall(r'\d+', filename)

            row_stat = {
                'classifier': [classifier],
                'mean-accuracy': [accuracy],
                'mean-time': [time],
                'core': [int(core)],
                'worker': [int(worker)],
                'config': [filename]
            }

            ci_acc = mu_confidence_interval(run_clf['Accuracy'].to_numpy(), 'accuracy')
            ci_time = mu_confidence_interval(run_clf['Time'].to_numpy(), 'time')

            row_stat.update(ci_acc)
            row_stat.update(ci_time)

            df_results = pd.concat([df_results, pd.DataFrame(row_stat)], ignore_index=True)

    print('Saving plot...')
    # classifiers = [f'{test_type.capitalize()} - {clf}' for clf in df_results['classifier'].unique()]
    classifiers = df_results['classifier'].unique()
    for classifier in classifiers:
        stats_clf = df_results[df_results['classifier'] == classifier].sort_values(by=['core'])

        fig, ax = _dark_subplots()

        plt.set_cmap('tab20')
        ax.set_title(classifier)
        ax.set_xlabel('# core')
        ax.set_ylabel('time (s)')

        ax.errorbar(
            stats_clf['core'],
            stats_clf['mean-time'],
            capsize=5,
            yerr=stats_clf['t_student_time'],
            color='green',
            ecolor='firebrick',
            zorder=1
        )
        ax.plot(stats_clf['core'], stats_clf['mean-time'], 'o', color='royalblue', zorder=2)
        # plt.show()

        plt.savefig(os.path.join(images_dir, f'{classifier.lower()}'))

    return 0


if __name__ == '__main__':
    exit(main())
