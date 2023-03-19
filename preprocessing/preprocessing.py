import csv
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer
from imblearn.under_sampling import RandomUnderSampler


def construct_line(label, line):
    new_line = []
    if float(label) == 0.0:
        label = '0'
    new_line.append(label)

    for i, item in enumerate(line):
        if item == '' or float(item) == 0.0:
            continue
        new_item = '%s:%s' % (i + 1, item)
        new_line.append(new_item)
    new_line = ' '.join(new_line)
    new_line += '\n'
    return new_line


def main():
    cwd = os.getcwd()
    data_dir = os.path.join(cwd, '..', 'data')
    input_filepath = os.path.join(data_dir, 'weatherAUS-original.csv')
    output_filepath = os.path.join(data_dir, 'weatherAUS-final.csv')

    raw = pd.read_csv(input_filepath)

    target_col = 'RainTomorrow'
    categorical_cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
    many_missing_cols = ['Date', 'Location', 'Cloud9am', 'Cloud3pm', 'Evaporation', 'Sunshine']
    to_fill_cols = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindSpeed3pm', 'WindSpeed9am', 'Humidity3pm', 'Humidity9am',
                    'Temp3pm', 'Temp9am', 'Pressure3pm', 'Pressure9am', 'WindGustSpeed']

    to_fill_values = raw[to_fill_cols].mean().round(1).to_dict()

    print('Processing...')
    final = (pd.get_dummies(data=raw, columns=categorical_cols)
             .replace({target_col: {'No': 0, 'Yes': 1}})
             .drop(columns=many_missing_cols)
             .dropna(subset=([target_col]))
             .fillna(to_fill_values))

    if args.balancing:
        print('Balancing...')
        columns = final.columns
        y = final[target_col].to_numpy().astype('int32')
        X = final.iloc[:, final.columns != target_col]
        # TODO: review type of balancing
        under_sampler = RandomUnderSampler(random_state=42)
        X_res, y_res = under_sampler.fit_resample(X, y)
        y_res = y_res.reshape(-1, 1)
        interim = np.concatenate((X_res, y_res), axis=1)
        final = pd.DataFrame(data=interim, columns=columns)

    if args.normalize:
        print('Normalizing...')
        columns = final.columns
        y = final[target_col].to_numpy().reshape(-1, 1)

        X = final.iloc[:, final.columns != target_col]
        X_norm = Normalizer().fit_transform(X)

        interim = np.concatenate((X_norm, y), axis=1)
        final = pd.DataFrame(data=interim, columns=columns)

    final.to_csv(output_filepath, index=False)
    print(f'Exported dataset [CSV]: {os.path.basename(output_filepath)}')

    if args.libsvm:
        libsvm_filepath = os.path.join(data_dir, 'weatherAUS-final.data')
        input_file = open(output_filepath)
        output_file = open(libsvm_filepath, 'wb')

        reader = csv.reader(input_file)

        # skip headers
        _ = next(reader)

        for line in reader:
            target_idx = 62
            label = line.pop(target_idx)

            try:
                new_line = construct_line(label, line).encode(encoding='utf-8')
                output_file.write(new_line)
            except ValueError:
                print('Problem with the following line, skipping...')
                print(line)
        print(f'Exported dataset [LIBSVM]: {os.path.basename(libsvm_filepath)}')

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocessing weatherAUS dataset',
        usage='%(prog)s [--normalize] [--libsvm]'
    )

    parser.add_argument(
        '-n', '--normalize', action='store_true',
        help='apply normalization to the dataset'
    )

    parser.add_argument(
        '-l', '--libsvm', action='store_true',
        help='create .data file with LIBSVM format'
    )
    parser.add_argument(
        '-b', '--balancing', action='store_true',
        help='balancing dataset'
    )

    args = parser.parse_args()
    exit(main())
