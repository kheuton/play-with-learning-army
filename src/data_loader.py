import pandas as pd
import os

def read_raw(file_path, data_config, problem_id):
    """
    Read data from an Excel file
    :param file_path: str, path to the csv file
    :param sheet_name: str, name of the sheet to read
    :return: pd.DataFrame, data read from the csv file
    """
    
    # read csv, converting string list into actual list object
    data = pd.read_csv(file_path)

    y = data[data_config['label_col']].apply(eval)

    x = data[data_config['response_col']]

    problem_ids = [problem_id] * len(data)
    
    student_ids = data[data_config['student_id_col']]

    return x, y, problem_ids, student_ids

def load_dataset(data_config, hyper_config, train=False, val=False, test=False):
    
    data_dir = data_config['processed_data_dir']

    outputs = ()

    if train:
        processed_train_x_path = os.path.join(data_dir, hyper_config['train_x_file'])
        processed_train_y_path = os.path.join(data_dir, hyper_config['train_y_file'])
        processed_train_features, processed_train_labels = load_processed(processed_train_x_path, processed_train_y_path)
        train_dataset = make_dataset(processed_train_features, processed_train_labels)

        outputs += (train_dataset,)

    if val:
        processed_val_x_path = os.path.join(data_dir, hyper_config['val_x_file'])
        processed_val_y_path = os.path.join(data_dir, hyper_config['val_y_file'])
        processed_val_features, processed_val_labels = load_processed(processed_val_x_path, processed_val_y_path)
        val_dataset = make_dataset(processed_val_features, processed_val_labels)

        outputs += (val_dataset,)

    if test:
        processed_test_x_path = os.path.join(data_dir, hyper_config['test_x_file'])
        processed_test_y_path = os.path.join(data_dir, hyper_config['test_y_file'])
        processed_test_features, processed_test_labels = load_processed(processed_test_x_path, processed_test_y_path)
        test_dataset = make_dataset(processed_test_features, processed_test_labels)

        outputs += (test_dataset,)
    
    return outputs

def load_processed(x_path, y_path):
    x = pd.read_csv(x_path)
    y = pd.read_csv(y_path)
    return x, y

def make_dataset(features, labels):

    # extract problem_id and student_id as last 2 columns of x
    problem_id = features.iloc[:, -2]
    student_id = features.iloc[:, -1]
    x = features.iloc[:, :-2]
    y = labels

    # create pytorch dataset
    dataset = (x, y, problem_id, student_id)

