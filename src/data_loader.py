import pandas as pd

def read_xlsx(file_path, config):
    """
    Read data from an Excel file
    :param file_path: str, path to the Excel file
    :param sheet_name: str, name of the sheet to read
    :return: pd.DataFrame, data read from the Excel file
    """
    data = pd.csv(file_path)
    # split to x and y
    x = data["text"]
    y = data.drop(columns=["text"])
    assert y.shape[1] == config.M

    return x, y