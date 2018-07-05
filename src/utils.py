from typing import List
import pandas as pd


def load_and_merge_dataframes(path: str, keys: List[str], merge_on: str='fname',
                              exclude_cols_all=None,
                              exclude_cols_except_first=None):
    """
    Load dataframes from a HDF5 table, merge into one, while dropping some columns
    :param path:
    :param keys:
    :param merge_on:
    :param exclude_cols_all:
    :param exclude_cols_except_first:
    :return:
    """
    if exclude_cols_except_first is None:
        exclude_cols_except_first = []
    if exclude_cols_all is None:
        exclude_cols_all = []

    df_list = [pd.read_hdf(path, key=key) for key in keys]

    if len(df_list) == 1:
        base_df = df_list[0]
    else:
        base_df = df_list[0]
        for df in df_list[1:]:
            if exclude_cols_except_first is not None:
                use_cols = [col for col in df.columns if col not in exclude_cols_except_first]
            else:
                use_cols = df.columns
            base_df = base_df.merge(df[use_cols], on=merge_on)

    if exclude_cols_all is not None:
        remove_cols = [col for col in base_df.columns if col in exclude_cols_all]
        base_df.drop(remove_cols, axis=1, inplace=True)

    return base_df
