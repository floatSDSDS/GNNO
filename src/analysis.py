import os
import pandas as pd


def parse_head(head_string, params_selected):
    head_string = head_string.replace('.pt', '')
    param_string = head_string.split('/')[-1]
    string_split = param_string.split('__')
    lst_split = [s.split('=') for s in string_split if '=' in s]
    lst_split = [s for s in lst_split if s[0] in params_selected]
    return lst_split


def parse_rst(rst_string, params_selected):
    string_metrix = rst_string.replace(')', '')
    string_metrix = string_metrix.split('(')[1]
    lst_split = string_metrix.split(',')
    lst_split = [s.split(':') for s in lst_split]
    lst_split = [s for s in lst_split if s[0] in params_selected]
    return lst_split


def extract_keys(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    str_head = [line for line in lines if 'Load model from' in line][-1]
    str_rst = [line for line in lines if 'Test After Training' in line][-1]
    params = parse_head(str_head, param_selected)
    metrics = parse_rst(str_rst, metric_selected)

    keys = []
    keys.extend(params)
    keys.extend(metrics)
    head = [key[0] for key in keys]
    rst = [eval(key[1]) for key in keys]
    df = pd.DataFrame(rst).transpose()
    df.columns = head
    return df


def get_summary(log_path):
    files = os.listdir(log_path)
    files = [file for file in files if file.endswith('.txt')]
    files_full = [os.path.join(log_path, file) for file in files]
    keys = [extract_keys(file) for file in files_full]
    summary = pd.concat(keys)
    return summary


def get_avg(df, param_avg):
    rst = df.groupby(param_avg).mean()
    rst = rst.round(4)
    return rst.reset_index()


if __name__ == '__main__':

    param_selected = ['random_seed', 'batch_size', 'gamma_st', 'gamma', 'min_st_freq']
    metric_selected = ['HR@5', 'NDCG@5', 'HR@10', "NDCG@10", 'HR@20', "NDCG@20", 'HR@50', "NDCG@50"]
    avg_param = ['batch_size', 'gamma_st', 'min_st_freq']

    target_path = '../log/STRec/Beauty'
    output_filename = 'STRecBeautyAvg.csv'

    # target_path = '../log/STRec/Grocery'
    # output_filename = 'STRecGroceryAvg.csv'

    rst_df_raw = get_summary(target_path)
    rst_df_avg = get_avg(rst_df_raw, avg_param)
    rst_df_avg.to_csv(os.path.join(target_path, output_filename), index=False)
    print('fin.')
