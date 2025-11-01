from sklearn.model_selection import train_test_split
import os
import logging
from Utils import check_dir, simulate_nan, scale_data
import pandas as pd
import logging
import datetime


def setup_logger(output_dir, dataset_name):
    dataset_output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    log_filename = f"error_log_{timestamp}.txt"
    log_path = os.path.join(dataset_output_dir, log_filename)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=log_path,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    return log_path

import json
if __name__ == '__main__':
    base_path = '../../data_set'
    with open('../../data_set/result.json', 'r', encoding='utf-8') as f:
        result = json.load(f)
    file_paths = []
    for folder, files in result.items():
        for filename in files:
            full_path = os.path.join(base_path, folder, filename)
            file_paths.append(full_path)

    for seed in range(40, 50):
        for dataset in file_paths:

            dataname = os.path.basename(dataset).split('.')[0]

            log_path = setup_logger(output_dir=f"../../log/{seed}", dataset_name=f"{dataname}")
            try:
                df_full_ori = pd.read_csv(dataset)
            except Exception as e:
                print(f'{dataset}',e)
                continue
            train_full, test_data = train_test_split(df_full_ori, test_size=0.2, random_state=seed)
            
            test_data_X = test_data.iloc[:, :-1]
            test_data_y = test_data.iloc[:, [-1]]
            test_X_path = f'../../dataset/test_data/{seed}/X_data'
            check_dir(test_X_path)
            test_y_path = f'../../dataset/test_data/{seed}/y_data'
            check_dir(test_y_path)
            test_data_X.to_csv(os.path.join(test_X_path, f'{dataname}.csv'), index=False)
            test_data_y.to_csv(os.path.join(test_y_path, f'{dataname}.csv'), index=False)

            df_full_ = train_full.iloc[:, :-1]

            X_path = f'../../dataset/train_data/{seed}/X_full'
            check_dir(X_path)
            df_full_.to_csv(os.path.join(X_path, f'{dataname}.csv'), index=False)

            df_full_ = df_full_.dropna(how='any')

            cols = df_full_.columns.tolist()
            df_full_scaled = scale_data(df_full_)
            
            y_data = train_full.iloc[:, [-1]]
            y_path = f'../../dataset/train_data/{seed}/y_data'
            
            check_dir(y_path)
            y_data.to_csv(os.path.join(y_path, f'{dataname}.csv'), index=False)
            y_data = y_data.dropna(how='any')

            cols_y = y_data.columns.tolist()
            y_data_scaled = scale_data(y_data)
            for method in ['MCAR', 'MAR', 'MNAR_lo']:

                if method == "MAR":
                    data_root_dir = os.path.join(f"../../dataset/train_data/{seed}/MAR",
                                                 os.path.basename(dataset).split(".")[0])
                    check_dir(data_root_dir)

                    for i in range(1, 6):
                        miss_ratio = i * 0.1
                        df_full = df_full_scaled.values
                        try:
                            df_miss = simulate_nan(df_full, miss_ratio, mecha="MAR", random_state=seed)
                        except Exception as e:
                            logging.error(f"Method '{method}'- data:'{dataname}' failed: {str(e)}", exc_info=True)
                            continue

                        X_with_nan = df_miss["X_incomp"]
                        mask = df_miss["mask"]
                        X_with_nan = pd.DataFrame(X_with_nan, columns=cols)
                        X_with_nan.to_csv(os.path.join(data_root_dir, f"{miss_ratio:.1f}.csv"), index=False)
                        logging.info(
                            f"Method '{method}'- data:'{dataname} - missing_p: {miss_ratio}' succesfully saved.")

                elif method == "MNAR_lo":
                    data_root_dir = os.path.join(f"../../dataset/train_data/{seed}/MNAR_lo",
                                                 os.path.basename(dataset).split(".")[0])
                    check_dir(data_root_dir)
                    for i in range(1, 6):
                        miss_ratio = i * 0.1
                        df_full = df_full_scaled.values
                        try:
                            df_miss = simulate_nan(df_full, miss_ratio, mecha="MNAR", opt="logistic", random_state=seed)
                        except Exception as e:
                            logging.error(f"Method '{method}'- data:'{dataname}' failed: {str(e)}", exc_info=True)
                            continue
                        X_with_nan = df_miss["X_incomp"]
                        mask = df_miss["mask"]
                        X_with_nan = pd.DataFrame(X_with_nan, columns=cols)

                        X_with_nan.to_csv(os.path.join(data_root_dir, f"{miss_ratio:.1f}.csv"), index=False)
                        logging.info(
                            f"Method '{method}'- data:'{dataname} - missing_p: {miss_ratio}' succesfully saved.")
                elif method == "MCAR":

                    data_root_dir = os.path.join(f"../../dataset/train_data/{seed}/MCAR",
                                                 os.path.basename(dataset).split(".")[0])
                    check_dir(data_root_dir)

                    for i in range(1, 6):
                        miss_ratio = i * 0.1
                        df_full = df_full_scaled.values
                        try:
                            df_miss = simulate_nan(df_full, miss_ratio, mecha="MCAR", random_state=seed)
                        except Exception as e:
                            logging.error(f"Method '{method}'- data:'{dataname}' failed: {str(e)}", exc_info=True)
                            continue
                        X_with_nan = df_miss["X_incomp"]
                        mask = df_miss["mask"]
                        X_with_nan = pd.DataFrame(X_with_nan, columns=cols)

                        X_with_nan.to_csv(os.path.join(data_root_dir, f"{miss_ratio:.1f}.csv"), index=False)
                        logging.info(
                            f"Method '{method}'- data:'{dataname} - missing_p: {miss_ratio}' succesfully saved.")


