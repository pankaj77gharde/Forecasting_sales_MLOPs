"""Performance analysis of model by using evaluation metrics i.e. MAPE
by building profet model for each item i.e. 50 models"""
"""Product wise Profet model is build and check its performance by using MAPE metrics.
Store all the model and metrics data with the help of mlflow"""
import pandas as pd
from pmdarima.model_selection import train_test_split
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime
import mlflow
import hydra
import omegaconf

eval_item_metrics = {}

cfg = omegaconf.OmegaConf.load(
    "./config/pross_config.yaml"  # "D:\MLOPs_POC\python_files_v2\config\pross_config.yaml"
)


def train_final_model(
    data,
    dict_mod_met,
):
    """considering whole data for traning final models"""
    # date time formate
    data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d")
    data = data.sort_values("date")
    final_model_dict = {}
    col_name = list(data.columns)
    col_name.remove("date")

    date_now = datetime.now().date()
    str_date_now = date_now.strftime("%Y_%m_%d")

    for prod_name in col_name:
        # work on algo one by one
        mod_name = dict_mod_met[prod_name]["best_model"]
        if mod_name == "Prophet":
            with mlflow.start_run(run_name=f"{prod_name}_run") as _:
                # train data processing as pere NP formate
                data_prod = data.loc[:, ["date", prod_name]]
                data_prod.columns = ["ds", "y"]

                # model traning
                mod = Prophet()
                mod.fit(data_prod)
                final_model_dict[prod_name] = mod
                # mlflow tracking metrics and model item wise i.e. 50
                mlflow.prophet.log_model(
                    mod, f"{prod_name}_model"
                )  # model train on whole data
                # using eval_model performance metrics as per item
                dict_eval = dict_mod_met[prod_name]
                mlflow.log_metric(f"MAPE_train", dict_eval["MAPE_train"])
                mlflow.log_metric(f"MAPE_test", dict_eval["MAPE_test"])

        elif mod_name == "SARIMAX":
            with mlflow.start_run(run_name=f"{prod_name}_run") as _:
                # train data processing as pere NP formate
                data_prod = data.loc[:, ["date", prod_name]]
                data_prod.set_index("date", inplace=True)

                # model traning
                mod = SARIMAX(
                    data_prod,
                    order=(
                        cfg.hyper_param_SRIMAX.p,
                        cfg.hyper_param_SRIMAX.d,
                        cfg.hyper_param_SRIMAX.q,
                    ),
                )  # , seasonal_order=(1, 1, 1, 7))
                result = mod.fit()
                final_model_dict[prod_name] = result
                # mlflow tracking metrics and model item wise i.e. 50
                mlflow.statsmodels.log_model(
                    result, f"{prod_name}_model"
                )  # model train on whole data
                # using eval_model performance metrics as per item
                dict_eval = dict_mod_met[prod_name]
                mlflow.log_metric(f"MAPE_train", dict_eval["MAPE_train"])
                mlflow.log_metric(f"MAPE_test", dict_eval["MAPE_test"])

    return final_model_dict


def eval_model(train_true, train_pred, test_true, test_pred):
    """MAPE metrics"""
    mape_train = mean_absolute_percentage_error(train_true, train_pred)
    mape_test = mean_absolute_percentage_error(test_true, test_pred)
    dict_eval = {"MAPE_train": mape_train, "MAPE_test": mape_test}
    return dict_eval


def train_test_data(data, split_test):
    """Train test split"""
    train, test = train_test_split(
        data, test_size=split_test
    )  # splite by dates
    return {"train": train, "test": test}


def eval_model_train(data_upd, list_algo, split_test=0.2):
    data = data_upd.copy()
    data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d")
    data = data.sort_values("date")
    """Using only train and test data"""
    data_dic = train_test_data(data, split_test)
    train = data_dic["train"]
    test = data_dic["test"]
    # convert to date time formate
    #     train["date"] = pd.to_datetime(train["date"], format="%Y-%m-%d")
    #     test["date"] = pd.to_datetime(test["date"], format="%Y-%m-%d")
    model_dict = {}
    col_names = train.columns
    # work on algo one by one
    for algo in list_algo:
        if algo == Prophet:
            algo_dict = {}
            mod_name = "Prophet"
            # work on profet part
            for y_indx in range(1, len(train.columns)):
                # item key for model_dict
                item_name = col_names[y_indx]

                # train data processing as pere NP formate
                tsdf_train = train.iloc[:, [0, y_indx]]
                tsdf_train.columns = ["ds", "y"]

                # test data processing as pere NP formate
                tsdf_test = test.iloc[:, [0, y_indx]]
                tsdf_test.columns = ["ds", "y"]

                # model traning
                mod = Prophet()
                mod.fit(tsdf_train)
                model_dict[item_name] = mod

                # train and test pred and true values
                pred_train = mod.predict(tsdf_train)
                pred_test = mod.predict(tsdf_test)
                train_true = pred_train.iloc[:, 1]
                train_pred = pred_train.iloc[:, 2]
                test_true = pred_test.iloc[:, 1]
                test_pred = pred_test.iloc[:, 2]

                # model evaluation metrics
                dict_eval = eval_model(
                    train_true, train_pred, test_true, test_pred
                )
                algo_dict[item_name] = dict_eval
        #                 eval_item_metrics[f"{mod_name}_{item_name}"] = dict_eval

        elif algo == SARIMAX:
            mod_name = "SARIMAX"
            algo_dict = {}
            # work on auto arima part
            for y_indx in range(1, len(train.columns)):
                # item key for model_dict
                item_name = col_names[y_indx]

                tsdf_train = train.iloc[:, [0, y_indx]]
                tsdf_train.set_index("date", inplace=True)
                tsdf_test = test.iloc[:, [0, y_indx]]
                tsdf_test.set_index("date", inplace=True)

                # model traning
                mod = SARIMAX(
                    tsdf_train,
                    order=(
                        cfg.hyper_param_SRIMAX.p,
                        cfg.hyper_param_SRIMAX.d,
                        cfg.hyper_param_SRIMAX.q,
                    ),
                )  # , seasonal_order=(1, 1, 1, 7))
                result = mod.fit()
                model_dict[item_name] = result

                # train and test pred and true values
                pred_train = result.predict(
                    start=tsdf_train.index[0], end=tsdf_train.index[-1]
                )
                pred_test = result.predict(
                    start=tsdf_test.index[0], end=tsdf_test.index[-1]
                )
                train_true = tsdf_train
                train_pred = pred_train
                test_true = tsdf_test
                test_pred = pred_test

                # model evaluation metrics
                dict_eval = eval_model(
                    train_true, train_pred, test_true, test_pred
                )
                algo_dict[item_name] = dict_eval
        #                 eval_item_metrics[f"{mod_name}_{item_name}"] = dict_eval
        eval_item_metrics[mod_name] = algo_dict
    # best model selection from yrained models
    lis_df = []
    for model in eval_item_metrics.keys():
        df_met = pd.DataFrame(eval_item_metrics[model]).loc[["MAPE_test"], :]
        df_met.index = [model]
        lis_df.append(df_met)
    conct_met_df = pd.concat(lis_df, axis=0)
    dict_best_mod = {}
    mode_names_indx = list(conct_met_df.index)
    for prod in conct_met_df.columns:
        index = conct_met_df[prod].argmin()
        mod_dict = {"best_model": mode_names_indx[index]}
        mod_dict.update(eval_item_metrics[mode_names_indx[index]][prod])
        dict_best_mod[prod] = mod_dict

    # final model devlopment and storing with mlflow
    final_best_mod = train_final_model(data_upd, dict_best_mod)

    return final_best_mod
