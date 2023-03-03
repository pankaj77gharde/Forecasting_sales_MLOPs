"""In pre process first grouping by items i.e. 50 then by
droping unique columns geting final X and Y dataframe
for current data no need of null or nan value handling
but we can add fun to deal with that in case data change"""
import os
import sys
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra import utils
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import omegaconf

sys.path.insert(0, "./src/Model")
from model_eval import eval_model_train


# @hydra.main(
#     version_base=None,
#     config_path=r"D:\MLOPs_POC\python_files_v2",
#     config_name="pross_config.yaml",
# )


cfg = omegaconf.OmegaConf.load("./config/pross_config.yaml")


def pre_pros(raw_df):  # , cfg: DictConfig) -> None:
    # cfg = current_app.config["config"]
    """By using group by working on one item at a time"""
    y = {}
    X = {}
    for item, data in raw_df.groupby("item"):
        item_data = data.drop(
            [col for col in data.columns if data[col].nunique() == 1], axis=1
        )
        item_data.rename(columns={"sales": f"sales_{item}"}, inplace=True)
        X.update({"date": item_data["date"].values})
        y.update({f"sales_{item}": item_data[f"sales_{item}"].values})
    df_X = pd.DataFrame(X)
    df_y = pd.DataFrame(y)
    df_final = pd.concat([df_X, df_y], axis=1)
    df_final["date"] = pd.to_datetime(df_final["date"], format="%Y-%m-%d")
    df_final = df_final.resample("D", on="date").sum()
    df_final = df_final.reset_index()
    df_final = df_final.sort_values("date")

    # base_path = cfg.dataset.path
    list_mods = [
        mod
        for mod in [SARIMAX, Prophet]
        if str(mod)[1:-1].split(".")[-1].replace("'", "") in cfg.models
    ]  # cfg.models
    # r"D:\MLOPs_POC\python_files_v2"  # cfg.dataset.path
    upd_file_path = rf"./data/final/updated_sales_data.csv"
    if not os.path.exists(upd_file_path):
        os.mkdir(rf"./data/final")
        df_final.to_csv(
            upd_file_path,
            index=False,
        )
        eval_model_train(
            df_final,
            list_mods,
            split_test=cfg.data_ratio.split_test,
        )
    else:
        df_old = pd.read_csv(upd_file_path)
        df_new = df_final.copy()
        df_final_con = pd.concat([df_old, df_new], axis=0, ignore_index=True)
        df_final_con.to_csv(
            upd_file_path,
            index=False,
        )

        eval_model_train(
            df_final_con,
            list_mods,  # [SARIMAX, Prophet]
            split_test=cfg.data_ratio.split_test,
        )

        return df_final
