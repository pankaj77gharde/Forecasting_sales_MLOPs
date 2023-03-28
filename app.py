"""It provides input from user and follow pipe line flow for loading input files,
data processing, stornig process updated data, model building, best model selection
and forecasting with api links"""
import os
import dvc.api
import sys
import matplotlib
import plotly.graph_objects as go
import plotly.express as px
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra import utils
import omegaconf

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, render_template, request
from flask import current_app

sys.path.insert(0, "./src/Model_selection")
from model_selection import best_model

sys.path.insert(0, "./src/Forecasting")
from forecasting import forecast_with_run_model

sys.path.insert(0, "./src/Preprocess")
from preprocess import pre_pros

# sys.path.insert(0, "./python_files_v2")
# from test import data_fetch

app = Flask(__name__, static_folder="")

# app = Flask(__name__)


# @app.route("/")
@app.route("/", methods=["GET", "POST"])
def index():
    cfg = omegaconf.OmegaConf.load("./config/pross_config.yaml")
    # return f"Hello {cfg.app.username}"
    if request.method == "POST":
        """check the presence of new data if it presesnt then update current data..
        if not then use the current data"""

        # base_path = "./data"  #cfg.dataset.path
        # r"D:\MLOPs_POC\python_files_v2"  # data_fetch() config.dataset.path

        img_save_fold = "./images"
        # img_folder = os.path.join(base_path, img_save_fold)
        try:
            os.mkdir(img_save_fold)
        except:
            pass
        pick_fold_path = rf"{img_save_fold}\\forecast.png"

        path_ = rf"./data/final/updated_sales_data.csv"
        # pick_store = os.path.join(app.config["UPLODE_FOLDER"], "forecast.svg")
        if os.path.exists(path_):
            data_df = pd.read_csv(rf"{path_}").iloc[-3:, :]
        else:
            # raw_df_path = rf"./data/raw/{cfg.dataset.raw_data}"
            ################################dvc ################
            raw_df_path = rf"data/raw/{cfg.dataset.raw_data}"
            path_ = r"data/raw/data_2013_to_2016.csv"
            with dvc.api.open(
                repo="https://github.com/pankaj77gharde/Forecasting_sales_MLOPs.git",
                path=raw_df_path,
                mode="r",
            ) as fd:
                raw_df = pd.read_csv(fd)
            #################end################################
            # raw_df = pd.read_csv(raw_df_path)
            pre_pros(raw_df)
            data_df = pd.read_csv(rf"{path_}").iloc[-3:, :]
            data_df = data_df.iloc[-3:, :]

        # need to mension location of new data
        new_path = rf"./data/new_update/{cfg.dataset.new_data}"
        if os.path.exists(new_path):
            new_path_df = pd.read_csv(new_path)
            if new_path_df.iloc[-1, 0] != data_df.iloc[-1, 0][:10]:
                pre_pros(new_path_df)
                data_df = pd.read_csv(rf"{path_}").iloc[-3:, :]
                data_df = data_df.iloc[-3:, :]

        # gey proc data last date by using updated file
        month_count = request.form.get("month_count")
        prod_name = request.form.get("product_name")
        month_count = int(month_count)
        run_model_dict = best_model()
        forecasted_month = forecast_with_run_model(
            run_model_dict, data_df, month_count
        )[["date", prod_name]]

        # Create a line plot of the output data
        plot_df = forecasted_month.set_index("date")
        plt.figure(figsize=(15, 10), dpi=80)
        plt.plot(plot_df)
        plt.title(f"{prod_name} Sales Forecasting")
        plt.xlabel("Month")
        plt.ylabel("Sales")
        plt.xticks(rotation=45)
        plt.savefig(pick_fold_path)
        plt.clf()

        # fig = go.Figure()
        # fig = px.line(
        #     forecasted_month,
        #     x="date",
        #     y=forecasted_month.columns.drop(["date"]),
        #     title="Sales",
        # )
        # fig.update_xaxes(rangeslider_visible=True)
        # fig.update_layout(
        #     template="plotly_dark",
        #     hovermode="x unified",
        #     width=1400,
        #     height=800,
        # )
        # fig.show()
        # fig.write_image(pick_fold_path)
    else:
        forecasted_month = pd.DataFrame()

    """Need to add code for geting product wise result"""

    return render_template(
        "index.html",
        forecast=forecasted_month,
        # plot_graph=pick_store,
    )


@hydra.main(
    config_name="pross_config",
    config_path="./config",  # "D:\MLOPs_POC\python_files_v2\config"
    version_base=None,
)
def main(cfg: DictConfig):
    app.config["config"] = cfg
    app.run(
        debug=True,
        use_reloader=True,
        host=cfg.server.host,
        port=cfg.server.port,
    )


if __name__ == "__main__":
    main()
