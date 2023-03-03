"""First forecasting dataframe is created as per input argument provided then
by using mlflow profet flavor fetching model and return forecasted data in
standard dataframe formate"""
from datetime import timedelta, datetime
import calendar
import pandas as pd
import mlflow
import mlflow.pyfunc


def forecast_with_run_model(
    run_model_dict, latest_proc_data, next_month_count
):
    """create dataframe for next month forecasting with current data used for traning model"""
    try:
        latest_proc_data["date"] = pd.to_datetime(latest_proc_data["date"])
    finally:
        year = list(latest_proc_data["date"])[-1].year
        month = list(latest_proc_data["date"])[-1].month

        list_mon_yr = []
        for _ in range(1, next_month_count + 1):
            new_month = month + 1
            if new_month > 12:
                month = 1
                year += 1
            else:
                month = new_month
            day_count = calendar.monthrange(year, month)[1]
            list_mon_yr.append([month, year, day_count])

        final_day_count = 0
        for val in list_mon_yr:
            final_day_count += val[-1]

        start_ = datetime(list_mon_yr[0][1], list_mon_yr[0][0], 1)
        end_ = start_ + timedelta(final_day_count - 1)

        data_next_month = pd.date_range(start=start_, end=end_, freq="D")
        data_next_month = pd.DataFrame({"ds": data_next_month})

        # fetch each best model from dict and forecast
        dict_forecast = {}
        for name, run_id in run_model_dict.items():
            model_folder_name = f"{name[:-3]}model"
            pyfunc_uri = f"runs:/{run_id}/{model_folder_name}"
            pyfunc_model = mlflow.pyfunc.load_model(
                pyfunc_uri
            )  # we are using  flavor from mlflow
            flavor = [
                i.split(":")[1].strip()
                for i in str(pyfunc_model).split("\n")
                if i.strip().startswith("flavor")
            ]
            if flavor[0] == "mlflow.statsmodels":
                pred_df = pyfunc_model.predict(
                    pd.DataFrame(
                        {
                            "start": data_next_month.iloc[0, :],
                            "end": data_next_month.iloc[-1, :],
                        }
                    )
                )  ### sarima and profet same ????
                col_val = pred_df.values
            elif flavor[0] == "mlflow.prophet":
                pred_df = pyfunc_model.predict(data_next_month)
                col_val = pred_df["yhat"].values

            # final single datafarem with each item sales for next month
            name_col = name[:-4]
            dict_forecast[name_col] = col_val
            df_fourcast = pd.DataFrame(dict_forecast)
            final_forecast = pd.concat(
                [data_next_month["ds"], df_fourcast], axis=1
            )
            final_forecast.rename(columns={"ds": "date"}, inplace=True)

        # make dataframe in standered formate
        col_name_order = ["date"]
        col_name_order.extend([f"sales_{i}" for i in range(1, 51)])
        final_forecast = final_forecast.reindex(col_name_order, axis=1)
        final_forecast = final_forecast.round()
        final_forecast.columns = ["date"] + [
            f"sales_{i}_forecast" for i in range(1, 51)
        ]
    return final_forecast
