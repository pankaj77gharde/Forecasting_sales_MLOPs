"""Fetch the best model wrt each product by using metrics i.e. MAPE from mlflow track data"""
import mlflow


def best_model():
    """# path of mlflow folder"""
    list_run = mlflow.search_runs()
    run_name_list = set(list_run["tags.mlflow.runName"])
    best_model_runs = {}
    for run_name in run_name_list:
        df_item = list_run[list_run["tags.mlflow.runName"] == run_name]
        df_item = df_item.reset_index()
        df_item = df_item.iloc[:, 1:]
        df_item = df_item.sort_values("metrics.MAPE_test")
        best_model_runs[run_name] = df_item.loc[0]["run_id"]
    #  delete rest of run id in future

    return best_model_runs
