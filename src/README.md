# Sales Forecasting MLOPs POC

## Install Dependencies and setup applications
flask : 

### Dependency handling
It recommended to use virtual environments (conda, miniconda, venv) and have separate environments. 

### Logic 
Raw inpute data is fetch from the source directry then process the data and use the process data for model building for each product. Once models are availble then go for respective performance analysis with mape metrics. All the requred data, files and informations are saved with the help of mlflow tracking and use those data with mlflow help.
Now when new data (new month data) is updated in specifed folder then we will process it and merge with availble processed data dynamicaly. In case no new data availble then just go with the availble processed data and build the model.
***Right now we are working with single algo model but we are looking to train and test more than one model each and select best out of it for forcasting. Thats why we are doing model building each time we go for forecasting.