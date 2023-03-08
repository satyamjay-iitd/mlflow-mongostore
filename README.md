# Mlflow-MongoStore
Mlflow plugin to use MongoDB as backend for MLflow tracking service.


# Install
    git clone git clone [https://github.com/satyamjay-iitd/mlflow-mongostore.git](https://github.com/satyamjay-iitd/mlflow-mongostore/)
    cd mlflow-mongostore
    pip install .

# Use
    $ mlflow server --backend-store-uri mongodb://$USER:$PASSWORD@$MONGO_HOST/$DB_NAME
    OR
    $ mlflow server --backend-store-uri mongodb+srv://$USER:$PASSWORD@$MONGO_HOST/$DB_NAME
