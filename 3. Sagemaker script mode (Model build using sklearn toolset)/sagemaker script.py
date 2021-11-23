from __future__ import print_function
import argparse
import os
import pandas as pd

from sklearn import tree
from sklearn.externals import joblib


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Sagemaker specific arguments. Defaults are set in the environment variables.

    #Saves Checkpoints and graphs
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    #Save model artifacts
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    #Train data
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    file = os.path.join(args.train, "insurance.csv")
    insurance_df = pd.read_csv(file, engine="python")

    # labels are in the first column
    X=insurance_df.iloc[:,:-1].values
    y=insurance_df.iloc[:,-1].values
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
    
    #transforming column 
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    transformer = ColumnTransformer(transformers=[
    ('tnf3',OneHotEncoder(sparse=False),[1,4,5])
    ],remainder='passthrough')
    
    #scaling the data before feeding the model
    from sklearn.preprocessing import StandardScaler
    scaler_x = StandardScaler()
    
    # using linear regression model
    from sklearn.linear_model import LinearRegression
    regresssion_model_sklearn = LinearRegression()
    
    #Pipeline
    from sklearn.pipeline import make_pipeline
    model=make_pipeline(transformer,scaler_x,regresssion_model_sklearn)

    model.fit(X_train, y_train)

    # Print the coefficients of the trained regressor, and save the coefficients
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))


def model_fn(model_dir):
    """Deserialized and return fitted model
    
    Note that this should have the same name as the serialized model in the main method
    """
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model