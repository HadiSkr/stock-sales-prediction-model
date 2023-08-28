import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from datetime import datetime




# the path of the dataset 
DATAPATH = "C:\\Users\\Hadi\\Downloads\\merged_data.csv"
# the e number of decision trees to be included in the random forest (into the model) this number affects the performance of the model depending on our dataset. We can experiment many numbers to reach the best performance.
NUM_ESTIMATORS = 200
# the ratio of the training and testing samples
Split = 0.8
# the type of evaluating the model
EVALUATION_METRIC = mean_absolute_error
# Define the number of folds for cross-validation
K = 8



# loading the sales dataset
# The easiest  way to use the data that we used in the last task was to export the data frame into a csv file (merged_data.csv) and use it in this file.
# this is what I use to load the dataframe into csv file: "merged_df.to_csv('merged_data.csv', index=False)"
data_df = pd.read_csv(DATAPATH)
data_df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
#print(data_df.info())
# Craeting a function to asign the predictors to X and target to y
def create_target_and_predictors(
    data: data_df = None, 
    target: str = "estimated_stock_pct"
):
    if target not in data.columns:
        raise Exception(f"Target: {target} is not present in the data")
    
    X = data.drop(columns=[target])
    y = data[target]
    return X, y

# calling the function:
X, y = create_target_and_predictors(data_df, target="estimated_stock_pct")



def train_algorithm_with_cross_validation(
    X: X, 
    y: y
):
    accuracy = []

    # Enter a loop to run K folds of cross-validation
    for fold in range(0, K):

        # Instantiate algorithm and scaler
        model = RandomForestRegressor(NUM_ESTIMATORS)
        scaler = StandardScaler()

        # Create training and test samples
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=Split, random_state=42)

        # Scale X data, we scale the data because it helps the algorithm to converge
        # and helps the algorithm to not be greedy with large values
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Train model
        trained_model = model.fit(X_train, y_train)

        # Generate predictions on test sample
        y_pred = trained_model.predict(X_test)

        # Compute accuracy, using mean absolute error
        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        accuracy.append(mae)
        print(f"Fold {fold + 1}: MAE = {mae:.3f}")
    # Finish by computing the average MAE across all folds
    print(f"Average MAE: {(sum(accuracy) / len(accuracy)):.2f}")

# After performing Hyperparameter Tuning, I determined that a suitable number for the NUM_ESTIMATORS parameter in our model is 200. Since our dataset is not particularly large, I had the freedom to increase the number of estimators to improve model performance.
# I reached an Average MAE of 0.23 which is lower than the we reached in the last task. 
train_algorithm_with_cross_validation(X, y)

 