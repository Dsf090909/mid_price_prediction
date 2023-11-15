import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

class PricePredictionModel:
    def __init__(self, data_files):
        self.data_files = data_files
        self.features = ['total_bid_depth', 'total_ask_depth', 'normalized_spread', 'layer_1_imbalance',
                         'layer_2_imbalance', 'layer_3_imbalance', 'layer_4_imbalance', 'layer_5_imbalance',
                         'exp_ma_roc_imbalance_span50', 'gap_difference_l1l2',
                         'log_returns', 'realized_variance', 'realized_bipower_variation']
        self.model = None

    def read_data_and_transform(self, day_file):
        columns_to_convert = ['timestamp', 'price', 'bp0', 'bq0', 'bp1', 'bq1', 'bp2', 'bq2', 'bp3', 'bq3', 
                              'bp4', 'bq4', 'ap0', 'aq0', 'ap1', 'aq1', 'ap2', 'aq2', 'ap3', 'aq3', 'ap4', 'aq4']
        column_types = {column: 'float64' for column in columns_to_convert}
        df = pd.read_csv(day_file, usecols=columns_to_convert, dtype=column_types)
        df.fillna(0, inplace=True)
        return df

    def normalize_feature(self, df):
        """
        Normalizes specified features in the DataFrame.
        Args:
            df (pandas.DataFrame): The DataFrame with data to be normalized.
        Returns:
            pandas.DataFrame: The DataFrame with normalized features.
        """
        df_normalized = df.copy()
        for column_name in self.features:
            quantile_value_up = df_normalized[column_name].quantile(0.999)
            quantile_value_down = df_normalized[column_name].quantile(0.001)
            #Remove outliers
            df_normalized = df_normalized[(df_normalized[column_name] <= quantile_value_up) & (df_normalized[column_name] >= quantile_value_down)]
            mean_value = df_normalized[column_name].mean()
            max_abs_value = df_normalized[column_name].abs().max()
            df_normalized.loc[:, column_name] = (df_normalized[column_name] - mean_value) / max_abs_value
        return df_normalized

    def feature_engineering(self, df):
        """
        Performs feature engineering on the given DataFrame.
        Args:
            df (pandas.DataFrame): The DataFrame containing the market data.
        Returns:
            pandas.DataFrame: The DataFrame with additional engineered features.
        """

        df['mid_price'] = (df['bp0'] + df['ap0']) / 2
        df['bid_ask_spread'] = df['ap0'] - df['bp0']
        df['normalized_spread'] = df['bid_ask_spread'] / df['mid_price']
        df['total_bid_depth'] = df[['bq0', 'bq1', 'bq2', 'bq3', 'bq4']].sum(axis=1)
        df['total_ask_depth'] = df[['aq0', 'aq1', 'aq2', 'aq3', 'aq4']].sum(axis=1)

        # Imbalance at different levels
        for level in range(1, 6):
            bid_col = f'bq{level-1}'
            ask_col = f'aq{level-1}'
            df[f'layer_{level}_imbalance'] = (df[bid_col] - df[ask_col]) / (df[bid_col] + df[ask_col]).replace(0, np.nan).fillna(0)
        #Define rate of change of the first layer imbalance
        df['roc_layer_1_imbalance'] = 0.0
        df.loc[30:, 'roc_layer_1_imbalance'] = df['layer_1_imbalance'].diff(periods=1).loc[30:]
        df['exp_ma_roc_imbalance_span50'] = df['roc_layer_1_imbalance'].ewm(span=50).mean()
        # Difference of the gap between bp0 and bp1 and ap0 and ap1
        df['gap_difference_l1l2'] = (df['bp0'] - df['bp1']) - (df['ap0'] - df['ap1'])
        # Log Returns
        df['log_returns'] = np.log(df['mid_price']).diff()
        # Realized Variance (simplified version)
        df['realized_variance'] = df['log_returns'].rolling(window=30).var()
        # Difference of the gap between bp0 and bp1 and ap0 and ap1
        df['gap_difference'] = (df['bp0'] - df['bp1']) - (df['ap0'] - df['ap1'])
        df['realized_bipower_variation'] = (df['log_returns'].abs() * df['log_returns'].abs().shift(1)).rolling(window=30).sum()
        #Prepare the data for the prediction model
        features = ['total_bid_depth','total_ask_depth','normalized_spread','layer_1_imbalance','layer_2_imbalance', 'layer_3_imbalance','layer_4_imbalance','layer_5_imbalance','exp_ma_roc_imbalance_span50','gap_difference_l1l2', 'log_returns','realized_variance', 'realized_bipower_variation']
        df = self.normalize_feature(df)
        return df

    def calculate_next_price_change_on_change(self, df):
        df['mid_price_change'] = df['mid_price'].diff().fillna(0) 
        df = df[df['mid_price_change'] != 0].copy()
        df['next_mid_price'] = df['mid_price'].shift(-1)
        df['next_price_change'] = df['next_mid_price'] - df['mid_price']
        df = df.dropna()
        return df
    

    def prepare_data(self):
        processed_dfs = [self.calculate_next_price_change_on_change(self.feature_engineering(self.read_data_and_transform(file))) for file in self.data_files]
        #Create the tradining set using the initial 3 days of dta, use the 4th day for validation and the 5th for testing. 
        self.X_train, self.y_train = pd.concat(processed_dfs[:3])[self.features], np.sign(pd.concat(processed_dfs[:3])['next_price_change'])
        self.X_validation, self.y_validation = processed_dfs[3][self.features], np.sign(processed_dfs[3]['next_price_change'])
        self.X_test, self.y_test = processed_dfs[4][self.features], np.sign(processed_dfs[4]['next_price_change'])


    def train_model(self):
            """
            Train the logistic regression model with hyperparameter tuning.

            This method finds the best hyperparameter for the regularization strength (C)
            and trains the logistic regression model using this best value.
            """
            # Define the range of C values for hyperparameter tuning
            C_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
            best_score = 0
            best_C = None

            # Iterate over the range of C values to find the best C value
            for C in C_values:
                model = LogisticRegression(penalty='l1', C=C, solver='saga', max_iter=10000)
                model.fit(self.X_train, self.y_train)
                score = model.score(self.X_validation, self.y_validation)
                print(f"Accuracy for C={C}: {score}")
                if score > best_score:
                    best_score = score
                    best_C = C

            # Print the best C value and its score
            print("Best C value:", best_C)
            print("Best validation set score:", best_score)

            # Train the final model with the best C value
            self.model = LogisticRegression(penalty='l1', C=best_C, solver='saga', max_iter=10000)
            self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """
        Evaluates the trained model on the test dataset.
        Args:
            X_test (DataFrame): Test features.
            y_test (Series): True values for the test set.
        """
        X_test, y_test = self.X_test, self.y_test
        # Predicting the labels for the test set
        y_pred_test = self.model.predict(X_test)

        # Printing the classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred_test))

        # Generating and displaying the confusion matrix
        cm = confusion_matrix(y_test, y_pred_test)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(np.unique(y_test)))
        plt.xticks(tick_marks, np.unique(y_test))
        plt.yticks(tick_marks, np.unique(y_test))
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
                
    def plot_coefficients(self):
        """
        Plots the coefficients of the logistic regression model.
        """
        # Check if the model has been trained and has coefficients
        if self.model is None or not hasattr(self.model, 'coef_'):
            print("Model is not trained or doesn't have coefficients.")
            return
        # Plotting the coefficients
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(self.features)), self.model.coef_[0])
        plt.xlabel('Features')
        plt.ylabel('Coefficient Value')
        plt.title('Feature Coefficients from the Model')
        plt.xticks(range(len(self.features)), self.features, rotation=45)
        plt.show()

    def perform_feature_selection(self, n_features):
        """
        Perform feature selection using Sequential Feature Selector (SFS).

        This method uses backward selection to identify the best set of features
        for the logistic regression model.
        """
        # Ensure the model has been trained and best_C is available
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("Model has not been trained. Please run train_model first.")

        # Extract the best C value from the trained model
        best_C = self.model.C

        # Initialize and fit the Sequential Feature Selector
        sbfs = SFS(LogisticRegression(penalty='l1', C=best_C, solver='saga', max_iter=10000),
                   k_features=n_features,  # Minimum number of features to select
                   forward=False,  # Perform backward selection
                   floating=True,  # Enable floating
                   scoring='accuracy',
                   cv=0)
        sbfs = sbfs.fit(self.X_train, self.y_train)

        # Retrieve the names of the selected features
        selected_features = [self.features[i] for i in sbfs.k_feature_idx_]

        # Print results of feature selection
        print('\nSequential Backward Floating Selection (k=2):')
        print('Selected features:', selected_features)
        print('Cross-validation score of selected subset:', sbfs.k_score_)

        # Create a DataFrame for detailed results and print it
        # feature_selection_df = pd.DataFrame.from_dict(sbfs.get_metric_dict()).T
        # print(feature_selection_df)

        # Optionally, return the indices of selected features and the SFS object
        return sbfs.k_feature_idx_, sbfs

    def plot_timestamp_difference_distribution(self, quantile_list):
        """
        Plots the distribution of the differences in timestamps.
        """
        dataframes = [self.calculate_next_price_change_on_change(self.feature_engineering(self.read_data_and_transform(file))) for file in self.data_files]
        for index, dataframe in enumerate(dataframes):
            # Calculate differences
            time_differences = dataframe['timestamp'].diff().dropna()
            # Convert to a suitable time unit, e.g., microseconds, if not already
            time_differences_in_microseconds = time_differences
            #Calculate Quantiles
            quantiles = time_differences_in_microseconds.quantile(quantile_list)
            # Plot the quantiles
            plt.figure(figsize=(10, 6))
            quantiles.plot(kind='bar')
            plt.title('Quantiles of Time Differences for Day {}'.format(index + 1))
            plt.xlabel('Quantiles')
            plt.ylabel('Time Difference (micro-seconds)')
            plt.grid(True)
            plt.show()

# Usage example
data_files = ['output_dfres_20190610.csv', 'output_dfres_20190611.csv', 'output_dfres_20190612.csv',
              'output_dfres_20190613.csv', 'output_dfres_20190614.csv']

model = PricePredictionModel(data_files)
model.prepare_data()
model.train_model()
model.evaluate_model()  #Pass test data as needed
model.plot_coefficients()
# model.perform_feature_selection(3)

#Example to see the quantiles of the time between changes in mid-price
list_q = [0.1, 0.2, 0.3, 0.4, 0.5]
model.plot_timestamp_difference_distribution(list_q)