# Mid Price Prediction

The `mid_price_prediction` package provides tools for analyzing and predicting mid prices in financial trading using order book data. It includes classes and methods for processing trading data, constructing order books, and predicting price movements.

## Installation

To install the `mid_price_prediction` package, run the following command:

<pre><div class="bg-black rounded-md"><div class="flex items-center relative text-gray-200 bg-gray-800 gizmo:dark:bg-token-surface-primary px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>bash</span><button class="flex ml-auto gizmo:ml-0 gap-2 items-center"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M9 5H7C5.89543 5 5 5.89543 5 7V19C5 20.1046 5.89543 21 7 21H17C18.1046 21 19 20.1046 19 19V7C19 5.89543 18.1046 5 17 5H15" stroke="currentColor" stroke-width="2"></path><path d="M9 6C9 4.34315 10.3431 3 12 3V3C13.6569 3 15 4.34315 15 6V6C15 6.55228 14.5523 7 14 7H10C9.44772 7 9 6.55228 9 6V6Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"></path></svg>Copy code</button></div><div class="p-4 overflow-y-auto"><code class="!whitespace-pre hljs language-bash">pip install mid_price_prediction
</code></div></div></pre>

## Modules

### OrderBook Class

The `OrderBook` class in `OrderBookClass.py` represents and manipulates an order book for trading data. It provides functionalities to update the order book based on transaction data, retrieve top price levels, and process trading data for a given day.

#### Features

* Update the order book with new transaction data.
* Retrieve top k price levels from the order book.
* Process and output data for trading days.

#### Usage

<pre><div class="bg-black rounded-md"><div class="flex items-center relative text-gray-200 bg-gray-800 gizmo:dark:bg-token-surface-primary px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>python</span><button class="flex ml-auto gizmo:ml-0 gap-2 items-center"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M9 5H7C5.89543 5 5 5.89543 5 7V19C5 20.1046 5.89543 21 7 21H17C18.1046 21 19 20.1046 19 19V7C19 5.89543 18.1046 5 17 5H15" stroke="currentColor" stroke-width="2"></path><path d="M9 6C9 4.34315 10.3431 3 12 3V3C13.6569 3 15 4.34315 15 6V6C15 6.55228 14.5523 7 14 7H10C9.44772 7 9 6.55228 9 6V6Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"></path></svg>Copy code</button></div><div class="p-4 overflow-y-auto"><code class="!whitespace-pre hljs language-python">from mid_price_prediction.OrderBookClass import OrderBook

# Create an instance of OrderBook
order_book = OrderBook()

# Example: Processing a day's trading data
day_files = ['res_20190610.csv', 'res_20190611.csv', 'res_20190612.csv']
for day_file in day_files:
    order_book.create_output_for_dayfile(day_file, 5)
</code></div></div></pre>

### PricePredictionModel Class

The `PricePredictionModel` class in `PredictionModel.py` is used for predicting future price movements based on historical order book data.

#### Features

* Reads and transforms trading data.
* Normalizes features for model training.
* Performs feature engineering.
* Trains a logistic regression model for price prediction.
* Evaluates the model and plots the coefficients.
* Performs feature selection and quantile analysis.

#### Usage

<pre><div class="bg-black rounded-md"><div class="flex items-center relative text-gray-200 bg-gray-800 gizmo:dark:bg-token-surface-primary px-4 py-2 text-xs font-sans justify-between rounded-t-md"><span>python</span><button class="flex ml-auto gizmo:ml-0 gap-2 items-center"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M9 5H7C5.89543 5 5 5.89543 5 7V19C5 20.1046 5.89543 21 7 21H17C18.1046 21 19 20.1046 19 19V7C19 5.89543 18.1046 5 17 5H15" stroke="currentColor" stroke-width="2"></path><path d="M9 6C9 4.34315 10.3431 3 12 3V3C13.6569 3 15 4.34315 15 6V6C15 6.55228 14.5523 7 14 7H10C9.44772 7 9 6.55228 9 6V6Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"></path></svg>Copy code</button></div><div class="p-4 overflow-y-auto"><code class="!whitespace-pre hljs language-python">from mid_price_prediction.PredictionModel import PricePredictionModel

# Initialize the model with data files
data_files = ['day1.csv', 'day2.csv', 'day3.csv', 'day4.csv', 'day5.csv']
model = PricePredictionModel(data_files)

# Prepare data and train the model
model.prepare_data()
model.train_model()

# Evaluate the model
model.evaluate_model()  # Pass test data as needed
model.plot_coefficients()

# Perform feature selection
model.perform_feature_selection(3)

# Quantile analysis on timestamp differences
quantile_list = [0.1, 0.2, 0.3, 0.4, 0.5]
model.plot_timestamp_difference_distribution(quantile_list)
</code></div></div></pre>

## Requirements

The `mid_price_prediction` package requires the following libraries:

* numpy
* pandas
* scikit-learn
* matplotlib
* mlxtend

---
