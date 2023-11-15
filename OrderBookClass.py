import pandas as pd
import numpy as np
from collections import OrderedDict

class OrderBook:
    """
    A class to represent and manipulate an order book for trading data.

    Attributes:
    -----------
    buy_orders : OrderedDict
        A dictionary to store buy orders with price as key.
    sell_orders : OrderedDict
        A dictionary to store sell orders with price as key.
    """

    def __init__(self):
        """
        Initializes the OrderBook with empty buy and sell orders.
        """
        self.buy_orders = OrderedDict()
        self.sell_orders = OrderedDict()

    def update_order_book(self, row):
        """
        Updates the order book based on a given transaction row.
        Parameters:
        -----------
        row : pd.Series
            A row from the dataframe representing a single transaction.
        Returns:
        --------
        None
        """
        side = row['side']
        action = row['action']
        price = row['price']
        quantity = row['quantity']
        id = row['id']

        # Select the appropriate order book side
        orders = self.buy_orders if side == 'b' else self.sell_orders

        if action == 'a':  # Add
            if price not in orders:
                orders[price] = OrderedDict()
            orders[price][id] = quantity

        elif action == 'd':  # Delete
            if price in orders and id in orders[price]:
                del orders[price][id]
                if not orders[price]:  # If no orders left at this price level
                    del orders[price]

        elif action == 'm':  # Modify
            # Find and update the order with the given ID
            for p in orders:
                if id in orders[p]:
                    # If the price is modified, move the order to the new price level
                    if p != price:
                        del orders[p][id]
                        if not orders[p]:  # If no orders left at this old price level
                            del orders[p]
                        if price not in orders:
                            orders[price] = OrderedDict()
                        orders[price][id] = quantity
                    else:
                        # If only the quantity is modified
                        orders[p][id] = quantity
                    break  # Exit loop after finding and updating the order

        # Reassign sorted orders to ensure correct order
        if side == 'b':
            self.buy_orders = OrderedDict(sorted(self.buy_orders.items(), key=lambda x: x[0], reverse=True))
        else:
            self.sell_orders = OrderedDict(sorted(self.sell_orders.items(), key=lambda x: x[0]))

    def get_top_k_levels(self, k, is_buy=True):
        """
        Retrieves the top k price levels from the order book using NumPy.
        Parameters:
        -----------
        k : int
            The number of top levels to retrieve.
        is_buy : bool
            Flag to determine whether to fetch from buy orders or sell orders.
            True for buy orders, False for sell orders.
        Returns:
        --------
        list
            A list of tuples representing the top k levels. Each tuple contains
            (price, total quantity at this price level).
        """
        orders = self.buy_orders if is_buy else self.sell_orders
        # Flatten the orders into a list of (price, quantity) tuples
        order_list = [(price, sum(orders_at_price.values())) for price, orders_at_price in orders.items()]
        # Convert to a NumPy array
        order_array = np.array(order_list, dtype=[('price', float), ('quantity', int)])
        # Sort the array by price in the desired order
        order_array.sort(order='price', axis=0)
        if not is_buy:  # If sell orders, reverse the array to get top k highest prices
            order_array = order_array[::-1]
        # Get the top k levels
        top_k_levels = order_array[:k]
        # If there are fewer than k levels, pad with (0, 0)
        if len(top_k_levels) < k:
            padding = np.array([(0, 0)] * (k - len(top_k_levels)), dtype=[('price', float), ('quantity', int)])
            top_k_levels = np.concatenate((top_k_levels, padding))
        return top_k_levels.tolist()



    def create_output_for_dayfile(self, dayfile, ob_levels):
        """
        Processes a day file to create and output a dataframe.

        Parameters:
        -----------
        dayfile : str
            The filename of the day file to process.
        ob_levels : int
            The number of order book levels to include in the output.

        Returns:
        --------
        None
            The method creates and saves a CSV file with the output data.
        """
        input_df = pd.read_csv(dayfile) 
        columns = ['timestamp', 'price', 'side', 'bp0', 'bq0', 'bp1', 'bq1', 'bp2', 'bq2', 'bp3', 'bq3', 'bp4', 'bq4', 'ap0', 'aq0', 'ap1', 'aq1', 'ap2', 'aq2', 'ap3', 'aq3', 'ap4', 'aq4']
        output_data = []
        # Process each row in the input dataframe
        for index, row in input_df.iterrows():
            self.update_order_book(row)
            top_buy = self.get_top_k_levels(ob_levels, is_buy=True)
            top_sell = self.get_top_k_levels(ob_levels, is_buy=False)

            # Flatten the list of tuples for buy and sell orders
            top_buy_flat = [element for tupl in top_buy for element in tupl]
            top_sell_flat = [element for tupl in top_sell for element in tupl]

            # Create a new row for the output dataframe
            output_row = [row['timestamp'], row['price'], row['side']] + top_buy_flat + top_sell_flat
            output_data.append(output_row)

        # Create and save the output dataframe
        output_df = pd.DataFrame(output_data, columns=columns)
        output_df.to_csv('output_' + dayfile, index=False)

# Usage demonstration
order_book = OrderBook()
day_files = ['res_20190610.csv', 'res_20190611.csv', 'res_20190612.csv', 'res_20190613.csv', 'res_20190614.csv']
for day_file in day_files:
    order_book.create_output_for_dayfile(day_file, 5)