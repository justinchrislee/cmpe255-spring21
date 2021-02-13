import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Solution:
    def __init__(self) -> None:
        # TODO: 
        # Load data from data/chipotle.tsv file using Pandas library and 
        # assign the dataset to the 'chipo' variable.
        file = 'data/chipotle.tsv'
        self.chipo = pd.read_csv(file, '\t')
    
    def top_x(self, count) -> None:
        # TODO
        # Top x number of entries from the dataset and display as markdown format.
        topx = self.chipo.head(count)
        print(topx.to_markdown())
        
    def count(self) -> int:
        # TODO
        # The number of observations/entries in the dataset.
        return self.chipo.shape[0]
    
    def info(self) -> None:
        # TODO
        # print data info.
        print(self.chipo.info())
        pass
    
    def num_column(self) -> int:
        # TODO return the number of columns in the dataset
        return self.chipo.shape[1]
    
    def print_columns(self) -> None:
        # TODO Print the name of all the columns.
        print(self.chipo.columns)
        pass
    
    def most_ordered_item(self):
        # TODO
        item_quants = self.chipo.groupby('item_name').agg({'quantity': 'sum', 'order_id': 'sum'})
        most_ordered_item = item_quants.sort_values('quantity', ascending=False)[:1]
        item_name = most_ordered_item.index[0]
        order_id = most_ordered_item.values[0][1]
        item_quants_choice_description = self.chipo.groupby('choice_description').agg({'quantity': 'sum'})
        most_ordered_item_choice_description = item_quants_choice_description.sort_values('quantity', ascending=False)[:1]
        quantity = most_ordered_item_choice_description.values[0][0]
        return item_name, order_id, quantity

    def total_item_orders(self) -> int:
       # TODO How many items were orderd in total?
       return self.chipo.quantity.sum()
   
    def total_sales(self) -> float:
        # TODO 
        # 1. Create a lambda function to change all item prices to float.
        # 2. Calculate total sales.
        total_sales = 0.0
        lam = lambda x: float(x[1:])
        price_to_float = self.chipo.copy(deep=False)
        price_to_float.item_price = price_to_float.item_price.apply(lam)
        for index, row in price_to_float.iterrows():
            total_sales += (row['quantity'] * row['item_price'])
        return round(total_sales, 2)
   
    def num_orders(self) -> int:
        # TODO
        # How many orders were made in the dataset?
        orders = self.chipo.groupby('order_id')
        return len(orders)
    
    def average_sales_amount_per_order(self) -> float:
        # TODO
        average = self.total_sales() / self.num_orders()
        return round(average, 2)

    def num_different_items_sold(self) -> int:
        # TODO
        # How many different items are sold?
        items = self.chipo.groupby('item_name')
        return len(items)
    
    def plot_histogram_top_x_popular_items(self, x:int) -> None:
        from collections import Counter
        letter_counter = Counter(self.chipo.item_name)
        # TODO
        # 1. convert the dictionary to a DataFrame
        # 2. sort the values from the top to the least value and slice the first 5 items
        # 3. create a 'bar' plot from the DataFrame
        # 4. set the title and labels:
        #     x: Items
        #     y: Number of Orders
        #     title: Most popular items
        # 5. show the plot. Hint: plt.show(block=True).
        dict_to_data_frame = pd.DataFrame({"Items": letter_counter.keys(), "Number of Orders": letter_counter.values()})
        dict_to_data_frame = dict_to_data_frame.sort_values('Number of Orders', ascending=False)[:5]
        bar_plot = dict_to_data_frame.plot.bar(x="Items", y="Number of Orders", title="Most Popular Items", rot=0)
        plt.show(block=True)
        pass
        
    def scatter_plot_num_items_per_order_price(self) -> None:
        # TODO
        # 1. create a list of prices by removing dollar sign and trailing space.
        # 2. groupby the orders and sum it.
        # 3. create a scatter plot:
        #       x: orders' item price
        #       y: orders' quantity
        #       s: 50
        #       c: blue
        # 4. set the title and labels.
        #       title: Numer of items per order price
        #       x: Order Price
        #       y: Num Items
        
        self.chipo.item_price = [float(price[1:].strip()) for price in self.chipo.item_price]
        orders_price_totals = self.chipo.groupby('order_id').agg({'item_price': 'sum'})['item_price'].tolist()
        orders_item_totals = self.chipo.groupby('order_id').agg('count')['quantity'].tolist()
        df = pd.DataFrame({"Order Price": orders_price_totals, "Num Items": orders_item_totals})
        df.plot.scatter(x="Order Price", y="Num Items", s=50, c="blue", title="Number of items per order price")
        plt.show(block=True)
        pass
    
        

def test() -> None:
    solution = Solution()
    solution.most_ordered_item()
    solution = Solution()
    solution.top_x(10)
    count = solution.count()
    print(count)
    assert count == 4622
    solution.info()
    count = solution.num_column()
    assert count == 5
    item_name, order_id, quantity = solution.most_ordered_item()
    assert item_name == 'Chicken Bowl'
    assert order_id == 713926	
    assert quantity == 159
    total = solution.total_item_orders()
    assert total == 4972
    assert 39237.02 == solution.total_sales()
    assert 1834 == solution.num_orders()
    assert 21.39 == solution.average_sales_amount_per_order()
    assert 50 == solution.num_different_items_sold()
    solution.plot_histogram_top_x_popular_items(5)
    solution.scatter_plot_num_items_per_order_price()
    
if __name__ == "__main__":
    # execute only if run as a script
    test()
    
    