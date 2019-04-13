"""
Predict House Price
"""

import graphlab
import matplotlib.pyplot as plt

def task_1(data):
    """Task 1
    """

    print "##########"
    print "# Task 1 #"
    print "##########"

    # Explore the data and show a bar chart of zipcode with highest average price
    # 98039
    #sales.show(view="Bar Chart", x="zipcode", y="price")

    # Filter out zipcode with highest average price
    zipcode = "98039"
    filter_data = data[data["zipcode"] == zipcode]
    # Get the average price and convert to currency format
    zipcode_average = "${:,.2f}".format(filter_data["price"].mean())
    # Print result
    #print filter_data[["zipcode", "price"]]
    print "Average Price For ZIP Code {0}: {1}".format(zipcode, zipcode_average)

def task_2(data):
    """Task 2
    """

    print "##########"
    print "# Task 2 #"
    print "##########"

    # Get total number of rows
    total_row = data.num_rows()
    #print "Total Row: {0}".format(total_row)

    # Select data with "sqft_living" higher than 2000 sqft but no larger than 4000 sqft
    filter_data = data[(data["sqft_living"] > 2000) & (data["sqft_living"] <= 4000)]
    #print filter_data

    # Get filtered number of rows
    filter_row = filter_data.num_rows()
    #print "Filter Row: {0}".format(filter_row)

    # Calculate the percent of house between 2000 and 4000 square feet living
    percent_range = "{:.2%}".format(float(filter_row) / float(total_row))
    print "Houses Between 2000 Square Feet and 4000 Square Feet: {0}".format(percent_range)

def task_3(data):
    """Task 3
    """

    print "##########"
    print "# Task 3 #"
    print "##########"

    # Small set of my features
    my_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']

    # Large set of advance features
    advanced_features = [
        "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "zipcode",
        "condition", # condition of house
        "grade", # measure of quality of construction
        "waterfront", # waterfront property
        "view", # type of view
        "sqft_above", # square feet above ground
        "sqft_basement", # square feet in basement
        "yr_built", # the year built
        "yr_renovated", # the year renovated
        "lat", "long", # the lat-long of the parcel
        "sqft_living15", # average sq.ft. of 15 nearest neighbors
        "sqft_lot15", # average lot size of 15 nearest neighbors
    ]

    # Split the training and test data 80/20
    train_data, test_data = data.random_split(.8, seed=0)
    
    # Create the model for the my_features set
    my_features_model = graphlab.linear_regression.create(train_data, target="price", features=my_features, validation_set=None)
    my_features_rmse = my_features_model.evaluate(test_data)["rmse"]

    # Create the model for the advanced_features set
    advanced_features_model = graphlab.linear_regression.create(train_data, target="price", features=advanced_features, validation_set=None)
    advanced_features_rmse = advanced_features_model.evaluate(test_data)["rmse"]

    # Calculate the difference
    rmse_difference = "${:,.2f}".format(my_features_rmse - advanced_features_rmse)
    print "RMSE Difference: {0}".format(rmse_difference)

def main():
    """Main Method
    """

    # Load the house sales data
    sales = graphlab.SFrame("home_data.gl/")

    # Task 1
    task_1(sales)

    # Task 2
    task_2(sales)

    # Task 3
    task_3(sales)

# Main
if __name__ == "__main__":
    main()