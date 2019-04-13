import graphlab
import matplotlib.pyplot as plt

# Load some house sales data
sales = graphlab.SFrame("home_data.gl/")

# Exploring the data for housing sales
sales.show(view="Scatter Plot", x="sqft_living", y="price")

# Create a simple regression model of sqft_living to price
train_data, test_data = sales.random_split(.8, seed=0)

# Build the regression model using only sqft_living as a feature
sqft_model = graphlab.linear_regression.create(train_data, target="price", features=["sqft_living"], validation_set=None)

# Evaluate the simple model
print test_data["price"].mean()
print sqft_model.evaluate(test_data)

# Let"s show what our predictions look like
%matplotlib inline
plt.plot(test_data["sqft_living"], test_data["price"], ".", test_data["sqft_living"], sqft_model.predict(test_data), "-")
sqft_model.get("coefficients")

# Explore other features in the data
my_features = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "zipcode"]
sales[my_features].show()
sales.show(view="BoxWhisker Plot", x="zipcode", y="price")

# Build a regression model with more features
my_features_model = graphlab.linear_regression.create(train_data, target="price", features=my_features, validation_set=None)
print my_features

# Comparing the results of the simple model with adding more features
print sqft_model.evaluate(test_data)
print my_features_model.evaluate(test_data)

# Apply learned models to predict prices of 3 houses
house1 = sales[sales["id"]=="5309101200"]
house1
print house1["price"]
print sqft_model.predict(house1)
print my_features_model.predict(house1)

# Prediction for a second, fancier house
house2 = sales[sales["id"]=="1925069082"]
house2
print sqft_model.predict(house2)
print my_features_model.predict(house2)

# Last house, super fancy
bill_gates = {"bedrooms": [8], 
              "bathrooms": [25], 
              "sqft_living": [50000], 
              "sqft_lot": [225000],
              "floors": [4], 
              "zipcode": ["98039"], 
              "condition": [10], 
              "grade": [10],
              "waterfront": [1],
              "view": [4],
              "sqft_above": [37500],
              "sqft_basement": [12500],
              "yr_built": [1994],
              "yr_renovated": [2010],
              "lat": [47.627606],
              "long": [-122.242054],
              "sqft_living15": [5000],
              "sqft_lot15": [40000]}
print my_features_model.predict(graphlab.SFrame(bill_gates))