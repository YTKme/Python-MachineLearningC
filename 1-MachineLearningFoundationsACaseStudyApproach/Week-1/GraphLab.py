import graphlab

# Output active product key.
graphlab.product_key.get_product_key()

# Load a tabular data set
sf = graphlab.SFrame('people-example.csv')
print sf

#graphlab.canvas.set_target('ipynb')

sf['age'].show(view='Categorical')