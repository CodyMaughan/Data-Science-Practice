import urllib
import pandas as pd
import numpy as np
import sklearn
import matplotlib as plt
from io import StringIO
import requests

#Problem 2
r = requests.get('https://github.com/cs109/2014_data/raw/master/countries.csv').content
test = StringIO(r.decode('utf-8'))
country_df = pd.read_csv(test)
print(country_df)