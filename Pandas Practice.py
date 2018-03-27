import pandas as pd
from io import BytesIO
from zipfile import ZipFile
import urllib.request
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sklearn.linear_model
import numpy as np

# Problem 1
def get_database_files_from_zip_online(url):
    url_file = urllib.request.urlopen(url)
    with ZipFile(BytesIO(url_file.read())) as my_file:
        dataframes = {}
        for contained_file in my_file.namelist():
            print(contained_file)
            try:
                dataframes.update({contained_file: pd.read_csv(my_file.open(contained_file))})
            except (pd.errors.EmptyDataError, pd.errors.ParserError) as error:
                # print(contained_file)
                pass

    return dataframes

baseball_url = 'http://seanlahman.com/files/database/baseballdatabank-2017.1.zip'
dataframes = get_database_files_from_zip_online(baseball_url)
salaries_df = dataframes['baseballdatabank-2017.1/core/Salaries.csv']
teams_df = dataframes['baseballdatabank-2017.1/core/Teams.csv']

print(salaries_df.columns.values)
print(teams_df.columns.values)

print(salaries_df.groupby(['teamID'])['salary'].sum())
print(salaries_df.merge(teams_df, how='inner', left_on=['yearID', 'teamID'], right_on=['yearID', 'teamID']).groupby(['teamID', 'yearID'])['W','salary'].sum().reset_index())

my_df = salaries_df.merge(teams_df, how='inner', left_on=['yearID', 'teamID'], right_on=['yearID', 'teamID']).groupby(['teamID', 'yearID'])['W','salary'].sum().reset_index()

fig = plt.figure()
ax = Axes3D(fig)
ax.set_title('Salary vs Wins')
ax.set_xlabel('Team Yearly Salary ($)')
ax.set_ylabel('Number of Wins')
ax.set_zlabel('Year')

all_points = ax.scatter(my_df.loc[my_df['teamID'] != 'OAK']['salary'], my_df.loc[my_df['teamID'] != 'OAK']['W'], my_df.loc[my_df['teamID'] != 'OAK']['yearID'])

oakland_team_df = my_df.loc[my_df['teamID'] == 'OAK']
oak_points = ax.scatter(oakland_team_df['salary'], oakland_team_df['W'], oakland_team_df['yearID'], c='red')
plt.legend((oak_points, all_points), ('OAK', 'Other Teams'), scatterpoints=1)
plt.show()

years = my_df['yearID'].drop_duplicates()
lm = sklearn.linear_model.LinearRegression()
r_squared = []
for year in years:
    data = my_df.loc[my_df['yearID'] == year]
    lm.fit(np.array(data['salary']).reshape(-1, 1), data['W'])
    r_squared.append(lm.score(np.array(data['salary']).reshape(-1, 1), data['W']))

plt.scatter(years, r_squared)
plt.title('R-Squared of Linear Model by Year')
plt.xlabel('Year')
plt.ylabel('R-Squared')
plt.show()

#Problem 2
url_file = urllib.request.urlopen('https://github.com/cs109/2014_data/blob/master/countries.csv')
country_df = pd.read_csv(url_file)

