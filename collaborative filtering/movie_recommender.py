import pandas as pd
import numpy as np

ratings = pd.read_csv('user_data', sep='\t',names=['user_id','movie_id','rating'], usecols=range(3), encoding = "ISO-8859-1",)

movies = pd.read_csv('movie_data',sep='|',names=['movie_id','title'], usecols = range(2), encoding="ISO-8859-1")

ratings = pd.merge(movies,ratings)

user_ratings = ratings.pivot_table(index=['user_id'],columns = ['title'],values='rating')
x = ['Four Weddings and a Funeral (1994)','Silence of the Lambs, The (1991)','Annie Hall (1977)']
y = [4,5,4]
my_ratings = pd.Series(y,name=0,index=x)

correlation_matrix = user_ratings.corr(method='spearman',min_periods=100)
similar_movies = pd.Series()
for i in range(0,len(my_ratings.index)):
    sims = correlation_matrix[my_ratings.index[i]].dropna()
    sims = sims.map(lambda x: (x * my_ratings[i] * my_ratings[i]) - 5)
    similar_movies = similar_movies.append(sims)

similar_movies = similar_movies.groupby(similar_movies.index).sum()
similar_movies.sort_values(inplace=True,ascending =False)
filtered_list = similar_movies.drop(my_ratings.index)

print(filtered_list.head(10))
