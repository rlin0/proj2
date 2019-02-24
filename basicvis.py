import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

def load_data(filename):
    """
    Function loads data stored in the file filename and returns it as a numpy ndarray.
    Input:
        filename: given as a string.
    Output:
        Data contained in the file, returned as a numpy ndarray
    """
    return np.loadtxt(filename)

data = load_data('data/data.txt')
mov_ratings = data[:, 1:]
mov_ids = data[:, 1]
ratings = data[:, 2]

# dictionary of movie id: number of ratings
mov_cnt = Counter(mov_ids)
# sort by number of ratings movie has
ten_movies = sorted(mov_cnt.items(), key=lambda kv: kv[1], reverse=True)
# top 10 most popular movies
ten_movies = [x[0] for x in ten_movies[:10]]
# ratings for top 10 most popular movies
ten_popular_ratings = [x[1] for x in mov_ratings if x[0] in ten_movies]
n, bins, patches = plt.hist(ten_popular_ratings, 4, alpha=0.5,density=True, label='Popular')
plt.title('Top 10 Movies')
plt.xlabel('Rating')
plt.ylabel('Proportion of Movies')

# dictionary of movie id: avg rating, initialized to 0
avg_rating = defaultdict(int)
# get sum of ratings for each movie id and divide by the number of ratings
for i in range(len(mov_ids)):
    avg_rating[mov_ids[i]] += ratings[i]
for key in avg_rating:
    avg_rating[key] /= mov_cnt[key]
# top 10 highest rated moveis
ten_best = sorted(avg_rating.items(), key=lambda kv: kv[1], reverse=True)
ten_best = [x[0] for x in ten_best[:10]]
# ratings for top 10 best movies
ten_best_ratings = [x[1] for x in mov_ratings if x[0] in ten_best]
print(ten_best_ratings)
n, bins, patches = plt.hist(ten_best_ratings, 4, range=(1,5), alpha=0.5, density=True,label='Highest Rated')
plt.legend(loc='best')
plt.show()

n, bins, patches = plt.hist(ratings, 4, density=True)
plt.title('Entire Movie Lens Dataset')
plt.xlabel('Rating')
plt.ylabel('Proportion of Movies')
plt.show()
