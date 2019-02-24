import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

def pltlabels(title):
    plt.title(title)
    plt.xlabel('Rating')
    plt.ylabel('Proportion of Movies')

def top10(movie_dict, title, png):
    # sort by value of movie dict has
    ten_movies = sorted(movie_dict.items(), key=lambda kv: kv[1], reverse=True)
    # get top 10 movie ids
    ten_movies = [x[0] for x in ten_movies[:10]]
    # ratings for top 10 movies
    ten_ratings = [x[1] for x in mov_ratings if x[0] in ten_movies]
    n, bins, patches = plt.hist(ten_ratings, 4, range=(1, 5), density=True)
    pltlabels(title)
    plt.savefig(png)
    plt.close()

data = np.loadtxt('data/data.txt')
mov_ratings = data[:, 1:]
mov_ids = data[:, 1]
ratings = data[:, 2]


# dictionary of movie id: number of ratings
mov_cnt = Counter(mov_ids)
top10(mov_cnt, 'Top 10 Most Popular Movies', 'pop.png')

# dictionary of movie id: avg rating, initialized to 0
avg_rating = defaultdict(int)
# get sum of ratings for each movie id and divide by the number of ratings
for x in mov_ratings:
    avg_rating[x[0]] += x[1]
for key in avg_rating:
    avg_rating[key] /= mov_cnt[key]
top10(avg_rating, 'Top 10 Highest Rated Movies', 'high.png')

n, bins, patches = plt.hist(ratings, 4, density=True)
pltlabels('Entire Movie Lens Dataset')
plt.savefig('all.png')
plt.close()

data = np.loadtxt('data/movies.txt', delimiter="\t", encoding="latin_1", usecols = (0,3,4,5))
cat = ['Action', 'Adventure', 'Animation']
for i in range(1, 4):
    cat_movies = [x[0] for x in data if x[i] == 1]
    cat_ratings = [x[1] for x in mov_ratings if x[0] in cat_movies]
    n, bins, patches = plt.hist(cat_ratings, 4, range=(1, 5), density=True)
    pltlabels(cat[i-1])
    plt.savefig(cat[i-1] + '.png')  # saves the current figure into a pdf page
    plt.close()
