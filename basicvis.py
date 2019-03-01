import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import pandas as pd

def pltlabels(title):
    plt.title(title)
    plt.xlabel('Rating')
    plt.ylabel('Proportion of Movies')

def getPopHighMovieIds():
    data = np.loadtxt('data/data.txt')
    mov_ratings = data[:, 1:]
    mov_ids = data[:, 1]
    ratings = data[:, 2]
    # dictionary of movie id: number of ratings
    mov_cnt = Counter(mov_ids)
    # sort by value of movie dict has
    pop_ten = sorted(mov_cnt.items(), key=lambda kv: kv[1], reverse=True)
    # get top 10 movie ids
    pop_ten = [x[0] for x in pop_ten[:10]]
    # dictionary of movie id: avg rating, initialized to 0
    avg_rating = defaultdict(int)
    # get sum of ratings for each movie id and divide by the number of ratings
    for x in mov_ratings:
        avg_rating[x[0]] += x[1]
    for key in avg_rating:
        avg_rating[key] /= mov_cnt[key]
    # sort by value of movie dict has
    high_ten = sorted(avg_rating.items(), key=lambda kv: kv[1], reverse=True)
    # get top 10 movie ids
    high_ten = [x[0] for x in high_ten[:10]]
    return list(map(int, pop_ten)), list(map(int, high_ten))

def graph(movie_ids, title, png):
    ten_ratings = [x[1] for x in mov_ratings if x[0] in movie_ids]
    n, bins, patches = plt.hist(ten_ratings, 4, range=(1, 5), density=True)
    pltlabels(title)
    plt.savefig(png)
    plt.close()

def visualize(X, Y, filename):
    movies = pd.read_csv('data/movies.txt', delimiter="\t", header=None, encoding="latin_1", usecols = (0,1,3,4,5)).values
    for i in range(10):
        plt.scatter(X[i], Y[i], marker='x')
        plt.text(X[i]+.005, Y[i]+.005, movies[i][1])
        plt.title('Any 10 Movies')
    plt.savefig(filename + 'any10.png')
    plt.close()
    pop_mov_ids, high_mov_ids = getPopHighMovieIds()

    for id in pop_mov_ids:
        plt.scatter(X[id-1], Y[id-1], marker='x')
        plt.text(X[id-1]+.005, Y[id-1]+.005, movies[id-1][1])
    plt.title('Popular 10 Movies')
    plt.savefig(filename + 'pop10.png')
    plt.close()

    for id in high_mov_ids:
        plt.scatter(X[id-1], Y[id-1], marker='x')
        plt.text(X[id-1]+.005, Y[id-1]+.005, movies[id-1][1])
    plt.title('Highest Rated 10 Movies')
    plt.savefig(filename + 'high10.png')
    plt.close()

    cat = ['Action', 'Adventure', 'Animation']
    for i in range(3):
        cat_ids = [x[0] for x in movies if x[i+2] == 1]
        for id in cat_ids[:10]:
            plt.scatter(X[id-1], Y[id-1], marker='x')
            plt.text(X[id-1]+.005, Y[id-1]+.005, movies[id-1][1])
        plt.title(cat[i] + ' Movies')
        plt.savefig(filename + cat[i] + '.png')
        plt.close()

def interesting_vis(X, Y, filename):
    movies = pd.read_csv('data/movies.txt', delimiter="\t", header=None, encoding="latin_1", usecols = (0, 1, 9, 13, 14)).values
    cat = ['Documentary', 'Horror', 'Musical']
    color = ['y','b','m']
    red_dot, = plt.plot(1, "y.")
    purple_dot, = plt.plot(1, "b.")
    green_dot, = plt.plot(1, "m.")

    plt.legend([red_dot, purple_dot, green_dot], cat)
    avg = [[0,0], [0,0], [0,0]]
    for i in range(3):
        cat_ids = [x[0] for x in movies if x[i+2] == 1]
        for id in cat_ids:
            avg[i][0] += X[id-1]
            avg[i][1] += Y[id-1]
            plt.scatter(X[id-1], Y[id-1], marker='.', c=color[i])
        avg[i][0] /= len(cat_ids)
        avg[i][1] /= len(cat_ids)
        plt.scatter(avg[i][0], avg[i][1], marker='*', s=20, c=color[i])
    plt.title('Documentary, Horror, Musical Movies')
    plt.savefig(filename + '3genres.png')
    plt.close()

data = np.loadtxt('data/data.txt')
mov_ratings = data[:, 1:]
mov_ids = data[:, 1]
ratings = data[:, 2]

pop_mov_ids, high_mov_ids = getPopHighMovieIds()
graph(pop_mov_ids, 'Top 10 Most Popular Movies', 'pop.png')
graph(high_mov_ids, 'Top 10 Highest Rated Movies', 'high.png')

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
