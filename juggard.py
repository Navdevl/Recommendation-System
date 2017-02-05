import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
from paths import *
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity

FILE_NAME = 'file.csv'


class Recommend():

    def __init__(self):
        self.read_file(FILE_NAME)
        self.set_matrix_data()
        # self.modify_matrix_data(self.train_data_matrix)

    def read_file(self, file):
        with open(file, 'r') as data:
            self.data = pd.read_csv(data, ';')

    def set_matrix_data(self):
        train_data, test_data = cv.train_test_split(self.data, test_size=1)
        n_users = self.data.user_id.unique().shape[0]
        n_items = self.data.item_id.unique().shape[0]

        self.train_data_matrix = np.zeros((n_users, n_items))
        self.test_data_matrix = np.zeros((n_users, n_items))
        self.similarity = np.zeros((n_users, n_items))

        for line in train_data.itertuples():
            # print line[1]-1 , line[2]-1
            self.train_data_matrix[line[1] - 1, line[2] - 1] = 1

        for line in test_data.itertuples():
            # print line[1]-1 , line[2]-1
            self.test_data_matrix[line[1] - 1, line[2] - 1] = 1

    def modify_matrix_data(self, ratings):
        mean_rating = []
        for i in range(len(ratings)):
            num = np.sum(ratings[i])
            den = np.count_nonzero(ratings[i])
            mean_rating.append(num / den)

        for index in range(len(ratings)):
            self.modified_train_data_matrix[index] = np.subtract(
                ratings[index], mean_rating[index])
        return self.modified_train_data_matrix

    def calculate_num(self, row, col, sim, ratings):
        sum = 0
        for i in range(len(ratings)):
            sum += (sim[row][i]) * (ratings[i][col])
        return sum

    def calculate_den(self, row, sim):
        sum = 0
        for i in range(len(sim)):
            sum += sim[row][i]
        return sum

    def count_intersection(self, first, second):
        inc = 0
        for index in range(len(first)):
            if((first[index] == second[index]) and first[index] == 1):
                inc += 1

        return inc

    def similarity_juggard(self, ratings):
        for row in range(len(ratings)):
            for col in range(row + 1, len(ratings[row])):
                intersection = self.count_intersection(
                    ratings[row], ratings[col])
                union = np.count_nonzero(ratings[row])
                union += np.count_nonzero(ratings[col])
                if(union == intersection):
                    result = 1
                else:
                    result = intersection / float(union - intersection)
                self.similarity[row][col] = result
                self.similarity[col][row] = result
        return self.similarity

    def predicts(self, ratings):
        print self.similarity_juggard(ratings)
        # print pairwise_distances([[0,1,1,1],[1,1,0,1]], metric='cosine')
        # print ''
        # print cosine_similarity([[0,1,1,1],[1,1,0,1]])

n = Recommend()

print n.train_data_matrix
print ''
print n.predicts(n.train_data_matrix)
