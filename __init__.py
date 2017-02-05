import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
from paths import *
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity

FILE_NAME = 'data/file1.csv'


class Recommend():

    def __init__(self):
        self.read_file(FILE_NAME)
        self.set_matrix_data()
        self.modified_data = self.modify_matrix_data(self.train_data_matrix.copy())
        self.calculate_similarity()


    def read_file(self, file):
        with open(file, 'r') as data:
            self.data = pd.read_csv(data, ';')

    def set_matrix_data(self):
        train_data, test_data = cv.train_test_split(self.data, test_size=0)
        n_users = self.data.user_id.unique().shape[0]
        n_items = self.data.item_id.unique().shape[0]

        self.train_data_matrix = np.zeros((n_users, n_items))
        self.test_data_matrix = np.zeros((n_users, n_items))
        self.modified_train_data_matrix = np.zeros((n_users, n_items))

        for line in train_data.itertuples():
            # print line[1]-1 , line[2]-1
            self.train_data_matrix[line[1] - 1, line[2] - 1] = line[3]

        for line in test_data.itertuples():
            # print line[1]-1 , line[2]-1
            self.test_data_matrix[line[1] - 1, line[2] - 1] = line[3]

    def modify_matrix_data(self, ratings):
        mean_rating = []
        for i in range(len(ratings)):
            num = np.sum(ratings[i])
            den = np.count_nonzero(ratings[i])
            mean_rating.append(num / den)

        for line in range(len(ratings)):
            for index in range(len(ratings[line])):
                if(ratings[line][index] != 0):
                    ratings[line][index] -= mean_rating[line]

        self.mean = mean_rating
        self.modified_train_data_matrix = ratings
        return self.modified_train_data_matrix

    def calculate_similarity(self):
        # modified_sim = cosine_similarity(self.modified_data)
        self.similarity = pairwise_distances(self.modified_train_data_matrix, metric = 'cosine')

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

    def predicts(self, ratings, similarity):
        for line in range(len(ratings)):
            for index in range(len(ratings[line])):
                if(self.train_data_matrix[line][index]==0):
                    self.train_data_matrix[line][index] = n.calculate_num(line,index,similarity,ratings)/n.calculate_den(line,similarity) + self.mean[line]

        return self.train_data_matrix

        # print pairwise_distances([[0,1,1,1],[1,1,0,1]], metric='cosine')
        # print ''
        # print cosine_similarity([[0,1,1,1],[1,1,0,1]])

# To run the program into prediction
# n = Recommend()
# print n.train_data_matrix
# print n.predicts(n.modified_data, n.similarity)
