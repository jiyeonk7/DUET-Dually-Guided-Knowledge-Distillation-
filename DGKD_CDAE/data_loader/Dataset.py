import os
import unittest
from data_loader import DataSplitter
import numpy as np

class Dataset(object):
    def __init__(self, data_dir, dataset, separator, implicit, split_type="fcv", popularity_order=True):
        self.data_dir = data_dir
        self.data_name = dataset
        self.filename = os.path.join(self.data_dir, self.data_name, self.data_name + '.rating')
        self.split_type = split_type
        
        train_file = "./data/ml1m/u1.base"
        valid_file = "./data/ml1m/u1.valid"
        test_file = "./data/ml1m/u1.test"
        info_file = "./data/ml1m/info"
        UIRT = False

        DataSplitter.save_fcv(self.filename, info_file, separator, popularity_order)

        self.num_users = 0
        self.num_items = 0
        self.train_matrix = None
        self.valid_matrix = None
        self.test_matrix = None
        self.train_dict = None
        self.neg_dict = None
        self.neg_user_dict = None

        if split_type == "fcv":
            self.num_users, self.num_items, self.train_matrix, self.valid_matrix, self.test_matrix, self.train_dict = DataSplitter.read_data_file(train_file, valid_file, test_file, info_file, implicit, UIRT)
        else:
            raise Exception("Please choose a splitter.")

    @property
    def neg_items(self):
        if self.neg_dict == None:
            self.neg_dict = {u: [] for u in range(self.num_users)}
            all_items = set(list(range(self.num_items)))
            for u, items in enumerate(self.train_dict):
                self.neg_dict[u] += list(all_items - set([x[0] for x in items]))

        return self.neg_dict

    @property
    def neg_users(self):
        if self.neg_user_dict == None:
            self.neg_user_dict = {u: [] for u in range(self.num_items)}
            all_users = set(list(range(self.num_users)))
            for i in range(self.num_items):
                pos_users = self.train_matrix[:, i].nonzero()[0].tolist()
                self.neg_user_dict[i] += list(all_users - set(pos_users))

        return self.neg_user_dict

    def switch_mode(self, MODE):
        if self.split_type == 'fcv':
            if MODE.lower() == 'valid':
                self.eval_input = self.train_matrix.toarray()
                self.eval_target = self.valid_matrix.toarray()
                self.mode = 'valid'
            elif MODE.lower() == 'test':
                self.eval_input = (self.train_matrix + self.valid_matrix).toarray()
                self.eval_target = self.test_matrix.toarray()
                self.mode = 'test'
        else:
            raise ValueError('Choose correct dataset mode. (valid or test)')

    def __str__(self):
        ret = '\n============= [Dataset] =============\n'
        ret += 'Filename: %s\n' % self.filename
        ret += 'Split type: %s\n' % self.split_type
        if self.split_type == 'ratio':
            ret += 'Split ratio: %s\n' % str(self.split_ratio)
        ret += 'Popularity order: %s\n' % str(self.popularity_order)
        ret += '# of User, Items: %d, %d\n' % (self.num_users, self.num_items)
        return ret


class TestDataset(unittest.TestCase):
    def runTest(self):
        filename = "..\\data\\ml1m\\ml1m.rating"
        separator = "::"
        implicit = True
        split_type = "fcv"
        split_ratio = [0.8, 0.1, 0.1]
        popularity_order = False
        dataset = Dataset(filename, separator, implicit, split_type, split_ratio, popularity_order)


if __name__ == '__main__':
    unittest.main()
