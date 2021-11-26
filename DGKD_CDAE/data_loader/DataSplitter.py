import numpy as np
import math
import scipy.sparse as sp


def read_data_file(train_file, valid_file, test_file, info_file, implicit, UIRT):
    """
    Reads data files.
    Returns {train, valid, test} matrices, dictionary of train data with number of users and items.

    :param str train_file: Filepath of train data
    :param str test_file: Filepath of test data
    :param bool implicit: Boolean indicating if rating should be converted to 1

    :return int num_users: Number of users
    :return int num_items: Number of items
    :return np.dok_matrix train_matrix: (num_users, num_items) shaped matrix with training ratings are stored
    :return np.dok_matrix test_matrix: (num_users, num_items) shaped matrix with test ratings are stored
    :return dict train_dict: Dictionary of training data. Key: Value = User id: List of related items.
    """

    # Read the meta file.
    separator = '\t'
    with open(info_file+"_total", "r") as f:
        # The first line is the basic information for the dataset.
        num_users, num_items, num_ratings = list(map(int, f.readline().split(separator)))
    
    # Build training and test matrices.
    train_dict = [[] for _ in range(num_users)]
    train_matrix = sp.dok_matrix((num_users, num_items), dtype=np.float32)
    valid_matrix = sp.dok_matrix((num_users, num_items), dtype=np.float32)
    test_matrix = sp.dok_matrix((num_users, num_items), dtype=np.float32)

    # Read the training file.
    print("Loading the train data from \"%s\"" % train_file)
    with open(train_file, "r") as f:
        for line in f.readlines():
            if UIRT:
                u, i, r, t = line.strip().split(separator)
                user_id, item_id, rating, time = int(u)-1, int(i)-1, float(r), int(t)
            else:
                u, i, r = line.strip().split(separator)
                user_id, item_id, rating = int(u)-1, int(i)-1, float(r)
            if implicit:
                rating = 1
            #print(user_id, item_id, train_matrix.shape)
            train_dict[user_id].append([item_id, rating])
            train_matrix[user_id, item_id] = rating

    # Read the valid file.
    print("Loading the valid data from \"%s\"" % valid_file)
    with open(valid_file, "r") as f:
        for line in f.readlines():
            if UIRT:
                u, i, r, t = line.strip().split(separator)
                user_id, item_id, rating, time = int(u), int(i), float(r), int(t)
            else:
                u, i, r = line.strip().split(separator)
                user_id, item_id, rating = int(u)-1, int(i)-1, float(r)
            if implicit:
                rating = 1
            valid_matrix[user_id, item_id] = rating

    # Read the test file.
    print("Loading the test data from \"%s\"" % test_file)
    with open(test_file, "r") as f:
        for line in f.readlines():
            if UIRT:
                u, i, r, t = line.strip().split(separator)
                user_id, item_id, rating, time = int(u), int(i), float(r), int(t)
            else:
                u, i, r = line.strip().split(separator)
                user_id, item_id, rating = int(u)-1, int(i)-1, float(r)
            if implicit:
                rating = 1
            test_matrix[user_id, item_id] = rating

    print("\"num_users\": %d, \"num_items\": %d, \"num_ratings\": %d" % (num_users, num_items, num_ratings))
    return num_users, num_items, train_matrix, valid_matrix, test_matrix, train_dict


def save_fcv(data_file, info_file, separator, popularity_order=True):
    """
    Read data and split it into train, valid and test in leave-one-out manner.

    :param str data_file: File path of data to read
    :param str info_file: File path of data information to save
    :param str separator: String by which UIRT line is seperated
    :param bool popularity_order:

    :return: None
    """
    # Read the data and reorder it by popularity.
    num_users, num_items, num_ratings, user_ids, item_ids, UIRTs_per_user = order_by_popularity(data_file, separator, popularity_order)

    num_ratings_per_user, num_ratings_per_item = {}, {}
    new_user_ids, new_item_ids = {}, {}

    # Assign new user_id for each user.
    for cnt, u in enumerate(user_ids):
        new_user_ids[u[0]] = cnt
        num_ratings_per_user[cnt] = u[1]
    # Assign new item_id for each item.
    for cnt, i in enumerate(item_ids):
        new_item_ids[i[0]] = cnt
        num_ratings_per_item[cnt] = i[1]
    # Convert UIRTs with new user_id and item_id.
    for u in UIRTs_per_user.keys():
        for UIRT in UIRTs_per_user[u]:
            i = UIRT[1]
            UIRT[0] = str(new_user_ids[u])
            UIRT[1] = str(new_item_ids[i])

    # Build info lines, user_idx_lines, and item_idx_lines.
    info_lines = []
    info_lines.append('\t'.join([str(num_users), str(num_items), str(num_ratings)]))
    info_lines.append("Sparsity : %.2f%%" % ((1 - (num_ratings / (num_users * num_items))) * 100))
    ratings_per_user = list(num_ratings_per_user.values())
    info_lines.append("Min/Max/Avg. ratings per users : %d %d %.2f" %
                      (min(ratings_per_user), max(ratings_per_user), np.mean(ratings_per_user)))
    ratings_per_item = list(num_ratings_per_item.values())
    info_lines.append("Min/Max/Avg. ratings per items : %d %d %.2f" %
                      (min(ratings_per_item), max(ratings_per_item), np.mean(ratings_per_item)))
    info_lines.append('User_id\tNumber of ratings')
    for u in range(num_users):
        info_lines.append("\t".join([str(u), str(num_ratings_per_user[u])]))
    info_lines.append('\nItem_id\tNumber of ratings')
    for i in range(num_items):
        info_lines.append("\t".join([str(i), str(num_ratings_per_item[i])]))

    with open(info_file + '_total', 'w') as f:
        f.write('\n'.join(info_lines))

    user_idx_lines, item_idx_lines = [], []
    user_idx_lines.append('Original_user_id\tCurrent_user_id')
    for u, v in user_ids:
        user_idx_lines.append("\t".join([str(u), str(new_user_ids[u])]))
    item_idx_lines.append('Original_item_id\tCurrent_item_id')
    for i, v in item_ids:
        item_idx_lines.append("\t".join([str(i), str(new_item_ids[i])]))

    with open(info_file + '_user_id', 'w') as f:
        f.write('\n'.join(user_idx_lines))
    with open(info_file + '_item_id', 'w') as f:
        f.write('\n'.join(item_idx_lines))
    print("Save leave-one-out files.")


def order_by_popularity(data_file, separator, popularity_order=True):
    """
    Reads data file.
    Returns Item-Rating-Time, dictionary of train data with number of users and items.

    :param data_file:
    :param separator:
    :param popularity_order:

    :return int num_users:
    :return int num_items
    :return int num_ratings
    :return int sorted_user_ids, sorted_item_ids, UIRTs_per_user
    :return int sorted_item_ids, UIRTs_per_user
    :return int UIRTs_per_user
    """
    num_users, num_items, num_ratings = 0, 0, 0
    user_ids, item_ids, UIRTs_per_user = {}, {}, {}

    # Read the data file.
    print("Loading the dataset from \"%s\"" % data_file)
    with open(data_file, "r") as f:
        for line in f.readlines():
            #user_id, item_id, rating, time = line.strip().split(separator)
            user_id, item_id, rating = line.strip().split(separator)
            user_id, item_id = user_id, item_id

            # Update the number of ratings per user
            if user_id not in user_ids:
                user_ids[user_id] = 1
                UIRTs_per_user[user_id] = []
                num_users += 1
            else:
                user_ids[user_id] += 1

            # Update the number of ratings per item
            if item_id not in item_ids:
                item_ids[item_id] = 1
                num_items += 1
            else:
                item_ids[item_id] += 1

            num_ratings += 1
            #line = [str(user_id), str(item_id), str(rating), str(time)]
            line = [str(user_id), str(item_id), str(rating)]
            UIRTs_per_user[user_id].append(line)
    print("\"num_users\": %d, \"num_items\": %d, \"num_ratings\": %d" % (num_users, num_items, num_ratings))

    if popularity_order:
        # Sort the user_ids and item_ids by the popularity
        sorted_user_ids = sorted(user_ids.items(), key=lambda x: x[-1], reverse=True)
        sorted_item_ids = sorted(item_ids.items(), key=lambda x: x[-1], reverse=True)
    else:
        sorted_user_ids = user_ids.items()
        sorted_item_ids = item_ids.items()

    return num_users, num_items, num_ratings, sorted_user_ids, sorted_item_ids, UIRTs_per_user
