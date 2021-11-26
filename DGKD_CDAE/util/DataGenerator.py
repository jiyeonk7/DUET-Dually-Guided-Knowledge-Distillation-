import numpy as np


def get_pairwise_all_data2(dataset, num_negatives, dual_side):
    ref_input, pos_input, neg_input = [], [], []
    train_matrix = dataset.train_matrix
    num_users = dataset.num_users
    num_items = dataset.num_items

    # 0 ~ (num_users - 1): user index
    # num_users ~ (num_users + num_items - 1): item index
    for (u, i) in train_matrix.keys():
        i += num_users
        for _ in range(num_negatives):
            # Insert user side triplets.
            ref_input.append(u)
            pos_input.append(i)
            j = np.random.randint(num_items)
            while (u, j) in train_matrix.keys():
                j = np.random.randint(num_items)
            j += num_users
            neg_input.append(j)

            if dual_side:
                # Insert item side triplets.
                ref_input.append(i)
                pos_input.append(u)
                v = np.random.randint(num_users)
                while (v, i) in train_matrix.keys():
                    v = np.random.randint(num_users)
                neg_input.append(v)

    ref_input = np.array(ref_input, dtype=np.int32)
    pos_input = np.array(pos_input, dtype=np.int32)
    neg_input = np.array(neg_input, dtype=np.int32)

    num_training_instances = len(ref_input)
    shuffle_index = np.arange(num_training_instances, dtype=np.int32)
    np.random.shuffle(shuffle_index)
    ref_input = ref_input[shuffle_index]
    pos_input = pos_input[shuffle_index]
    neg_input = neg_input[shuffle_index]

    return ref_input, pos_input, neg_input


def get_pairwise_all_data(dataset, num_negatives):
    ref_input, pos_input, neg_input = [], [], []
    train_matrix = dataset.train_matrix
    num_users = dataset.num_users
    num_items = dataset.num_items

    for (u, i) in train_matrix.keys():
        for _ in range(num_negatives):
            # Insert user side triplets.
            ref_input.append(u)
            pos_input.append(i)
            j = np.random.randint(num_items)
            while (u, j) in train_matrix.keys():
                j = np.random.randint(num_items)
            neg_input.append(j)

    ref_input = np.array(ref_input, dtype=np.int32)
    pos_input = np.array(pos_input, dtype=np.int32)
    neg_input = np.array(neg_input, dtype=np.int32)

    num_training_instances = len(ref_input)
    shuffle_index = np.arange(num_training_instances, dtype=np.int32)
    np.random.shuffle(shuffle_index)
    ref_input = ref_input[shuffle_index]
    pos_input = pos_input[shuffle_index]
    neg_input = neg_input[shuffle_index]

    return ref_input, pos_input, neg_input


def get_pointwise_all_data(dataset, num_negatives):
    user_input, item_input, labels = [], [], []

    train_matrix = dataset.train_matrix
    num_items = dataset.num_items

    for (u, i) in train_matrix.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)

        # negative instance
        for _ in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train_matrix.keys():
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)

    user_input = np.array(user_input, dtype=np.int32)
    item_input = np.array(item_input, dtype=np.int32)
    labels = np.array(labels, dtype=np.float32)
    num_training_instances = len(user_input)
    shuffle_index = np.arange(num_training_instances,dtype=np.int32)
    np.random.shuffle(shuffle_index)
    user_input = user_input[shuffle_index]
    item_input = item_input[shuffle_index]
    labels = labels[shuffle_index]

    return user_input, item_input, labels


def get_pointwise_batch_data(user_input, item_input, labels, num_batch, batch_size):
    num_training_instances = len(user_input)
    id_start = num_batch * batch_size
    id_end = (num_batch + 1) * batch_size

    if id_end > num_training_instances:
        id_end = num_training_instances

    bat_users = user_input[id_start:id_end]
    bat_items = item_input[id_start:id_end]
    bat_labeles = labels[id_start:id_end]

    return bat_users, bat_items, bat_labeles


def get_pairwise_batch_data(user_input, item_input_pos, item_input_neg, num_batch, batch_size):
    num_training_instances =len(user_input)
    id_start = num_batch * batch_size
    id_end = (num_batch + 1) * batch_size

    if id_end>num_training_instances:
        id_end=num_training_instances

    bat_users = user_input[id_start:id_end]
    bat_items_pos = item_input_pos[id_start:id_end]
    bat_items_neg = item_input_neg[id_start:id_end]

    return bat_users, bat_items_pos, bat_items_neg

def get_pointwise_all_highorder_data(dataset,high_order,num_negatives):
    user_input, item_input,item_input_recents,lables = [], [], [], []
    trainMatrix = dataset.train_matrix
    num_items = dataset.num_items
    num_users = dataset.num_users
    trainDict = dataset.train_dict
    for u in range(num_users):
        items_by_user = trainDict[u]
        for idx in range(high_order, len(items_by_user)):
            i = items_by_user[idx][0]
            # item id
            # positive instance
            user_input.append(u)
            item_input.append(i)
            item_input_recent = []

            for t in range(1, high_order+1):
                item_input_recent.append(items_by_user[idx-t][0])
            item_input_recents.append(item_input_recent)
            lables.append(1)

            for _ in range(num_negatives):
                j = np.random.randint(num_items)
                while (u, j) in trainMatrix.keys():
                    j = np.random.randint(num_items)
                user_input.append(u)
                item_input.append(j)
                item_input_recent = []
                for t in range(1, high_order+1):
                    item_input_recent.append(items_by_user[idx-t][0])
                item_input_recents.append(item_input_recent)
                lables.append(0)

    user_input = np.array(user_input, dtype=np.int32)
    item_input = np.array(item_input, dtype=np.int32)
    item_input_recents = np.array(item_input_recents)
    lables = np.array(lables, dtype=np.int32)
    num_training_instances = len(user_input)

    shuffle_index = np.arange(num_training_instances,dtype=np.int32)
    np.random.shuffle(shuffle_index)
    user_input = user_input[shuffle_index]
    item_input = item_input[shuffle_index]
    item_input_recents = item_input_recents[shuffle_index]
    lables = lables[shuffle_index]

    return user_input, item_input, item_input_recents, lables


def get_pointwise_batch_seqdata(user_input, item_input, item_input_recent, lables, num_batch, batch_size):
    num_training_instances = len(user_input)
    id_start = num_batch * batch_size
    id_end = (num_batch + 1) * batch_size
    if id_end > num_training_instances:
        id_end = num_training_instances
    bat_users = user_input[id_start:id_end]
    bat_items = item_input[id_start:id_end]
    bat_item_recent = item_input_recent[id_start:id_end]
    bat_lables = lables[id_start:id_end]
    return bat_users, bat_items, bat_item_recent, bat_lables
