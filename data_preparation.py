import pandas as pd
import numpy as np
import os
from collections import defaultdict
import pickle
import random

datasets_fake = ["FSF", "INT", "TWT"]
datasets_real = ["TFP", "E13"]
number_of_test_tweets = 1
number_of_test_users = 220


# fake dataset preparation
path_dataset = "/home/chiluen/Desktop/Fake_User_Detection/data/{}.csv".format(datasets_fake[0])
users = pd.read_csv(os.path.join(path_dataset, "users.csv"))
tweets = pd.read_csv(os.path.join(path_dataset,"tweets.csv"), dtype={"geo": str}, encoding='ISO-8859-1')
friends = pd.read_csv(os.path.join(path_dataset, "friends.csv"))
followers = pd.read_csv(os.path.join(path_dataset, "followers.csv"))
for i in datasets_fake[1:]:
    print(i)
    path_dataset = "/home/chiluen/Desktop/Fake_User_Detection/data/{}.csv".format(i)
    users = pd.concat([users, pd.read_csv(os.path.join(path_dataset, "users.csv"))])
    tweets = pd.concat([tweets, pd.read_csv(os.path.join(path_dataset,"tweets.csv"), dtype={"geo": str}, encoding='ISO-8859-1')])
    friends = pd.concat([friends, pd.read_csv(os.path.join(path_dataset, "friends.csv"))])
    followers = pd.concat([followers, pd.read_csv(os.path.join(path_dataset, "followers.csv"))])  


users_list = users['id'].tolist()
random.shuffle(users_list)
test_users = users_list[:number_of_test_users]
train_users = users_list[number_of_test_users:]

# only select first n sentence for each users in testing
test_data = []
delete_list = []
for user in test_users:
    d = tweets[tweets['user_id'] == user]['text'][0:number_of_test_tweets].tolist()
    if d == []:
        delete_list.append(user)
        continue
    test_data += d

#deal with empty tweet user
for no_user in delete_list:
    test_users.remove(no_user)

train_data = tweets['text'][tweets['user_id'].isin(train_users)].tolist()

with open("./data/fake_test_users.pickle", "wb") as f:
    pickle.dump(test_users, f)
with open("./data/fake_train_users.pickle", "wb") as f:
    pickle.dump(train_users, f)
with open("./data/fake_test_data.pickle", "wb") as f:
    pickle.dump(test_data, f)
with open("./data/fake_train_data.pickle", "wb") as f:
    pickle.dump(train_data, f)


#real dataset preparation
path_dataset = "/home/chiluen/Desktop/Fake_User_Detection/data/{}.csv".format(datasets_real[0])
users = pd.read_csv(os.path.join(path_dataset, "users.csv"))
tweets = pd.read_csv(os.path.join(path_dataset,"tweets.csv"), dtype={"geo": str}, encoding='ISO-8859-1')
friends = pd.read_csv(os.path.join(path_dataset, "friends.csv"))
followers = pd.read_csv(os.path.join(path_dataset, "followers.csv"))
for i in datasets_real[1:]:
    print(i)
    path_dataset = "/home/chiluen/Desktop/Fake_User_Detection/data/{}.csv".format(i)
    users = pd.concat([users, pd.read_csv(os.path.join(path_dataset, "users.csv"))])
    tweets = pd.concat([tweets, pd.read_csv(os.path.join(path_dataset,"tweets.csv"), dtype={"geo": str}, encoding='ISO-8859-1')])
    friends = pd.concat([friends, pd.read_csv(os.path.join(path_dataset, "friends.csv"))])
    followers = pd.concat([followers, pd.read_csv(os.path.join(path_dataset, "followers.csv"))]) 
    
users_list = users['id'].tolist()
random.shuffle(users_list)
test_users = users_list[:number_of_test_users]
train_users = users_list[number_of_test_users:]


# only select first n sentence for each users in testing
test_data = []
delete_list = []
for user in test_users:
    d = tweets[tweets['user_id'] == user]['text'][0:number_of_test_tweets].tolist()
    if d == []:
        delete_list.append(user)
        continue
    test_data += d

#deal with empty tweet user
for no_user in delete_list:
    test_users.remove(no_user)

train_data = tweets['text'][tweets['user_id'].isin(train_users)].tolist()

with open("./data/real_test_users.pickle", "wb") as f:
    pickle.dump(test_users, f)
with open("./data/real_train_users.pickle", "wb") as f:
    pickle.dump(train_users, f)

with open("./data/real_test_data.pickle", "wb") as f:
    pickle.dump(test_data, f)
with open("./data/real_train_data.pickle", "wb") as f:
    pickle.dump(train_data, f)