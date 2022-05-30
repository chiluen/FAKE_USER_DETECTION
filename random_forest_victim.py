import pandas as pd
import numpy as np
import os
from collections import defaultdict
import random
from sklearn import ensemble, preprocessing, metrics
import numpy as np

datasets_real = ["TFP", "E13"]



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

users.sort_values(by=['followers_count'], inplace=True, ascending=[False])
users_list = users['id'].tolist()

users = users[['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count']]

# followers_count最多的10%為victim

number_of_victim = int(len(users) / 10)
victim = users[:number_of_victim] #195
normal = users[number_of_victim:] #1755

train_victim, train_normal = victim.sample(n=50), normal.sample(n=50)
train_label = np.array([1 for i in range(len(train_victim))] + [0 for i in range(len(train_normal))])
train_data = np.concatenate((np.array(train_victim),np.array(train_normal)), axis=0)

forest = ensemble.RandomForestClassifier(n_estimators = 100)
forest.fit(train_data, train_label)
result_victim = forest.predict(victim)
result_normal = forest.predict(normal)
final_result = np.append(result_victim, result_normal)
arr = np.array([users_list, final_result])

df = pd.DataFrame(arr.T, columns = ['user_id', 'result(1 for victim)'])
df.to_csv('output.csv', index=False)
print('finish')


