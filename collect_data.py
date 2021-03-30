from fourrooms_water import *
import numpy as np
import random
import pickle
import bz2

train_data_name = 'data_train'
test_data_name = 'data_test'
max_coins = 20
max_waters = 20
max_epilen = 100
trans_per_model = 5000
num_models = 100
num_train_models_of10 = 7

def random_model():
    Model = dict()
    Model['water'] = np.random.choice(['pass', 'block', 'left', 'right', 'forward'])
    Model['coin'] = np.random.choice(['pass', 'left', 'right', 'forward'])
    Model['action'] = np.random.choice(['normal', 'left', 'right', 'inverse'])
    Model['extra step'] = np.random.choice(['stay', 'left', 'right', 'forward'])
    return Model

def to_vector(state: FourroomsWaterState):
    # return list[pos_n, goal_n, coin_list, water_list]
    # coin_list and water_list with maximum element, -1 for empty, in descending order
    state_list = [state.position_n, state.goal_n]
    coin_list = []
    for coin, coin_state in state.coin_dict.items():
        if coin_state[1]:
            coin_list.append(coin)
    while len(coin_list) < max_coins:
        coin_list.append(-1)
    coin_list.sort(reverse=True)
    water_list = state.water_list.tolist()
    while len(water_list) < max_waters:
        water_list.append(-1)
    water_list.sort(reverse=True)
    state_list.extend(coin_list)
    state_list.extend(water_list)
    return tuple(state_list)

def collect_from_model(Model, num_trans, file):
    descr = FourroomsWater.todescr(Model)
    pickle.dump(descr, file)
    num_coins = random.randint(0, max_coins)
    num_waters = random.randint(0, max_waters)
    env = FourroomsWaterNorender(Model=Model, max_epilen=max_epilen, num_coins=num_coins, num_waters=num_waters, seed=None)
    current_trans = 0
    while current_trans < num_trans:
        if env.state.done:
            num_coins = random.randint(0, max_coins)
            num_waters = random.randint(0, max_waters)
            env = FourroomsWaterNorender(Model=Model,  max_epilen=max_epilen, num_coins=num_coins, num_waters=num_waters, seed=None)
        current_state = to_vector(env.state)
        action = random.randint(0, env.action_space.n-1)
        env.step(action)
        next_state = to_vector(env.state)
        trans = (current_state, action, next_state)
        pickle.dump(trans, file)
        current_trans += 1

def collect_data():
    # Collecting data, 7 for train, 3 for test by turns
    # No overlaps between train and test
    train_f = bz2.BZ2File(train_data_name, 'w')
    test_f = bz2.BZ2File(test_data_name, 'w')
    train_Model = dict()
    test_Model = dict()
    current_models = 0
    train_flag = 1
    test_flag = 0
    num_train_of10 = 0
    num_test_of10 = 0
    while current_models < num_models:
        if train_flag:
            while True:
                Model = random_model()
                if not test_Model.get(FourroomsWater.todescr(Model), False):
                    break
            train_Model[FourroomsWater.todescr(Model)] = True
            collect_from_model(Model, trans_per_model, train_f)
            num_train_of10 += 1
        if test_flag:
            while True:
                Model = random_model()
                if not train_Model.get(FourroomsWater.todescr(Model), False):
                    break
            test_Model[FourroomsWater.todescr(Model)] = True
            collect_from_model(Model, trans_per_model, test_f)
            num_test_of10 += 1
        if num_train_of10 >= num_train_models_of10:
            train_flag = 0
            test_flag = 1
            num_train_of10 = 0
        if num_test_of10 >= 10-num_train_models_of10:
            train_flag = 1
            test_flag = 0
            num_test_of10 = 0
        current_models += 1


if __name__ == '__main__':
    collect_data()
