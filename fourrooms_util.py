"""
Some utils for fourrooms
"""
import itertools
from copy import deepcopy

LU_pos = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 31, 32, 33, 34, 35, 41, 42, 43, 44, 45]
RU_pos = [5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 26, 27, 28, 29, 30, 36, 37, 38, 39, 40, 46, 47, 48, 49, 50, 52, 53, 54, 55,
          56]
LD_pos = [57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 73, 74, 75, 76, 77, 83, 84, 85, 86, 87, 94, 95, 96, 97, 98]
RD_pos = [68, 69, 70, 71, 72, 78, 79, 80, 81, 82, 89, 90, 91, 92, 93, 99, 100, 101, 102, 103]
rooms_pos = [LU_pos, RU_pos, LD_pos, RD_pos]
gates_pos = [25, 51, 62, 88]

train_models_value = [
    # 0: coin, 1: water, 2: wall
    (2, 0, 1, 1),
    (1, 1, 2, 0),
    (0, 1, 0, 2),
    (1, 2, 0, 0),
    (0, 0, 1, 1),
    (1, 0, 1, 0),
    (0, 1, 0, 0),
    (1, 1, 0, 1),
    (2, 0, 0, 0),
    (0, 2, 0, 0),
    (0, 0, 2, 0),
    (0, 0, 0, 2)
]

test_models_value = []
gate_value = (0, 1, 2)
for model_value in itertools.product(gate_value, gate_value, gate_value, gate_value):
    if model_value.count(2) > 1 or model_value in train_models_value:
        continue
    test_models_value.append(model_value)


def value2model(model_value):
    Model = dict()
    for i in range(4):
        if model_value[i] == 0:
            Model[gates_pos[i]] = 'coin'
        elif model_value[i] == 1:
            Model[gates_pos[i]] = 'water'
        else:
            Model[gates_pos[i]] = 'wall'
    return Model

def value2onehot(model_value):
    pass
    
train_models = [value2model(model_value) for model_value in train_models_value]
test_models = [value2model(model_value) for model_value in test_models_value]
