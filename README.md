# GridWorld

This is a simple reinforcement learning environment for verifying correctness of my model.

#### Features

This environment is designed to be

- portable
- following gym interface
- easy to modify

#### Specifics

`fourrooms.py`and `fourrooms_coin.py` define two main game classes.Some details are in the beginning of the program file.

`test_util.py` and `test_baseline.py` both provide some test scripts,where `test_baseline.py` is based on [Stable Baselines]https://github.com/hill-a/stable-baselines.

`value_iteration.py` can provide knowledge about the game.It has only been implemented for non-coin and one-coin cases.

#### Installation
run

`pip install -r requirements_nobaseline.txt`

Then you can use the environment without installation of `tensorflow` and `Stable Baselines`. 


#### Unfinished work
- Finish stable baseline test.

- Add a pygame-like interface.