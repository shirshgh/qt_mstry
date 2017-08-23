import tensorflow as tf
import numpy as np
import sys

sys.path.insert(0, '../src')
from agent import Agent

state = np.random.rand(3,3)
print("state")
print(state)

p = Agent(3,3)    
print("prediction")
print(p.Play(state))
