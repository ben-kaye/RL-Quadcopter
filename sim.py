import simfunc
import numpy as np
import time

# use as simulator for reinforcement learning

m = 10
g = 9.81
dt = 1e-3
thrust_max = 30
Tmax = 30 # 30 secs



state = np.array(np.float32([0, 1, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
u = np.array([m*g/4/thrust_max for i in range(4)])


exec_start = time.time()

for i in range(int(Tmax/dt)):
    simfunc.state_advance(state, u, dt)
    
delta_exec = time.time() - exec_start

print('done, exec time: {0:.3f}s'.format(delta_exec) )
