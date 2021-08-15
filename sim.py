import simfunc
import numpy as np
import time

# use as simulator for reinforcement learning

m = 10
g = 9.81
dt = 1e-3
Tmax = 30 # 30 secs

state = np.array(np.float64([0, 1, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
u = np.array([m*g/4*u for u in [1, 1, 1, 1]])


exec_start = time.time()

for i in range(int(Tmax/dt)):
    simfunc.state_advance(state, u, dt)

delta_exec = time.time() - exec_start

print('done, exec time: {0:.3f}s'.format(delta_exec) )
