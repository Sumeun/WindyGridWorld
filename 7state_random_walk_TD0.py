# env = "7state_random_walk_wind"
#        7states with the leftmost and the rightmost states as terminal states
#        only one action possibility : S(tay) 
#        stochastic windy with prob. 1/2 going left and prob. 1/2 going right.

# env.valueIteration() : find Qmax with value iteration, store it in Qsol
# env.bestPolicy()     : using the result of env.valueIteration() find best policy 
#                        with the same value action having the same selection probability.

# Q-learning example

import numpy as np
from env_windyGridWorld import gym_make
from env_windyGridWorld import prWindyGW2

env = gym_make('7state_random_walk_wind')

env.valueIteration(tol=1e-10)
Qsol = env.Qmax

env.bestPolicy()
#print(env.piBest)
#input("Check the best pi")
pi = env.piBest

epsilon=1
alpha = 0.3
Q = np.zeros((env.nstate, env.naction))
Q[1:6, 0] = 1/2
nQ = np.zeros((env.nstate, env.naction))
for i in range(10000):
    totalReward =0
    #epsilon=epsilon*0.999
    #alpha = 1/float(i+1)**(0.55)
    alpha = 0.1 # constant alpha, compare Sutton p.120 
    alpha = 1/float(i+1)**(0.7)
    epsilon = 0
    #max(epsilon*0.999, 0.1)

    # Prioritized replay???
    
    #stateBefore = None
    #actionBefore = None
    iteration = 1
    state = env.reset(4,1)

    while True:        
        
        if (np.random.uniform(0,1)<epsilon):
            action = np.random.choice(range(env.naction))
        else:
            vQ = Q[state,:]
            vQmax = np.ndarray.max(vQ)    
            #vQmax = Q[state, :].max()
            actionlist = np.arange(env.naction)
            action = np.random.choice(actionlist[vQ == vQmax])
            
        #nQ[state, action] +=1
        stateNext, reward, done, info=env.step(action)            
        Q[state,action]= Q[state,action]+ alpha*(reward + max(Q[stateNext,:])-Q[state,action])
                            
        totalReward += reward
                
        if (done): 
            break;
            
        state = stateNext
        
    if (i % 100 == 0):
        print(i, "-th iteration, with epsilon=",epsilon,", alpha=",alpha,", Total reward=", totalReward, "SSQ =", np.sqrt(np.mean((Q[1:6,0]-Qsol[1:6,0])**2)))

# initial RMS averaged over states,
# Vinit = np.array([[0, 1/2, 1/2, 1/2, 1/2, 1/2, 0]])
# Vinit = np.array([[1/2, 1/2, 1/2, 1/2, 1/2]])
# np.sqrt(np.mean((Vinit-Qsol[0,1:6])**2))