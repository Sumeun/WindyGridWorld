# env = "7state_random_walk_wind"
#        7states with the leftmost and the rightmost states as terminal states
#        only one action possibility : S(tay) 
#        stochastic windy with prob. 1/2 going left and prob. 1/2 going right.

# env.calculateV(pi= , gamma = )   : calculate Value function for policy pi, 
#                                    with discount factor = gamma, with # of iterations = n_iter 
#
# TD(0) learning V-function for given policy pi



import numpy as np
from env_windyGridWorld import gym_make
from env_windyGridWorld import prWindyGW2

env = gym_make('7state_random_walk_pi')

pi = 1/2*np.ones((env.nstate, env.naction))

pi[:,0]=1/3
pi[:,1]=2/3


env.calculateV(pi=pi, gamma=1)
Vsol = env.V
alpha = 0.1

piAction = 1/2*np.ones((env.nstate, env.naction))

V = np.zeros((env.nstate))
nV = np.ones((env.nstate))
for i in range(1000):

    state = env.reset(4,1)
    done = False

    while not done:        
        
        # Selecting action according to pi
        action = np.random.choice(range(env.naction), p=piAction[state,:])    
        
        
        # TD update for V function
        stateNext, reward, done, info=env.step(action)  
        #alpha can be either constant or 1/(nV[state]**k) with 1/2<k<=1
        #alpha = 0.01
        #alpha = 1/(nV[state]**0.6)
        #alpha = 1/((i+1)**0.6)
        #alpha = 1/((i+1)**0.51)
        #alpha = 1/((nV[state])**0.7)
        #alpha = 1/((nV[state])**0.6)
        #alpha = 1/((nV[state])**0.55)
        #alpha = 75/((nV[state])**0.8+100)
        #alpha = 1/((nV[state])**0.51)
        #alpha = np.log(nV[state])/(0.1*nV[state]+np.log(nV[state]))
        #alpha = np.log(nV[state])/(0.05*nV[state]+np.log(nV[state]))
        alpha = 150/(nV[state]+300)

        # rho: importance sampling ratio
        rho = pi[state,action]/piAction[state, action]
        V[state] = V[state] + alpha*rho*(reward + V[stateNext] - V[state])  
        nV += 1
            
        state = stateNext
        
    if (i % 100 == 0):
        print(i, "-th iterations, current alpha=",alpha,", current MSE =", np.sqrt(np.mean((V[1:6]-Vsol[1:6])**2)))        

print("Final Value function : ", V)

# initial RMS averaged over states,
# Vinit = np.array([[0, 1/2, 1/2, 1/2, 1/2, 1/2, 0]])
# Vinit = np.array([[1/2, 1/2, 1/2, 1/2, 1/2]])
# np.sqrt(np.mean((Vinit-Qsol[0,1:6])**2))

state = env.reset(4,1)
done = False

while not done:        
        
    # Selecting action according to pi
    action = np.random.choice(range(env.naction), p=piAction[state,:])    
    
    stateNext, reward, done, info=env.step(action)  
    env.render()
    
    state = stateNext