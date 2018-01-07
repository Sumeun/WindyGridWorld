import numpy as np

class prWindyGW2:
    # state means the number of the state ex. 0, 1, 2, 3, 4, ...
    # state has coordinate x, y : state 0 is at (0,0), state 1 is at (1,0)...
    def __init__(self, x=None, y=None, goals=[[4,3]], actionList = [1,2], actionMap=None, 
                 actionMoveX=[-1,1], actionMoveY=[0,0], sizeX=7, sizeY=5, 
                 transportMap = None,
                 rewardMap = None,
                 windMap = None):
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.nstate = self.sizeX*self.sizeY
        
        self.goalsState = []
        for goal in goals:            
            if (not (goal[0] <= self.sizeX and goal[0]>=1) or not (goal[1] <= self.sizeY and goal[1] >= 1)):
                print("Check the goals!")
                return(False)
            self.goalsState.append(self.stateNum(goal))
        self.goals = goals
        #print(self.goals)
        
        if (transportMap is None):
            self.transportOn = False
            #self.transportMap = np.zeros((self.sizeX, self.sizeY, 1, 3))
            #for x in range(self.sizeX):
            #    for y in range(self.sizeY):
            #        self.transportMap[x,y,0,0] = x
            #        self.transportMap[x,y,0,1] = y
            #        self.transportMap[x,y,0,2] = 1
        else:
            self.transportMap = transportMap
            self.transportOn = True
        
        if (rewardMap is None):
            self.rewardMap = np.zeros((self.sizeX, self.sizeY, 1, 2))
            for x in range(self.sizeX):
                for y in range(self.sizeY):
                    self.rewardMap[x,y,0,0] = -1
                    self.rewardMap[x,y,0,1] = 1
        else:
            self.rewardMap = rewardMap
            
        #for x, y in self.goals:   # You can't do this cause it's afterstate reward!!!
        #    self.rewardMap[x-1, y-1, :, 0] = 0
        #    self.rewardMap[x-1, y-1, :, 1] = 1
            
            
        self.ErewardMap = np.zeros((self.sizeX, self.sizeY)) # stateNext
             
        for x in range(self.sizeX):
            for y in range(self.sizeY):
                for r in range(self.rewardMap.shape[2]):
                    self.ErewardMap[x,y] = self.ErewardMap[x,y] + self.rewardMap[x,y,r,0]*self.rewardMap[x,y,r,1]
            
        if (windMap is None):
            self.windOn = False
        else:
            self.windOn = True
            self.windMap = windMap
            
        
        if (x == None):
            while True:
                self.x, self.y = np.random.choice(range(self.sizeX))+1, np.random.choice(range(self.sizeY))+1
                if not self.stateNum([self.x,self.y]) in self.goalsState:  # [self.x, self.y] -> self.coord???
                    break
        elif (y == None): 
            print("warning x is not given but y is given!!!")
            self.x = x
            self.y = np.random.choice(range(sizeY))+1
        else:
            self.x = x
            self.y = y
            
        self.observation_space = np.zeros(1)        
        #self.obs = stateNum(self.x,self.y)    
        
        self.actionList = actionList
        self.naction = len(self.actionList)
        
        self.actionMap = np.zeros((self.sizeX, self.sizeY, self.naction, 1, 3))
        
        for x in range(self.sizeX):
            for y in range(self.sizeY):
                for a in range(self.naction):
                    self.actionMap [x,y,a,0,0] = actionMoveX[a]
                    self.actionMap [x,y,a,0,1] = actionMoveY[a]
                    self.actionMap [x,y,a,0,2] = 1
        
        # rendering
        self.viewer = None
        self.xp = None
        self.yp = None
        
        self.Psas = None
        
    def reset(self, x=None, y=None):
        if (x == None):
            while True:
                self.x, self.y = np.random.choice(range(self.sizeX))+1, np.random.choice(range(self.sizeY))+1
                if not self.stateNum([self.x,self.y]) in self.goalsState:
                    break
        elif (y == None): 
            print("warning x is not given but y is given!!!")            
            self.x = x
            while True:
                self.y = np.random.choice(range(self.sizeY))+1                        
                self.x, self.y = np.random.choice(range(self.sizeX))+1, np.random.choice(range(self.sizeY))+1
                if not self.stateNum([self.x,self.y]) in self.goalsState:
                    break
        else:
            self.x = x
            self.y = y
            
        self.x = int(self.x)
        self.y = int(self.y)
        self.xp = None
        self.yp = None
        return(self.stateNum([self.x, self.y]))
    
    def moveAgent(self, a):   
        if not a in self.actionList:
            print("Unkown action!")
            return(False)
           
        if (a==1):
            self.x = self.x-1
        if (a==2):
            self.x = self.x + 1

            
    def stateNum(self, coord):
        x = coord[0]
        y = coord[1]
        return(int(x + (y-1) *self.sizeX -1))  
    
    def stateNumToCoord(self, st):
        return([st % self.sizeX +1, 1+st // self.sizeX])        
                
    def keepInside(self):
        if (self.x<1): 
            self.x=1
        elif (self.x>self.sizeX):
            self.x=self.sizeX
        
        if (self.y<1):
            self.y=1
        elif (self.y>self.sizeY):
            self.y=self.sizeY     
            
    def keepInsideXY(self, pos):
        x = pos[0]
        y = pos[1]
        if (x<1): 
            x=1
        elif (x>self.sizeX):
            x=self.sizeX
        
        if (y<1):
            y=1
        elif (y>self.sizeY):
            y=self.sizeY     
            
        return int(x),int(y)

    def wind(self):
        if (self.x in [1,6]):        
            self.y=self.y+np.random.choice([0,1,2])    
        if (self.x in [7]):        
            self.y=self.y+np.random.choice([1,2,3])            
        if (self.x in [2,5]):        
            self.y=self.y+np.random.choice([0,-1,-2])    
        if (self.x in [3,4]):
            self.y=self.y+np.random.choice([-1,-2,-3])
            
    def windBlows(self):
        #print(type(self.windMap))
        #print(self.windMap)
        #print(self.x,self.y)
        #print(type(self.x), type(self.y))
        p = self.windMap[self.x-1, self.y-1, :, 2]
        #p = self.windMap[1, 1, :, 2]

        blowsX = np.random.choice(self.windMap[self.x-1, self.y-1, :, 0], p=p)
        blowsY = np.random.choice(self.windMap[self.x-1, self.y-1, :, 1], p=p)
        
        return [int(blowsX), int(blowsY)]
        #self.x = int(self.x + blowsX)
        #self.y = int(self.y + blowsY)
        
        # wind[x, y, i, {blowX, blowY, p}] 
        
    def agentMoves(self, action):
       
        action = int(action)
       
        #print(self.x-1, self.y-1, action-1)
        p = self.actionMap[self.x-1, self.y-1, action, :, 2]      
       
        #print(self.actionMap[self.x-1, self.y-1, action-1, :, 0])
        #print(self.actionMap[self.x-1, self.y-1, action-1, :, 1])
        #print(self.actionMap[self.x-1, self.y-1, action-1, :, 2])
        movesX = np.random.choice(self.actionMap[self.x-1, self.y-1, action, :, 0], p=p)
        movesY = np.random.choice(self.actionMap[self.x-1, self.y-1, action, :, 1], p=p)
        return [int(movesX), int(movesY)]
        #self.x = int(self.x + movesX)
        #self.y = int(self.y + movesY)
        
    def transport(self):
        p = self.transportMap[self.x-1, self.y-1, :, 2]
        transportX = np.random.choice(self.transportMap[self.x-1, self.y-1, :, 0], p=p)
        transportY = np.random.choice(self.transportMap[self.x-1, self.y-1, :, 1], p=p)
        self.x = transportX+1
        self.y = transportY+1
        
    def getReward(self):
        p = self.rewardMap[self.x-1, self.y -1, :, 1]
        return(np.random.choice(self.rewardMap[self.x-1, self.y -1, :, 0], p=p))
                
    def step(self,action):  # observation, reward, done, info
        ##print("Starting step:", self.x, self.y)
        if [self.x, self.y] in self.goals:
            print("It in the terminal State. Nothing will change.")
            return([self.stateNum([self.x, self.y]), 0, True, None])
        #windX, windY = self.windBlows() 
        
        if self.windOn:
            windX, windY = self.windBlows()
        else:
            windX, windY = 0, 0
            
        moveX, moveY = self.agentMoves(action)
        #print("move X, move Y:", moveX, moveY)
        # the problem is if wind blows first, the agent could be outside the grid world,
        # and then you can't call agentMoves because it's already out side the grid world!!!
        #self.moveAgent(action)
        #print("self.x, self.y:",self.x, self.y)
        self.x = self.x + windX + moveX
        self.y = self.y + windY + moveY
        #print("self.x, self.y:",self.x, self.y)
        self.keepInside()  
        #print("self.x, self.y:",self.x, self.y)
        # check if the goal state is reached!!! and transport!!! so goal state overrides transport               
        reward = self.getReward()
        obs = self.stateNum([self.x, self.y])
        if (obs in self.goalsState):            
            done = True                
        else:            
            done = False
            if self.transportOn:
                self.transport()
            
        obs = self.stateNum([self.x, self.y])                  
        self.x, self.y = self.stateNumToCoord(obs)
            
        info=None
        
        
        return([obs, reward, done, info])   
    
    def render(self, screen_width=None, screen_height=None, mode='human', close=False, gridSize=50, trace=True):
        from gym.envs.classic_control import rendering
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        
        #screen_width=600
        #screen_height = 400

        if (screen_width is not None and screen_height is not None):
            gridSizeX = screen_width/self.sizeX
            gridSizeY = screen_height/self.sizeY
        else:
            gridSizeX = gridSize
            gridSizeY = gridSize
            screen_width = gridSizeX*self.sizeX
            screen_height = gridSizeY*self.sizeY
            #print(screen_width, screen_height)
                    
        if self.viewer is None:
            
            self.viewer = rendering.Viewer(screen_width, screen_height+1)
            #print(gridSizeX, gridSizeY)
            for x in range(self.sizeX+1):
                self.line = rendering.Line((gridSizeX*x, 0), (gridSizeX*x, screen_height))
                self.line.set_color(0,0,0)
                self.viewer.add_geom(self.line)                             
            for y in range(self.sizeY+1):                             
                theLine = rendering.Line((0, gridSizeY*y), (screen_width, gridSizeY*y))
                theLine.set_color(0,0,0)
                self.viewer.add_geom(theLine)
            for goal in self.goals:
                goalx = (goal[0]-1)*gridSizeX
                goaly = (self.sizeY-goal[1])*gridSizeY
                self.imgGoal = rendering.FilledPolygon([(goalx+0, goaly+0), 
                                                        (goalx+gridSizeX, goaly+0), 
                                                        (goalx+gridSizeX, goaly+gridSizeY), 
                                                        (goalx+0, goaly+gridSizeY)])
                self.imgGoal.set_color(0, 0, 1) # R, G, B?
                self.viewer.add_geom(self.imgGoal)
                #self.imgGoal_trans = rendering.Transform()            
                #self.imgGoal.add_attr(self.imgGoal_trans)
                        
            #self.agent = rendering.FilledPolygon([((self.x-1)*gridSizeX,(self.sizeY-self.y+1)*gridSizeY),
            #                                    (self.x*gridSizeX,(self.sizeY-self.y+1)*gridSizeY),
            #                                    (self.x*gridSizeX,(self.sizeY-self.y)*gridSizeY),
            #                                    ((self.x-1)*gridSizeX,(self.sizeY-self.y)*gridSizeY)])
            self.agent = rendering.FilledPolygon([(0,0), (gridSizeX, 0), (gridSizeX, gridSizeY), (0,gridSizeY)])
            
            self.xp = None
            
            self.agent.set_color(0.5, 0.5, 1) # R, G, B?
            self.agent_trans = rendering.Transform()            
            self.agent.add_attr(self.agent_trans)
            self.viewer.add_geom(self.agent)
            #self.viewer.render(return_rgb_array = False)
            
        #if self.x is None: return None
        
        self.agent_trans.set_translation((self.x-1)*gridSizeX,(self.sizeY-self.y)*gridSizeY)        
        
        if self.xp is None:
            self.xp = self.x
            self.yp = self.y
            
        self.trace = rendering.Line((gridSizeX*(self.xp-1/2), gridSizeY*(self.sizeY-self.yp+1/2)), 
                                    (gridSizeX*(self.x-1/2), gridSizeY*(self.sizeY-self.y+1/2)))
        self.trace.set_color(0,0,0)
        self.viewer.add_geom(self.trace)      
            
        self.xp = self.x
        self.yp = self.y
        
        return self.viewer.render(return_rgb_array = mode =='rgb_array')
    
    def calculatePsas(self):
        if self.Psas is None:
            self.Psas = np.zeros((self.nstate, self.naction, self.nstate))
            self.ER = np.zeros((self.nstate, self.naction))
            for stateNow in range(self.nstate):
                xNow, yNow = self.stateNumToCoord(stateNow)
                         
                for actionNow in range(self.naction):                    
                    if ([xNow, yNow] in self.goals):           
                        self.Psas[stateNow, actionNow, stateNow] = 1
                        self.ER[stateNow, actionNow] = 0
                    else:
                        for a in range(self.actionMap.shape[3]):
                            pa = self.actionMap[xNow-1, yNow-1, actionNow, a, 2]
                            moveX = self.actionMap[xNow-1, yNow-1, actionNow, a, 0]
                            moveY = self.actionMap[xNow-1, yNow-1, actionNow, a, 1]
                            if self.windOn:
                                for w in range(self.windMap.shape[2]):
                                    pw = self.windMap[xNow-1, yNow-1, w, 2]
                                    windX = self.windMap[xNow-1, yNow-1, w, 0]
                                    windY = self.windMap[xNow-1, yNow-1, w, 1]
                                    x, y = self.keepInsideXY([xNow+moveX+windX, yNow+moveY+windY])
                                    #
                                    self.ER[stateNow, actionNow] = self.ER[stateNow, actionNow] + pw*pa*self.ErewardMap[x-1, y-1]
                                    if self.transportOn:
                                        for t in range(self.transportMap.shape[2]):
                                            pt = self.transportMap[x-1, y-1, t, 2]
                                            x = self.transportMap[x-1, y-1, t, 0]
                                            y = self.transportMap[x-1, y-1, t, 1]
                                            self.Psas[stateNow, actionNow, self.stateNum([x+1,y+1])] = self.Psas[stateNow, actionNow, self.stateNum([x+1,y+1])] + pa*pw*pt
                                            #self.ER[stateNow, actionNow] = self.ER[stateNow, actionNow] + pa*pw*pt*self.ErewardMap[x, y]
                                    else:
                                        self.Psas[stateNow, actionNow, self.stateNum([x,y])] = self.Psas[stateNow, actionNow, self.stateNum([x,y])] + pa*pw                                    
                                        #self.ER[stateNow, actionNow] = self.ER[stateNow, actionNow] + pa*pw*self.ErewardMap[x-1, y-1]
                            else:
                                x, y = self.keepInsideXY([xNow+moveX, yNow+moveY])
                                self.ER[stateNow, actionNow] = self.ER[stateNow, actionNow] + pa*self.ErewardMap[x-1, y-1]
                                if self.transportOn:
                                    for t in range(self.transportMap.shape[2]):
                                        pt = self.transportMap[x-1, y-1, t, 2]
                                        x = self.transportMap[x-1, y-1, t, 0]
                                        y = self.transportMap[x-1, y-1, t, 1]
                                        self.Psas[stateNow, actionNow, self.stateNum([x+1,y+1])] = self.Psas[stateNow, actionNow, self.stateNum([x+1,y+1])] + pa*pt
                                        #self.ER[stateNow, actionNow] = self.ER[stateNow, actionNow] + pa*pt*self.ErewardMap[x, y]
                                else:                                    
                                    self.Psas[stateNow, actionNow, self.stateNum([x,y])] = self.Psas[stateNow, actionNow, self.stateNum([x,y])] + pa
                                    #self.ER[stateNow, actionNow] = self.ER[stateNow, actionNow] + pa*self.ErewardMap[x-1, y-1]
        return self.Psas
    
    def calculatePss(self, pi, check=True):
        # pi[s,a] = probability 
        if check:
            for iState in range(self.nstate):
                if sum(pi[iState,:]) != 1:
                    print("Pr(action | state) should sum to 1!")
                    return False
                
        self.Pss = np.zeros((self.nstate, self.nstate))
        
        for iState in range(self.nstate):
            for iAction in range(self.naction):
                for jState in range(self.nstate):
                    self.Pss[iState, jState] = self.Pss[iState, jState] + pi[iState, iAction]*self.Psas[iState, iAction, jState]
             
        return(self.Pss)                
    
    def calculatePsasa(self, pi, check=True):
        if check:
            for iState in range(self.nstate):
                if sum(pi[iState,:]) != 1:
                    print("Pr(action | state) should sum to 1!")
                    return False
                
        self.Psasa = np.zeros((self.nstate, self.naction, self.nstate, self.naction))

        if self.Psas is None:
            calculatePsas()

        for iState in range(self.nstate):
            for iAction in range(self.naction):
                for jState in range(self.nstate):
                    for jAction in range(self.naction):
                        self.Psasa[iState, iAction, jState, jAction] = self.Psas[iState, iAction, jState]*pi[jState,jAction]
        
        return(self.Psasa)
  
    
    def calculateV(self, pi, gamma = 0.9, check=True, n_iter = 100):
        self.calculatePsas()
        self.calculatePss(pi, check)        
        self.V = np.zeros((self.nstate))
        for i in range(n_iter):            
            for iState in range(self.nstate):
                V = 0
                if not(iState in self.goalsState):
                    for iAction in range(self.naction):                
                        V = V + pi[iState, iAction]*self.ER[iState, iAction]
                        for jState in range(self.nstate):
                            V = V + gamma*pi[iState, iAction]*self.Psas[iState,iAction,jState]*self.V[jState]
                    self.V[iState] = V
        return(self.V)    
    
    def calculateQ(self, pi, gamma = 0.9, check=True, n_iter = 100):
        self.calculatePss(pi, check)
        self.calculatePsas()
        self.Q = np.zeros((self.nstate, self.naction))
        for i in range(n_iter):                        
            for iState in range(self.nstate):                
                if not(iState in self.goalsState):
                    for iAction in range(self.naction):                                        
                        Q = self.ER[iState, iAction]
                        for jState in range(self.nstate):
                            for jAction in range(self.naction):
                                Q = Q + gamma*self.Psas[iState,iAction,jState]*pi[jState,jAction]*self.Q[jState,jAction]
                        self.Q[iState, iAction] = Q
        return(self.Q)  

    def solveBellmanV(self, pi, gamma = 0.9, check=True):
        self.calculatePss(pi, check)
        self.Vsol = np.zeros((self.nstate))  
        # V = R + gamma T V,  (I- gammaT)V = R
        I = np.diag(np.ones(self.nstate))
        T = self.Pss        
        R = np.zeros((self.nstate))
        for iState in range(self.nstate):
            for iAction in range(self.naction):
                R[iState] = R[iState] + self.ER[iState, iAction]*pi[iState, iAction]

        self.Vsol = np.matmul(np.linalg.inv(I - gamma*T), R)

    def solveBellmanQ(self, pi, gamma = 0.9, check=True):
        self.calculatePsasa(pi, check)
        self.Qsol = np.zeros((self.nstate, self.naction))  
        # Q = R + gamma T Q,  (I- gammaT)Q = R
        I = np.diag(np.ones(self.nstate*self.naction))
        T = np.copy(self.Psasa)  # np.array = .... the reference to the same object!!!
        T = np.reshape(T, (self.nstate*self.naction, self.nstate*self.naction))

        R = np.copy(self.ER)
        R = np.reshape(R, (self.nstate*self.naction))           

        self.Qsol = np.matmul(np.linalg.inv(I - gamma*T), R)
    
    def valueIteration(self, gamma = 1, check=True, n_iter = 100, tol = 1e-3):
        #self.calculatePss(pi, check)
        self.calculatePsas()
        self.Qmax = np.zeros((self.nstate, self.naction))    
        Qmax_ = np.zeros((self.nstate, self.naction))        
        #tol = np.zeros((self.nstate))
        #if n_iter is None:
        for i in range(n_iter):
            Qmax_[:] = self.Qmax
            for iState in range(self.nstate):           
                if not(iState in self.goalsState):
                    for iAction in range(self.naction):
                        Qmax = self.ER[iState, iAction]
                        for jState in range(self.nstate):
                            Qmax = Qmax + gamma*self.Psas[iState,iAction,jState]*self.Qmax[jState,:].max()
                        self.Qmax[iState, iAction] = Qmax            
            tol_current = (np.absolute(Qmax_ - self.Qmax)).max()
            if  tol_current < tol:
                print("tolerace", tol_current, " is less than max tol. ", tol, "with Itertaion ", i)
                return self.Qmax
        print("Warning: iteration reached. Current tolereance : ", tol_current)
        return self.Qmax

   

    def bestPolicy(self, deterministic = True):
        if self.Qmax is None:
            valueIteration()

        self.piBest = np.zeros((self.nstate, self.naction))

        if deterministic:
            for iState in range(self.nstate):
                bestAction = self.Qmax[iState,:].argmax()
                self.piBest[iState, bestAction] = 1
                
        else:
            for iState in range(self.nstate):            
                vQ = self.Qmax[iState,:]
                vQmax = np.ndarray.max(vQ)    
                self.piBest[iState, :] = 1/sum(vQ==vQmax)* np.ones((self.naction))*(vQ == vQmax)

        return(self.piBest)
    
def gym_make(envName, sizeX=None, sizeY=None):
    if envName == 'cliff_walking':
        if sizeX is None:
            cliffWalkingSizeX = 12
            cliffWalkingSizeY = 3
        else:
            cliffWalkingSizeX = sizeX
            cliffWalkingSizeY = sizeY

        env = prWindyGW2(1,cliffWalkingSizeY,sizeX=cliffWalkingSizeX, 
                 sizeY=cliffWalkingSizeY, 
                 goals=[[cliffWalkingSizeX,cliffWalkingSizeY]], 
                 actionList=[1,2,3,4],
                 actionMoveX=[0,-1,0,+1], actionMoveY=[-1,0,+1,0])

        transportMap = np.zeros((cliffWalkingSizeX, cliffWalkingSizeY, 1, 3), dtype=np.int)
        for x in range(cliffWalkingSizeX):
            for y in range(cliffWalkingSizeY):
                transportMap[x,y,0,0] = x
                transportMap[x,y,0,1] = y
                transportMap[x,y,0,2] = 1
        
        transportMap[1:(cliffWalkingSizeX-1),cliffWalkingSizeY-1,0,0]=0   # env.x != x of map!!!
        transportMap[1:(cliffWalkingSizeX-1),cliffWalkingSizeY-1,0,1]=cliffWalkingSizeY-1
        transportMap[1:(cliffWalkingSizeX-1),cliffWalkingSizeY-1,0,2]=1

        totalRewardM=0

        rewardMap = np.zeros((cliffWalkingSizeX, cliffWalkingSizeY, 1, 2))
        for x in range(cliffWalkingSizeX):
            for y in range(cliffWalkingSizeY):
                rewardMap[x,y,0,0] = -1
                rewardMap[x,y,0,1] = 1

        rewardMap[1:(cliffWalkingSizeX-1),cliffWalkingSizeY-1,0,0]= -100  # env.x != x of map!!!
        rewardMap[1:(cliffWalkingSizeX-1),cliffWalkingSizeY-1,0,1]= 1       

        return prWindyGW2(1,cliffWalkingSizeY,sizeX=cliffWalkingSizeX, 
                 sizeY=cliffWalkingSizeY, 
                 goals=[[cliffWalkingSizeX,cliffWalkingSizeY]], 
                 actionList=[1,2,3,4],
                 actionMoveX=[0,-1,0,+1], actionMoveY=[-1,0,+1,0],
                 transportMap=transportMap,
                 rewardMap = rewardMap)
    
    
    # Stochastic Windy Grid World 7x5x2 (sizeX, sizeY, action)
    if envName == 'windy_gridworld_2actions':
        if sizeX is None:
            worldSizeX = 7
            worldSizeY = 5
        else:
            worldSizeX = sizeX
            worldSizeY = sizeY
            
        windMap = np.zeros((worldSizeX, worldSizeY, 3, 3))
        windMap = np.zeros((worldSizeX, worldSizeY, 3, 3))
        windMap[:, :, :, 0] = 0 # doesn't move x-wise
        windMap[:, :, :, 2] = 1/3
        windMap[0, :, :, 1] = np.array([0,1,2])  
        
        # windMap[x, y, :, 1] = moveX, windMap[x, y, :, 2] = moveY, windMap[x, y, :, 3] = Prob
        windMap[1, :, :, 1] = np.array([0,-1,-2]) 
        windMap[2, :, :, 1] = np.array([-1,-2,-3]) 
        windMap[3, :, :, 1] = np.array([-1,-2,-3]) 
        windMap[4, :, :, 1] = np.array([0,-1,-2]) 
        windMap[5, :, :, 1] = np.array([0,1,2]) 
        windMap[6, :, :, 1] = np.array([1,2,3]) 

        return prWindyGW2(1,3,sizeX=worldSizeX, 
                 sizeY=worldSizeY, 
                 goals=[[4,3]], 
                 windMap = windMap,
                 actionList=[1,2],
                 actionMoveX=[-1,+1], actionMoveY=[0,0])

    if envName == '7state_random_walk_wind':
        if sizeX is None:
            worldSizeX = 7
            worldSizeY = 1
        else:
            worldSizeX = sizeX
            worldSizeY = sizeY
            
        windMap = np.zeros((worldSizeX, worldSizeY, 2, 3))        
        windMap[:, :, :, 1] = 0 # doesn't move y-wise
        windMap[:, :, :, 2] = 1/2

        windMap[:, :, :, 0] = np.array([-1,+1]) # either left or right

        windMap[0, :, :, 0] = np.array([0, 0])  
        windMap[6, :, :, 0] = np.array([0, 0])  

        rewardMap = np.zeros((worldSizeX, worldSizeY, 1, 2))
        rewardMap[:,:,0,1] = 1
        rewardMap[6,0,0,0] = 1
        
        return prWindyGW2(4,1,sizeX=worldSizeX, 
                 sizeY=worldSizeY, 
                 goals=[[1,1], [7,1]], 
                 windMap = windMap,
                 rewardMap = rewardMap,
                 actionList=[1],
                 actionMoveX=[0], actionMoveY=[0])

    if envName == '7state_random_walk_pi':
        if sizeX is None:
            worldSizeX = 7
            worldSizeY = 1
        else:
            worldSizeX = sizeX
            worldSizeY = sizeY
        
        rewardMap = np.zeros((worldSizeX, worldSizeY, 1, 2))
        rewardMap[:,:,0,1] = 1
        rewardMap[6,0,0,0] = 1
        
        return prWindyGW2(4,1,sizeX=worldSizeX, 
                 sizeY=worldSizeY, 
                 goals=[[1,1], [7,1]],                  
                 rewardMap = rewardMap,
                 actionList=[1,2],
                 actionMoveX=[-1,1], actionMoveY=[0,0])

    if envName == 'windy_gridworld_4actions':
        if sizeX is None:
            worldSizeX = 6
            worldSizeY = 5
        else:
            worldSizeX = sizeX
            worldSizeY = sizeY
        
        windMap = np.zeros((worldSizeX, worldSizeY, 3, 3))
        windMap[:, :, :, 0] = 0 # doesn't move x-wise
        windMap[:, :, :, 2] = 1/3

        windMap[0, :, :, 1] = np.array([0, 0, 0])  
        windMap[0, :, :, 2] = np.array([1, 0, 0])  
        
        windMap[1, :, :, 1] = np.array([0,-1,-2]) # wind blows upward
        windMap[2, :, :, 1] = np.array([-1,-2,-3]) 
        windMap[3, :, :, 1] = np.array([-1,-2,-3]) 
        windMap[4, :, :, 1] = np.array([0,-1,-2]) 

        windMap[5, :, :, 1] = np.array([0, 0, 0]) 
        windMap[5, :, :, 2] = np.array([1, 0, 0]) 
        #windMap[6, :, :, 1] = np.array([1,2,3]) 



        return prWindyGW2(1,3,sizeX=worldSizeX, 
                 sizeY=worldSizeY, 
                 goals=[[4,3]], 
                 windMap = windMap,                 
                 actionList=[1,2,3,4],
                 actionMoveX=[0,-1,0,+1], actionMoveY=[-1,0,+1,0])

        
    

                                
