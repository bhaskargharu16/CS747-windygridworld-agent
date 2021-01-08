import numpy as np

class KingsWindyGridWorld(object):
    def __init__(self,width,height,wind,start_state,goal_state,reward):
        self.width = width # int
        self.height = height # int
        self.wind = wind # list
        self.start_state = start_state # tuple (3,0)
        self.goal_state = goal_state # tuple (3,7)
        self.current_state = start_state
        self.reward = reward # int -1
        self.actions = {'U':0,'R':1,'D':2,'L':3,'UR':4,'DR':5, 'DL':6,'UL':7 } #up#right #down #left #up-right#down-right #down-left #up-left
        self.destination = np.zeros((height,width,8,2)).astype(int)
    
    def populate_destinations(self):
        for i in range(self.height):
            for j in range(self.width):
                self.destination[i,j,0,0] = max(i - 1 - self.wind[j], 0) #up
                self.destination[i,j,0,1] = j #up

                self.destination[i,j,1,0] = max(i - self.wind[j], 0) #right
                self.destination[i,j,1,1] = min(j + 1, self.width - 1) #right

                self.destination[i,j,2,0] = max(min(i + 1 - self.wind[j], self.height - 1), 0)#down
                self.destination[i,j,2,1] = j #down

                self.destination[i,j,3,0] = max(i - self.wind[j], 0) #left
                self.destination[i,j,3,1] = max(j - 1, 0) #left

                self.destination[i,j,4,0] =  max(i - 1 - self.wind[j], 0)#up right
                self.destination[i,j,4,1] =  min(j + 1, self.width - 1)#up right

                self.destination[i,j,5,0] =  max(min(i + 1 - self.wind[j],self.height - 1), 0)# down right
                self.destination[i,j,5,1] =  min(j + 1,self.width - 1)# down right

                self.destination[i,j,6,0] =  max(min(i + 1 - self.wind[j],self.height - 1), 0)#down left
                self.destination[i,j,6,1] =  max(j - 1, 0)#down left

                self.destination[i,j,7,0] =  max(i - 1 - self.wind[j], 0)#up left
                self.destination[i,j,7,1] =  max(j - 1, 0)#up left
    
    def take_step(self,action):
        i = int(self.current_state[0])
        j = int(self.current_state[1])
        a = int(self.actions[action])
        self.current_state = (self.destination[i,j,a,0],self.destination[i,j,a,1])
        return self.current_state, self.reward

class KingsSolver(object):
    def __init__(self,environment,epsilon,alpha):
        self.env = environment
        self.epsilon = epsilon
        self.gamma = 1
        self.alpha = alpha
        self.timestep = 0
        self.epsiode = 0
        self.data = [] #epsiode values
        self.actions = {'U':0,'R':1,'D':2,'L':3,'UR':4,'DR':5, 'DL':6,'UL':7 }
        self.action_map = {0:'U',1:'R',2:'D',3:'L',4:'UR',5:'DR', 6:'DL',7:'UL' }
        self.action_value = np.zeros((environment.height*environment.width,8))
        self.path = []
    
    def reset(self):
        self.action_value = np.zeros((self.env.height*self.env.width,8))
        self.data = []
        return self

    def choose_action(self,indices):
        action = -1
        i,j = int(self.env.current_state[0]),int(self.env.current_state[1])
        if np.random.binomial(1,self.epsilon):#explore
            action = self.action_map[np.random.choice([0,1,2,3,4,5,6,7],1)[0]]
        else:#exploit
            action = self.action_map[np.argmax(self.action_value[indices[i,j],:])]
        return action
    
    def sarsa(self):
        indices = np.arange(self.env.height*self.env.width).reshape(self.env.height,self.env.width)
        while self.timestep <= 9000:
            self.env.current_state = self.env.start_state
            # start epsiode 
            self.path = []
            action = self.choose_action(indices)
            while self.env.current_state != self.env.goal_state:
                self.data.append(self.epsiode)
                action1 = action
                self.path.append(self.actions[action1])
                i1,j1 = int(self.env.current_state[0]),int(self.env.current_state[1])
                self.env.current_state,reward =  self.env.take_step(action1)
                i2,j2 = int(self.env.current_state[0]),int(self.env.current_state[1])
                action2 = self.choose_action(indices)
                action = action2
                target = reward + self.gamma * self.action_value[indices[i2,j2],self.actions[action2]]
                temporal_diff = target - self.action_value[indices[i1,j1],self.actions[action1]]
                self.action_value[indices[i1,j1],self.actions[action1]] += self.alpha * temporal_diff
                self.timestep += 1
            # end episode
            self.epsiode += 1
        self.data = self.data[:8000]
        return self
    
    def policy_probability_distribution(self,indices):
        i,j = int(self.env.current_state[0]),int(self.env.current_state[1])
        best_action = np.argmax(self.action_value[indices[i,j],:])
        distribution = np.ones(8).astype(float) * (self.epsilon / 8)
        distribution[best_action] += 1.0 - self.epsilon
        return distribution
    
    def expected_sarsa(self):
        indices = np.arange(self.env.height*self.env.width).reshape(self.env.height,self.env.width)
        while self.timestep <= 9000:
            self.env.current_state = self.env.start_state
            # start epsiode 
            self.path = []
            while self.env.current_state != self.env.goal_state:
                self.data.append(self.epsiode)
                action1 = self.choose_action(indices)
                self.path.append(self.actions[action1])
                i1,j1 = int(self.env.current_state[0]),int(self.env.current_state[1])
                self.env.current_state,reward =  self.env.take_step(action1)
                i2,j2 = int(self.env.current_state[0]),int(self.env.current_state[1])
                prob_distribution = self.policy_probability_distribution(indices)
                target = reward + self.gamma * np.sum(self.action_value[indices[i2,j2],:] * prob_distribution)
                temporal_diff = target - self.action_value[indices[i1,j1],self.actions[action1]]
                self.action_value[indices[i1,j1],self.actions[action1]] += self.alpha * temporal_diff
                self.timestep += 1
            # end episode
            self.epsiode += 1
        self.data = self.data[:8000]
        return self
    
    def q_learning(self):
        indices = np.arange(self.env.height*self.env.width).reshape(self.env.height,self.env.width)
        while self.timestep <= 9000:
            self.env.current_state = self.env.start_state
            # start epsiode 
            self.path = []
            while self.env.current_state != self.env.goal_state:
                self.data.append(self.epsiode)
                action1 = self.choose_action(indices)
                self.path.append(self.actions[action1])
                i1,j1 = int(self.env.current_state[0]),int(self.env.current_state[1])
                self.env.current_state,reward =  self.env.take_step(action1)
                i2,j2 = int(self.env.current_state[0]),int(self.env.current_state[1])
                target = reward + self.gamma * np.max(self.action_value[indices[i2,j2],:])
                temporal_diff = target - self.action_value[indices[i1,j1],self.actions[action1]]
                self.action_value[indices[i1,j1],self.actions[action1]] += self.alpha * temporal_diff
                self.timestep += 1
            #end episode
            self.epsiode += 1
        self.data = self.data[:8000]
        return self