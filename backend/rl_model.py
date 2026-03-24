import numpy as np
import random

def train_agent():

    grid_size = 4
    start = (0,0)
    goal = (3,3)
    risky_cells = [(1,1),(2,3)]

    actions = [(-1,0),(1,0),(0,-1),(0,1)]

    Q = np.zeros((grid_size,grid_size,4))
    returns = {}

    epsilon = 0.2
    gamma = 1
    episodes = 5000

    def step(state, action):
        r,c = state
        dr,dc = actions[action]

        nr = r+dr
        nc = c+dc

        if nr<0 or nr>=grid_size or nc<0 or nc>=grid_size:
            return state,-1

        next_state = (nr,nc)

        if next_state == goal:
            return next_state,20
        elif next_state in risky_cells:
            return next_state,-10
        else:
            return next_state,-1

    for ep in range(episodes):

        state = start
        episode=[]
        step_count=0

        while state!=goal and step_count<100:

            step_count+=1

            if random.random()<epsilon:
                action=random.randint(0,3)
            else:
                action=np.argmax(Q[state[0],state[1]])

            next_state,reward=step(state,action)

            episode.append((state,action,reward))
            state=next_state

        G=0
        visited=set()

        for t in reversed(range(len(episode))):

            state,action,reward=episode[t]
            G = gamma*G + reward

            if (state,action) not in visited:

                visited.add((state,action))

                if (state,action) not in returns:
                    returns[(state,action)]=[]

                returns[(state,action)].append(G)

                Q[state[0],state[1],action]=np.mean(returns[(state,action)])

    policy=np.argmax(Q,axis=2)

    state=start
    path=[state]

    while state!=goal:
        action=np.argmax(Q[state[0],state[1]])
        r,c=state
        dr,dc=actions[action]
        state=(r+dr,c+dc)
        path.append(state)

    return policy.tolist(), path