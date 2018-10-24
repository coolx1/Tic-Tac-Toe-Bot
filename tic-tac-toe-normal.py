import numpy as np
import pickle
import copy
import matplotlib.pyplot as plt

class board:

    def __init__(self):
        self.config = [0]*9
        self.isInvalid = False

    def __repr__(self):
        string = ""
        for i in range(0,9,3):
            string +=str(self.config[i])+" "+str(self.config[i+1])+" "+str(self.config[i+2])+"\n"
        return string

    def number(self):
        sum = 0
        for i in range(9):
            sum += (3**i)*self.config[i]
        return sum

    def isGwin(self):
        configs = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
        for item in configs:
            if (self.config[item[0]] == self.config[item[1]] == self.config[item[2]]):
                return True
        return False

    def isWin(self,marker):
        configs = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
        for item in configs:
            if(self.config[item[0]]==self.config[item[1]]== self.config[item[2]]==marker):
                return True
        return False

    def isLoss(self, marker):
        configs = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
        for item in configs:
            if (self.config[item[0]] == self.config[item[1]] == self.config[item[2]] != 0 and self.config[item[0]]!=marker):
                return True
        return False

    def isDraw(self):
        out = True
        if self.isGwin():
            return False
        for i in range(9):
            if self.config[i]==0:
                out = False
                break
        return out

    def isInvalidf(self, posn):
        if (self.config[posn] > 0):
            return True
        else:
            return False

    def changeConfig(self,posn,marker):
        if self.isInvalidf(posn):
            self.isInvalid = True
            return self
        else:
            self.isInvalid = False
            self.config[posn] = marker
            return self

    def reset(self):
        self.config = [0]*9


class Environment:

    def __init__(self,state):
        self.state = state

    def rewardAgent(self, marker):
        if(self.state.isWin(marker)):
            return 100
        if(self.state.isLoss(marker)):
            return -200
        elif(self.state.isDraw()):
            return 50
        elif(self.state.isInvalid):
            return -5000
        else:
            return 0

    def executeMove(self, agent):
        self.state,action = agent.makeMove()
        agent.reward = self.rewardAgent(agent.marker)
        agent.updateQtable(action)

    def LossOrDraw(self,agent):
        lst = agent.stateArray[0]
        state = lst.number()
        act = agent.lastAction
        table = agent.Q_table
        if(self.state.isLoss(agent.marker)):
            table[state][act] = -200
        elif(self.state.isDraw()):
            table[state][act] = 50

class AIAgent:

    def __init__(self, marker,env,randomness):
        self.marker = marker
        self.reward = 0
        self.Q_table = np.zeros(shape=(19683, 9))
        self.gamma = 0.9
        self.env = env
        self.stateArray = [0]
        self.randomness = randomness

    def getCurrentState(self):
        self.state = self.env.state
        return self.state

    def makeMove(self):
        board = self.getCurrentState()
        bookKeeping = copy.deepcopy(board)
        self.stateArray.pop()
        self.stateArray.append(bookKeeping)
        action = self.selectNextMove()
        self.lastAction = action
        new = board.changeConfig(action, self.marker)
        return new,action

    def selectNextMove(self):
        n = self.state.number()
        r = self.Q_table[n]
        max = np.max(r)
        w = 0
        for j in range(len(r)):
            if r[j] == max:
                w = j
                break
        randAct = np.random.randint(0,9)
        randNum = 100* np.random.random_sample()
        if(randNum<=self.randomness):
            return randAct
        else:
            return w

    def updateQtable(self,action):
        n = self.state.number()
        nxtState = self.getCurrentState()
        nn = nxtState.number()
        max=np.max(self.Q_table[nn])
        self.Q_table[n][action]= self.reward + self.gamma*max

noOfgames = []
def playGame(env, agent1, agent2):
    board = env.state
    i = 0
    global noOfgames
    while( not(board.isDraw()) and not(board.isWin(agent1.marker)) and not(board.isWin(agent2.marker))):
        env.executeMove(agent1)
        i+=1
        if(not(board.isDraw()) and not(board.isWin(agent1.marker)) and not(board.isWin(agent2.marker))):
            env.executeMove(agent2)
            i+=1
        else:
            env.LossOrDraw(agent2)
            noOfgames.append(i)
            board.reset()
            return
    env.LossOrDraw(agent1)
    noOfgames.append(i)
    board.reset()
    return


if __name__ == "__main__":
    b = board()
    env = Environment(b)
    agent1 = AIAgent(1,env,40)
    agent2 = AIAgent(2, env,40)
    print("Training...")
    # Training for 200000 games
    nog = 100000
    for i in range(nog):
        playGame(env, agent1, agent2)

    lst = [i for i in range(nog)]
    plt.plot(lst, noOfgames,"-")
    plt.xlabel("Game Number")
    plt.ylabel("Number of moves")
    plt.show()

    pickle_out = open("Q_table-normal.pickle","wb")
    pickle.dump(agent1.Q_table, pickle_out)
    pickle_out.close()
    print("Training Completed!")