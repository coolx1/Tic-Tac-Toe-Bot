import pickle
import numpy as np
import copy
import random
import re

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
            return -300
        elif(self.state.isDraw()):
            return 50
        elif(self.state.isInvalid):
            return -5000
        else:
            return 0

    def parseInput(self,a):
        b,c = re.split(",| ",a)
        try:
            b = int(b)
        except ValueError:
            return 100
        try:
            c = int(c)
        except ValueError:
            return 100
        out = 3*(b-1)+(c-1)
        return out

    def executeMove(self, agent):
        print("Agent:")
        self.state,action = agent.makeMove()
        print(self.state)
        agent.reward = self.rewardAgent(agent.marker)
        agent.updateQtable(action)

    def executePlayerMove(self):
        game_moves = [i for i in range(9)]
        a = input("Your Move:")
        action = self.parseInput(a)

        while(action not in game_moves):
            print("Invalid Input!Try again!")
            a = input("Your Move:")
            action = self.parseInput(a)

        while(self.state.isInvalidf(action)):
            print("Invalid move!Try again!")
            a = input("Your Move:")
            action = self.parseInput(a)

        self.state = self.state.changeConfig(action, 2)
        print(self.state)

    def LossOrDraw(self,agent):
        lst = agent.stateArray[0]
        state = lst.number()
        act = agent.lastAction
        table = agent.Q_table
        if(self.state.isLoss(agent.marker)):
            table[state][act] += -200
        elif(self.state.isDraw()):
            table[state][act] = 50


class AI_bot:

    def __init__(self,marker, env, qtablePickle):
        Q_file = open(qtablePickle,"rb")
        Q_table = pickle.load(Q_file)
        self.Q_table = Q_table
        self.marker = marker
        self.gamma = 0.9
        self.reward = 0
        self.env = env
        self.stateArray = [0]
        Q_file.close()

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
        w = []
        for j in range(len(r)):
            if r[j] == max:
                w.append(j)
        return random.choice(w)

    def updateQtable(self,action):
        n = self.state.number()
        nxtState = self.getCurrentState()
        nn = nxtState.number()
        max=np.max(self.Q_table[nn])
        self.Q_table[n][action]= self.reward + self.gamma*max


def playGame(env, agent1):
    board = env.state
    while( not(board.isDraw()) and not(board.isWin(agent1.marker)) and not(board.isWin(2))):
        env.executeMove(agent1)
        if(not(board.isDraw()) and not(board.isWin(agent1.marker)) and not(board.isWin(2))):
            env.executePlayerMove()
        else:
            if(board.isDraw()):
                print("Draw game!")
            else:
                print("AI wins!\nI'm going to take over your world!\nhahahahah")
            board.reset()
            return
    if (board.isDraw()):
        env.LossOrDraw(agent1)
        print("Draw game!")
    else:
        env.LossOrDraw(agent1)
        print("You Win!")
    board.reset()
    return


if __name__ == "__main__":
    b = board()
    print("Enter input in the format row,column or row(space)column\nRows and columns start with 1")
    print(b)
    env = Environment(b)
    qtablePickle = "Q_table.pickle"
    agent = AI_bot(1,env,qtablePickle)
    playGame(env,agent)
    pickle_out = open("Q_table.pickle", "wb")
    pickle.dump(agent.Q_table, pickle_out)
    pickle_out.close()