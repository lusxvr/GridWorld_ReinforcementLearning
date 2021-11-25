import numpy as np
import time
import matplotlib.pyplot as plt
import statistics

HOEHE = 6
BREITE = 7
FINISH_STATE = (4,5)
HINDERNIS = {(2,3),(3,3),(4,3)}     #Hier Beliebig Hindernisse Einfügen
WAND = {(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(1,0),(2,0),(3,0),(4,0),(1,6),(2,6),(3,6),(4,6),(5,0),(5,1),(5,2),(5,3),(5,4),(5,5),(5,6)}
START = (4,1)
DETERMINISTIC = False

#   0 1 2 3 4 5 6 
# 0 - - - - - - -     - Wand    
# 1 -           -     | Hindernis
# 2 -     |     -     o Ziel
# 3 -     |     -     x Agent
# 4 - x   |   o -
# 5 - - - - - - -

class State:
    def __init__(self, state=START):
        self.board = np.zeros([HOEHE, BREITE])
        for a in HINDERNIS:
            self.board[a] = -1
        for b in WAND:
            self.board[b] = -2
        self.state = state
        self.isEnd = False
        self.determine = DETERMINISTIC

    def giveReward(self):
        if self.state == FINISH_STATE:
            return 1
        elif (self.state in HINDERNIS):
            return -0.1
        else:
            return 0

    def isEndFunc(self):
        if(self.state == FINISH_STATE) or (self.state in HINDERNIS):
            self.isEnd = True

    def _chooseActionProb(self, action):
        if action == "up":
            return np.random.choice(["up", "left", "right"], p=[0.8, 0.1, 0.1])
        if action == "down":
            return np.random.choice(["down", "left", "right"], p=[0.8, 0.1, 0.1])
        if action == "left":
            return np.random.choice(["left", "up", "down"], p=[0.8, 0.1, 0.1])
        if action == "right":
            return np.random.choice(["right", "up", "down"], p=[0.8, 0.1, 0.1])

    def nxtPosition(self, action):
        if self.determine:
            if action == "up":
                nxtState = (self.state[0] - 1, self.state[1])
            elif action == "down":
                nxtState = (self.state[0] + 1, self.state[1])
            elif action == "left":
                nxtState = (self.state[0], self.state[1] - 1)
            else:
                nxtState = (self.state[0], self.state[1] + 1)
            self.determine = False
        else:
            action = self._chooseActionProb(action)
            self.determine = True
            nxtState = self.nxtPosition(action)


        if (nxtState not in WAND):
                    return nxtState
        return self.state

    def visualize(self, i=0, curr_state=(4,1), action='', nxt_state=(4,1)):
        print("------------------------")
        print("Round:            ", i)            
        print("Current position:", curr_state)         
        print("Action:           ", action)
        print("Next state:      ", nxt_state)
        print("------------------------")

        self.board[self.state] = 1
        print("------------------------")
        for i in range(0, HOEHE):
            out = ' '
            for j in range(0, BREITE):
                if self.board[i,j] == 1:
                    token = '*'
                if self.board[i,j] == 0:
                    token = ' '
                if self.board[i,j] == -1:
                    token = '|'
                if self.board[i,j] == -2:
                    token = '-'
                out += token + ' '
            print(out)
        print("------------------------")


class Agent:

    def __init__(self):
        self.states = []
        self.actions = ["up", "down", "left", "right"]
        self.State = State()
        self.isEnd = self.State.isEnd
        self.lr = 0.3
        self.exp_rate = 0.2
        self.decay_gamma = 0.95
        self.r = 0
        self.first_r = 0
        self.rewards = {'round': [], 'reward': [], 'steps': []}

        self.Q_values = {}
        for i in range(HOEHE):
            for j in range(BREITE):
                self.Q_values[(i, j)] = {}
                for a in self.actions:
                    self.Q_values[(i, j)][a] = 0

    def chooseAction(self):
        mx_nxt_reward = -1
        action = ""

        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            for a in self.actions:
                current_position = self.State.state
                nxt_reward = self.Q_values[current_position][a]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
        return action

    def takeAction(self, action):
        position = self.State.nxtPosition(action)
        return State(state=position)

    def reset(self):
        self.states = []
        self.State = State()
        self.isEnd = self.State.isEnd

    def printQ_Values(self):
        for i in range(1, HOEHE-1):
            out = ''
            for j in range(1, BREITE-1):
                for a in ag.actions:
                    out += format(ag.Q_values[(i,j)][a])
                    if len(str(ag.Q_values[(i,j)][a])) < 6:
                        out += (' '*(6-len(str(ag.Q_values[(i,j)][a]))))
                    out += " | "
                print("({},{}) | ".format(i,j) + out)
                out = ''

    def roundWon(self, reward=0):
        if(reward == 1):
            self.r = self.r + 1
        return self.r

    def firstWonRound(self, i=0):
        if (self.first_r == 0):
            self.first_r = i
        return self.first_r

    def play(self, rounds=1000, expl_rate=0.3, lr=0.2, dec=0.9, v=0, t=0):
        i = 1
        s = 0
        self.exp_rate = expl_rate
        self.lr = lr
        self.decay_gamma = dec
        while i < rounds:
            if self.State.isEnd:
                reward = self.State.giveReward()
                for a in self.actions:
                    self.Q_values[self.State.state][a] = reward
                self.rewards['round'].append(i)
                self.rewards['reward'].append(reward)
                self.rewards['steps'].append(s)

                if v:    
                    print("\n--------")
                    print("Game End Reward", reward)
                    print("--------\n")
                self.roundWon(reward)
                if (reward == 1):
                    self.firstWonRound(i)
                for s in reversed(self.states):
                    current_q_value = self.Q_values[s[0]][s[1]]
                    reward = current_q_value + self.lr * (self.decay_gamma * reward - current_q_value)
                    self.Q_values[s[0]][s[1]] = round(reward, 3)
                self.reset()
                i += 1
                s = 0
            else:
                action = self.chooseAction()
                self.states.append([(self.State.state), action])
                curr_state = self.State.state

                self.State = self.takeAction(action)
                self.State.isEndFunc()

                nxt_state = self.State.state

                self.isEnd = self.State.isEnd

                if v:
                    self.State.visualize(i, curr_state, action, nxt_state)
                
                s += 1

            
            if v:
                time.sleep(t)
                print('\n'*60)


if __name__ == "__main__":
    ag = Agent()
    ag = Agent()

    print("-------------------------------------------")
    print("Initial Q-values: \n")
    
    ag.printQ_Values()
    
    print("-------------------------------------------")
    set = int(input("Möchten sie die Standarteinstellungen ändern? (1/0)\n 1000 Runden | 0.3 Exp | 0.2 LR | 0.9 Decay\n"))
    rounds = 1000
    exp = 0.3
    lr = 0.2
    dec = 0.9
    v = 0

    if set:
        print("-------------------------------------------")
        rounds = int(input("Wie viele Runden möchten sie trainieren? (z.B. 5000)\n"))
        print("-------------------------------------------")
        exp = float(input("Welcher Exploration Parameter soll verwendet werden? (z.B. 0.3)\n"))
        print("-------------------------------------------")
        lr = float(input("Welche Learning Rate soll verwendet werden? (z.B. 0.2)\n"))
        print("-------------------------------------------")
        dec = float(input("Welches Gamma Decay soll verwendet werden? (z.B. 0.9)\n"))
        print("-------------------------------------------")
        v = int(input("Soll die gelernte Strecke am Ende visualisiert werden? (1/0)\n"))

    #Training
    print("-------------------------------------------")
    print("Training . . .")
    print("-------------------------------------------")

    
    ag.play(rounds, exp, lr, dec)
    comp_tr = ag.r
    avg_steps = sum(ag.rewards['steps'])/rounds
    med_steps = statistics.median(ag.rewards['steps'])

    print("Vollendete Runden im Training: {}/{}".format(comp_tr, rounds))
    print("Erste Vollendete Runde: {}".format(ag.first_r))
    print("Average Steps: {}".format(avg_steps))
    print("Median Steps: {}".format(med_steps))
    
    print("-------------------------------------------")
    
    #Testing
    print("-------------------------------------------")
    print("Testing . . .")
    print("-------------------------------------------")

    ag.play(11, 0, lr, dec, v, 0.1)                                                #Bei zu schneller Visualisierung hier die 0.1 ändern, das ist die Zeit zwischen Schritten
    comp_tst = ag.r - comp_tr
    
    if v:
        print("\n")
        print("Vollendete Runden im Training: {}/{}".format(comp_tr, rounds))
        print("Erste Vollendete Runde: {}".format(ag.first_r))
        print("Average Steps: {}".format(avg_steps))
        print("Median Steps: {}".format(med_steps))
    #print("\n")

    print("Vollendete Runden im Test: {}/10".format(comp_tst))
    print("-------------------------------------------")
    print("Q-values:")
    print("(x,y) | up     | down   | left   | right  |")
    print("-------------------------------------------")
    
    ag.printQ_Values()

    print("\n")

    plt.hlines(avg_steps, color="r", xmin = 0, xmax = rounds, linestyle="-", label="Avg. Steps", zorder=10)
    plt.plot(ag.rewards['round'], ag.rewards['steps'], label="Steps")
    plt.legend(loc=1)
    plt.grid(True)
    plt.xlabel('Round')
    plt.show()
