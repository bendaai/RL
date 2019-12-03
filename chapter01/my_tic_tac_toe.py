import numpy as np
import pickle
## 环境
BOARD_ROWS,BOARD_COLS=3,3
BOARD_SIZE=BOARD_COLS*BOARD_ROWS

#棋盘及规则
class BOARD:
    def __init__(self):
        self.board=np.zeros((3,3))
        self.cur_symbol=1
        self.win=None
    def Who_Win(self):
        if self.win:
            return self.win
        for i in range(BOARD_ROWS):
            x=np.sum(self.board[i,:])
            if abs(x)==BOARD_COLS:
                self.win=x/BOARD_COLS
                return self.win
        for i in range(BOARD_COLS):
            x=np.sum(self.board[:,i])
            if abs(x)==BOARD_COLS:
                self.win=x/BOARD_COLS
                return self.win
        x=np.rot90(self.board).trace()
        if abs(x)==BOARD_COLS:
            self.win=x/BOARD_COLS
            return self.win       
        x=self.board.trace()
        if abs(x)==BOARD_COLS:
            self.win=x/BOARD_COLS
            return self.win
        if np.sum(np.abs(self.board))==9:
            self.win=0
            return self.win
        return self.win
    def Next_Step(self,i,j,symbol):
        if self.board[i,j]!=0:
            print('ERROR')
            return False
        self.board[i,j]=symbol
        return True
    def Judge(self,player1,player2):
        while True:
            if self.cur_symbol==1:
                i,j=player1.Get_Next(self)
                self.Next_Step(i,j,self.cur_symbol)
                winer=self.Who_Win()
                if winer is not None:
                    return winer
                self.cur_symbol=-1
            else:
                i,j=player2.Get_Next(self)
                self.Next_Step(i,j,self.cur_symbol)
                winer=self.Who_Win()
                if winer is not None:
                    return winer
                self.cur_symbol=1
    def Restart(self):
        self.__init__()
    def Print_Board(self):
        for i in range(BOARD_ROWS):
            print('-------------')
            out = '| '
            for j in range(BOARD_COLS):
                if self.board[i, j] == 1:
                    token = '*'
                elif self.board[i, j] == -1:
                    token = 'x'
                else:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-------------')

## 智能体


class Player:
    #state是棋盘board的状态
    def __init__(self,symbol,step_size=0.1,epsilon=0.1):
        self.symbol=symbol
        self.pi=dict()
        self.step_size=step_size
        self.epsilon=epsilon
        self.state_hash_proc=[]
        self.is_greedy_proc=[]
        
    def Restart(self):
        self.state_hash_proc=[]
        self.is_greedy_proc=[]
        
    def Get_Statehash(self,state):
        hash_val=0
        for i in np.nditer(state):
            hash_val=hash_val*3+i+1
        return hash_val 
        
    def Get_Next(self,Board):
        board=Board.board
        # return i j
        all_next_possible_statue=[]#元素为(state,i,j)
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if board[i,j]==0:
                    board[i,j]=self.symbol
                    
                    state=self.Get_Statehash(board)
                    score=self.pi.get(state,0.5)
                    all_next_possible_statue.append([score,state,i,j])
                    
                    board[i,j]=0
        e=np.random.rand()
        if e<self.epsilon:
            action=all_next_possible_statue[np.random.randint(len(all_next_possible_statue))]
        else:
            action=max(all_next_possible_statue)
        self.is_greedy_proc.append(e<self.epsilon)
        self.state_hash_proc.append(action[1])
        return action[2],action[3]
    def back_ward(self,final_score):
        self.pi[self.state_hash_proc[-1]]=final_score
        for k in reversed(range(len(self.state_hash_proc)-1)):
            state_hash=self.state_hash_proc[k]
            next_state_hash=self.state_hash_proc[k+1]
            is_greedy=self.is_greedy_proc[k]
            td_error=is_greedy*(self.pi.get(next_state_hash,0.5)-self.pi.get(state_hash,0.5))
            self.pi[state_hash]=self.pi.get(state_hash,0.5)+self.step_size*td_error
    def save_pi(self):
        with open(f'policy_{1 if self.symbol==1 else 2}.bin','wb') as f:
            pickle.dump(self.pi,f)
    def load_pi(self):
        with open(f'policy_{1 if self.symbol==1 else 2}.bin','rb') as f:
            self.pi=pickle.load(f)       

class HumanPlayer(Player):
    def __init__(self,symbol,step_size=0.1,epsilon=0.1):
        super(HumanPlayer,self).__init__(symbol,step_size=0.1,epsilon=0.1)
        self.keys=list('789456123')
        
    def Get_Next(self,Board):
        Board.Print_Board()
        board=Board.board
        key=input('请输入落子方位[1-9]:')
        idx=self.keys.index(key)
        i,j=idx//BOARD_COLS,idx%BOARD_COLS
        if board[i,j]!=0:
            print('ERROR,这个位置已经有子了')
            return self.Get_Next(Board)
        return i,j
        

## 训练、测试、人机对战

def train(epochs):
    p1=Player(1,epsilon=0.1)
    p2=Player(-1,epsilon=0.1)
    chess=BOARD()
    _sum1,_sum2=0,0
    for i in range(epochs):
        winner=chess.Judge(p1,p2)
        score1=(winner+1)/2
        p1.back_ward(score1)
        p2.back_ward(1-score1)
        p1.Restart()
        p2.Restart()
        chess.Restart()
        if winner==1:
            _sum1+=1
        if winner==-1:
            _sum2+=1
        if (i+1)%10000==0:
            print(f'目前{i+1}局中,play1赢了{_sum1}局,play2赢了{_sum2}局')
    p1.save_pi()
    p2.save_pi()
    print('训练结束')
def val(epochs):
    chess=BOARD()
    p1=Player(1,epsilon=0)
    p2=Player(-1,epsilon=0)
    p1.load_pi()
    p2.load_pi()
    _sum1,_sum2=0,0
    for i in range(epochs):
        chess.Restart()
        p1.Restart()
        p2.Restart()
        winner=chess.Judge(p1,p2)
        if winner==1:
            _sum1+=1
        if winner==-1:
            _sum2+=1
    print(f'{epochs}局中,play1赢了{_sum1}局,play2赢了{_sum2}局')
    print('测试结束')

def humanplay():
    chess=BOARD()
    p1=HumanPlayer(1)
    p2=Player(-1,epsilon=0)
    p2.load_pi()
    _sum1,_sum2=0,0
    while True:
        chess.Restart()
        p1.Restart()
        p2.Restart()
        winner=chess.Judge(p1,p2)
        chess.Print_Board()
        if winner==1:
            print('You Win!')
        elif winner==0:
            print('Tie!')
        else:
            print('You Loss!')
        x=input('请输入r重新开始游戏,否则结束游戏:')
        if x!='r':
            break

## Main

train(int(1e5))

val(int(1e4))

humanplay()#18679