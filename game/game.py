PIT_SIZE = 6
PIT_STONES = 4
TOTAL_PITS = 2 * PIT_SIZE + 2

class Game:
    def __init__(self):
        self.board=[]
        self.current_player = 0
        self.bank0i = PIT_SIZE
        self.bank1i = TOTAL_PITS-1
        self.game_finish = False
        self.old_stone_count = 0
        self.stones_earned = 0
        self.second_move = False

        for x in range(TOTAL_PITS):
            if x == PIT_SIZE or x == TOTAL_PITS - 1:
                self.board.append(0)
            else:
                self.board.append(PIT_STONES)

    def move_stones(self, index):
        self.stones_earned = 0
        self.old_stone_count = (
            self.board[self.bank0i]
            if self.current_player == 0
            else self.board[self.bank1i]
        )
        last_idx = self.distribute_stones(index)

        if last_idx is not self.bank0i and last_idx is not self.bank1i :
            self.get_if_opposite_stones(last_idx)    
            self.take_stones_if_even(last_idx)

        self.check_side_empty() 

        self.stones_earned = (
            self.board[self.bank0i]
            if self.current_player == 0
            else self.board[self.bank1i]
        ) - self.old_stone_count

        if last_idx == self.bank0i or last_idx == self.bank1i:
            self.second_move = True
        else:
            self.current_player = 1 - self.current_player
            self.second_move = False

    def distribute_stones(self, index):
        stones = self.board[index]
        if stones == 1:
            self.board[index] = 0
            self.board[index+1] += 1
            return index+1
        
        self.board[index] = 1
        stones -= 1
        
        while stones > 0:
            index = (index + 1) % TOTAL_PITS
            if (index == self.bank0i and self.current_player == 1) or (index == self.bank1i and self.current_player == 0):
                continue
            self.board[index] += 1
            stones -= 1
        return index        
   
    def take_stones_if_even(self, current_index):
        if self.board[current_index] % 2 == 0:
            if (
                self.current_player == 0
                and self.is_p1_side(current_index)
            )or (
                self.current_player == 1
                and self.is_p0_side(current_index)
            ):
                bank= self.bank0i if self.current_player == 0 else self.bank1i
                self.board[bank] += self.board[current_index]
                self.board[current_index] = 0

    def is_p0_side(self, index):return 0 <= index < PIT_SIZE
    def is_p1_side(self, index):return PIT_SIZE < index < TOTAL_PITS - 1            

    def get_if_opposite_stones(self, index):
        if self.board[index] != 1:
            return
        opposite_index =TOTAL_PITS-2 - index

        if self.board[opposite_index] == 0:
            return
        
        if (self.current_player == 0 and self.is_p0_side(index)) or (self.current_player == 1 and self.is_p1_side(index)):
            banki= self.bank0i if self.current_player == 0 else self.bank1i
            self.board[banki] += (
                self.board[index] + self.board[opposite_index]
            )
            self.board[index] = 0
            self.board[opposite_index] = 0
    def check_side_empty(self):
        p0_side= sum(self.board[0:PIT_SIZE])
        p1_side= sum(self.board[PIT_SIZE+1:TOTAL_PITS-1])
        if p0_side == 0 or p1_side == 0:
            self.board[self.bank0i] += p1_side
            self.board[self.bank1i] += p0_side
            for i in range(TOTAL_PITS):
                if i != self.bank0i and i != self.bank1i:
                    self.board[i] = 0
            self.game_finish = True

    def play(self, pit_index, symmetry=False):
        if symmetry:
            self.move_stones(pit_index)
            return True 
        if self.board[pit_index] == 0:
            print("Pit is empty")
            return False
        if self.current_player == 0 and self.is_p0_side(pit_index):
            self.move_stones(pit_index)
            return True
        elif self.current_player == 1 and self.is_p1_side(pit_index):
            self.move_stones(pit_index)
            return True
        print("Invalid pit selection")
        return False

    def display_board(self):
        print(f"Player {self.current_player}'s turn")
        print("      ", end="")
        for i in range(12, 6, -1):
            print(f"[{self.board[i]}]", end=" ")
        print()
        print(f"[{self.board[self.bank1i]}] ", end="")
        print(" " * 28, end="")
        print(f"[{self.board[self.bank0i]}]")
        print("      ", end="")
        for i in range(0, 6):
            print(f"[{self.board[i]}]", end=" ")
        print()
        print()

    def get_board(self, symmetry=False):
        if symmetry:
            if self.current_player == 0:
                return self.board.copy()
            else:
                return self.board[PIT_SIZE + 1:TOTAL_PITS - 1] + self.board[0:PIT_SIZE]
        return self.board.copy()

    def get_playable_pits(self,symmetry=False):
        if self.current_player == 0:
            return [i for i in range(0, PIT_SIZE) if self.board[i] > 0]
        else:
            if symmetry:
                return [i - (PIT_SIZE + 1) for i in range(PIT_SIZE + 1, TOTAL_PITS - 1) if self.board[i] > 0]
            return [i for i in range(PIT_SIZE + 1, TOTAL_PITS - 1) if self.board[i] > 0]     