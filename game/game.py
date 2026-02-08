from .bank import Bank
from .base import Base
from .pit import Pit
from typing import Dict


class Game:
    def __init__(self,verbose=False):
        self.linked_node_start = None
        self.node_dict: Dict[str, Base] = {}
        self.bank1 = None
        self.bank2 = None
        self.old_stone_count = 0
        self.old_player = None
        self.current_player = 0
        self.game_finish = False
        self.stones_earned = 0
        self.second_move = False
        self.verbose = verbose
        
    def reset(self):
        self.linked_node_start = None
        self.node_dict = {}
        self.bank1 = None
        self.bank2 = None
        self.old_stone_count = 0
        self.old_player = None
        self.current_player = 0
        self.game_finish = False
        self.stones_earned = 0  
        self.second_move = False  

    def initialize(self):
        last_node = None
        for x in range(14):
            if x == 6:
                node = Bank(x)
                self.bank1 = node
            elif x == 13:
                node = Bank(x)
                self.bank2 = node
            else:
                node = Pit(x)

            self.node_dict[node.index] = node

            if self.linked_node_start is None:
                self.linked_node_start = node
            else:
                last_node.next = node
            if x == 13:
                node.next = self.linked_node_start

            last_node = node

    def move_stones(self, start_index):
        self.stones_earned = 0
        self.old_player = self.current_player
        self.old_stone_count = (
            self.bank1.stone_count
            if self.current_player == 0
            else self.bank2.stone_count
        )
        current_pit = self.distribute_stones(start_index)

        if isinstance(current_pit, Pit) and current_pit.stone_count == 1:
            self.get_if_opposite_stones(current_pit)

        self.take_stones_if_even(start_index, current_pit)

        self.check_side_empty() 

        self.stones_earned = (
            self.bank1.stone_count
            if self.current_player == 0
            else self.bank2.stone_count
        ) - self.old_stone_count

        if self.current_player == 0 and current_pit.index == 6:
            self.second_move = True
        elif self.current_player == 1 and current_pit.index == 13:
            self.second_move = True
        elif self.current_player == 0:
            self.current_player = 1
        else:
            self.current_player = 0
   

    def take_stones_if_even(self, start_index, current_pit: Base):
        if (
            start_index in range(0, 6)
            and current_pit.index in range(7, 13)
            and current_pit.stone_count % 2 == 0
        ):
            self.bank1.stone_count += current_pit.stone_count
            current_pit.stone_count = 0

        elif (
            start_index in range(7, 13)
            and current_pit.index in range(0, 6)
            and current_pit.stone_count % 2 == 0
        ):
            self.bank2.stone_count += current_pit.stone_count
            current_pit.stone_count = 0

    def distribute_stones(self, start_index):
        current_pit = self.node_dict[start_index]
        stones_to_distribute = current_pit.stone_count
        current_pit.stone_count = 0

        while stones_to_distribute > 0:
            if self.verbose: print("Distributing stones...")
            current_pit: Base = current_pit.next
            if (
                (current_pit.index == 6 and self.current_player == 0)
                or (current_pit.index == 13 and self.current_player == 1)
            ) or (
                not (current_pit.index == 6 and self.current_player == 0)
                and not (current_pit.index == 13 and self.current_player == 1)
            ):
                current_pit.stone_count += 1
                stones_to_distribute -= 1
        return current_pit

    def get_if_opposite_stones(self, last_pit: Pit):
        opposite_index = 12 - last_pit.index

        if self.node_dict[opposite_index].stone_count == 0:
            return
        
        if self.current_player == 0 and last_pit.index in range(0, 6):
            self.bank1.stone_count += (
                last_pit.stone_count + self.node_dict[opposite_index].stone_count
            )
            last_pit.stone_count = 0
            self.node_dict[opposite_index].stone_count = 0
            if self.verbose: print(f"Captured stones from pit {opposite_index}")
        elif self.current_player == 1 and last_pit.index in range(7, 13):
            self.bank2.stone_count += (
                last_pit.stone_count + self.node_dict[opposite_index].stone_count
            )
            last_pit.stone_count = 0
            self.node_dict[opposite_index].stone_count = 0
            if self.verbose: print(f"Captured stones from pit {opposite_index}")

    def check_side_empty(self):
        if all(self.node_dict[i].stone_count == 0 for i in range(7, 13)):
            if self.verbose: print("Player 1 side is empty.")
            self.bank2.stone_count += sum(
                self.node_dict[i].stone_count for i in range(0, 6)
            )
            for i in range(0, 6):
                self.node_dict[i].stone_count = 0
            self.game_finish = True
        elif all(self.node_dict[i].stone_count == 0 for i in range(0, 6)):
            if self.verbose: print("Player 2 side is empty.")
            self.bank1.stone_count += sum(
                self.node_dict[i].stone_count for i in range(7, 13)
            )
            for i in range(7, 13):
                self.node_dict[i].stone_count = 0
            self.game_finish = True

    def play(self, pit_index):
        if self.node_dict[pit_index].stone_count == 0:
            print("Pit is empty")
            return False
        if self.current_player == 0 and pit_index in range(0, 6):
            self.move_stones(pit_index)
            return True
        elif self.current_player == 1 and pit_index in range(7, 13):
            self.move_stones(pit_index)
            return True
        print("Invalid pit selection")
        return False

    def display_board(self):
        print(f"Player {self.current_player}'s turn")
        self.print_board()

    def print_board(self):
        print("      ", end="")
        for i in range(12, 6, -1):
            print(f"[{self.node_dict[i].stone_count}]", end=" ")
        print()
        print(f"[{self.bank2.stone_count}] ", end="")
        print(" " * 28, end="")
        print(f"[{self.bank1.stone_count}]")
        print("      ", end="")
        for i in range(0, 6):
            print(f"[{self.node_dict[i].stone_count}]", end=" ")
        print()
        print()

    def get_board(self):
        return [self.node_dict[i].stone_count for i in range(14)]


    def get_playable_pits(self):
        if self.current_player == 0:
            return [i for i in range(0, 6) if self.node_dict[i].stone_count > 0]
        else:
            return [i for i in range(7, 13) if self.node_dict[i].stone_count > 0]     
