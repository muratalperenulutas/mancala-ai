from bank import Bank
from base import Base
from pit import Pit
from typing import Dict


class GameBoard:
    def __init__(self):
        self.linked_node_start = None
        self.node_dict: Dict[str, Base] = {}
        self.bank1 = None
        self.bank2 = None
        self.old_stone_count = 0
        self.old_player = None
        self.current_player = 1
        self.game_finish = False
        self.movements = []
        self.stones_earned = 0
        self.second_move = False
        
    def reset(self):
        self.linked_node_start = None
        self.node_dict = {}
        self.bank1 = None
        self.bank2 = None
        self.old_stone_count = 0
        self.old_player = None
        self.current_player = 1
        self.game_finish = False
        self.movements = []
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
            if self.current_player == 1
            else self.bank2.stone_count
        )
        self.movements.append(start_index)
        current_pit = self.distribute_stones(start_index)

        if isinstance(current_pit, Pit) and current_pit.stone_count == 1:
            self.get_if_opposite_stones(current_pit)

        self.take_stones_if_even(start_index, current_pit)

        self.check_side_empty() 

        self.stones_earned = (
            self.bank1.stone_count
            if self.current_player == 1
            else self.bank2.stone_count
        ) - self.old_stone_count

        if self.current_player == 1 and current_pit.index == 6:
            self.second_move = True
            self.current_player = 1
        elif self.current_player == 2 and current_pit.index == 13:
            self.second_move = True
            self.current_player = 2
        elif self.current_player == 1:
            self.current_player = 2
        else:
            self.current_player = 1
   

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
            current_pit: Base = current_pit.next
            if (
                (current_pit.index == 6 and self.current_player == 1)
                or (current_pit.index == 13 and self.current_player == 2)
            ) or (
                not (current_pit.index == 6 and self.current_player == 1)
                and not (current_pit.index == 13 and self.current_player == 2)
            ):
                current_pit.stone_count += 1
                stones_to_distribute -= 1
        return current_pit

    def get_if_opposite_stones(self, last_pit: Pit):
        if self.current_player == 1 and last_pit.index in range(0, 6):
            opposite_index = 12 - last_pit.index
            self.bank1.stone_count += (
                last_pit.stone_count + self.node_dict[opposite_index].stone_count
            )
            last_pit.stone_count = 0
            self.node_dict[opposite_index].stone_count = 0
        elif self.current_player == 2 and last_pit.index in range(7, 13):
            opposite_index = 12 - last_pit.index
            self.bank2.stone_count += (
                last_pit.stone_count + self.node_dict[opposite_index].stone_count
            )
            last_pit.stone_count = 0
            self.node_dict[opposite_index].stone_count = 0

    def check_side_empty(self):
        if all(self.node_dict[i].stone_count == 0 for i in range(7, 13)):
            self.bank2.stone_count += sum(
                self.node_dict[i].stone_count for i in range(0, 6)
            )
            for i in range(0, 6):
                self.node_dict[i].stone_count = 0
            self.game_finish = True
            self.save()
        elif all(self.node_dict[i].stone_count == 0 for i in range(0, 6)):
            self.bank1.stone_count += sum(
                self.node_dict[i].stone_count for i in range(7, 13)
            )
            for i in range(7, 13):
                self.node_dict[i].stone_count = 0
            self.game_finish = True
            self.save()

    def play(self, pit_index, player=None):
        if player is not self.current_player and player is not None:
            print("Not your turn")
            return False
        if self.node_dict[pit_index].stone_count == 0:
            print("Pit is empty")
            return False
        if self.current_player == 1 and pit_index in range(0, 6):
            self.move_stones(pit_index)
            return True
        elif self.current_player == 2 and pit_index in range(7, 13):
            self.move_stones(pit_index)
            return True
        print("Invalid pit selection")
        return False

    def save(self, filename="game_history.txt"):
        with open(filename, "a") as file:
            file.write(f"{self.movements}\n")
        print(f"Game saved: {filename}")

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

    def get_status(self):
        array = []
        for i in range(0, 14):
            array.append(self.node_dict[i].stone_count)
        array.append(self.current_player)
        array.append(1 if self.game_finish else 0)
        return array

    def get_score(self):
        score = 0
        score += self.stones_earned / 10
        score += 0.2 if self.second_move else 0
        if self.game_finish:
            if self.old_player == 1 and self.bank1.stone_count > self.bank2.stone_count:
                score += 1
            elif self.old_player == 2 and self.bank2.stone_count > self.bank1.stone_count:
                score += 1
            elif self.bank1.stone_count == self.bank2.stone_count:
                score += 0.4      
            else:
                score += -1
        return score

    def get_playable_pits(self):
        if self.current_player == 1:
            return [i for i in range(0, 6) if self.node_dict[i].stone_count > 0]
        else:
            return [i for i in range(7, 13) if self.node_dict[i].stone_count > 0]

board = GameBoard()
board.initialize()
print(board.get_status())
