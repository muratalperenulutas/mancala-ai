import unittest
from game_board import GameBoard


class TestGameBoard(unittest.TestCase):

    def setUp(self):
        self.board = GameBoard()
        self.board.initialize()

    def test_initialization(self):
        for i in range(14):
            if i == 6 or i == 13:
                self.assertEqual(self.board.node_dict[i].stone_count, 0)
            else:
                self.assertEqual(self.board.node_dict[i].stone_count, 4)

    def test_player1_simple_move(self):
        self.board.play(0)
        self.assertEqual(self.board.node_dict[0].stone_count, 0)
        self.assertEqual(self.board.node_dict[1].stone_count, 5)
        self.assertEqual(self.board.node_dict[2].stone_count, 5)
        self.assertEqual(self.board.node_dict[3].stone_count, 5)
        self.assertEqual(self.board.node_dict[4].stone_count, 5)
        self.assertEqual(self.board.current_player, 2)

    def test_player2_simple_move(self):

        self.board.current_player = 2

        self.board.play(7)

        self.assertEqual(self.board.node_dict[7].stone_count, 0)

        self.assertEqual(self.board.node_dict[8].stone_count, 5)
        self.assertEqual(self.board.node_dict[9].stone_count, 5)
        self.assertEqual(self.board.node_dict[10].stone_count, 5)
        self.assertEqual(self.board.node_dict[11].stone_count, 5)

        self.assertEqual(self.board.current_player, 1)

    def test_player1_lands_in_own_bank(self):
        self.board.node_dict[2].stone_count = 4
        self.board.play(2)
        self.assertEqual(self.board.node_dict[2].stone_count, 0)
        self.assertEqual(self.board.bank1.stone_count, 1)
        self.assertEqual(self.board.current_player, 1)

    def test_player2_lands_in_own_bank(self):
        self.board.current_player = 2
        self.board.node_dict[9].stone_count = 4
        self.board.play(9)
        self.assertEqual(self.board.node_dict[9].stone_count, 0)

        self.assertEqual(self.board.bank2.stone_count, 1)

        self.assertEqual(self.board.current_player, 2)

    def test_player1_capture(self):

        self.board.node_dict[3].stone_count = 1
        self.board.node_dict[4].stone_count = 0
        self.board.node_dict[8].stone_count = 5
        self.board.play(3)

        self.assertEqual(self.board.node_dict[3].stone_count, 0)
        self.assertEqual(self.board.node_dict[8].stone_count, 0)

        self.assertEqual(self.board.bank1.stone_count, 6)

    def test_game_over(self):
        for i in range(0, 5):
            self.board.node_dict[i].stone_count = 0
        self.board.node_dict[5].stone_count = 1
        self.board.current_player = 1
        self.board.play(5, player=1)

        self.assertTrue(self.board.gameover)

if __name__ == "__main__":
    unittest.main()
