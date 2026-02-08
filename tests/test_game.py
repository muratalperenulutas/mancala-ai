import unittest
from game.game import Game


class TestGame(unittest.TestCase):

    def setUp(self):
        self.game = Game()

    def test_initialization(self):
        self.game = Game()

    def test_player1_simple_move(self):
        self.game.play(0)
        self.assertEqual(self.game.board[0], 1)
        self.assertEqual(self.game.board[1], 5)
        self.assertEqual(self.game.board[2], 5)
        self.assertEqual(self.game.board[3], 5)
        self.assertEqual(self.game.current_player, 1)

    def test_player2_simple_move(self):

        self.game.current_player = 1

        self.game.play(7)

        self.assertEqual(self.game.board[7], 1)
        self.assertEqual(self.game.board[8], 5)
        self.assertEqual(self.game.board[9], 5)
        self.assertEqual(self.game.board[10], 5)

        self.assertEqual(self.game.current_player, 0)

    def test_player1_lands_in_own_bank(self):
        self.game.play(3)
        self.assertEqual(self.game.board[3], 1)
        self.assertEqual(self.game.board[self.game.bank0i], 1)
        self.assertEqual(self.game.current_player, 0)

    def test_player2_lands_in_own_bank(self):
        self.game.current_player = 1
        self.game.play(10)
        self.assertEqual(self.game.board[10], 1)

        self.assertEqual(self.game.board[self.game.bank1i], 1)
        self.assertEqual(self.game.current_player, 1)

    def test_player1_capture(self):
        self.game.board[3] = 1
        self.game.board[4] = 0
        self.game.board[8] = 5
        self.game.play(3)

        self.assertEqual(self.game.board[3], 0)
        self.assertEqual(self.game.board[8], 0)

        self.assertEqual(self.game.board[self.game.bank0i], 6)

    def test_game_over(self):
        for i in range(0, 5):
            self.game.board[i] = 0
        self.game.board[5] = 1
        self.game.current_player = 0
        self.game.play(5)

        self.assertTrue(self.game.game_finish)
if __name__ == "__main__":
    unittest.main()
