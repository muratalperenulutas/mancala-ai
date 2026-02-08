class BoardUtils:
    @staticmethod
    def action_to_board_index(action, current_player):
        if current_player == 0:
            return action
        else:
            return action + 7

    @staticmethod
    def canonicalize_board(board14, player):
        b = list(board14)
        if player == 1:
            return b[7:14] + b[0:7]
        else:
            return b 
