import random

class Bot:
    """
    Two-ply minimax bot using make/undo moves to avoid deepcopy.
    Heuristics: material, positional, king-safety.
    Requires Board to support handle_move and direct piece manipulation.
    """
    def __init__(self):
        self.piece_values = {' ' : 0,'P': 1, 'N': 3, 'B': 3, 'R': 5,
                             'S': 5, 'Q': 9, 'J': 9, 'K': 100}
        self.center_bonus = {(2, 2): 0.5, (2, 3): 0.5,
                             (3, 2): 0.5, (3, 3): 0.5}
        self.check_penalty = -1000

    def get_possible_moves(self, side, board):
        return board.get_all_valid_moves(side)

    def evaluate_board(self, side, state):
        score = 0
        for y in range(6):
            for x in range(6):
                p = state[y][x]
                if not p:
                    continue
                v = self.piece_values[p[1]]
                score += v if p[0] == side[0] else -v
        return score

    def is_in_check(self, side, board):
        # find king
        state = board.get_board_state()
        king_pos = next(((x, y)
                         for y in range(6)
                         for x in range(6)
                         if state[y][x] == side[0] + 'K'), None)
        if not king_pos:
            return True
        # any opponent move hits king
        opp = 'white' if side == 'black' else 'black'
        for _, end in board.get_all_valid_moves(opp):
            if end == king_pos:
                return True
        return False

    def _make_move(self, board, move):
        # apply move, return undo info
        start, end = move
        from_sq = board.get_square_from_pos(start)
        to_sq = board.get_square_from_pos(end)
        moved = from_sq.occupying_piece
        captured = to_sq.occupying_piece
        # perform
        from_sq.occupying_piece = None
        to_sq.occupying_piece = moved
        moved.pos = end
        # toggle turn
        board.turn = 'white' if board.turn == 'black' else 'black'
        board.num_moves += 1
        return (start, end, moved, captured)

    def _undo_move(self, board, info):
        start, end, moved, captured = info
        from_sq = board.get_square_from_pos(start)
        to_sq = board.get_square_from_pos(end)
        # revert
        to_sq.occupying_piece = captured
        from_sq.occupying_piece = moved
        moved.pos = start
        board.turn = 'white' if board.turn == 'black' else 'black'
        board.num_moves -= 1

    def evaluate_move(self, side, board, move):
        state = board.get_board_state()
        sx, sy = move[0]; ex, ey = move[1]
        target = state[ey][ex]
        material_gain = self.piece_values[target[1]] if target else 0
        pos_bonus = self.center_bonus.get((ex, ey), 0)

        # make, evaluate, undo
        undo_info = self._make_move(board, move)
        if self.is_in_check(side, board):
            score = self.check_penalty
        else:
            new_state = board.get_board_state()
            board_eval = self.evaluate_board(side, new_state)
            score = material_gain * 10 + pos_bonus + board_eval * 0.1
        self._undo_move(board, undo_info)
        return score

    def move(self, side, board):
        my_moves = self.get_possible_moves(side, board)
        best_val = float('-inf'); best = []
        opp = 'white' if side == 'black' else 'black'
        for m in my_moves:
            val = self.evaluate_move(side, board, m)
            # opponent replies minimax
            if val > self.check_penalty:
                # simulate my move
                undo1 = self._make_move(board, m)
                replies = board.get_all_valid_moves(opp)
                worst = float('inf')
                for o in replies:
                    undo2 = self._make_move(board, o)
                    if self.is_in_check(side, board): leaf = self.check_penalty
                    else:
                        leaf_state = board.get_board_state()
                        leaf = self.evaluate_board(side, leaf_state)
                    self._undo_move(board, undo2)
                    worst = min(worst, leaf)
                val = min(val, worst)
                self._undo_move(board, undo1)
            if val > best_val:
                best_val = val; best = [m]
            elif val == best_val:
                best.append(m)
        return random.choice(best) if best else random.choice(my_moves)