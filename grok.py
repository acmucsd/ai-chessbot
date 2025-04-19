import random
import time

class Bot:
    def __init__(self):
        # Piece values tuned for StarChess
        self.piece_values = {
            'P': 1, 'N': 3, 'B': 3, 'R': 5, 'S': 4, 'Q': 9, 'J': 10, 'K': 10000
        }
        self.center_squares = [(2, 2), (2, 3), (3, 2), (3, 3)]
        self.transposition_table = {}
        self.time_limit = 0.095  # Slightly under 0.1s for safety
        self.start_time = 0
        self.max_depth = 10  # Limited by time

    def move(self, side, board):
        """Select the best move within the time limit."""
        self.start_time = time.time()
        self.side = side
        self.opp_side = 'white' if side == 'black' else 'black'
        best_move = None
        depth = 1
        try:
            while time.time() - self.start_time < self.time_limit and depth <= self.max_depth:
                score, move = self.iterative_deepening_search(board, depth)
                if move:
                    best_move = move
                depth += 1
        except TimeoutError:
            pass
        return best_move if best_move else random.choice(board.get_all_valid_moves(side))

    def iterative_deepening_search(self, board, depth):
        """Search to a specific depth, optimizing with alpha-beta."""
        best_move = None
        alpha = float('-inf')
        beta = float('inf')
        moves = self.get_ordered_moves(board, self.side)
        highest_score = float('-inf')
        for move in moves:
            undo_info = self._make_move(board, move)
            if self.is_king_capturable(self.side, board):
                score = float('-inf')
            else:
                score = -self.alpha_beta(board, depth - 1, -beta, -alpha, self.opp_side)
            self._undo_move(board, undo_info)
            if score > highest_score:
                highest_score = score
                best_move = move
            alpha = max(alpha, score)
            if time.time() - self.start_time > self.time_limit:
                raise TimeoutError
        return highest_score, best_move

    def alpha_beta(self, board, depth, alpha, beta, side):
        """Alpha-beta pruning search."""
        if time.time() - self.start_time > self.time_limit:
            raise TimeoutError
        board_hash = self.hash_board(board)
        if board_hash in self.transposition_table and self.transposition_table[board_hash]['depth'] >= depth:
            return self.transposition_table[board_hash]['score']
        if depth == 0:
            return self.quiescence_search(board, alpha, beta, side)
        moves = self.get_ordered_moves(board, side)
        if not moves:
            return -self.evaluate_board(board, side)
        max_score = float('-inf')
        for move in moves:
            undo_info = self._make_move(board, move)
            if self.is_king_capturable(side, board):
                self._undo_move(board, undo_info)
                continue
            score = -self.alpha_beta(board, depth - 1, -beta, -alpha, 'white' if side == 'black' else 'black')
            self._undo_move(board, undo_info)
            max_score = max(max_score, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        self.transposition_table[board_hash] = {'depth': depth, 'score': max_score}
        return max_score

    def quiescence_search(self, board, alpha, beta, side):
        """Evaluate captures to stabilize the search."""
        stand_pat = self.evaluate_board(board, side)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
        capture_moves = [move for move in board.get_all_valid_moves(side) if board.get_piece_from_pos(move[1]) is not None]
        for move in capture_moves:
            undo_info = self._make_move(board, move)
            score = -self.quiescence_search(board, -beta, -alpha, 'white' if side == 'black' else 'black')
            self._undo_move(board, undo_info)
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def evaluate_board(self, board, side):
        """Evaluate the board state with StarChess-specific heuristics."""
        state = board.get_board_state()
        score = 0
        for y in range(6):
            for x in range(6):
                piece = state[y][x]
                if piece:
                    piece_color, piece_type = piece[0], piece[1]
                    value = self.piece_values.get(piece_type, 0)
                    if piece_color == side[0]:
                        score += value
                        # Central control bonus
                        if (x, y) in self.center_squares:
                            score += 0.5
                        # Pawn advancement bonus (closer to promotion)
                        if piece_type == 'P':
                            dist_to_promote = y if side == 'black' else (5 - y)
                            score += (5 - dist_to_promote) * 0.2
                    else:
                        score -= value
                        if (x, y) in self.center_squares:
                            score -= 0.5
                        if piece_type == 'P':
                            dist_to_promote = (5 - y) if side == 'black' else y
                            score -= (5 - dist_to_promote) * 0.2
        # King safety: penalize exposure
        king_pos = self.find_king(side, state)
        if king_pos:
            threats = self.count_threats(king_pos, 'white' if side == 'black' else 'black', board)
            score -= threats * 2
        return score

    def get_ordered_moves(self, board, side):
        """Order moves to optimize pruning."""
        moves = board.get_all_valid_moves(side)
        return sorted(moves, key=lambda move: self.move_score(board, move), reverse=True)

    def move_score(self, board, move):
        """Score moves for ordering."""
        score = 0
        start, end = move
        piece = board.get_piece_from_pos(start)
        target = board.get_piece_from_pos(end)
        if target:
            score += self.piece_values.get(target.notation, 0)
        if piece.notation == 'P' and (end[1] == 0 or end[1] == 5):
            score += 10  # Promotion to Joker
        return score

    def hash_board(self, board):
        """Hash the board state for transposition table."""
        return str(board.get_board_state()) + board.turn

    def is_king_capturable(self, side, board):
        """Check if the King can be captured next move."""
        state = board.get_board_state()
        king_pos = self.find_king(side, state)
        if not king_pos:
            return True
        opp = 'white' if side == 'black' else 'black'
        for _, end in board.get_all_valid_moves(opp):
            if end == king_pos:
                return True
        return False

    def find_king(self, side, state):
        """Locate the King’s position."""
        king_symbol = side[0] + 'K'
        for y in range(6):
            for x in range(6):
                if state[y][x] == king_symbol:
                    return (x, y)
        return None

    def count_threats(self, pos, opp_side, board):
        """Count threats to a position."""
        threats = 0
        for _, end in board.get_all_valid_moves(opp_side):
            if end == pos:
                threats += 1
        return threats

    def _make_move(self, board, move):
        """Apply a move and return undo info."""
        start, end = move
        from_sq = board.get_square_from_pos(start)
        to_sq = board.get_square_from_pos(end)
        moved = from_sq.occupying_piece
        captured = to_sq.occupying_piece
        from_sq.occupying_piece = None
        to_sq.occupying_piece = moved
        moved.pos = end
        board.turn = 'white' if board.turn == 'black' else 'black'
        board.num_moves += 1
        return (start, end, moved, captured)

    def _undo_move(self, board, info):
        """Revert a move."""
        start, end, moved, captured = info
        from_sq = board.get_square_from_pos(start)
        to_sq = board.get_square_from_pos(end)
        to_sq.occupying_piece = captured
        from_sq.occupying_piece = moved
        moved.pos = start
        board.turn = 'white' if board.turn == 'black' else 'black'
        board.num_moves -= 1