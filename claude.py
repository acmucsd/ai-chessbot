import random
import time

class Bot:
    """
    Advanced chess bot featuring:
    - Iterative deepening search
    - Alpha-beta pruning
    - Transposition table
    - Enhanced evaluation function
    - Time management
    - Move ordering optimization

    Optimized for performance to stay within 0.1s time constraint.
    """
    def __init__(self):
        # Piece values
        self.piece_values = {
            'P': 100, 'N': 320, 'B': 330, 'R': 500,
            'S': 520, 'Q': 900, 'J': 950, 'K': 10000
        }

        # Position bonuses for each piece type
        self.position_tables = {
            'P': [  # Pawn positional table
                [0,  0,  0,  0,  0,  0],
                [50, 50, 50, 50, 50, 50],
                [10, 20, 30, 30, 20, 10],
                [5,  5, 10, 20, 20,  5],
                [0,  0,  0, 10, 10,  0],
                [0,  0,  0,  0,  0,  0]
            ],
            'N': [  # Knight positional table
                [-50, -40, -30, -30, -40, -50],
                [-40, -20,   0,   0, -20, -40],
                [-30,   0,  15,  15,   0, -30],
                [-30,   5,  15,  15,   5, -30],
                [-40, -20,   0,   0, -20, -40],
                [-50, -40, -30, -30, -40, -50]
            ],
            'B': [  # Bishop positional table
                [-20, -10, -10, -10, -10, -20],
                [-10,   0,   0,   0,   0, -10],
                [-10,   0,  10,  10,   0, -10],
                [-10,   5,  10,  10,   5, -10],
                [-10,   0,   5,   5,   0, -10],
                [-20, -10, -10, -10, -10, -20]
            ],
            'R': [  # Rook positional table
                [0,  0,  0,  0,  0,  0],
                [5, 10, 10, 10, 10,  5],
                [-5,  0,  0,  0,  0, -5],
                [-5,  0,  0,  0,  0, -5],
                [-5,  0,  0,  0,  0, -5],
                [0,  0,  5,  5,  0,  0]
            ],
            'Q': [  # Queen positional table
                [-20, -10, -10, -5, -10, -20],
                [-10,   0,   0,  0,   0, -10],
                [-10,   0,   5,  5,   0, -10],
                [ -5,   0,   5,  5,   0,  -5],
                [-10,   0,   0,  0,   0, -10],
                [-20, -10, -10, -5, -10, -20]
            ],
            'K': [  # King positional table - early/mid game
                [-30, -40, -40, -50, -40, -30],
                [-30, -40, -40, -50, -40, -30],
                [-30, -40, -40, -50, -40, -30],
                [-30, -40, -40, -50, -40, -30],
                [-20, -30, -30, -40, -30, -20],
                [-10, -20, -20, -20, -20, -10]
            ],
            'S': [  # Star positional table
                [-10, -10, -10, -10, -10, -10],
                [-10,   0,   0,   0,   0, -10],
                [-10,   0,  10,  10,   0, -10],
                [-10,   0,  10,  10,   0, -10],
                [-10,   0,   0,   0,   0, -10],
                [-10, -10, -10, -10, -10, -10]
            ],
            'J': [  # Joker positional table
                [-10, -5, -5, -5, -5, -10],
                [ -5,  0,  5,  5,  0,  -5],
                [ -5,  5, 10, 10,  5,  -5],
                [ -5,  5, 10, 10,  5,  -5],
                [ -5,  0,  5,  5,  0,  -5],
                [-10, -5, -5, -5, -5, -10]
            ]
        }

        # Transposition table
        self.tt_table = {}

        # Killer moves - stores good moves that caused beta cutoffs
        self.killer_moves = [None, None]

        # Time management
        self.move_time_limit = 0.095  # 95ms to be safe
        self.start_time = 0

        # Search depth parameters
        self.max_depth = 4  # Will be limited by time
        self.quiescence_depth = 2

    def move(self, side, board):
        """Main method to select the best move within time constraints"""
        self.start_time = time.time()
        self.tt_table = {}  # Clear transposition table at the start of a new move
        self.side = side
        self.opp_side = 'white' if side == 'black' else 'black'

        # Get all possible moves
        possible_moves = self.get_possible_moves(side, board)
        if not possible_moves:
            return random.choice(board.get_all_valid_moves(side))

        # For safety, always have a move ready
        best_move = random.choice(possible_moves)

        # Iterative deepening
        for depth in range(1, self.max_depth + 1):
            if time.time() - self.start_time > self.move_time_limit * 0.5:
                break  # Stop if we've used half our time budget

            try:
                current_best = self.iterative_deepening_search(board, depth, side)
                if current_best:
                    best_move = current_best
            except TimeoutError:
                break

        return best_move

    def iterative_deepening_search(self, board, depth, side):
        """Perform an iterative deepening search with alpha-beta pruning"""
        best_move = None
        alpha = float('-inf')
        beta = float('inf')

        # Order moves first for better pruning
        moves = self.get_ordered_moves(side, board)

        highest_score = float('-inf')

        for move in moves:
            # Check time frequently to avoid timeout
            if time.time() - self.start_time > self.move_time_limit:
                raise TimeoutError

            # Make move
            undo_info = self._make_move(board, move)

            # Check if we're in check after our move
            if self.is_in_check(side, board):
                score = float('-inf')  # Very bad move
            else:
                # Negamax call with alpha-beta pruning
                score = -self.negamax(board, depth-1, -beta, -alpha, self.opp_side)

            # Undo move
            self._undo_move(board, undo_info)

            if score > highest_score:
                highest_score = score
                best_move = move

            alpha = max(alpha, score)

        return best_move

    def negamax(self, board, depth, alpha, beta, side):
        """Negamax algorithm with alpha-beta pruning"""
        # Check time frequently
        if time.time() - self.start_time > self.move_time_limit:
            raise TimeoutError

        # Check for transposition table hit
        board_hash = self.hash_board(board)
        if board_hash in self.tt_table and self.tt_table[board_hash]['depth'] >= depth:
            return self.tt_table[board_hash]['score']

        # Base case: depth reached or game over
        if depth == 0:
            return self.quiescence_search(board, alpha, beta, side, self.quiescence_depth)

        opp = 'white' if side == 'black' else 'black'

        # Check if king is captured (game over)
        if self.is_king_captured(side, board):
            return float('-inf')
        if self.is_king_captured(opp, board):
            return float('inf')

        # Get and order moves
        moves = self.get_ordered_moves(side, board)
        if not moves:
            return -self.evaluate_position(opp, board)  # No moves, use opponent perspective

        max_score = float('-inf')

        for move in moves:
            undo_info = self._make_move(board, move)

            # Skip moves that leave us in check
            if self.is_in_check(side, board):
                self._undo_move(board, undo_info)
                continue

            score = -self.negamax(board, depth-1, -beta, -alpha, opp)
            self._undo_move(board, undo_info)

            if score > max_score:
                max_score = score

            alpha = max(alpha, score)
            if alpha >= beta:
                # Store killer move
                if not self.is_capture(board, move):
                    self.killer_moves[1] = self.killer_moves[0]
                    self.killer_moves[0] = move
                break

        # Store result in transposition table
        self.tt_table[board_hash] = {'depth': depth, 'score': max_score}
        return max_score

    def quiescence_search(self, board, alpha, beta, side, depth):
        """Quiescence search to avoid horizon effect"""
        # Basic evaluation first
        stand_pat = self.evaluate_position(side, board)

        if depth == 0:
            return stand_pat

        if stand_pat >= beta:
            return beta

        if alpha < stand_pat:
            alpha = stand_pat

        # Only look at capture moves
        capture_moves = self.get_capture_moves(side, board)

        for move in capture_moves:
            undo_info = self._make_move(board, move)

            opp = 'white' if side == 'black' else 'black'
            score = -self.quiescence_search(board, -beta, -alpha, opp, depth-1)

            self._undo_move(board, undo_info)

            if score >= beta:
                return beta

            if score > alpha:
                alpha = score

        return alpha

    def get_ordered_moves(self, side, board):
        """Order moves to optimize alpha-beta pruning"""
        moves = self.get_possible_moves(side, board)
        scored_moves = []

        for move in moves:
            score = 0
            start_pos, end_pos = move

            # Check if it's a killer move
            if move == self.killer_moves[0]:
                score += 900000
            elif move == self.killer_moves[1]:
                score += 800000

            # Prioritize captures
            end_piece = board.get_piece_from_pos(end_pos)
            if end_piece:
                # MVV-LVA (Most Valuable Victim - Least Valuable Aggressor)
                piece = board.get_piece_from_pos(start_pos)
                victim_value = self.piece_values.get(end_piece.notation, 0)
                aggressor_value = self.piece_values.get(piece.notation, 100)
                score += 1000000 + (victim_value * 100) - aggressor_value

            # Promotion is good
            piece = board.get_piece_from_pos(start_pos)
            if piece and piece.notation == 'P':
                # Check if it's a promotion move (pawn reaches the end)
                if (side == 'white' and end_pos[1] == 0) or (side == 'black' and end_pos[1] == 5):
                    score += 700000

            # Position improvement
            if piece:
                # Basic positional improvement check
                start_x, start_y = start_pos
                end_x, end_y = end_pos

                # Flip coordinates for black pieces to match position tables
                if side == 'black':
                    start_y = 5 - start_y
                    end_y = 5 - end_y

                try:
                    pos_start = self.position_tables.get(piece.notation, [[0]*6]*6)[start_y][start_x]
                    pos_end = self.position_tables.get(piece.notation, [[0]*6]*6)[end_y][end_x]
                    score += (pos_end - pos_start) * 10
                except (IndexError, TypeError):
                    pass

            scored_moves.append((score, move))

        # Sort by score in descending order
        scored_moves.sort(reverse=True)
        return [move for _, move in scored_moves]

    def get_capture_moves(self, side, board):
        """Get only capturing moves for quiescence search"""
        all_moves = self.get_possible_moves(side, board)
        capture_moves = []

        for move in all_moves:
            _, end_pos = move
            if board.get_piece_from_pos(end_pos) is not None:
                capture_moves.append(move)

        return capture_moves

    def is_capture(self, board, move):
        """Check if a move is a capture"""
        _, end_pos = move
        return board.get_piece_from_pos(end_pos) is not None

    def evaluate_position(self, side, board):
        """Evaluate the current board position"""
        state = board.get_board_state()

        my_side_char = side[0]  # 'w' or 'b'
        opp_side_char = 'b' if my_side_char == 'w' else 'w'

        # Material and positional score
        material_score = 0
        position_score = 0

        # Check if kings are present
        my_king_present = False
        opp_king_present = False

        for y in range(6):
            for x in range(6):
                piece = state[y][x]
                if not piece:
                    continue

                piece_type = piece[1]
                piece_color = piece[0]

                # Track kings
                if piece_type == 'K':
                    if piece_color == my_side_char:
                        my_king_present = True
                    else:
                        opp_king_present = True

                # Calculate material value
                piece_value = self.piece_values.get(piece_type, 0)

                # Adjust position evaluation for the perspective
                pos_y = y
                if piece_color == 'b':  # If black, flip the position
                    pos_y = 5 - y

                # Get position value
                pos_value = 0
                if piece_type in self.position_tables:
                    try:
                        pos_value = self.position_tables[piece_type][pos_y][x]
                    except IndexError:
                        pass

                # Add to total score depending on piece color
                if piece_color == my_side_char:
                    material_score += piece_value
                    position_score += pos_value
                else:
                    material_score -= piece_value
                    position_score -= pos_value

        # Check for winning/losing positions
        if not opp_king_present:
            return float('inf')  # Win
        if not my_king_present:
            return float('-inf')  # Loss

        # Combine scores with different weights
        total_score = (material_score * 1.0) + (position_score * 0.1)

        return total_score

    def hash_board(self, board):
        """Create a simple hash of the board state for the transposition table"""
        state = str(board.get_board_state())
        return hash(state + board.turn)

    def get_possible_moves(self, side, board):
        """Get all valid moves for the given side"""
        return board.get_all_valid_moves(side)

    def is_king_captured(self, side, board):
        """Check if the king of the given side is captured"""
        state = board.get_board_state()
        king_symbol = side[0] + 'K'  # 'wK' or 'bK'

        for row in state:
            if king_symbol in row:
                return False
        return True

    def is_in_check(self, side, board):
        """Check if the side is in check"""
        # Find the king
        king_pos = None
        state = board.get_board_state()
        king_symbol = side[0] + 'K'  # 'wK' or 'bK'

        for y in range(6):
            for x in range(6):
                if state[y][x] == king_symbol:
                    king_pos = (x, y)
                    break
            if king_pos:
                break

        if not king_pos:
            return True  # King is captured

        # Check if any opponent's move can capture the king
        opp_side = 'white' if side == 'black' else 'black'
        for start_pos, end_pos in board.get_all_valid_moves(opp_side):
            if end_pos == king_pos:
                return True

        return False

    def _make_move(self, board, move):
        """Apply a move and return undo information"""
        start, end = move
        from_sq = board.get_square_from_pos(start)
        to_sq = board.get_square_from_pos(end)
        moved = from_sq.occupying_piece
        captured = to_sq.occupying_piece

        # Perform the move
        from_sq.occupying_piece = None
        to_sq.occupying_piece = moved
        moved.pos = end

        # Toggle turn
        board.turn = 'white' if board.turn == 'black' else 'black'
        board.num_moves += 1

        return (start, end, moved, captured)

    def _undo_move(self, board, info):
        """Undo a move using the provided undo information"""
        start, end, moved, captured = info
        from_sq = board.get_square_from_pos(start)
        to_sq = board.get_square_from_pos(end)

        # Revert the move
        to_sq.occupying_piece = captured
        from_sq.occupying_piece = moved
        moved.pos = start

        # Toggle turn back
        board.turn = 'white' if board.turn == 'black' else 'black'
        board.num_moves -= 1