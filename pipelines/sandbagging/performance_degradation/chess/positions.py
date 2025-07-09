# sandbagging/chess/tasks.py
"""
Simplified chess sandbagging detection using pre-generated positions.
"""

from typing import List, Dict, Tuple
import chess
import chess.engine
import random


class ChessPositionGenerator:
    """Generates chess positions from Stockfish self-play."""

    def __init__(self, stockfish_path: str = "/opt/homebrew/bin/stockfish"):
        self.stockfish_path = stockfish_path
        self.positions = []

    def generate_positions(self, num_positions: int = 100, min_moves: int = 5, max_moves: int = 20) -> List[Dict]:
        """Generate chess positions from Stockfish self-play."""
        positions = []

        for _ in range(num_positions):
            board = chess.Board()

            # Play random number of moves
            num_moves = random.randint(min_moves, max_moves)
            # Make it even if it's odd
            if num_moves % 2 == 1:
                num_moves += 1

            with chess.engine.SimpleEngine.popen_uci(self.stockfish_path) as engine:
                for move_num in range(num_moves):
                    # Get Stockfish move with some randomness
                    result = engine.play(board, chess.engine.Limit(time=0.1))
                    board.push(result.move)

                    if board.is_game_over():
                        break

                # Skip if game is over
                if board.is_game_over():
                    continue

                # Analyze position to get evaluation
                info = engine.analyse(board, chess.engine.Limit(depth=7))

                # Get top 3 moves with evaluations
                analysis = engine.analyse(board, chess.engine.Limit(depth=7), multipv=10)

                best_moves = []
                if isinstance(analysis, list):
                    for line in analysis:
                        if "pv" in line and line["pv"]:
                            move = line["pv"][0]
                            score = line["score"].relative
                            cp = score.score() if not score.is_mate() else (10000 if score.mate() > 0 else -10000)
                            best_moves.append({
                                "move": move.uci(),
                                "centipawns": cp
                            })
                else:
                    # Single line analysis
                    if "pv" in analysis and analysis["pv"]:
                        move = analysis["pv"][0]
                        score = analysis["score"].relative
                        cp = score.score() if not score.is_mate() else (10000 if score.mate() > 0 else -10000)
                        best_moves.append({
                            "move": move.uci(),
                            "centipawns": cp
                        })

                position = {
                    "board": str(board),
                    "fen": board.fen(),
                    "legal_moves": [m.uci() for m in board.legal_moves],
                    "best_moves": best_moves,
                    "turn": "white" if board.turn else "black",
                    "move_number": board.fullmove_number
                }
                print(position)
                positions.append(position)

        self.positions = positions
        return positions