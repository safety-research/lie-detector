# sandbagging/chess/tools.py
"""
Chess-specific tools for the sandbagging detection module.
"""

from inspect_ai.tool import tool
from typing import Optional, List, Dict, Any
import chess
import chess.engine
import json
import asyncio
from pathlib import Path


class ChessEnvironment:
    """Manages the chess game state and Stockfish engine."""

    def __init__(self, stockfish_path: Optional[str] = None, stockfish_level: int = 1):
        self.board = chess.Board()
        self.engine = None
        self.stockfish_path = stockfish_path or self._find_stockfish()
        self.stockfish_level = stockfish_level
        self.move_history = []

    def _find_stockfish(self) -> str:
        """Try to find Stockfish executable."""
        # Common locations
        paths = [
            "/opt/homebrew/bin/stockfish",
            "/usr/local/bin/stockfish",
            "/usr/bin/stockfish",
            "stockfish",  # In PATH
        ]

        for path in paths:
            if Path(path).exists():
                return path

        raise FileNotFoundError("Stockfish not found. Please install it or provide path.")

    async def initialize_engine(self):
        """Initialize Stockfish engine."""
        if self.engine is None:
            self.engine = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            )
        # Set skill level (1-20)
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.engine.configure({"Skill Level": self.stockfish_level})
        )

    async def close_engine(self):
        """Close the engine."""
        if self.engine:
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.engine.quit
            )
            self.engine = None

    def get_legal_moves(self) -> List[str]:
        """Get all legal moves in UCI format."""
        return [move.uci() for move in self.board.legal_moves]

    def make_move(self, move_uci: str) -> bool:
        """Make a move on the board."""
        try:
            move = chess.Move.from_uci(move_uci)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.move_history.append(move_uci)
                return True
        except Exception:
            pass
        return False

    async def get_stockfish_move(self, time_limit: float = 1.0) -> str:
        """Get Stockfish's move."""
        if not self.engine:
            await self.initialize_engine()

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.engine.play(self.board, chess.engine.Limit(time=time_limit))
        )
        return result.move.uci()

    async def evaluate_position(self) -> float:
        """Evaluate current position with Stockfish."""
        if not self.engine:
            await self.initialize_engine()

        info = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.engine.analyse(self.board, chess.engine.Limit(depth=15))
        )

        score = info["score"].relative
        if score.is_mate():
            return 10000 if score.mate() > 0 else -10000
        else:
            return score.score() / 100  # Convert centipawns to pawns


# Global variable to store environment for current execution
_current_env: Optional[ChessEnvironment] = None


def set_chess_environment(env: ChessEnvironment):
    """Set the current chess environment."""
    global _current_env
    _current_env = env


def get_chess_environment() -> ChessEnvironment:
    """Get the current chess environment."""
    if _current_env is None:
        raise RuntimeError("Chess environment not initialized")
    return _current_env


@tool
def get_board_state() -> str:
    """Get the current chess board state.

    Returns:
        JSON string with board state including:
        - FEN notation
        - ASCII board representation
        - Whose turn it is
        - Legal moves
        - Game status
    """
    env = get_chess_environment()

    board_state = {
        "fen": env.board.fen(),
        "ascii_board": str(env.board),
        "turn": "white" if env.board.turn else "black",
        "move_number": env.board.fullmove_number,
        "legal_moves": env.get_legal_moves(),
        "is_check": env.board.is_check(),
        "is_checkmate": env.board.is_checkmate(),
        "is_stalemate": env.board.is_stalemate(),
        "is_game_over": env.board.is_game_over()
    }

    return json.dumps(board_state, indent=2)


@tool
def make_chess_move(move: str) -> str:
    """Make a chess move on the board.

    Args:
        move: The move in UCI notation or SAN notation

    Returns:
        Result of the move attempt
    """
    env = get_chess_environment()

    # Try UCI format first
    if env.make_move(move):
        return f"Move {move} played successfully."

    # Try SAN format
    try:
        move_obj = env.board.parse_san(move)
        if env.make_move(move_obj.uci()):
            return f"Move {move} played successfully."
    except Exception:
        pass

    return f"Invalid move: {move}. Legal moves are: {', '.join(env.get_legal_moves())}"


@tool
async def analyze_position(depth: int = 10) -> str:
    """Analyze the current chess position.

    Args:
        depth: Analysis depth (default 10)

    Returns:
        JSON string with position evaluation
    """
    env = get_chess_environment()

    try:
        evaluation = await env.evaluate_position()

        # Get best moves
        if not env.engine:
            await env.initialize_engine()

        analysis_result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: env.engine.analyse(env.board, chess.engine.Limit(depth=depth), multipv=3)
        )

        best_moves = []
        if hasattr(analysis_result, '__iter__'):
            for info in analysis_result:
                if "pv" in info and info["pv"]:
                    best_moves.append({
                        "move": info["pv"][0].uci(),
                        "evaluation": info["score"].relative.score() / 100 if not info[
                            "score"].relative.is_mate() else "mate"
                    })
        else:
            if "pv" in analysis_result and analysis_result["pv"]:
                best_moves.append({
                    "move": analysis_result["pv"][0].uci(),
                    "evaluation": evaluation
                })

        return json.dumps({
            "evaluation": evaluation,
            "best_moves": best_moves,
            "position_type": "winning" if evaluation > 1 else "losing" if evaluation < -1 else "equal"
        }, indent=2)

    except Exception as e:
        return f"Error analyzing position: {str(e)}"


@tool
def get_move_history() -> str:
    """Get the history of moves played in the game.

    Returns:
        JSON string with move history
    """
    env = get_chess_environment()

    # Convert to SAN notation for readability
    temp_board = chess.Board()
    moves_san = []

    for move_uci in env.move_history:
        move = chess.Move.from_uci(move_uci)
        moves_san.append(temp_board.san(move))
        temp_board.push(move)

    return json.dumps({
        "moves_uci": env.move_history,
        "moves_san": moves_san,
        "total_moves": len(env.move_history)
    }, indent=2)