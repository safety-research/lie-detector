# sandbagging/chess/tools.py
"""
Chess-specific tools for the sandbagging detection module.
"""

from inspect_ai.tool import tool, Tool
from typing import Optional, List, Dict, Any
import bulletchess
import chess
import chess.engine
import json
import asyncio
from pathlib import Path


class ChessEnvironment:
    """Manages the chess game state and Stockfish engine."""

    def __init__(self, stockfish_path: Optional[str] = None):
        self.board = chess.Board()
        self.engine = None
        self.stockfish_path = stockfish_path or self._find_stockfish()
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

    async def initialize_engine(self, level: int = 1):
        """Initialize Stockfish engine."""
        if self.engine is None:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)

        # Set skill level (1-20)
        self.engine.configure({"Skill Level": level})

    async def close_engine(self):
        """Close the engine."""
        if self.engine:
            self.engine.quit()
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
        except:
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


# Global chess environment for current game
_chess_env: Optional[ChessEnvironment] = None


def get_chess_env() -> ChessEnvironment:
    """Get or create the chess environment."""
    global _chess_env
    if _chess_env is None:
        _chess_env = ChessEnvironment()
    return _chess_env



def get_board_state() -> str:
    """Get the current chess board state.

    Get the current state of the chess board.

    Returns:
        JSON string with board state including:
        - FEN notation
        - ASCII board representation
        - Whose turn it is
        - Legal moves
        - Game status
    """
    env = get_chess_env()

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
def make_chess_move():
    """Make a chess move."""

    async def execute(move: str) -> str:
        """
        Make a chess move on the board.

        Args:
            move: The move in UCI notation (e.g., "e2e4") or SAN notation (e.g., "e4")

        Returns:
            Result of the move attempt
        """
        env = get_chess_env()

        # Try UCI format first
        if env.make_move(move):
            return f"Move {move} played successfully."

        # Try SAN format
        try:
            move_obj = env.board.parse_san(move)
            if env.make_move(move_obj.uci()):
                return f"Move {move} played successfully."
        except:
            pass

        return f"Invalid move: {move}. Legal moves are: {', '.join(env.get_legal_moves())}"

    return execute


@tool
def analyze_position():
    """Analyze the current position."""

    async def execute(depth: int = 10) -> str:
        """
        Analyze the current chess position.

        Args:
            depth: Analysis depth (default 10)

        Returns:
            JSON string with position evaluation
        """
        env = get_chess_env()

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

    return execute


@tool
def get_move_history():
    """Get the game move history."""

    async def execute() -> str:
        """
        Get the history of moves played in the game.

        Returns:
            JSON string with move history
        """
        env = get_chess_env()

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

    return execute
