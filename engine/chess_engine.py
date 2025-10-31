from stockfish import Stockfish
import os
STOCKFISH_PATH = r"H:\Minor Project 2\ChessAI-Glasses\Backend\stockfish\stockfish.exe"
stockfish = Stockfish(path = STOCKFISH_PATH,
    depth = 15,
    parameters={
        "Threads":2,
        "Minimum Thinking Time": 30 
    })
def get_best_move(fen: str) -> dict:
    if not stockfish.is_fen_valid(fen):
        return{
            "error":"FEN is invalid",
            "fen": fen
        }

    stockfish.set_fen_position(fen)
    best_move = stockfish.get_best_move()
    eval_info = stockfish.get_evaluation()

    return{
        "fen": fen,
        "best_move": best_move,
        "evaluation": eval_info
    }