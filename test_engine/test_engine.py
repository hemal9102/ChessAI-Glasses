from stockfish import Stockfish

stockfish = Stockfish(path="backend/stockfish/stockfish.exe", depth=15)
stockfish.set_position(["e2e4", "e7e5"])
print(stockfish.get_best_move())
