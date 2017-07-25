from chess_env import Chess
import chess.uci


def main():
    engine = chess.uci.popen_engine("/Users/adam/Documents/Stockfish/src/stockfish")
    info_handler = chess.uci.InfoHandler()
    engine.info_handlers.append(info_handler)

    env = Chess(load_pgn=True, random_position=True)
    idx = 0
    while True:
        print(idx)
        env.reset()
        fen = env.board.fen()
        engine.position(env.board)
        engine.go(movetime=100)
        try:
            score = info_handler.info["score"][1].cp / 100.0
            with open('stockfish_positions.fen', 'a') as f:
                f.write(fen + ',' + str(score) + '\n')
            idx += 1
        except TypeError:
            print(env.board)
            print(fen)

if __name__ == "__main__":
    main()
