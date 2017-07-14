from random import random, choice


def generate_random_fen(density):
    pieces = ['p', 'n', 'b', 'r', 'q', 'P', 'N', 'B', 'R', 'Q']
    rank_fens = []
    for rank in range(8):
        rank_fen = ''
        empty_count = 0
        for file in range(8):
            if random() > density:
                empty_count += 1
            else:
                if empty_count > 0:
                    rank_fen += str(empty_count)
                    empty_count = 0
                piece = choice(pieces)
                rank_fen += piece
        if empty_count > 0:
            rank_fen += str(empty_count)
        rank_fens.append(rank_fen)
    fen = '/'.join(rank_fens)

    if random() < .5:
        fen += ' w '
    else:
        fen += ' b '
    casteling_rights = ''
    if random() < .5:
        casteling_rights += 'K'
    if random() < .5:
        casteling_rights += 'Q'
    if random() < .5:
        casteling_rights += 'k'
    if random() < .5:
        casteling_rights += 'q'
    if casteling_rights == '':
        casteling_rights = '-'
    fen += casteling_rights

    fen += ' - 0 1'
    return fen

def simple_value_from_fen(fen):
    value = 0

    value += fen.count('P') * 1
    value += fen.count('N') * 3
    value += fen.count('B') * 3
    value += fen.count('R') * 5
    value += fen.count('Q') * 9

    value -= fen.count('p') * 1
    value -= fen.count('n') * 3
    value -= fen.count('b') * 3
    value -= fen.count('r') * 5
    value -= fen.count('q') * 9

    return value


def main():
    idx = 0
    while True:
        density = random()
        fen = generate_random_fen(density)
        value = simple_value_from_fen(fen)
        with open('simple_positions.fen', 'a') as f:
            f.write(fen + ',' + str(value) + '\n')
        idx += 1
        # print(fen)
        # print(value)
        print(idx)

if __name__ == "__main__":
    main()
