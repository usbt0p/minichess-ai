'''Tests for the chess rule checking. This serves as proof of correctness
and shows the usage of the python fairy stockfish bindings. 
I'ts also good to build understanding
'''

import sys
import os
import pytest
import pyffish
from src.chess.arena import get_game_status_with_reason

# Add project root to python path to import pyffish correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# these are fixtures. they are set up once for all tests and the
# result is passed as argument to the tests. this way we dont have
# to set up the pyffish engine for each test.
@pytest.fixture(autouse=True)
def setup_pyffish():
    pyffish.set_option("UCI_Variant", "gardner")

@pytest.fixture
def start_fen():
    return pyffish.start_fen("gardner")

def test_checkmate1():
    fen = "5/5/1k3/r1b2/Kr3 w - 0 1"
    ended, result, reason = get_game_status_with_reason(fen, [])
    assert ended is True
    assert result == "black"
    assert reason == "checkmate"

def test_checkmate2():
    fen = "2Q1k/p2pp/1p3/PP2P/RN2K b - - 0 7"
    ended, result, reason = get_game_status_with_reason(fen, [])
    assert ended is True
    assert result == "white"
    assert reason == "checkmate"

def test_stalemate_rook():
    fen = "5/5/2k2/1r3/K4 w - 0 1"
    ended, result, reason = get_game_status_with_reason(fen, [])
    assert ended is True
    assert result == "draw"
    assert reason == "stalemate"

def test_stalemate_knights():
    fen = "4k/1N3/2K2/3N1/5 b - - 10 5"
    ended, result, reason = get_game_status_with_reason(fen, [])
    assert ended is True
    assert result == "draw"
    assert reason == "stalemate"

def test_insufficient_material1():
    # King vs King: insufficient material.
    fen = "k4/5/5/5/4K w - 0 1"
    ended, result, reason = get_game_status_with_reason(fen, [])
    assert ended is True
    assert result == "draw"
    assert reason == "insufficient_material"

def test_insufficient_material2():
    # King vs King and Knight: insufficient material.
    fen = "k4/5/2n2/5/4K w - 0 1"
    ended, result, reason = get_game_status_with_reason(fen, [])
    assert ended is True
    assert result == "draw"
    assert reason == "insufficient_material"

def test_sufficient_material_black():
    # King vs King &Rook: sufficient material.
    fen = "5/5/1k3/2r2/K4 w - 0 1"
    white_insufficient, black_insufficient = pyffish.has_insufficient_material("gardner", fen, [])
    assert black_insufficient is False
    assert white_insufficient is True

def test_sufficient_material_white():
    # King vs King &Rook: sufficient material.
    fen = "5/5/1K3/2R2/k4 b - 0 1"
    white_insufficient, black_insufficient = pyffish.has_insufficient_material("gardner", fen, [])
    assert white_insufficient is False
    assert black_insufficient is True

def test_50_move_rule():

    fen_98 = "rnbqk/ppppp/5/PPPPP/RNBQK w - - 98 49"
    game_ended, result = pyffish.is_optional_game_end("gardner", fen_98, [])
    assert game_ended is False
    # this seems to be not very useful
    print(result)

    # advance one move
    fen_99 = "rnbqk/ppppp/5/PPPPP/RNBQK w - - 99 49"
    game_ended, result = pyffish.is_optional_game_end("gardner", fen_99, [])
    assert game_ended is False
    print(result)

    # advance another move, now it ends in draw
    fen_100 = "rnbqk/ppppp/5/PPPPP/RNBQK w - - 100 50"
    game_ended, result = pyffish.is_optional_game_end("gardner", fen_100, [])
    assert game_ended is True
    print(result)

def test_50_move_rule_with_status():

    fen_98 = "rnbqk/ppppp/5/PPPPP/RNBQK w - - 98 49"
    ended, result, reason = get_game_status_with_reason(fen_98, [])
    assert result == "ongoing"
    assert ended is False
    assert reason == "none"

    # advance one move
    fen_99 = "rnbqk/ppppp/5/PPPPP/RNBQK w - - 99 49"
    ended, result, reason = get_game_status_with_reason(fen_99, [])
    assert result == "ongoing"
    assert ended is False
    assert reason == "none"

    # Clock 100 is 50 full moves (100 half-moves) since last pawn move or capture
    fen_100 = "rnbqk/ppppp/5/PPPPP/RNBQK w - - 100 50"
    ended, result, reason = get_game_status_with_reason(fen_100, [])
    assert result == "draw"
    assert ended is True
    assert reason == "50_move_rule"

# this is good to understand the internals of pyffish repetition and state
def test_repetition_rule_without_status(start_fen):
    # move pawns to allow repetitions
    setup_moves = ["e2e3", "a4a3"]
    current_fen = pyffish.get_fen("gardner", start_fen, setup_moves)
    assert current_fen == "rnbqk/1pppp/p3P/PPPP1/RNBQK w - - 0 2"

    # first rep
    rep_cycle = ["e1e2", "a5a4", "e2e1", "a4a5"] 
    current_fen = pyffish.get_fen("gardner", current_fen, rep_cycle)
    assert current_fen == "rnbqk/1pppp/p3P/PPPP1/RNBQK w - - 4 4"

    # second repetition
    current_fen = pyffish.get_fen("gardner", current_fen, rep_cycle)
    assert current_fen == "rnbqk/1pppp/p3P/PPPP1/RNBQK w - - 8 6"

    # third repetition, now it should end
    current_fen = pyffish.get_fen("gardner", current_fen, rep_cycle)
    assert current_fen == "rnbqk/1pppp/p3P/PPPP1/RNBQK w - - 12 8"

    # Why does this return False?
    # since FEN is a stateless representation of a single position, it does not carry the history of previous moves.
    # Therefore, when we call is_optional_game_end with an empty move list [], the engine sees the position for the first time
    # and cannot detect any repetition. to verify repetition we need the whole move history
    ended_stateless, result_stateless = pyffish.is_optional_game_end(
        "gardner", current_fen, []
    )
    assert ended_stateless is False

    # The correct way to detect repetition without get_game_status is to pass the full history of moves
    # from the starting position (or any point before the repetitions occurred)
    full_history = setup_moves + rep_cycle * 3
    ended_with_history, result_with_history = pyffish.is_optional_game_end(
        "gardner", start_fen, full_history
    )
    assert ended_with_history is True
    assert result_with_history == 0

def test_repetition_rule_with_status(start_fen):
    # Repeat the sequence: b1c3, b5a3, c3b1, a3b5
    movelist = []
    moves = ["b1c3", "b5a3", "c3b1", "a3b5"]
    
    # At move count 0 to 7: ongoing
    for cycle in range(1, 3):
        for move in moves:
            movelist.append(move)
            ended, result, reason = get_game_status_with_reason(start_fen, movelist)
            if len(movelist) < 8:
                assert ended is False, f"Ended prematurely at move {len(movelist)}"
                assert result == "ongoing"
                assert reason == "none"
            else:
                assert ended is True, f"Should have ended by 3-fold repetition at move {len(movelist)}"
                assert result == "draw"
                assert reason == "3_repetition_rule"
                break
