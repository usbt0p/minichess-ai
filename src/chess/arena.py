"""
Functions to facilitate playing chess between agents and gathering statistics.
Includes helper functions for game status checking, game playing, and match playing.

Intended to be used as an API for other scripts to use, e.g. `agentVSagent.py` and `run_agent_tournament.py`.
"""

import pyffish
import numpy as np
import json
from src.chess.agents.base import ChessAgent

def get_game_status_with_reason(start_fen: str, movelist: list) -> tuple:
    """
    Determines if the game has ended and returns (ended, result, reason).
    result: "white", "black", "draw", or "ongoing"
    reason: "checkmate", "stalemate", "insufficient_material", "50_move_rule",
            "3_repetition_rule", "none"
    """
    current_fen = pyffish.get_fen("gardner", start_fen, movelist)
    
    # check legal moves
    legal = pyffish.legal_moves("gardner", current_fen, [])
    if not legal:
        in_check = pyffish.gives_check("gardner", current_fen, [])
        active_player = current_fen.split(" ")[1]
        if in_check:
            # Checkmate: active player loses, opponent wins
            return True, "black" if active_player == 'w' else "white", "checkmate"
        else:
            # Stalemate: draw
            return True, "draw", "stalemate"
            
    # check insufficient material
    insufficient = pyffish.has_insufficient_material("gardner", start_fen, movelist)
    if insufficient == (True, True): # if neither player has enough material to win
        return True, "draw", "insufficient_material"
        
    # check repetition or 50-move rule
    parts = current_fen.split(" ")
    halfmove_clock = int(parts[4]) if len(parts) > 4 else 0
    
    opt_ended, _ = pyffish.is_optional_game_end("gardner", start_fen, movelist)
    if opt_ended:
        if halfmove_clock >= 100:
            return True, "draw", "50_move_rule"
        else:
            return True, "draw", "3_repetition_rule"
        
    return False, "ongoing", "none"

def get_game_status(start_fen: str, movelist: list) -> tuple:
    """Legacy helper for testing compatibility, returns (ended, result) only."""
    ended, result, _ = get_game_status_with_reason(start_fen, movelist)
    return ended, result

def play_game(agent_white: ChessAgent, agent_black: ChessAgent, max_moves=100, temperature=0.1):
    pyffish.set_option("UCI_Variant", "gardner")
    start_fen = pyffish.start_fen("gardner")
    
    movelist = []
    move_history = []
    entropies_white = []
    entropies_black = []
    
    while True:
        # Check current game status using full move history for repetition & 50-move rules
        ended, result, reason = get_game_status_with_reason(start_fen, movelist)
        if ended:
            return result, reason, len(movelist), move_history, entropies_white, entropies_black
            
        # Hard limit to prevent infinite loops (max_moves treated as max half-moves)
        if len(movelist) >= max_moves:
            return "draw", "max_moves", len(movelist), move_history, entropies_white, entropies_black
            
        current_fen = pyffish.get_fen("gardner", start_fen, movelist)
        legal = pyffish.legal_moves("gardner", current_fen, [])
        
        # Select active agent
        parts = current_fen.split(" ")
        active_player = parts[1]
        
        if active_player == 'w':
            move, ent, top_6 = agent_white.select_move(current_fen, legal, temperature)
            entropies_white.append(ent)
            move_history.append({"move": move, "player": "white", "entropy": ent, "top_6": top_6})
        else:
            move, ent, top_6 = agent_black.select_move(current_fen, legal, temperature)
            entropies_black.append(ent)
            move_history.append({"move": move, "player": "black", "entropy": ent, "top_6": top_6})
            
        movelist.append(move)

def play_matchup(agent1: ChessAgent, agent2: ChessAgent, num_games=20, max_moves=100, temperature=0.1, save_log=None):
    print(f"\n=== Matchup: {agent1.name} vs {agent2.name} ({num_games} games) ===")
    
    agent1_wins = 0
    agent2_wins = 0
    draws = 0
    
    entropies_agent1 = []
    entropies_agent2 = []
    
    reasons_count = {
        "checkmate": 0,
        "stalemate": 0,
        "insufficient_material": 0,
        "50_move_rule": 0,
        "3_repetition_rule": 0,
        "max_moves": 0
    }
    
    color_stats = {
        "agent1": {
            "white": {"wins": 0, "losses": 0, "draws": 0},
            "black": {"wins": 0, "losses": 0, "draws": 0}
        },
        "agent2": {
            "white": {"wins": 0, "losses": 0, "draws": 0},
            "black": {"wins": 0, "losses": 0, "draws": 0}
        }
    }
    
    games_log = []
    
    for game_idx in range(num_games):
        # Alternate colors
        if game_idx % 2 == 0:
            white, black = agent1, agent2
            agent1_color = "white"
        else:
            white, black = agent2, agent1
            agent1_color = "black"
            
        winner, reason, moves_len, movelist, ent_w, ent_b = play_game(white, black, max_moves, temperature)
        
        # Collect entropies
        if agent1_color == "white":
            entropies_agent1.extend(ent_w)
            entropies_agent2.extend(ent_b)
        else:
            entropies_agent1.extend(ent_b)
            entropies_agent2.extend(ent_w)
            
        # Update statistics
        reasons_count[reason] = reasons_count.get(reason, 0) + 1
        
        if winner == "draw":
            draws += 1
            result_str = "Draw"
            if agent1_color == "white":
                color_stats["agent1"]["white"]["draws"] += 1
                color_stats["agent2"]["black"]["draws"] += 1
            else:
                color_stats["agent1"]["black"]["draws"] += 1
                color_stats["agent2"]["white"]["draws"] += 1
        elif winner == agent1_color:
            agent1_wins += 1
            result_str = f"{agent1.name} won"
            if agent1_color == "white":
                color_stats["agent1"]["white"]["wins"] += 1
                color_stats["agent2"]["black"]["losses"] += 1
            else:
                color_stats["agent1"]["black"]["wins"] += 1
                color_stats["agent2"]["white"]["losses"] += 1
        else:
            agent2_wins += 1
            result_str = f"{agent2.name} won"
            if agent1_color == "white":
                color_stats["agent1"]["white"]["losses"] += 1
                color_stats["agent2"]["black"]["wins"] += 1
            else:
                color_stats["agent1"]["black"]["losses"] += 1
                color_stats["agent2"]["white"]["wins"] += 1
            
        game_detail = {
            "game_idx": game_idx + 1,
            "white": white.name,
            "black": black.name,
            "winner": winner,
            "reason": reason,
            "num_moves": moves_len,
            "moves": movelist,
            "entropies_white": [float(e) for e in ent_w],
            "entropies_black": [float(e) for e in ent_b]
        }
        games_log.append(game_detail)
            
        print(f"  Game {game_idx + 1:02d}: Winner: {result_str:15s} | Reason: {reason:22s} | Moves: {moves_len}")
        
    # Calculate win rates
    total_games = num_games
    win_rate1 = agent1_wins / total_games * 100
    win_rate2 = agent2_wins / total_games * 100
    draw_rate = draws / total_games * 100
    
    # Calculate Elo difference
    score1 = agent1_wins + 0.5 * draws
    p1 = score1 / total_games
    if p1 >= 0.99:
        elo_diff = 400
    elif p1 <= 0.01:
        elo_diff = -400
    else:
        elo_diff = -400 * np.log10((1 - p1) / p1)
        
    avg_entropy1 = np.mean(entropies_agent1) if entropies_agent1 else 0.0
    avg_entropy2 = np.mean(entropies_agent2) if entropies_agent2 else 0.0
    
    print("\n" + "="*50)
    print(" MATCHUP RESULTS ")
    print("="*50)
    print(f"{agent1.name:25s}: {agent1_wins} wins ({win_rate1:.1f}%)")
    print(f"{agent2.name:25s}: {agent2_wins} wins ({win_rate2:.1f}%)")
    print(f"Draws                    : {draws} ({draw_rate:.1f}%)")
    print("-" * 50)
    print("COLOR STATS BREAKDOWN:")
    w1 = color_stats["agent1"]["white"]
    b1 = color_stats["agent1"]["black"]
    print(f"  {agent1.name} (Agent 1):")
    print(f"    As White: {w1['wins']} wins, {w1['losses']} losses, {w1['draws']} draws")
    print(f"    As Black: {b1['wins']} wins, {b1['losses']} losses, {b1['draws']} draws")
    w2 = color_stats["agent2"]["white"]
    b2 = color_stats["agent2"]["black"]
    print(f"  {agent2.name} (Agent 2):")
    print(f"    As White: {w2['wins']} wins, {w2['losses']} losses, {w2['draws']} draws")
    print(f"    As Black: {b2['wins']} wins, {b2['losses']} losses, {b2['draws']} draws")
    print("-" * 50)
    print("Termination Reasons Breakdown:")
    for r, count in reasons_count.items():
        print(f"  {r:22s}: {count:3d} ({count / total_games * 100:.1f}%)")
    print("-" * 50)
    print(f"Approx Elo Difference (Model1 - Model2): {elo_diff:+.1f}")
    print(f"Average Policy Entropy ({agent1.name}): {avg_entropy1:.4f}")
    print(f"Average Policy Entropy ({agent2.name}): {avg_entropy2:.4f}")
    print("="*50 + "\n")
    
    results = {
        "agent1_wins": agent1_wins,
        "agent2_wins": agent2_wins,
        "draws": draws,
        "elo_diff": elo_diff,
        "avg_entropy1": avg_entropy1,
        "avg_entropy2": avg_entropy2,
        "reasons": reasons_count,
        "color_stats": color_stats
    }
    
    if save_log:
        log_data = {
            "agent1": {
                "name": agent1.name,
                "type": agent1.__class__.__name__,
            },
            "agent2": {
                "name": agent2.name,
                "type": agent2.__class__.__name__,
            },
            "summary": {
                "total_games": total_games,
                "agent1_wins": agent1_wins,
                "agent2_wins": agent2_wins,
                "draws": draws,
                "elo_diff": float(elo_diff),
                "avg_entropy1": float(avg_entropy1),
                "avg_entropy2": float(avg_entropy2),
                "reasons": reasons_count,
                "color_stats": color_stats
            },
            "games": games_log
        }
        with open(save_log, "w") as f:
            json.dump(log_data, f, indent=2)
        print(f"[INFO] Detailed matchup log saved to {save_log}")
        
    return results
