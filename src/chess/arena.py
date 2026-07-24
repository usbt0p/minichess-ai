"""
Functions to facilitate playing chess between agents and gathering statistics.
Includes helper functions for game status checking, game playing, and match playing.

Intended to be used as an API for other scripts to use, e.g. `agentVSagent.py` and `run_agent_tournament.py`.
"""

import pyffish
import numpy as np
import json
from src.chess.agents.base import ChessAgent

def get_repetition_count(fen_history: list[str]) -> int:
    """
    Computes the repetition count of the last FEN in the history.
    A state is defined by the board layout and the active player.
    Repetitions reset when a pawn move or capture occurs (halfmove clock == 0).
    """
    if not fen_history:
        return 0
        
    state_history = []
    for fen in fen_history:
        parts = fen.split(" ")
        board = parts[0]
        player = parts[1]
        halfmove = int(parts[4]) if len(parts) > 4 else 0
        
        # If a pawn move or capture occurs, previous positions can no longer be repeated
        if halfmove == 0:
            state_history = []
            
        state_history.append((board, player))
        
    current_state = state_history[-1]
    occurrences = state_history.count(current_state)
    return max(0, occurrences - 1)

def get_game_status_with_reason(start_fen: str, movelist: list, current_fen: str = None, current_repetition: int = None) -> tuple:
    """
    Determines if the game has ended and returns (ended, result, reason).
    result: "white", "black", "draw", or "ongoing"
    reason: "checkmate", "stalemate", "insufficient_material", "50_move_rule",
            "3_repetition_rule", "none"
    """
    if current_fen is None:
        # Backward compatibility mode
        current_fen = pyffish.get_fen("gardner", start_fen, movelist)
        
        # Reconstruct FEN history to calculate repetition count
        fen_history = [start_fen]
        temp_fen = start_fen
        for m in movelist:
            temp_fen = pyffish.get_fen("gardner", temp_fen, [m])
            fen_history.append(temp_fen)
        current_repetition = get_repetition_count(fen_history)
        
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
    insufficient = pyffish.has_insufficient_material("gardner", current_fen, [])
    if insufficient == (True, True): # if neither player has enough material to win
        return True, "draw", "insufficient_material"
        
    # check repetition or 50-move rule
    parts = current_fen.split(" ")
    halfmove_clock = int(parts[4]) if len(parts) > 4 else 0
    
    if halfmove_clock >= 100:
        return True, "draw", "50_move_rule"
    elif current_repetition is not None and current_repetition >= 2:
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
    
    current_fen = start_fen
    fen_history = [current_fen]
    
    parts = current_fen.split(" ")
    state_history = [(parts[0], parts[1])]
    
    while True:
        # Calculate current repetition count (occurrences in state_history minus 1)
        current_repetition = max(0, state_history.count(state_history[-1]) - 1)
        
        # Check current game status using optimized signature
        ended, result, reason = get_game_status_with_reason(
            start_fen, movelist, current_fen=current_fen, current_repetition=current_repetition
        )
        if ended:
            return result, reason, len(movelist), move_history, entropies_white, entropies_black
            
        # Hard limit to prevent infinite loops (max_moves treated as max half-moves)
        if len(movelist) >= max_moves:
            return "draw", "max_moves", len(movelist), move_history, entropies_white, entropies_black
            
        legal = pyffish.legal_moves("gardner", current_fen, [])
        
        # Select active agent
        parts = current_fen.split(" ")
        active_player = parts[1]
        
        if active_player == 'w':
            try:
                move, ent, top_6 = agent_white.select_move(current_fen, legal, temperature, repetition=current_repetition)
            except TypeError:
                move, ent, top_6 = agent_white.select_move(current_fen, legal, temperature)
            entropies_white.append(ent)
            move_history.append({"move": move, "player": "white", "entropy": ent, "top_6": top_6})
        else:
            try:
                move, ent, top_6 = agent_black.select_move(current_fen, legal, temperature, repetition=current_repetition)
            except TypeError:
                move, ent, top_6 = agent_black.select_move(current_fen, legal, temperature)
            entropies_black.append(ent)
            move_history.append({"move": move, "player": "black", "entropy": ent, "top_6": top_6})
            
        movelist.append(move)
        
        # Update FEN and repetition state incrementally
        current_fen = pyffish.get_fen("gardner", current_fen, [move])
        fen_history.append(current_fen)
        
        parts = current_fen.split(" ")
        board = parts[0]
        player = parts[1]
        halfmove = int(parts[4]) if len(parts) > 4 else 0
        
        if halfmove == 0:
            state_history = []
        state_history.append((board, player))

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
        "color_stats": color_stats,
        "avg_game_length": float(np.mean([g["num_moves"] for g in games_log])) if games_log else 0.0
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
