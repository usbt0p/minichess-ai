import pytest
import torch
import pyffish
import time
from src.chess.env import MinichessEnv, batch_parse_fens
from src.models.dataset_parser import uci_to_index, index_to_uci
from src.chess.arena import get_game_status_with_reason

@pytest.fixture(autouse=True)
def setup_pyffish():
    pyffish.set_option("UCI_Variant", "gardner")

def test_index_to_uci_bijection():
    """Test that index_to_uci and uci_to_index form a mathematical bijection for all 704 actions."""
    for i in range(704):
        move_str = index_to_uci(i)
        reconstructed_i = uci_to_index(move_str)
        assert reconstructed_i == i, f"Bijection failed at index {i}: got move {move_str} reconstructing to {reconstructed_i}"

def test_minichess_env_reset():
    """Test that reset initializes the state correctly."""
    env = MinichessEnv()
    start_fen = env.reset()
    assert start_fen == pyffish.start_fen("gardner")
    assert env.movelist == []
    assert len(env.state_history) == 1

def test_minichess_env_step_legal():
    """Test making legal steps in the environment."""
    env = MinichessEnv()
    legal_moves = env.get_legal_moves()
    assert len(legal_moves) > 0
    
    first_move = legal_moves[0]
    next_fen, reward, ended = env.step(first_move)
    
    assert next_fen == env.current_fen
    assert len(env.movelist) == 1
    assert env.movelist[0] == first_move
    assert ended is False
    assert reward == 0.0

def test_minichess_env_repetition():
    """Test that 3-fold repetition results in a draw."""
    env = MinichessEnv()
    # To cause repetition, we make setup moves that don't reset clocks (no pawn move or capture)
    moves = ["b1c3", "b5a3", "c3b1", "a3b5", "b1c3", "b5a3", "c3b1", "a3b5"]
    
    for i, m in enumerate(moves[:-1]):
        _, reward, ended = env.step(m)
        assert ended is False, f"Game ended prematurely at step {i} with move {m}"
        assert reward == 0.0
        
    # The last move triggers 3-fold repetition
    next_fen, reward, ended = env.step(moves[-1])
    assert ended is True
    assert reward == 0.0 # Draw reward is 0.0

def test_batch_parse_fens():
    """Test batch_parse_fens output shapes and content."""
    env = MinichessEnv()
    fens = [env.reset() for _ in range(5)]
    
    # Simple parse with no repetitions provided
    features = batch_parse_fens(fens, device="cpu")
    assert features.shape == (5, 28)
    
    # Specific assertions about Gardner starting FEN features
    # Check active player (white = 1) is at index 27
    assert (features[:, 27] == 1).all()
    # Check halfmove clock is at index 26
    assert (features[:, 26] == 0).all()
    # Check repetition count is at index 25
    assert (features[:, 25] == 0).all()

    # Pass custom repetitions list
    reps = [0, 1, 2, 0, 1]
    features_reps = batch_parse_fens(fens, repetitions=reps, device="cpu")
    assert torch.equal(features_reps[:, 25], torch.tensor(reps, dtype=torch.long))

def test_vectorized_environments():
    """Test synchronous vectorized loop with auto-reset."""
    num_envs = 8
    envs = [MinichessEnv() for _ in range(num_envs)]
    current_fens = [env.reset() for env in envs]
    
    # Run a simulated vectorized loop for 15 steps
    for step in range(15):
        # 1. Batched FEN parsing to features
        features = batch_parse_fens(current_fens, device="cpu")
        assert features.shape == (num_envs, 28)
        
        # 2. Step all environments
        for i, env in enumerate(envs):
            legal_moves = env.get_legal_moves()
            if not legal_moves:
                current_fens[i] = env.reset()
                continue
                
            # Pick first legal move
            move = legal_moves[0]
            next_fen, reward, ended = env.step(move)
            
            if ended:
                # in this case, we would store the game result in a buffer to later forwarding 
                current_fens[i] = env.reset() 
            else:
                current_fens[i] = next_fen
                
    # Verify all environments are stateful and independent
    for i, env in enumerate(envs):
        assert len(env.movelist) > 0
        assert env.current_fen == current_fens[i]

def test_throughput_benchmark_cpu():
    """Profile throughput of vectorized environment list looping."""
    num_envs = 64
    steps = 50
    envs = [MinichessEnv() for _ in range(num_envs)]
    current_fens = [env.reset() for env in envs]
    
    start_time = time.time()
    total_transitions = 0
    
    for _ in range(steps):
        features = batch_parse_fens(current_fens, device="cpu")
        
        for i, env in enumerate(envs):
            legal_moves = env.get_legal_moves()
            if not legal_moves:
                current_fens[i] = env.reset()
                continue
                
            # this is an oversimplification, the real one requires us to use policy to select moves, 
            # which in turn requires us to send stuff to the gpu and forward...
            move = legal_moves[0] 
            next_fen, reward, ended = env.step(move)
            
            if ended:
                current_fens[i] = env.reset()
            else:
                current_fens[i] = next_fen
                
            total_transitions += 1
            
    elapsed = time.time() - start_time
    throughput = total_transitions / elapsed
    print(f"\nGenerated {total_transitions} steps in {elapsed:.2f}s ({throughput:.1f} steps/sec)")
    assert throughput > 100.0, "Throughput is too slow" # aim for 100 transitions per second

def test_throughput_benchmark_gpu():
    """Profile throughput of vectorized environment list looping using actual model forward passes on GPU/device."""
    num_envs = 64 # higher gives slightly better troughput, but not insane amounts
    steps = 100
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[BENCHMARK] Running GPU benchmark on device: {device}")
    
    from src.models.transformerEncoder import MiniChessTransformerEncoder, EncoderConfig
    from src.chess.agents.transformer import TransformerAgent

    config = {
        "embed_dim": 64,
        "num_blocks": 3,
        "representation": "spatial",
        "factorized_policy": False
    }
    
    model_path = "/home/usbt0p/TFG/experiments/test_value_refactor/20260620_013832_d4_val_trnsf_dk64_n3_value_refactor_spatial_nofact_dk64_depth3_lr3.00e-03_bs512/best_model.pth"
    agent = TransformerAgent(model_path, config, device=device)
    
    envs = [MinichessEnv() for _ in range(num_envs)]
    current_fens = [env.reset() for env in envs]
    current_repetitions = [0] * num_envs
    
    start_time = time.time()
    total_transitions = 0
    
    for _ in range(steps):
        # 1. Batched FEN parsing to device
        features = batch_parse_fens(current_fens, repetitions=current_repetitions, device=device)
        
        # 2. Batched model inference
        with torch.no_grad():
            outputs = agent.model(features)
            if len(outputs) == 5:
                policy_logits, value_pred, _, _, _ = outputs
            else:
                policy_logits, value_pred = outputs
                
        # 3. Step all environments
        for i, env in enumerate(envs):
            legal_moves = env.get_legal_moves()
            if not legal_moves:
                current_fens[i] = env.reset()
                current_repetitions[i] = 0
                continue
                
            legal_indices = [uci_to_index(m) for m in legal_moves]
            logits = policy_logits[i][legal_indices]
            
            probs = torch.softmax(logits, dim=-1)
            sampled_idx = torch.multinomial(probs, 1).item()
            best_move = legal_moves[sampled_idx]
            
            next_fen, reward, ended = env.step(best_move)
            
            if ended:
                current_fens[i] = env.reset()
                current_repetitions[i] = 0
            else:
                current_fens[i] = next_fen
                current_repetitions[i] = env._get_repetition_count()
                
            total_transitions += 1
            
    elapsed = time.time() - start_time
    throughput = total_transitions / elapsed
    print(f"\n[BENCHMARK] GPU Vectorized Throughput: {total_transitions} steps in {elapsed:.2f}s ({throughput:.1f} steps/sec)")
    assert throughput > 10.0, "Throughput with GPU models is too slow"

    