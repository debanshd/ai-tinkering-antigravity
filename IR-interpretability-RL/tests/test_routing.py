import torch
from src.models import BipolarSAE, LinearController, DEVICE

def test_sae_instantiation():
    """Ensure BipolarSAE initializes and propagates tensors without dimension mismatch."""
    sae = BipolarSAE(768, 4096).to(DEVICE)
    dummy_input = torch.randn(10, 768).to(DEVICE)
    x_reconstructed, f = sae(dummy_input)
    assert x_reconstructed.shape == (10, 768), "SAE output mismatch"
    assert f.shape == (10, 4096), "Latent dimension mismatch"

def test_controller_instantiation():
    """Ensure LinearController correctly operates on SAE outputs without Name/Type errors."""
    controller = LinearController(4096).to(DEVICE)
    dummy_features = torch.randn(5, 4096).to(DEVICE)
    action_means, _ = controller(dummy_features)
    assert action_means.shape == (5, 4096), "Controller action output mismatch"

def test_v8_routing_latency_logic():
    """
    Replicates the exact routing logic injected in V8 to ensure no runtime
    TypeError with the `state_enc` lambda function or NameError with the agent.
    """
    first_pass_enc = torch.randn(8, 768).to(DEVICE)
    dummy_sae = BipolarSAE(768, 4096).to(DEVICE)
    dummy_agent = LinearController(4096).to(DEVICE)
    
    # Simulating the exact V8 notebook fix inject lambda:
    state_enc = lambda *args: args[0]
    
    # 1. First Pass Fake Return
    f_prf = torch.randn(8, 768).to(DEVICE)
    
    # 2. SAE Encoding
    _, f_exp = dummy_sae(first_pass_enc)
    
    # 3. Policy Forward Pass
    state = state_enc(f_exp, f_prf)  # State should receive f_exp
    assert state.shape == f_exp.shape, "Lambda mock failed to return f_exp correctly."
    
    action_means, _ = dummy_agent(state)
    delta_M = torch.zeros_like(f_exp).to(DEVICE)
    
    # Simulating the torque masking
    mask = action_means > 0.0
    delta_M[mask] = action_means[mask]
    assert delta_M.shape == f_exp.shape
    
    # 4. Final vector summation
    new_query_embeddings = first_pass_enc - torch.matmul(delta_M, dummy_sae.dec.weight.T)
    assert new_query_embeddings.shape == (8, 768), "Routing combination failed."
