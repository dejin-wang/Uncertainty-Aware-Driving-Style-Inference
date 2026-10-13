import torch

def linear_bound(W, b, l, u):
    W_pos = torch.clamp(W, min=0)
    W_neg = torch.clamp(W, max=0)
    lower = W_pos @ l + W_neg @ u + b
    upper = W_pos @ u + W_neg @ l + b
    return lower, upper

def tanh_linear_relaxation(l, u):
    tanh_l, tanh_u = torch.tanh(l), torch.tanh(u)
    k = (tanh_u - tanh_l) / (u - l + 1e-12)
    k = torch.clamp(k, 0.0, 1.0)
    b_u = tanh_u - k * u
    b_l = tanh_l - k * l
    return k, b_l, b_u

def tanh_bound(l, u):
    k, b_l, b_u = tanh_linear_relaxation(l, u)
    l_new = k * l + b_l
    u_new = k * u + b_u
    return l_new, u_new

def crown_analyze_style(model, obs_center, style_center, eps_style):
    """
    Only style is perturbed: style in [style_center - eps_style, style_center + eps_style].
    obs_center is fixed.
    """
    device = next(model.parameters()).device
    obs_center = obs_center.to(device)
    style_center = style_center.to(device)

    # If eps_style is scalar, expand to vector
    if not torch.is_tensor(eps_style):
        eps_style = torch.tensor(eps_style, device=device, dtype=style_center.dtype)
    if eps_style.dim() == 0:
        eps_style = torch.full_like(style_center, eps_style)

    # ---- First two layers: obs only, no uncertainty ----
    l_obs = obs_center
    u_obs = obs_center

    W1, b1 = model.layer1.weight.data, model.layer1.bias.data
    l1, u1 = linear_bound(W1, b1, l_obs, u_obs)
    l1, u1 = tanh_bound(l1, u1)  # here l1 == u1, effectively exact

    W2, b2 = model.layer2.weight.data, model.layer2.bias.data
    l2, u2 = linear_bound(W2, b2, l1, u1)
    l2, u2 = tanh_bound(l2, u2)

    # ---- Style interval ----
    style_l = style_center - eps_style
    style_u = style_center + eps_style

    # Concatenate fixed feature (l2/u2) with uncertain style
    l2s = torch.cat([l2, style_l], dim=-1)
    u2s = torch.cat([u2, style_u], dim=-1)

    # Layer 3
    W3, b3 = model.layer3.weight.data, model.layer3.bias.data
    l3, u3 = linear_bound(W3, b3, l2s, u2s)
    l3, u3 = tanh_bound(l3, u3)

    # Layer 4
    W4, b4 = model.layer4.weight.data, model.layer4.bias.data
    l4, u4 = linear_bound(W4, b4, l3, u3)
    l4, u4 = tanh_bound(l4, u4)

    # Policy head
    Wp, bp = model.policy_head.weight.data, model.policy_head.bias.data
    l_out, u_out = linear_bound(Wp, bp, l4, u4)
    return l_out, u_out
