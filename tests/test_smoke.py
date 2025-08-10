import torch
from abcd_clsloss import CLsLoss

def test_forward_backward_smoke():
    torch.manual_seed(0)

    # toy models
    class Toy(nn.Module):
        def __init__(self): super().__init__(); self.lin = nn.Linear(4,1)
        def forward(self, x): return self.lin(x)

    import torch.nn as nn
    m1, m2 = Toy(), Toy()

    N = 512
    X = torch.randn(N,4)
    y = (torch.rand(N) > 0.8).long()  # ~20% signal
    w = torch.ones(N)

    mt_edges = torch.linspace(200., 2000., steps=9)
    mt_scaled = torch.rand(N)

    loss_fn = CLsLoss(mt_edges, 200., 2000.)
    out = loss_fn(models=(m1,m2),
                  features=X,
                  cuts=(0.0, 0.0),
                  weights_xs=w,
                  weights_train=w,
                  target=y,
                  data_dict={"constraint_MT01FatJetMET": mt_scaled})
    assert torch.isfinite(out).all()
    out.backward()
