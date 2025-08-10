# Minimal runnable example (toy data)
import torch
from torch import nn
from abcd_clsloss import CLsLoss

class Toy(nn.Module):
    def __init__(self): super().__init__(); self.net = nn.Sequential(nn.Linear(10,16), nn.ReLU(), nn.Linear(16,1))
    def forward(self, x): return self.net(x)

N = 2000
torch.manual_seed(42)
X = torch.randn(N,10)
y = (torch.sigmoid(X[:,0] - 0.5 + 0.3*X[:,1]) > 0.0).float().bernoulli(0.2).long()
mt_scaled = torch.rand(N)  # pretend final discriminant scaled to [0,1]
w_xs = torch.ones(N)

f1_model, f2_model = Toy(), Toy()
mt_edges = torch.linspace(200., 2000., steps=17)
loss_fn = CLsLoss(mt_edges, 200., 2000.)

optim = torch.optim.Adam(list(f1_model.parameters()) + list(f2_model.parameters()), lr=3e-3)

for step in range(3):
    optim.zero_grad()
    cls = loss_fn(models=(f1_model, f2_model),
                  features=X,
                  cuts=(0.5, 0.5),
                  weights_xs=w_xs,
                  weights_train=w_xs,
                  target=y,
                  data_dict={"constraint_MT01FatJetMET": mt_scaled})
    print("step", step, "loss", cls.item())
    cls.backward()
    optim.step()
