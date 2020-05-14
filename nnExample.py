import torch
import random
N, D_in, D_out = 10, 5, 1

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
for i in range(10):
    y[i] = random.randint(0,1)
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, D_out)
)
loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-4
for t in range(501):
    for i in range(10):
        y_pred = model(x[i])
        loss = loss_fn(y_pred, y[i])
        if t % 100 == 0:
            print(t, loss.item())
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad
                
 #example
W=torch.tensor([0.7572, 1.6134, 0.3783, 0.0668, 0.8177])
print(model(W))
