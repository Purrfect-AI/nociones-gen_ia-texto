import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn.functional as F
import os

# Configura Torch para usar la cantidad de threads igual a la cantidad de cores físicos menos uno.
#  Si el host tiene 12 cores, usa 11 para torch.
print(f'Setting number of cores to {os.cpu_count() - 1}')
torch.set_num_threads(os.cpu_count() - 1)

import idna

# def utf8_to_punycode(text: str) -> str:
#     """Encodes a UTF-8 string to its Punycode representation."""
#     return idna.encode(text).decode('ascii')

def punyencode(text: str) -> str:
    """Encodes a UTF-8 string to its Punycode representation, handling spaces by encoding each word separately."""
    
    return " ".join([idna.encode(word).decode('ascii') for word in text.split()])
    
def punydecode(punycode: str) -> str:
    """Decodes a Punycode string back to UTF-8."""
    #return idna.decode(punycode)
    return " ".join([idna.decode(word) for word in punycode.split()])

def process_name(name):
    name = name.lower()
    for n in name.split():
        if len(n) < 2:
            return ''
    try:
        return punyencode(name)
    except:
        #print(f'Cant convert {name}')
        return ''

def nll(logits, Y):
    return F.cross_entropy(logits, Y)

dataset = open("data/city_names_full.txt", 'r').read().split('\n')
with open('data/city_names_puny.txt', 'w') as f:
    for n in dataset:
        name = process_name(n)
        if name != '':
            f.write(name+'\n')
dataset = open("data/city_names_puny.txt", 'r').read().split('\n')
puny = [x for x in dataset if 'xn--' in x]
nopuny = [x for x in dataset if 'xn--' not in x]
np.random.seed(42)
dataset = [x.item() for x in np.random.choice(nopuny, 100000,replace=False)]

charset = ['*'] + sorted(list(set([y for x in dataset for y in x])))
ctoi = {c:i for i, c in enumerate(charset)}
itoc = {i:c for i, c in enumerate(charset)}
charset_size = len(charset)

def build_dataset(dataset: list):
    X, Y  = [], []
    for d in dataset:
        example = list(d) + ['*']
        context = [0] * context_size
        for c in example:
            X.append(context)
            Y.append(ctoi[c])
            context = context[1:] + [ctoi[c]] 
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

# build the dataset
context_size = 8
np.random.seed(42)
np.random.shuffle(dataset)
n1 = int(.8 * len(dataset))  # límite para el 80% del dataset
n2 = int(.9 * len(dataset))  # límite para el 90% del dataset
Xtr, Ytr = build_dataset(dataset[:n1])    # 80%
Xva, Yva = build_dataset(dataset[n1:n2])  # 10%
Xte, Yte = build_dataset(dataset[n2:])    # 10%

class Linear:
    def __init__(self, input_dim, output_dim, bias=True):
        self.W = torch.randn(input_dim, output_dim)/(input_dim ** 0.5)
        self.b = torch.zeros(output_dim) if bias else None

    def __call__(self, x):
        self.out = x @ self.W
        if self.b is not None:
            self.out += self.b
        return self.out

    def parameters(self):
        return [self.W] + ([] if self.b is None else [self.b])

class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []

class Embedding:
    def __init__(self, vocab_size, embidding_size):
        self.w = torch.randn(vocab_size, embidding_size)

    def __call__(self, x):
        self.out = self.w[x]
        return self.out

    def parameters(self):
        return [self.w]

class Flatten:
    def __init__(self, start_dim=1, end_dim=-1):
        self.start = start_dim
        self.end = end_dim
        
    def __call__(self, x):
        self.out = torch.flatten(x, self.start, self.end)
        return self.out

    def parameters(self):
        return []

class Group:
    def  __init__(self, group_size=1):
        self.group_size = group_size
        
    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T//self.group_size, C*self.group_size)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out

    def parameters(self):
        return []

class BatchNorm1d:
    def __init__(self, input_size, eps=1e-5, momentum=0.1):
        self.momentum = momentum
        self.eps = eps
        self.training_mode_on = True
        # los parametros
        self.gamma = torch.ones(input_size)
        self.beta = torch.zeros(input_size)
        self.running_mean = torch.zeros(input_size)
        self.runnint_std = torch.ones(input_size)

    def __call__(self, x):
        if x.ndim == 2:
            dim = 0
        else:
            dim = (0,1)
        if self.training_mode_on:
            xmean = x.mean(dim, keepdims=True)
            xstd = x.std(dim, keepdims=True)
            with torch.no_grad():
                self.running_mean = self.running_mean * (1 - self.momentum) + xmean * self.momentum
                self.runnint_std = self.runnint_std * (1 - self.momentum) + xstd * self.momentum
        else:
            xmean = self.running_mean
            xstd = self.runnint_std
        # normalizamos x para que tenga distribución N(0, 1)
        xhat = (x - xmean)/ (xstd + self.eps)
        self.out = self.gamma * xhat + self.beta
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

class ModelWaveNet:
    def __init__(self, charset_size, context_size, emb_size, hidden_size, g=torch.Generator().manual_seed(42)):
        self.charset_size = charset_size
        self.context_size = context_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        #self.C = torch.randn(self.charset_size, self.emb_size, generator=g)
        self.layers = [Embedding(self.charset_size, self.emb_size),
                       Group(2), Linear(self.emb_size*2, self.hidden_size, bias=False), BatchNorm1d(self.hidden_size), Tanh(),
                       Group(2), Linear(self.hidden_size*2, self.hidden_size, bias=False), BatchNorm1d(self.hidden_size), Tanh(),
                       Group(2), Linear(self.hidden_size*2, self.hidden_size, bias=False), BatchNorm1d(self.hidden_size), Tanh(),
                       Linear(self.hidden_size, self.charset_size)
                      ]

        # Kaiming init para todas las capas menos la última
        for l in self.layers[:-1]:
            if isinstance(l,Linear):
                l.W *= 5/3
        self.layers[-1].W *= 0.1  # La última capa es menos confianzuda

        # require_grad para todos los parámetros
        for p in self.parameters():
            p.requires_grad = True
    
    def __call__(self, x):
        # self.emb = self.C[x]
        # x = self.emb.view(-1, self.emb_size*self.context_size)
        
        for l in self.layers:
            x = l(x)
        return x
        
    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

    def to(self, device):
        for p in self.parameters():
            p.to(device)

    def count_parameters(self):
        return sum([p.nelement() for p in self.parameters()])

    def train(self, value):
        for p in model.layers:
            if isinstance(p, BatchNorm1d):
                p.training_mode_on = value

torch.manual_seed(42)
emb_size = 10
n_hidden = 256
model = ModelWaveNet(charset_size, context_size, emb_size, n_hidden)
model.count_parameters()

batch_size = 128
lr = 0.1
steps = 300000
losses = []
val_losses = []
st_time = time.time()
for i in range(steps):
    # 1. Forwards pass
    idx = torch.randint(0, len(Xtr), (batch_size, ))
    idx_val = torch.randint(0, len(Xva), (batch_size, ))
    
    logits = model(Xtr[idx])
    
    # 2. loss
    loss = F.cross_entropy(logits, Ytr[idx])
    losses.append(loss.item())
    
    # 3. zero grad
    for p in model.parameters():
        p.grad = None
    #for layer in model.layers:
    #    layer.out.retain_grad()
    
    # 4. backward pass
    loss.backward()

    with torch.no_grad():
        model.train(False)
        val_logits = model(Xva[idx_val])
        val_loss = F.cross_entropy(val_logits, Yva[idx_val])
        val_losses.append(val_loss.item())
        model.train(True)
    
    # 5. update
    if i > 100000 and i < 200000: # learing rate decay
        lr = 0.01
    if i > 200000:
        lr = 0.005
    for p in model.parameters():
        p.data -= p.grad * lr
    if i%10000 == 0:
        print(f'epoch {i}/{steps} loss: {loss.item():0.4f} val_loss: {val_loss.item():0.4f}  time/10K step: {(time.time() - st_time):0.4f}')
        st_time = time.time()

print(f'Train loss {F.cross_entropy(model(Xtr), Ytr).item()}')
print(f'Vtion loss {F.cross_entropy(model(Xva), Yva).item()}')

for p in model.layers:
    if isinstance(p, BatchNorm1d):
        p.training_mode_on = False
samples = []
for n in range(100):
    context = [0] * context_size
    out = []
    while True:
        logits = model(torch.tensor([context]))
        probs = F.softmax(logits, dim=1)
        idx = torch.multinomial(probs, num_samples=1, replacement=True).item()
        context = context[1:] + [idx]
        out.append(itoc[idx])
        if idx == 0:
            samples.append(''.join(out[:-1]))
            break
print(samples)