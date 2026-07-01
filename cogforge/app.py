import numpy
from . import backend
# from backend import np,USE_GPU,NO_GRAD,_NUMEXPR_AVAILABLE,USE_NUMEXPR,_ne,_cp

def to_cpu(a):
    """Return a numpy array regardless of backend."""
    return backend._cp.asnumpy(a) if (backend.USE_GPU and backend._cp is not None) else numpy.asarray(a)

def scatter_add(target, indices, values):
    """target[indices] += values."""
    if backend.USE_GPU:
        import cupyx
        cupyx.scatter_add(target, indices, values)
    else:
        numpy.add.at(target, indices, values)


def set_numexpr(enabled, threads=None):
    """User switch: set_numexpr(True, threads=8) or set_numexpr(False)."""
    # global USE_NUMEXPR
    if enabled and not backend._NUMEXPR_AVAILABLE:
        raise RuntimeError("numexpr not installed — run: pip install numexpr")
    backend.USE_NUMEXPR = bool(enabled)
    if backend.USE_NUMEXPR and threads:
        backend._ne.set_num_threads(threads)
    return backend.USE_NUMEXPR

def _stable_softmax(sd):
    mx = sd.max(axis=-1, keepdims=True)          # reduction stays numpy
    if backend.USE_NUMEXPR and not backend.USE_GPU:
        e = backend._ne.evaluate("exp(sd - mx)")
        s = e.sum(axis=-1, keepdims=True)
        return backend._ne.evaluate("e / s")
    e = backend.np.exp(sd - mx)
    return e / e.sum(axis=-1, keepdims=True)

def _iadd_scaled(dst, src, coef):                # dst += src * coef, in place
    if backend.USE_NUMEXPR and not backend.USE_GPU:
        backend._ne.evaluate("dst + src * coef", out=dst)
    else:
        dst += src * coef    
        
def needGradientHence(state:bool):
    backend.NO_GRAD = not state

class Tensor:
    def __init__(self, array,children=(),requires_grad = True,typed="compressed"):
        self.data = backend.np.asarray(array, dtype=backend.np.float32 if typed == "compressed" else backend.np.float64)
        self.shape = self.data.shape
        self.requires_grad = requires_grad
        # if self.requires_grad:
        self.grad = None if backend.NO_GRAD else backend.np.zeros(self.shape,dtype=backend.np.float32 if typed == "compressed" else backend.np.float64)
        self._backwards = lambda:None
        self._children = set(children)
        self.typed = typed
        
    def __getitem__(self, key):
        out_data = self.data[key]
        
        out = Tensor(out_data,children=(self,))
        
        def _backward():
            self.grad[key] += out.grad
            # backend.np.add.at(self.grad, key, out.grad)
            
        out._backwards = _backward
        
        return out
    
    @classmethod
    def cat(cls,tensors:tuple,axis=-1):
        """
        Concatenates a sequence of Tensors along a specified axis.
        """
        arrays = tuple(t.data for t in tensors)
        out = Tensor(backend.np.concatenate(arrays,axis=axis),children=tensors)
        sizes = [a.shape[axis] for a in arrays[:-1]]
        split_indices = backend.np.cumsum(sizes)
        
        
        def _backward():
            split_grads = backend.np.split(out.grad, split_indices, axis=axis)
            
            for t,g in zip(tensors,split_grads):
                if t.requires_grad:
                    t.grad += g
            
        out._backwards = _backward
        
        return out
        
        
        
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = self.data + other.data
        out = Tensor(out_data, (self, other))

        def _backward():
            if self.shape != out_data.shape:
                axis = tuple(range(out.grad.ndim - self.data.ndim))
                self.grad += backend.np.sum(out.grad, axis=axis).reshape(self.shape)
            else:
                self.grad += out.grad

            if other.shape != out_data.shape:
                axis = tuple(range(out.grad.ndim - other.data.ndim))
                other.grad += backend.np.sum(out.grad, axis=axis).reshape(other.shape)
            else:
                other.grad += out.grad

        out._backwards = _backward
        return out
    
    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = self.data * other.data
        
        out = Tensor(out_data, children=(self, other))
        
        def _backward():
            grad_self = other.data * out.grad
            if self.shape != out_data.shape:
                axis = tuple(range(out.grad.ndim - self.data.ndim))
                grad_self = backend.np.sum(grad_self, axis=axis).reshape(self.shape)
            self.grad += grad_self

            # gradient w.r.t. other
            grad_other = self.data * out.grad
            if other.shape != out_data.shape:
                axis = tuple(range(out.grad.ndim - other.data.ndim))
                grad_other = backend.np.sum(grad_other, axis=axis).reshape(other.shape)
            other.grad += grad_other
            
        out._backwards = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    # def __matmul__(self, other):
    #     out_data = self.data @ other.data
        
    #     out = Tensor(out_data, children=(self,other))
        
    #     # def _backward():
    #     #     self.grad += out.grad @ backend.np.transpose(other.data)
    #     #     other.grad +=  backend.np.transpose(self.data) @ out.grad
        
    #     def _backward():
    #         self.grad += out.grad @ backend.np.swapaxes(other.data, -1, -2)
    #         other.grad += backend.np.swapaxes(self.data, -1, -2) @ out.grad
        
    #     out._backwards = _backward    
    #     return out
    
    @staticmethod
    def unbroadcast(grad, shape):
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        for i, s in enumerate(shape):
            if s == 1 and grad.shape[i] != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def __matmul__(self, other):    
        out = Tensor(self.data @ other.data, children=(self, other))
        def _backward():
            gs = out.grad @ backend.np.swapaxes(other.data, -1, -2)
            go = backend.np.swapaxes(self.data, -1, -2) @ out.grad
            self.grad  += Tensor.unbroadcast(gs, self.data.shape)
            other.grad += Tensor.unbroadcast(go, other.data.shape)
        out._backwards = _backward
        return out
    
    def relu(self):
        out_data = backend.np.maximum(0,self.data)
        
        out = Tensor(out_data, children=(self,))
        
        def _backward():
            mask = out_data > 0
            self.grad+=(mask*out.grad)
        
        out._backwards = _backward    
        return out
    
    
    def view(self,shape):
        curr_shape = self.data.shape
        
        if not self.data.flags['C_CONTIGUOUS']:
            arr = backend.np.ascontiguousarray(self.data)
        else:
            arr = self.data
        
        out_data = arr.reshape(shape)
        out = Tensor(out_data, children=(self,))
        
        def _backward():
            self.grad+=(out.grad.reshape(curr_shape))
            
            
        out._backwards = _backward
        return out
        
        
        
        
    def sigmoid(self):
        out_data = 1 / (1 + backend.np.exp(-self.data))
        out = Tensor(out_data, children=(self,))
        def _backward():
            local_derivative = out_data * (1 - out_data)
            
            self.grad += local_derivative * out.grad
            
        out._backwards = _backward
        return out
    
    def tanh(self):
        out_data = backend.np.tanh(self.data)
        out = Tensor(out_data,children=(self,))
        def _backward():
            dervitive = 1-(out_data*out_data)
            self.grad+=(out.grad*dervitive)
            
        out._backwards = _backward
        return out
    
    def softmax(self,axis=-1):
        maxVal = backend.np.max(self.data,axis=axis,keepdims=True)
        expVal = backend.np.exp(self.data-maxVal)
        
        out_data = expVal/backend.np.sum(expVal,axis=axis,keepdims=True)
        out=Tensor(out_data,(self,))
        
        def _backward():
            g = out.grad
            self.grad += out_data * (g - backend.np.sum(g * out_data, axis=axis, keepdims=True))
        
        out._backwards = _backward
        return out
    
    #optimized for batches
    @classmethod
    def cross_entropy_loss(cls,predictions, targets):
       
        probabilities = backend.np.clip(predictions.data, 1e-15, 1 - 1e-15)
        batch_size = predictions.shape[0]
        
        loss_data = - backend.np.sum(targets * backend.np.log(probabilities)) / batch_size
        loss = Tensor(loss_data, children=(predictions,))
        
        def _backward():
            
            # predictions.grad += (predictions.data - targets) / batch_size * loss.grad
            predictions.grad += (-(targets / probabilities)) / batch_size * loss.grad

            
        loss._backwards = _backward
        return loss
    
    #LEGACY
    def backwards_recursive(self):
        topo = []
        visited = set()
        def topoSort(v):
            if v in visited:
                return
            visited.add(v)
            for edges in v._children:
                topoSort(edges)
            
            topo.append(v)
            
        topoSort(self)
        
        self.grad = backend.np.ones(self.shape,dtype=backend.np.float32 if self.typed == "compressed" else backend.np.float64)
        
        for node in reversed(topo):
            node._backwards()
            
    def backwards(self):
        topo = []
        visited = set()
        stack = [(self,False)]
        
        while stack:
            node,processed = stack.pop()
            if processed:
                topo.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)            
                
            stack.append((node,True))
            for child in node._children:
                stack.append((child,False))
                
        self.grad = backend.np.ones(self.shape,dtype=backend.np.float32 if self.typed == "compressed" else backend.np.float64)
        
        for node in reversed(topo):
            node._backwards()
                    
            
    def flatten(self):
        batch_size = self.shape[0]
        out_data = self.data.reshape(batch_size, -1)
        
        out = Tensor(out_data, children=(self,))
        def _backward():
            self.grad += out.grad.reshape(self.shape)
            
        out._backwards = _backward
        return out
    
    def flatten_consective(self,num):
        """THIS ONLY FLATTENS A 3 DIMENSIONAL TENSOR. CAREFULL.
            EXPLECTS (BATCH, CONTEXT, DIMENSION) AS SHAPE"""
        self.flatten_dim =  num
        if(len(self.shape)) != 3:
            raise ValueError("THIS FUNCTION ONLY EXPECTS A 3 DIMENSIONAL TENSOR.")
        
        B,T,C = self.shape
        
        if T % num != 0:
            raise ValueError(f"Time dimension {T} must be cleanly divisible by num {num}")
        
        out_data = self.data.reshape(B,T//num,C*num)
        
        out = Tensor(out_data, children=(self,))
        
        def _backward():
            self.grad += out.grad.reshape(self.shape)
            
        out._backwards = _backward
        return out
    
    #TODO
    @classmethod
    def cross_entropy_loss_masked(cls, predictions, targets, mask):
        """
        predictions: (B, V) softmax probs
        targets:     (B, V) one-hot
        mask:        (B,) 1.0 for real positions, 0.0 for <PAD>
        """
        probs = backend.np.clip(predictions.data, 1e-15, 1 - 1e-15)
        mask = mask.reshape(-1, 1)                          # (B, 1)
        n_real = mask.sum()
        n_real = n_real if n_real > 0 else 1.0

        # only real rows contribute to the loss value
        loss_data = -backend.np.sum(mask * targets * backend.np.log(probs)) / n_real
        loss = cls(loss_data, children=(predictions,))

        def _backward():
            # grad = (predictions.data - targets) / n_real
            grad = -(targets / probs) / n_real
            grad = grad * mask                              # <-- padded rows get exactly zero gradient
            predictions.grad += grad * loss.grad

        loss._backwards = _backward
        return loss

    @classmethod
    def sparse_softmax_cross_entropy_legacy(cls, scores, target_ids):
        """ scores: (B, T, V) logits ; target_ids: (B, T) int """
        z = scores.data - scores.data.max(axis=-1, keepdims=True)
        e = backend.np.exp(z)
        p = e / e.sum(axis=-1, keepdims=True)
        N = int(backend.np.prod(scores.shape[:-1]))
        flat, idx, rows = p.reshape(N, -1), target_ids.reshape(N), backend.np.arange(N)
        loss = cls(-backend.np.log(backend.np.clip(flat[rows, idx], 1e-15, 1.0)).sum() / N, children=(scores,))
        def _backward():
            # g = p.reshape(N, -1).copy()
            g = p.reshape(N, -1)
            g[rows, idx] -= 1.0                      # (softmax - onehot)
            scores.grad += (g.reshape(scores.shape) / N) * loss.grad
        loss._backwards = _backward
        return loss
    
    @classmethod
    def sparse_softmax_cross_entropy(cls, scores, target_ids):
        p = _stable_softmax(scores.data)
        N = int(backend.np.prod(scores.shape[:-1]))
        flat, idx, rows = p.reshape(N, -1), target_ids.reshape(N), backend.np.arange(N)
        loss = cls(-backend.np.log(backend.np.clip(flat[rows, idx], 1e-15, 1.0)).sum() / N, children=(scores,))
        def _backward():
            g = p.reshape(N, -1)
            g[rows, idx] -= 1.0
            _iadd_scaled(scores.grad, g.reshape(scores.shape), backend.np.float32(loss.grad / N))
        loss._backwards = _backward
        return loss
    
    #Legacy
    @classmethod
    def softmax_cross_entropy_old(cls, scores, targets):
        z = scores.data
        z = z - z.max(axis=1, keepdims=True)          # stability
        e = backend.np.exp(z)
        p = e / e.sum(axis=1, keepdims=True)          # softmax, computed privately
        B = scores.shape[0]

        loss = cls(-backend.np.sum(targets * backend.np.log(backend.np.clip(p, 1e-15, 1.0))) / B, children=(scores,))

        def _backward():
            scores.grad += (p - targets) / B * loss.grad
        loss._backwards = _backward
        return loss
    
    @classmethod
    def softmax_cross_entropy(cls, scores, targets):
        z = scores.data
        z = z - z.max(axis=-1, keepdims=True)          
        e = backend.np.exp(z)
        p = e / e.sum(axis=-1, keepdims=True)
        N = backend.np.prod(scores.shape[:-1])                  # B for 2D, B*T for 3D
        loss = cls(-backend.np.sum(targets * backend.np.log(backend.np.clip(p, 1e-15, 1.0))) / N, children=(scores,))
        def _backward():
            scores.grad += (p - targets) / N * loss.grad
        loss._backwards = _backward
        return loss


    @classmethod
    def softmax_cross_entropy_masked(cls, scores, targets, mask):
        z = scores.data
        z = z - z.max(axis=1, keepdims=True)
        e = backend.np.exp(z)
        p = e / e.sum(axis=1, keepdims=True)

        mask = mask.reshape(-1, 1)                     
        n_real = mask.sum()
        n_real = n_real if n_real > 0 else 1.0

        loss = cls(-backend.np.sum(mask * targets * backend.np.log(backend.np.clip(p, 1e-15, 1.0))) / n_real, children=(scores,))

        def _backward():
            grad = (p - targets) / n_real
            grad = grad * mask                         
            scores.grad += grad * loss.grad
        loss._backwards = _backward
        return loss
    
    def transpose(self,axes):
        out = Tensor(backend.np.transpose(self.data,axes=axes),children=(self,))
        inv = backend.np.argsort(axes)
        def _backward():
            self.grad+=backend.np.transpose(out.grad,inv)
            
        out._backwards = _backward
        return out
    
    def masked_fill(self, mask, value):
        out_data = backend.np.where(mask, value, self.data)
        out = Tensor(out_data, children=(self,))
        def _backward():
            self.grad += backend.np.where(mask, 0.0, 1.0) * out.grad
        out._backwards = _backward
        return out
                            
        
            
            
class Linear:
    def __init__(self, nin, nout):
        self.W = Tensor(backend.np.random.randn(nin, nout) * backend.np.sqrt(2.0/nin))
        self.v_W = backend.np.zeros(self.W.shape)
        self.b = Tensor(backend.np.zeros((nout,)))
        self.v_b = backend.np.zeros(self.b.shape)
        
    def __call__ (self,x):
        if(x.shape[-1]!=self.W.shape[0]):
            raise TypeError("Dimensions mismatched")        
        out = x@self.W + self.b
        return out
    
    def parameters(self):
        return [self.W, self.b]
    
    
class MLP:
   
    
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes)-1):
            self.layers.append(Linear(layer_sizes[i], layer_sizes[i+1]))
        
    
    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x).relu()
        
        out = self.layers[-1](x)
        return out
    
    
    def save(self, filename="best_model.npz"):
        weights_dict = {}
        for i, layer in enumerate(self.layers):
            weights_dict[f'W_{i}'] = layer.W.data
            weights_dict[f'b_{i}'] = layer.b.data
            
        backend.np.savez_compressed(filename, **weights_dict)
        
    def load(self, filename="best_model.npz"):
        with backend.np.load(filename) as data:
            for i, layer in enumerate(self.layers):
                layer.W.data = data[f'W_{i}']
                layer.b.data = data[f'b_{i}']
        print(f"Model loaded successfully from {filename}!")
        
        
class Embedding:
    def __init__(self, vocab_size, embedding_dim):
        self.weights = Tensor(backend.np.random.randn(vocab_size, embedding_dim))
        
    def __call__(self, input_indices):
        idx = backend.np.asarray(input_indices)   
        out_data = self.weights.data[idx]
        
        out = Tensor(out_data, children=(self.weights,))
        
        def _backward():
            # backend.np.add.at(self.weights.grad,input_indices,out.grad)
            scatter_add(self.weights.grad, idx, out.grad)
        
        out._backwards = _backward
        return out
    
    def parameters(self):
        return [self.weights]
    
    
class RNNCell:
    def __init__(self,input_dim, hidden_dim):
        self.i2h = Linear(input_dim,hidden_dim)
        self.h2h = Linear(hidden_dim,hidden_dim)
        
    def __call__(self, x,h_prev):
        input_transform = self.i2h(x)
        hidden_transform = self.h2h(h_prev)
        
        h_next = Tensor.tanh(input_transform + hidden_transform)
        
        return h_next
    
    def parameters(self):
        return self.i2h.parameters() + self.h2h.parameters()
    
    
class RNN:
    def __init__(self,input_dim, hidden_dim):
        self.hidden_dim = hidden_dim
        self.cell = RNNCell(input_dim=input_dim,hidden_dim=hidden_dim)
        
    def __call__(self,xs,prev_hidden = None):
        """
        xs: A list of Tensors, each of shape (B, input_dim) — the same
            timestep across all sequences in the batch.

        Returns: A list of Tensors (hidden states) at every time step,
                 each of shape (B, hidden_dim).
        """
        
        if prev_hidden is not None:
            h = prev_hidden
        else:
            B = xs[0].shape[0]
            h = Tensor(backend.np.zeros((B, self.hidden_dim)))
        
        
        hidden_states = []
        
        for x in xs:
            h = self.cell(x,h)
            hidden_states.append(h)
            
        return hidden_states
    
    def parameters(self):
        return self.cell.parameters()
    
            
class StackedRNN:
    def __init__(self,input_dim,hidden_dim,num_layers):
        self.layers = []
        self.layers.append(RNN(input_dim,hidden_dim))
        
        for _ in range(1,num_layers):
            self.layers.append(RNN(input_dim=hidden_dim, hidden_dim=hidden_dim))
            
    def __call__(self,xs,prev_hidden = None):
        """
        xs: A list of Tensors (the initial word embeddings)
        
        Returns: A list of Tensors representing the hidden states 
                 from the very TOP layer.
        """
        if prev_hidden is not None:
            assert len(prev_hidden) == len(self.layers), \
                f"prev_hidden has {len(prev_hidden)} states but stack has {len(self.layers)} layers"


        curr = xs
        h_outs = []
        
        for i,layer in enumerate(self.layers):
            initial = prev_hidden[i] if prev_hidden is not None else None
            curr = layer(curr,initial)
            
            h_outs.append(curr[-1])
            
        return curr,h_outs
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    
class SGD:
    def __init__(self,parameters,learning_rate=0.01):
        self.parameters = parameters
        self.lr = learning_rate
        
    def clip_grads(self, max_norm=5.0):
        total = backend.np.sqrt(sum(backend.np.sum(p.grad ** 2) for p in self.parameters))
        if total > max_norm:
            scale = max_norm / (total + 1e-6)
            for p in self.parameters:
                p.grad *= scale
        
    def step(self):
        for p in self.parameters:
            p.data-=self.lr*(p.grad)
            
    def zero_grad(self):
        for p in self.parameters:
            p.grad = backend.np.zeros_like(p.grad)
            
class Sequential:
    def __init__(self,layers):
        self.layers = layers
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'W'): params.append(layer.W)
            if hasattr(layer, 'b'): params.append(layer.b)
            if hasattr(layer, 'gamma'): params.append(layer.gamma)
            if hasattr(layer, 'beta'): params.append(layer.beta)
            #TODO: add rest of parameters
        return params
    
    def train(self):
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = True
                
    def test(self):
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = False
    
    
class BatchNorm1D:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        
        self.gamma = Tensor(backend.np.ones(dim))
        self.beta = Tensor(backend.np.zeros(dim))
        
        self.running_mean = backend.np.zeros(dim)
        self.running_var = backend.np.ones(dim)
        
    
    def __call__(self,x:Tensor):
        reduce_dims = (0,) if x.data.ndim==2 else (0,1)
        
        if self.training:
            mean = x.data.mean(axis=reduce_dims, keepdims=True)
            var = x.data.var(axis=reduce_dims, keepdims=True)
            
            self.x_centered = x.data - mean
            self.std_inv = 1.0 / backend.np.sqrt(var + self.eps)
            self.x_hat = self.x_centered * self.std_inv
            
            N = backend.np.prod([x.data.shape[d] for d in reduce_dims])
            unbiased_var = var.squeeze() * (N / (N - 1)) if N > 1 else var.squeeze()
            
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbiased_var
            
        else:
            self.x_hat = (x.data-self.running_mean)/backend.np.sqrt(self.running_var + self.eps)
        
        out_data = self.gamma.data * self.x_hat + self.beta.data
        
        out = Tensor(out_data, children=(x, self.gamma, self.beta))
        
        def _backward():
            if not self.training:
                return
            
            dout = out.grad
            self.gamma.grad += backend.np.sum(dout * self.x_hat, axis=reduce_dims)
            self.beta.grad += backend.np.sum(dout, axis=reduce_dims)
            
            N = backend.np.prod([x.data.shape[d] for d in reduce_dims])

            dx_hat = dout * self.gamma.data
            dx = (1.0 / N) * self.std_inv * (
                N * dx_hat 
                - backend.np.sum(dx_hat, axis=reduce_dims, keepdims=True) 
                - self.x_hat * backend.np.sum(dx_hat * self.x_hat, axis=reduce_dims, keepdims=True)
            )

            x.grad += dx
            
            
        out._backwards = _backward
        
        return out
            
           
           
class Bridge:
    """
    Maps encoder final hidden states to decoder initial hidden states.
    Handles asymmetry in:
      - num_layers   (encoder count != decoder count)
      - hidden_dim   (enc_hidden != dec_hidden)
    Embedding-dim asymmetry needs no bridge: encoder/decoder have
    independent Embeddings and independent first-layer i2h Linears.

    mode:
      "project" : one learned Linear(enc_H -> dec_H) per decoder layer (general, recommended)
      "tie"     : no params; requires enc_H == dec_H. Selects/repeats raw states.
    """
    def __init__(self, enc_hidden, dec_hidden, enc_layers, dec_layers, mode="project"):
        self.enc_hidden = enc_hidden
        self.dec_hidden = dec_hidden
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.mode = mode
        
        if mode == "tie":
            if enc_hidden != dec_hidden:
                raise ValueError(
                    f"mode='tie' needs enc_hidden==dec_hidden, "
                    f"got {enc_hidden} vs {dec_hidden}. Use mode='project'."
                )
            self.projections = None
        elif mode == "project":
            # one projection per decoder layer
            self.projections = [Linear(enc_hidden, dec_hidden) for _ in range(dec_layers)]
        else:
            raise ValueError(f"unknown bridge mode: {mode}")
        
    def __select(self,enc_hidden):
        """Pick/pad encoder states to exactly dec_layers, bottom→top aligned at the TOP."""
        n = len(enc_hidden) #enc_hidden is the final layer of each level
        if n == self.dec_layers:
            return enc_hidden
        elif n > self.dec_layers:
            return enc_hidden[-self.dec_layers:]
        else:
            top = enc_hidden[-1]
            return enc_hidden + [top] * (self.dec_layers - n)
        
    def __call__(self,enc_hidden):
        final_layers = self.__select(enc_hidden)
        if self.mode=="tie":
            return final_layers
        
        return [proj(h) for proj,h in zip(self.projections,final_layers)]
    
    def parameters(self):
        if self.projections is None:
            return []
        return [p for proj in self.projections for p in proj.parameters()]
        
    
class Adam:
    def __init__(self, parameters, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = parameters
        self.lr = lr; self.beta1 = beta1; self.beta2 = beta2; self.eps = eps
        self.m = [backend.np.zeros_like(p.data) for p in parameters]
        self.v = [backend.np.zeros_like(p.data) for p in parameters]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.parameters):
            self.m[i] = self.beta1*self.m[i] + (1-self.beta1)*p.grad
            self.v[i] = self.beta2*self.v[i] + (1-self.beta2)*(p.grad**2)
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            p.data -= self.lr * m_hat / (backend.np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.parameters:
            p.grad = backend.np.zeros_like(p.grad)

    def clip_grads(self, max_norm=5.0):
        total = backend.np.sqrt(sum(backend.np.sum(p.grad**2) for p in self.parameters))
        if total > max_norm:
            scale = max_norm / (total + 1e-6)
            for p in self.parameters:
                p.grad *= scale
                
                
class Attention:
    def __init__(self,dk):
        self.scale = 1.0/backend.np.sqrt(dk)
    
    def __call__(self, Q:Tensor,K:Tensor,V:Tensor,mask=None):
        axes = list(range(Q.data.ndim)); axes[-1], axes[-2] = axes[-2], axes[-1]
        scores:Tensor = (Q @ K.transpose(tuple(axes)))*(self.scale)
        
        if mask is not None:
            scores = scores.masked_fill(mask,-1e9)
            
        intermediate = scores.softmax(axis=-1)
        
        return intermediate @ V
        
class MultiHeadAttention:
    def __init__(self,dinp,dmodel, dout,n,rope=None):
        assert dmodel%n==0
        self.h = n
        self.dk = dmodel//n
        self.q = Linear(dinp,dmodel) #dmodel = dk*h, we will slice it 
        self.k = Linear(dinp,dmodel)
        self.rope = rope
        self.v = Linear(dinp,dmodel)
        self.o = Linear(dmodel,dout)
        self.attention = Attention(self.dk)

    def split(self,x:Tensor):
        B,T,_ = x.data.shape
        return x.view((B,T,self.h,self.dk)).transpose((0,2,1,3)) #(B, h, T, d_k)
    
    def merge(self,x:Tensor):
        B, h, T, d_k = x.data.shape
        return x.transpose((0, 2, 1, 3)).view((B, T, h * d_k))

    def __call__(self, query,key,value,mask=None):
        Q = self.split(self.q(query))
        K = self.split(self.k(key))
        V = self.split(self.v(value))
        if self.rope is not None:
            Q = self.rope(Q)
            K = self.rope(K)
        out = self.attention(Q=Q,K=K,V=V,mask=mask)
        return self.o(self.merge(out))
    
    def parameters(self):
        return (self.q.parameters() + self.k.parameters()
                + self.v.parameters() + self.o.parameters())
    
    
class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = Tensor(backend.np.ones(dim))      
        self.beta  = Tensor(backend.np.zeros(dim))     

    def __call__(self, x):
        mu  = x.data.mean(axis=-1, keepdims=True)
        var = x.data.var(axis=-1, keepdims=True)
        std_inv = 1.0 / backend.np.sqrt(var + self.eps)
        x_hat = (x.data - mu) * std_inv
        out_data = self.gamma.data * x_hat + self.beta.data
        out = Tensor(out_data, children=(x, self.gamma, self.beta))

        D = x.data.shape[-1]
        def _backward():
            dout = out.grad
            axes = tuple(range(dout.ndim - 1))            
            self.gamma.grad += backend.np.sum(dout * x_hat, axis=axes)
            self.beta.grad  += backend.np.sum(dout, axis=axes)
            dxhat = dout * self.gamma.data
            dx = std_inv / D * (
                D * dxhat
                - backend.np.sum(dxhat, axis=-1, keepdims=True)
                - x_hat * backend.np.sum(dxhat * x_hat, axis=-1, keepdims=True)
            )
            x.grad += dx
        out._backwards = _backward
        return out

    def parameters(self):
        return [self.gamma, self.beta]
    
class FeedForward:
    def __init__(self,dmodel,dff=None):
        dff = dff if dff is not None else 4 * dmodel
        self.fc1 = Linear(dmodel,dff)
        self.fc2 = Linear(dff,dmodel)
        
    def __call__(self, x):
        return self.fc2(self.fc1(x).relu())
    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters()

class Transformer:
    """The transformer block has no bridge. So demb = dmodel"""
    
    def __init__(self, dmodel, n, dff = None,rope=None):
        self.ln1 = LayerNorm(dmodel)
        self.attn = MultiHeadAttention(dinp=dmodel,dmodel=dmodel,dout=dmodel,n=n,rope=rope)
        self.ln2 = LayerNorm(dmodel)
        self.ff  = FeedForward(dmodel=dmodel,dff=dff)
        
    def __call__(self,x,mask=None):
        a = self.ln1(x)
        x = x + self.attn(a,a,a,mask=mask)
        f = self.ln2(x)
        x = x+ self.ff(f)
        
        return x
    
    def parameters(self):
        return (self.ln1.parameters() + self.attn.parameters()
                + self.ln2.parameters() + self.ff.parameters())
        
        
class PositionalEncoding:
    def __init__(self,max_len,dmodel):
        pe = backend.np.zeros((max_len,dmodel))
        pos = backend.np.arange(max_len).reshape(-1,1)
        div = backend.np.exp(backend.np.arange(0, dmodel, 2) * (-backend.np.log(10000.0) / dmodel))
        pe[:,0::2] = backend.np.sin(pos*div)
        pe[:,1::2] = backend.np.cos(pos*div)
        self.pe = pe
        
    
    def __call__(self, x):
        T = x.shape[1]
        return x + Tensor(self.pe[:T])
    
    def parameters(self):
        return []
        
class RotatoryPositionalEncoding:
    def __init__(self,max_len, dim, base=10000.0):
        assert dim % 2 == 0, "RoPE dim must be even"
        theta = 1.0/(base**(backend.np.arange(0,dim,2)/float(dim)))
        m = backend.np.arange(max_len)
        
        freq = backend.np.outer(m,theta)
        cos = backend.np.cos(freq)
        sin = backend.np.sin(freq)
        
        self.cos = backend.np.concatenate([cos, cos], axis=-1)               
        self.sin = backend.np.concatenate([sin, sin], axis=-1)
        
    def xbar(self,x):
        d = x.shape[-1]
        half = d // 2
        
        x1 = x[...,:half]
        x2 = x[...,half:]
        
        return Tensor.cat((-x2,x1),axis=-1)
    
    def __call__(self,x:Tensor):
        T = x.shape[-2]
        cos = self.cos[:T]
        sin = self.sin[:T]
        return x*cos + self.xbar(x)*sin
    
    def parameters(self):
        return []
