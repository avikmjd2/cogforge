from . import backend
from .app import Embedding, PositionalEncoding,Transformer,LayerNorm,Linear,RotatoryPositionalEncoding,to_cpu
import numpy


class GPTV1:
    def __init__(self, vocab, d_model, n_heads, n_layers, max_len, d_ff=None):
        self.tok = Embedding(vocab, d_model)
        self.pos    = PositionalEncoding(max_len, d_model)
        self.blocks = [Transformer(dmodel=d_model,n=n_heads,dff=d_ff) for _ in range(n_layers)]
        self.ln_f   = LayerNorm(d_model) 
        self.head   = Linear(d_model, vocab)
        self.max_len = max_len
        
    def __call__(self, idx):
        T = idx.shape[1]
        casual = backend.np.triu((backend.np.ones((T,T))),k=1).astype(bool)
        x = self.pos(self.tok(idx))
        for block in self.blocks:
            x = block(x,mask=casual)
            
        x = self.ln_f(x)
        return self.head(x)
    
    def parameters(self):
        ps = self.tok.parameters() + self.ln_f.parameters() + self.head.parameters()
        for b in self.blocks: ps += b.parameters()
        return ps
    
    def generate(self, idx, n_new, temperature=1.0, top_k=None):
        idx = numpy.asarray(idx)
        for _ in range(n_new):
            cond = idx[:, -self.max_len:]
            logits = to_cpu(self(cond).data)[:, -1, :] / temperature
            if top_k is not None:
                kth = numpy.sort(logits, axis=-1)[:, -top_k][:, None]
                logits = numpy.where(logits < kth, -1e9, logits)
            z = logits - logits.max(-1, keepdims=True)
            p = numpy.exp(z).astype(numpy.float64); p /= p.sum(-1, keepdims=True)
            nxt = numpy.array([[numpy.random.choice(len(pr), p=pr)] for pr in p])
            idx = numpy.concatenate([idx, nxt], axis=1)
        return idx                                          # <-- was a bare `return` 
        
        
        
        
class GPT2:
    def __init__(self, vocab, d_model, n_heads, n_layers, max_len, d_ff=None, base=10000.0):
        assert d_model % n_heads == 0
        self.tok    = Embedding(vocab, d_model)
        self.rope   = RotatoryPositionalEncoding(max_len, d_model // n_heads, base)  # dim = d_k
        self.blocks = [Transformer(d_model, n_heads, dff=d_ff, rope=self.rope)
                       for _ in range(n_layers)]
        self.ln_f   = LayerNorm(d_model)
        self.head   = Linear(d_model, vocab)
        self.max_len = max_len


    def __call__(self, idx):
        T = idx.shape[1]
        causal = backend.np.triu(backend.np.ones((T, T)), k=1).astype(bool)
        x = self.tok(idx)                          # straight from token embedding
        for block in self.blocks:
            x = block(x, mask=causal)
        return self.head(self.ln_f(x))

    def parameters(self):
        ps = self.tok.parameters() + self.ln_f.parameters() + self.head.parameters()
        for b in self.blocks: ps += b.parameters()
        return ps   
    
    def generate(self, idx, n_new, temperature=1.0, top_k=None):
        idx = numpy.asarray(idx)                                 # token ids live on host
        for _ in range(n_new):
            cond = idx[:, -self.max_len:]
            logits = to_cpu(self(cond).data)[:, -1, :] / temperature   # forward on GPU, logits to host
            if top_k is not None:
                kth = numpy.sort(logits, axis=-1)[:, -top_k][:, None]
                logits = numpy.where(logits < kth, -1e9, logits)
            z = logits - logits.max(-1, keepdims=True)
            p = numpy.exp(z).astype(numpy.float64); p /= p.sum(-1, keepdims=True)
            nxt = numpy.array([[numpy.random.choice(len(pr), p=pr)] for pr in p])
            idx = numpy.concatenate([idx, nxt], axis=1)
            # backend._cp.get_default_memory_pool().free_all_blocks() if backend.USE_GPU else None
        return idx                              