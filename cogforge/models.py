from . import backend
from .app import Embedding, PositionalEncoding,Transformer,LayerNorm,Linear,RotatoryPositionalEncoding

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
        """
        idx: (B, T) int array — the prompt (B is usually 1)
        returns: (B, T + n_new) int array
        """
        for _ in range(n_new):
            cond = idx[:, -self.max_len:]                    # crop to context window
            logits = self(cond).data[:, -1, :] / temperature # (B, V) — last position only
            if top_k is not None:
                kth = backend.np.sort(logits, axis=-1)[:, -top_k][:, None]
                logits = backend.np.where(logits < kth, -1e9, logits)  # keep only top-k choices
            z = logits - logits.max(-1, keepdims=True)
            p = backend.np.exp(z).astype(backend.np.float64); p /= p.sum(-1, keepdims=True)     # softmax → probabilities
            nxt = backend.np.array([[backend.np.random.choice(len(pr), p=pr)] for pr in p])  # sample
            idx = backend.np.concatenate([idx, nxt], axis=1)         # append, feed back in
        return idx                                            # <-- was a bare `return` 
        
        
        
        
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
        """
        idx: (B, T) int array — the prompt (B is usually 1)
        returns: (B, T + n_new) int array
        """
        for _ in range(n_new):
            cond = idx[:, -self.max_len:]                    # crop to context window
            logits = self(cond).data[:, -1, :] / temperature # (B, V) — last position only
            if top_k is not None:
                kth = backend.np.sort(logits, axis=-1)[:, -top_k][:, None]
                logits = backend.np.where(logits < kth, -1e9, logits)  # keep only top-k choices
            z = logits - logits.max(-1, keepdims=True)
            p = backend.np.exp(z).astype(backend.np.float64); p /= p.sum(-1, keepdims=True)     # softmax → probabilities
            nxt = backend.np.array([[backend.np.random.choice(len(pr), p=pr)] for pr in p])  # sample
            idx = backend.np.concatenate([idx, nxt], axis=1)         # append, feed back in
        return idx                               