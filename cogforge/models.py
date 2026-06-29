import numpy as np
from app import Embedding, PositionalEncoding,Transformer,LayerNorm,Linear

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
        casual = np.triu((np.ones((T,T))),k=1).astype(bool)
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
                kth = np.sort(logits, axis=-1)[:, -top_k][:, None]
                logits = np.where(logits < kth, -1e9, logits)  # keep only top-k choices
            z = logits - logits.max(-1, keepdims=True)
            p = np.exp(z).astype(np.float64); p /= p.sum(-1, keepdims=True)     # softmax → probabilities
            nxt = np.array([[np.random.choice(len(pr), p=pr)] for pr in p])  # sample
            idx = np.concatenate([idx, nxt], axis=1)         # append, feed back in
        return idx                                            # <-- was a bare `return` 
        