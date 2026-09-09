import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", app_title="Minimal LLM -- NumPy")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return math, mo, np, plt


@app.cell
def _(mo):
    get_st, set_st = mo.state({
        "params": None, "adam": None, "step": 0, "losses": [],
        "tok": None, "cfg": None,
        "last_init_click": 0,
        "last_train_click": 0,
    })
    return get_st, set_st


@app.cell
def _(math, np):
    def softmax(x, ax=-1):
        x = x - np.max(x, axis=ax, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=ax, keepdims=True)

    def rmsnorm(x, eps=1e-5):
        return x * (np.mean(x * x, axis=-1, keepdims=True) + eps) ** -0.5

    def rmsnorm_backward(dy, x, eps=1e-5):
        s = (np.mean(x * x, axis=-1, keepdims=True) + eps) ** -0.5
        return s * (dy - x * s * s * np.mean(dy * x, axis=-1, keepdims=True))

    def display_token(i, i2c, bos):
        if i == bos:
            return "<BOS>"
        ch = i2c.get(i, "?")
        if ch == " ":
            return "<space>"
        if ch == "\n":
            return "<newline>"
        return ch

    def tokenize(docs):
        chars = sorted(set("".join(docs)))
        bos = len(chars)
        c2i = {c:i for i,c in enumerate(chars)}
        i2c = {i:c for i,c in enumerate(chars)}
        return c2i, i2c, bos, len(chars)+1

    def init_model(vocab, n_embd, n_head, n_layer, block_size, seed=42):
        rng = np.random.default_rng(seed)
        def g(r,c): return rng.normal(0, 0.08, (r,c))
        p = {"wte":g(vocab,n_embd), "wpe":g(block_size,n_embd), "lm_head":g(vocab,n_embd)}
        for li in range(n_layer):
            for name in ("wq","wk","wv","wo"):
                p[f"l{li}.{name}"] = g(n_embd,n_embd)
            p[f"l{li}.w1"] = g(4*n_embd,n_embd)
            p[f"l{li}.w2"] = g(n_embd,4*n_embd)
        return p

    def adam_init(p):
        return {k:[np.zeros_like(v),np.zeros_like(v)] for k,v in p.items()}

    def forward(tokens,p,n_head):
        inp=np.asarray(tokens[:-1],dtype=np.int32)
        tgt=np.asarray(tokens[1:],dtype=np.int32)
        T=len(inp); E=p["wte"].shape[1]; D=E//n_head
        L=sum(k.endswith(".wq") for k in p)
        if T<1 or T>p["wpe"].shape[0]:
            return None,None
        xp=p["wte"][inp]+p["wpe"][:T]
        x=rmsnorm(xp)
        caches=[]
        for li in range(L):
            c={"xi":x}
            xn=rmsnorm(x)
            Q=xn@p[f"l{li}.wq"].T
            K=xn@p[f"l{li}.wk"].T
            V=xn@p[f"l{li}.wv"].T
            Qh=Q.reshape(T,n_head,D).transpose(1,0,2)
            Kh=K.reshape(T,n_head,D).transpose(1,0,2)
            Vh=V.reshape(T,n_head,D).transpose(1,0,2)
            scores=Qh@Kh.transpose(0,2,1)/math.sqrt(D)
            mask=np.triu(np.ones((T,T),dtype=bool),1)[None]
            scores=np.where(mask,-1e9,scores)
            attn=softmax(scores)
            attended=(attn@Vh).transpose(1,0,2).reshape(T,E)
            x=x+attended@p[f"l{li}.wo"].T
            xim=x
            xnm=rmsnorm(x)
            h1=xnm@p[f"l{li}.w1"].T
            h=np.maximum(0,h1)
            x=x+h@p[f"l{li}.w2"].T
            c.update({
                "xna":xn,"Qh":Qh,"Kh":Kh,"Vh":Vh,
                "attn":attn,"attended":attended,
                "xim":xim,"xnm":xnm,"h1":h1,"h":h
            })
            caches.append(c)
        logits=x@p["lm_head"].T
        probs=softmax(logits)
        loss=-np.mean(np.log(probs[np.arange(T),tgt]+1e-12))
        return loss,{
            "inp":inp,"tgt":tgt,"T":T,
            "n_head":n_head,"head_dim":D,
            "xp":xp,"xf":x,"probs":probs,"layers":caches
        }

    def backward(p,c):
        T,H,D=c["T"],c["n_head"],c["head_dim"]
        inp,tgt=c["inp"],c["tgt"]
        g={k:np.zeros_like(v) for k,v in p.items()}
        dl=c["probs"].copy()
        dl[np.arange(T),tgt]-=1
        dl/=T
        g["lm_head"]+=dl.T@c["xf"]
        dx=dl@p["lm_head"]

        for li in reversed(range(len(c["layers"]))):
            z=c["layers"][li]

            g[f"l{li}.w2"]+=dx.T@z["h"]
            dh=dx@p[f"l{li}.w2"]
            dh1=dh*(z["h1"]>0)
            g[f"l{li}.w1"]+=dh1.T@z["xnm"]
            dx=dx+rmsnorm_backward(dh1@p[f"l{li}.w1"],z["xim"])

            g[f"l{li}.wo"]+=dx.T@z["attended"]
            da=dx@p[f"l{li}.wo"]
            dah=da.reshape(T,H,D).transpose(1,0,2)
            dVh=z["attn"].transpose(0,2,1)@dah
            datt=dah@z["Vh"].transpose(0,2,1)

            ds=z["attn"]*(datt-(datt*z["attn"]).sum(-1,keepdims=True))
            mask=np.triu(np.ones((T,T),dtype=bool),1)[None]
            ds=np.where(mask,0.0,ds)/math.sqrt(D)

            dQh=ds@z["Kh"]
            dKh=ds.transpose(0,2,1)@z["Qh"]

            dQ=dQh.transpose(1,0,2).reshape(T,-1)
            dK=dKh.transpose(1,0,2).reshape(T,-1)
            dV=dVh.transpose(1,0,2).reshape(T,-1)

            g[f"l{li}.wq"]+=dQ.T@z["xna"]
            g[f"l{li}.wk"]+=dK.T@z["xna"]
            g[f"l{li}.wv"]+=dV.T@z["xna"]

            dxna=(
                dQ@p[f"l{li}.wq"]+
                dK@p[f"l{li}.wk"]+
                dV@p[f"l{li}.wv"]
            )
            dx=dx+rmsnorm_backward(dxna,z["xi"])

        dxp=rmsnorm_backward(dx,c["xp"])
        np.add.at(g["wte"],inp,dxp)
        g["wpe"][:T]+=dxp
        return g

    def adam_step(p,g,state,step,lr,beta1=.85,beta2=.99,eps=1e-8):
        for k in p:
            m,v=state[k]
            m=beta1*m+(1-beta1)*g[k]
            v=beta2*v+(1-beta2)*g[k]**2
            mh=m/(1-beta1**(step+1))
            vh=v/(1-beta2**(step+1))
            p[k]-=lr*mh/(np.sqrt(vh)+eps)
            state[k]=[m,v]
        return p,state

    def explain_forward(p,ids,n_head,block_size,temperature=1.0):
        ids=list(ids[-block_size:])
        T=len(ids)
        E=p["wte"].shape[1]
        D=E//n_head
        L=sum(k.endswith(".wq") for k in p)

        x0=p["wte"][ids]+p["wpe"][:T]
        x=rmsnorm(x0)
        layers=[]

        for li in range(L):
            x_in=x.copy()
            xn=rmsnorm(x)
            Q=xn@p[f"l{li}.wq"].T
            K=xn@p[f"l{li}.wk"].T
            V=xn@p[f"l{li}.wv"].T

            Qh=Q.reshape(T,n_head,D).transpose(1,0,2)
            Kh=K.reshape(T,n_head,D).transpose(1,0,2)
            Vh=V.reshape(T,n_head,D).transpose(1,0,2)

            scores=Qh@Kh.transpose(0,2,1)/math.sqrt(D)
            mask=np.triu(np.ones((T,T),dtype=bool),1)[None]
            scores=np.where(mask,-1e9,scores)
            attn=softmax(scores)

            attended=(attn@Vh).transpose(1,0,2).reshape(T,E)
            x_attn=x+attended@p[f"l{li}.wo"].T

            xnm=rmsnorm(x_attn)
            h1=xnm@p[f"l{li}.w1"].T
            h=np.maximum(0,h1)
            x=x_attn+h@p[f"l{li}.w2"].T

            layers.append({
                "x_in":x_in,"xn":xn,"Q":Qh,"K":Kh,"V":Vh,
                "scores":scores,"attn":attn,"attended":attended,
                "x_attn":x_attn,"h1":h1,"h":h,"x_out":x
            })

        logits=x[-1]@p["lm_head"].T
        probs=softmax(logits/max(temperature,.01))
        return {
            "ids":ids,"embedding":x0,"layers":layers,
            "final":x,"logits":logits,"probs":probs
        }

    def next_probs(p,ids,n_head,block_size,temperature):
        return explain_forward(
            p,ids,n_head,block_size,temperature
        )["probs"]

    def generate(p,prompt_ids,bos,i2c,n_head,block_size,max_new,temperature):
        ids=list(prompt_ids)
        for _ in range(max_new):
            probs=next_probs(p,ids,n_head,block_size,temperature)
            nxt=int(np.random.choice(len(probs),p=probs))
            if nxt==bos:
                break
            ids.append(nxt)
        return "".join(i2c.get(t,"") for t in ids[len(prompt_ids):])

    return (
        adam_init,adam_step,backward,forward,generate,
        explain_forward,init_model,next_probs,tokenize,display_token
    )


@app.cell
def _(mo):
    mo.md("""# Minimal LLM -- NumPy

This starts from `ann_numpy.py` and makes the Transformer visible.

`character → embedding → attention → MLP → next-character probability`

Everything below is produced by the same small NumPy model used for training.
There is no autograd.""")


@app.cell
def _(mo):
    data="""hello world
hello there
hello world
how are you
how is the world
the world is small
the world is beautiful
we learn neural networks
we learn machine learning
machine learning is fun
"""
    expl_data=mo.ui.text_area(
        value=data,label="Training data (one document per line)",
        rows=8,full_width=True
    )
    expl_n_embd=mo.ui.slider(8,64,step=8,value=16,label="embedding size")
    expl_n_head=mo.ui.slider(1,8,value=4,label="attention heads")
    expl_n_layer=mo.ui.slider(1,4,value=1,label="transformer layers")
    expl_block=mo.ui.slider(8,64,step=8,value=32,label="context / block size")
    expl_lr=mo.ui.slider(1,50,value=10,label="learning rate (× 1e-3)")
    expl_steps=mo.ui.slider(10,500,step=10,value=100,label="steps per click")
    expl_prompt=mo.ui.text(value="the wor",label="Prompt")
    expl_temp=mo.ui.slider(1,20,value=5,label="temperature (× 0.1)")
    expl_maxnew=mo.ui.slider(10,100,step=10,value=30,label="new characters")
    expl_layer=mo.ui.slider(1,4,value=1,label="inspect layer")
    expl_head=mo.ui.slider(1,8,value=1,label="inspect head")
    expl_pos=mo.ui.slider(1,32,value=1,label="inspect token position")
    expl_init=mo.ui.button(label="Initialize",kind="success")
    expl_train=mo.ui.button(label="Train",kind="warn")
    expl_generate=mo.ui.button(label="Generate")
    return (
        expl_block,expl_data,expl_generate,expl_head,expl_init,expl_layer,
        expl_lr,expl_maxnew,expl_n_embd,expl_n_head,expl_n_layer,expl_pos,
        expl_prompt,expl_steps,expl_temp,expl_train
    )


@app.cell
def _(
    expl_block,expl_data,expl_head,expl_init,expl_layer,
    expl_lr,expl_n_embd,expl_n_head,expl_n_layer,expl_pos,
    expl_steps,expl_train,mo
):
    mo.vstack([
        mo.md("## Controls"),
        mo.md("### 1. Training data"),
        expl_data,
        mo.md("### 2. Model architecture"),
        mo.hstack([expl_n_embd, expl_n_head, expl_n_layer, expl_block], gap=1),
        mo.md("### 3. Train"),
        mo.hstack([expl_lr, expl_steps, expl_init, expl_train], gap=1),
        mo.md("### 4. Inspect the Transformer"),
        mo.hstack([expl_layer, expl_head, expl_pos], gap=1),
    ])



# State updates trigger reactive re-execution. Each button click is consumed
# once, so the state update cannot execute the same action again.
@app.cell
def _(
    adam_init, expl_block, expl_data, expl_init, expl_n_embd, expl_n_head,
    expl_n_layer, get_st, init_model, mo, set_st, tokenize
):
    _st = get_st()
    _click = expl_init.value

    mo.stop(
        _click == 0 or _click == _st["last_init_click"],
        None,
    )

    _docs = [x.strip() for x in expl_data.value.splitlines() if x.strip()]
    mo.stop(
        not _docs,
        mo.callout("Enter at least one training document.", kind="danger"),
    )

    _c2i, _i2c, _bos, _vocab = tokenize(_docs)
    _H = expl_n_head.value
    _E = (expl_n_embd.value // _H) * _H
    _L = expl_n_layer.value
    _B = expl_block.value
    _longest = max(len(x) for x in _docs)

    mo.stop(
        _longest + 1 > _B,
        mo.callout(
            f"Longest document has {_longest} characters. "
            f"Increase block size to at least {_longest + 1}.",
            kind="danger",
        ),
    )

    _p = init_model(_vocab, _E, _H, _L, _B)
    _adam = adam_init(_p)

    set_st({
        **_st,
        "params": _p,
        "adam": _adam,
        "step": 0,
        "losses": [],
        "tok": (_c2i, _i2c, _bos, _vocab),
        "cfg": (_E, _H, _L, _B),
        "last_init_click": _click,
    })

    mo.callout(
        f"Ready. {_vocab} characters | {_E} embedding | "
        f"{_H} heads | {_L} layer(s) | {_B} context",
        kind="success",
    )


@app.cell
def _(
    adam_step, backward, expl_data, expl_lr, expl_steps, expl_train,
    forward, get_st, mo, set_st
):
    _st = get_st()
    _click = expl_train.value

    mo.stop(
        _click == 0 or _click == _st["last_train_click"],
        None,
    )

    mo.stop(
        _st["params"] is None or _st["tok"] is None or _st["cfg"] is None,
        mo.callout("Initialize the model first.", kind="danger"),
    )

    _p = {k: v.copy() for k, v in _st["params"].items()}
    _adam = {k: [m.copy(), v.copy()] for k, (m, v) in _st["adam"].items()}
    _step = _st["step"]
    _losses = list(_st["losses"])
    _c2i, _i2c, _bos, _vocab = _st["tok"]
    _E, _H, _L, _B = _st["cfg"]
    _docs = [x.strip() for x in expl_data.value.splitlines() if x.strip()]

    mo.stop(
        not _docs,
        mo.callout("Enter at least one training document.", kind="danger"),
    )

    _unknown = sorted(
        set(c for _doc in _docs for c in _doc if c not in _c2i)
    )
    mo.stop(
        _unknown,
        mo.callout(
            f"Training data contains character(s) not in the initialized "
            f"vocabulary: {_unknown}. Click **Initialize** again.",
            kind="danger",
        ),
    )

    for _local in range(expl_steps.value):
        _doc = _docs[(_step + _local) % len(_docs)]
        _tokens = [_bos] + [_c2i[c] for c in _doc] + [_bos]
        _loss, _cache = forward(_tokens, _p, _H)

        if _cache is not None:
            _grads = backward(_p, _cache)
            _p, _adam = adam_step(
                _p, _grads, _adam, _step + _local, expl_lr.value * 1e-3
            )
            _losses.append(float(_loss))

    _new_step = _step + expl_steps.value

    set_st({
        **_st,
        "params": _p,
        "adam": _adam,
        "step": _new_step,
        "losses": _losses,
        "last_train_click": _click,
    })

    mo.callout(
        f"Trained {_new_step} steps | loss {_losses[-1]:.4f}",
        kind="success",
    )


@app.cell
def _(display_token, expl_generate, expl_maxnew, expl_prompt, expl_temp, get_st, mo, next_probs, np):
    _st = get_st()

    _playground = mo.vstack([
        mo.md("## Interactive playground"),
        mo.md("Enter a prompt, then inspect the next-character prediction or generate text."),
        mo.hstack([expl_prompt, expl_temp, expl_maxnew, expl_generate], gap=1),
    ])

    mo.stop(
        _st["params"] is None or _st["tok"] is None or _st["cfg"] is None,
        mo.vstack([_playground, mo.md("Initialize the model to inspect next-character probabilities.")]),
    )

    _c2i, _i2c, _bos, _vocab = _st["tok"]
    _E, _H, _L, _B = _st["cfg"]
    _unknown = sorted(set(c for c in expl_prompt.value if c not in _c2i))

    mo.stop(
        _unknown,
        mo.vstack([_playground, mo.callout(f"Unknown character(s): {_unknown}", kind="danger")]),
    )

    _ids = ([_bos] + [_c2i[c] for c in expl_prompt.value] if expl_prompt.value else [_bos])
    _probs = next_probs(_st["params"], _ids, _H, _B, expl_temp.value * 0.1)
    _order = np.argsort(_probs)[::-1][:min(12, _vocab)]

    mo.vstack([
        _playground,
        mo.md(
            "### Next-character prediction\n\n"
            + "\n".join(
                f"`{display_token(int(i), _i2c, _bos)}` — **{_probs[i]:.3f}**"
                for i in _order
            )
        ),
    ])



@app.cell
def _(
    display_token, explain_forward, expl_head, expl_layer, expl_pos,
    expl_prompt, expl_temp, get_st, mo, np
):
    _st = get_st()

    mo.stop(
        _st["params"] is None or _st["tok"] is None or _st["cfg"] is None,
        mo.md("Initialize the model to see the Transformer."),
    )

    _c2i, _i2c, _bos, _vocab = _st["tok"]
    _E, _H, _L, _B = _st["cfg"]
    _prompt = expl_prompt.value
    _unknown = sorted(set(c for c in _prompt if c not in _c2i))

    mo.stop(
        _unknown,
        mo.callout(f"Unknown character(s): {_unknown}", kind="danger"),
    )

    _ids = [_bos] + [_c2i[c] for c in _prompt] if _prompt else [_bos]
    _tr = explain_forward(
        _st["params"], _ids, _H, _B, expl_temp.value * 0.1
    )

    _T = len(_tr["ids"])
    _layer = min(expl_layer.value - 1, _L - 1)
    _head = min(expl_head.value - 1, _H - 1)
    _pos = min(expl_pos.value - 1, _T - 1)

    _tokens = [display_token(i, _i2c, _bos) for i in _tr["ids"]]

    _row = _tr["layers"][_layer]["attn"][_head, _pos]
    _top = np.argsort(_row)[::-1][:min(5, _T)]
    _attention_text = "\n".join(
        f"`{_tokens[i]}`  {_row[i]:.3f}" for i in _top
    )

    _h = _tr["layers"][_layer]["h"][_pos]
    _active = np.where(_h > 0)[0]
    _top_neurons = (
        _active[np.argsort(_h[_active])[::-1][:10]]
        if len(_active)
        else []
    )
    _mlp_text = (
        "\n".join(
            f"neuron {int(i)}  {_h[i]:.3f}" for i in _top_neurons
        )
        or "no active ReLU neurons"
    )

    _out = np.argsort(_tr["probs"])[::-1][:min(8, _vocab)]
    _output_text = "\n".join(
        f"`{display_token(int(i), _i2c, _bos)}`  **{_tr['probs'][i]:.3f}**"
        for i in _out
    )

    mo.vstack([
        mo.md(f"""### Transformer flow

**Input** `{''.join(_tokens)}`

```text
token IDs
    ↓
token + position embedding   {_T} × {_E}
    ↓
Transformer block {_layer + 1}
    ├── self-attention: {_H} heads
    └── MLP: {_E} → {4 * _E} → {_E}
    ↓
final representation → logits → softmax → next character
```"""),
        mo.hstack([
            mo.callout(
                mo.md(
                    f"### Attention\n"
                    f"Layer {_layer + 1}, head {_head + 1}, "
                    f"position {_pos + 1}\n\n{_attention_text}"
                ),
                kind="info",
            ),
            mo.callout(
                mo.md(f"### MLP\nTop active neurons\n\n{_mlp_text}"),
                kind="info",
            ),
            mo.callout(
                mo.md(f"### Output\n\n{_output_text}"),
                kind="info",
            ),
        ], gap=1),
    ])


@app.cell
def _(
    display_token, explain_forward, expl_head, expl_layer, expl_prompt,
    expl_temp, get_st, mo, plt
):
    _st = get_st()

    mo.stop(
        _st["params"] is None or _st["tok"] is None or _st["cfg"] is None,
        mo.md("Initialize the model to see attention."),
    )

    _c2i, _i2c, _bos, _vocab = _st["tok"]
    _E, _H, _L, _B = _st["cfg"]
    _prompt = expl_prompt.value

    mo.stop(
        any(c not in _c2i for c in _prompt),
        mo.md("Unknown character in prompt."),
    )

    _ids = [_bos] + [_c2i[c] for c in _prompt] if _prompt else [_bos]
    _tr = explain_forward(
        _st["params"], _ids, _H, _B, expl_temp.value * 0.1
    )
    _layer = min(expl_layer.value - 1, _L - 1)
    _head = min(expl_head.value - 1, _H - 1)
    _attn = _tr["layers"][_layer]["attn"][_head]
    _tokens = [display_token(i, _i2c, _bos) for i in _tr["ids"]]

    _fig, _ax = plt.subplots(
        figsize=(max(5, len(_tokens) * 0.55),
                 max(4, len(_tokens) * 0.45))
    )
    _im = _ax.imshow(_attn, vmin=0, vmax=1, aspect="auto")
    _ax.set_xticks(range(len(_tokens)), _tokens)
    _ax.set_yticks(range(len(_tokens)), _tokens)
    _ax.set_xlabel("keys: tokens being read")
    _ax.set_ylabel("queries: current token")
    _ax.set_title(
        f"Layer {_layer + 1}, head {_head + 1}: causal self-attention"
    )
    _fig.colorbar(_im, ax=_ax, label="attention weight")
    _fig.tight_layout()
    plt.close(_fig)
    _fig


@app.cell
def _(display_token, explain_forward, expl_prompt, expl_temp, get_st, mo, plt):
    _st = get_st()

    mo.stop(
        _st["params"] is None or _st["tok"] is None or _st["cfg"] is None,
        mo.md("Initialize the model to see embeddings."),
    )

    _c2i, _i2c, _bos, _vocab = _st["tok"]
    _E, _H, _L, _B = _st["cfg"]
    _prompt = expl_prompt.value

    mo.stop(
        any(c not in _c2i for c in _prompt),
        mo.md("Unknown character in prompt."),
    )

    _ids = [_bos] + [_c2i[c] for c in _prompt] if _prompt else [_bos]
    _tr = explain_forward(
        _st["params"], _ids, _H, _B, expl_temp.value * 0.1
    )
    _tokens = [display_token(i, _i2c, _bos) for i in _tr["ids"]]

    _fig, _ax = plt.subplots(
        figsize=(10, max(2.5, len(_tokens) * 0.42))
    )
    _im = _ax.imshow(_tr["embedding"], aspect="auto")
    _ax.set_yticks(range(len(_tokens)), _tokens)
    _ax.set_xlabel("embedding dimension")
    _ax.set_ylabel("token")
    _ax.set_title("Token + positional embedding")
    _fig.colorbar(_im, ax=_ax, label="value")
    _fig.tight_layout()
    plt.close(_fig)
    _fig


@app.cell
def _(
    explain_forward, expl_head, expl_layer, expl_pos, expl_prompt,
    expl_temp, get_st, mo, np, plt
):
    _st = get_st()

    mo.stop(
        _st["params"] is None or _st["tok"] is None or _st["cfg"] is None,
        mo.md("Initialize the model to see Q/K/V."),
    )

    _c2i, _i2c, _bos, _vocab = _st["tok"]
    _E, _H, _L, _B = _st["cfg"]
    _prompt = expl_prompt.value

    mo.stop(
        any(c not in _c2i for c in _prompt),
        mo.md("Unknown character in prompt."),
    )

    _ids = [_bos] + [_c2i[c] for c in _prompt] if _prompt else [_bos]
    _tr = explain_forward(
        _st["params"], _ids, _H, _B, expl_temp.value * 0.1
    )

    _layer = min(expl_layer.value - 1, _L - 1)
    _head = min(expl_head.value - 1, _H - 1)
    _pos = min(expl_pos.value - 1, len(_ids) - 1)

    _q = _tr["layers"][_layer]["Q"][_head, _pos]
    _k = _tr["layers"][_layer]["K"][_head]
    _v = _tr["layers"][_layer]["V"][_head]

    _q_norm = float(np.sqrt(np.sum(_q * _q)))
    _k_norm = float(np.mean(np.sqrt(np.sum(_k * _k, axis=1))))
    _v_norm = float(np.mean(np.sqrt(np.sum(_v * _v, axis=1))))

    _fig, _ax = plt.subplots(figsize=(6, 3))
    _ax.bar(["Q", "K norm", "V norm"], [_q_norm, _k_norm, _v_norm])
    _ax.set_ylabel("vector norm")
    _ax.set_title(
        f"Layer {_layer + 1}, head {_head + 1}, position {_pos + 1}: Q / K / V"
    )
    _ax.grid(True, axis="y", alpha=.2)
    _fig.tight_layout()
    plt.close(_fig)
    _fig


@app.cell
def _(get_st, mo, plt):
    _st = get_st()
    _losses = _st["losses"]

    if not _losses:
        _loss_output = mo.md(
            "### Training\nTrain the model to see the loss fall."
        )
    else:
        _fig, _ax = plt.subplots(figsize=(10, 3))
        _ax.plot(_losses, lw=.8)
        _ax.set(
            xlabel="training step",
            ylabel="cross-entropy",
            title=f"training loss | step {_st['step']}",
        )
        _ax.grid(True, alpha=.2)
        _fig.tight_layout()
        plt.close(_fig)
        _loss_output = _fig

    _loss_output


@app.cell
def _(
    expl_generate, expl_maxnew, expl_prompt, expl_temp, generate,
    get_st, mo
):
    _st = get_st()

    mo.stop(
        expl_generate.value == 0,
        mo.md("Click **Generate** to sample from the trained model."),
    )

    mo.stop(
        _st["params"] is None or _st["tok"] is None or _st["cfg"] is None,
        mo.callout("Initialize the model first.", kind="danger"),
    )

    _c2i, _i2c, _bos, _vocab = _st["tok"]
    _E, _H, _L, _B = _st["cfg"]
    _raw = expl_prompt.value
    _unknown = sorted(set(c for c in _raw if c not in _c2i))

    mo.stop(
        _unknown,
        mo.callout(f"Unknown character(s): {_unknown}", kind="danger"),
    )

    _prompt_ids = (
        [_bos] + [_c2i[c] for c in _raw]
        if _raw
        else [_bos]
    )
    _generated = generate(
        _st["params"],
        _prompt_ids,
        _bos,
        _i2c,
        _H,
        _B,
        expl_maxnew.value,
        expl_temp.value * 0.1,
    )

    mo.callout(
        mo.md(
            f"### `{_raw or '<BOS>'}` → "
            f"`{_generated or '<nothing>'}`"
        ),
        kind="success",
    )


@app.cell
def _(get_st, mo):
    _st = get_st()

    if _st["params"] is None or _st["tok"] is None or _st["cfg"] is None:
        _state_output = mo.md(
            "### Model state\nInitialize the model to see parameter counts."
        )
    else:
        _n = sum(v.size for v in _st["params"].values())
        _E, _H, _L, _B = _st["cfg"]
        _state_output = mo.md(f"""### Model state

**Parameters:** {_n:,}  
**Embedding:** {_E}  
**Heads:** {_H} × {_E // _H}  
**Layers:** {_L}  
**Context:** {_B}  
**Training steps:** {_st["step"]}""")

    _state_output


if __name__ == "__main__":
    app.run()
