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
    # One persistent state object.  Keep this in one cell so Initialize and
    # Train always operate on the same model.
    get_st, set_st = mo.state({
        "params": None,
        "adam": None,
        "step": 0,
        "losses": [],
        "tok": None,       # (char -> id, id -> char, BOS, vocabulary size)
        "cfg": None,       # (embedding, heads, layers, block size)
    })
    return get_st, set_st


@app.cell
def _(math, np):
    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def softmax(x, ax=-1):
        x = x - x.max(axis=ax, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=ax, keepdims=True)


    def rmsnorm(x, eps=1e-5):
        return x * (np.mean(x**2, axis=-1, keepdims=True) + eps) ** -0.5


    def rmsnorm_backward(dy, x, eps=1e-5):
        s = (np.mean(x**2, axis=-1, keepdims=True) + eps) ** -0.5
        return s * (dy - x * s**2 * np.mean(dy * x, axis=-1, keepdims=True))


    # -------------------------------------------------------------------------
    # Character tokenizer
    # -------------------------------------------------------------------------
    def tokenize(docs):
        chars = sorted(set("".join(docs)))
        bos = len(chars)
        c2i = {c: i for i, c in enumerate(chars)}
        i2c = {i: c for i, c in enumerate(chars)}
        return c2i, i2c, bos, len(chars) + 1


    # -------------------------------------------------------------------------
    # Model initialization
    # -------------------------------------------------------------------------
    def init_model(vocab, n_embd, n_head, n_layer, block_size, seed=42):
        rng = np.random.default_rng(seed)

        def g(rows, cols):
            return rng.normal(0, 0.08, (rows, cols))

        p = {
            "wte": g(vocab, n_embd),
            "wpe": g(block_size, n_embd),
            "lm_head": g(vocab, n_embd),
        }

        for li in range(n_layer):
            for name in ("wq", "wk", "wv", "wo"):
                p[f"l{li}.{name}"] = g(n_embd, n_embd)
            p[f"l{li}.w1"] = g(4 * n_embd, n_embd)
            p[f"l{li}.w2"] = g(n_embd, 4 * n_embd)

        return p


    def adam_init(p):
        return {k: [np.zeros_like(v), np.zeros_like(v)] for k, v in p.items()}


    # -------------------------------------------------------------------------
    # Forward pass
    # -------------------------------------------------------------------------
    def forward(tokens, p, n_head):
        inp = np.asarray(tokens[:-1], dtype=np.int32)
        tgt = np.asarray(tokens[1:], dtype=np.int32)
        T = len(inp)
        n_embd = p["wte"].shape[1]
        head_dim = n_embd // n_head
        n_layer = sum(k.endswith(".wq") for k in p)

        if T < 1 or T > p["wpe"].shape[0]:
            return None, None

        # token embedding + position embedding
        xp = p["wte"][inp] + p["wpe"][:T]
        x = rmsnorm(xp)

        caches = []

        for li in range(n_layer):
            c = {"xi": x}

            # ---------------- attention ----------------
            xn = rmsnorm(x)
            Q = xn @ p[f"l{li}.wq"].T
            K = xn @ p[f"l{li}.wk"].T
            V = xn @ p[f"l{li}.wv"].T

            Qh = Q.reshape(T, n_head, head_dim).transpose(1, 0, 2)
            Kh = K.reshape(T, n_head, head_dim).transpose(1, 0, 2)
            Vh = V.reshape(T, n_head, head_dim).transpose(1, 0, 2)

            scores = Qh @ Kh.transpose(0, 2, 1) / math.sqrt(head_dim)
            mask = np.triu(np.ones((T, T), dtype=bool), 1)[None]
            scores = np.where(mask, -1e9, scores)
            attn = softmax(scores)

            attended = (attn @ Vh).transpose(1, 0, 2).reshape(T, n_embd)
            x = x + attended @ p[f"l{li}.wo"].T

            # ---------------- MLP ----------------
            xim = x
            xnm = rmsnorm(x)
            h1 = xnm @ p[f"l{li}.w1"].T
            h = np.maximum(0, h1)
            x = x + h @ p[f"l{li}.w2"].T

            c.update({
                "xna": xn,
                "Qh": Qh, "Kh": Kh, "Vh": Vh,
                "attn": attn, "attended": attended,
                "xim": xim, "xnm": xnm,
                "h1": h1, "h": h,
            })
            caches.append(c)

        logits = x @ p["lm_head"].T
        probs = softmax(logits)
        loss = -np.mean(np.log(probs[np.arange(T), tgt] + 1e-12))

        cache = {
            "inp": inp, "tgt": tgt, "T": T,
            "n_head": n_head, "head_dim": head_dim,
            "xp": xp, "xf": x, "probs": probs,
            "layers": caches,
        }
        return loss, cache


    # -------------------------------------------------------------------------
    # Explicit backpropagation
    # -------------------------------------------------------------------------
    def backward(p, c):
        T = c["T"]
        n_head = c["n_head"]
        head_dim = c["head_dim"]
        n_layer = len(c["layers"])
        inp, tgt = c["inp"], c["tgt"]

        g = {k: np.zeros_like(v) for k, v in p.items()}

        # softmax + cross entropy
        dlogits = c["probs"].copy()
        dlogits[np.arange(T), tgt] -= 1
        dlogits /= T

        g["lm_head"] += dlogits.T @ c["xf"]
        dx = dlogits @ p["lm_head"]

        for li in reversed(range(n_layer)):
            lc = c["layers"][li]

            # ================================================================
            # MLP backward
            # ================================================================
            g[f"l{li}.w2"] += dx.T @ lc["h"]
            dh = dx @ p[f"l{li}.w2"]
            dh1 = dh * (lc["h1"] > 0)
            g[f"l{li}.w1"] += dh1.T @ lc["xnm"]
            dxnm = dh1 @ p[f"l{li}.w1"]

            # residual path + normalized path
            dx = dx + rmsnorm_backward(dxnm, lc["xim"])

            # ================================================================
            # Attention backward
            # ================================================================
            g[f"l{li}.wo"] += dx.T @ lc["attended"]
            dattended = dx @ p[f"l{li}.wo"]

            dah = dattended.reshape(T, n_head, head_dim).transpose(1, 0, 2)
            dVh = lc["attn"].transpose(0, 2, 1) @ dah
            datt = dah @ lc["Vh"].transpose(0, 2, 1)

            # softmax Jacobian-vector product
            dscores = lc["attn"] * (
                datt - (datt * lc["attn"]).sum(-1, keepdims=True)
            )

            mask = np.triu(np.ones((T, T), dtype=bool), 1)[None]
            dscores = np.where(mask, 0.0, dscores) / math.sqrt(head_dim)

            dQh = dscores @ lc["Kh"]
            dKh = dscores.transpose(0, 2, 1) @ lc["Qh"]

            dQ = dQh.transpose(1, 0, 2).reshape(T, -1)
            dK = dKh.transpose(1, 0, 2).reshape(T, -1)
            dV = dVh.transpose(1, 0, 2).reshape(T, -1)

            g[f"l{li}.wq"] += dQ.T @ lc["xna"]
            g[f"l{li}.wk"] += dK.T @ lc["xna"]
            g[f"l{li}.wv"] += dV.T @ lc["xna"]

            dxna = (
                dQ @ p[f"l{li}.wq"]
                + dK @ p[f"l{li}.wk"]
                + dV @ p[f"l{li}.wv"]
            )

            dx = dx + rmsnorm_backward(dxna, lc["xi"])

        # embeddings
        dxp = rmsnorm_backward(dx, c["xp"])
        np.add.at(g["wte"], inp, dxp)
        g["wpe"][:T] += dxp

        return g


    # -------------------------------------------------------------------------
    # Adam
    # -------------------------------------------------------------------------
    def adam_step(p, grads, state, step, lr, beta1=0.85, beta2=0.99, eps=1e-8):
        for k in p:
            m, v = state[k]
            m = beta1 * m + (1 - beta1) * grads[k]
            v = beta2 * v + (1 - beta2) * grads[k] ** 2

            mh = m / (1 - beta1 ** (step + 1))
            vh = v / (1 - beta2 ** (step + 1))

            p[k] -= lr * mh / (np.sqrt(vh) + eps)
            state[k] = [m, v]

        return p, state


    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    def next_probs(p, ids, n_head, block_size, temperature):
        ids = list(ids[-block_size:])
        T = len(ids)
        n_embd = p["wte"].shape[1]
        head_dim = n_embd // n_head
        n_layer = sum(k.endswith(".wq") for k in p)

        x = rmsnorm(p["wte"][ids] + p["wpe"][:T])

        for li in range(n_layer):
            xn = rmsnorm(x)
            Q = xn @ p[f"l{li}.wq"].T
            K = xn @ p[f"l{li}.wk"].T
            V = xn @ p[f"l{li}.wv"].T

            Qh = Q.reshape(T, n_head, head_dim).transpose(1, 0, 2)
            Kh = K.reshape(T, n_head, head_dim).transpose(1, 0, 2)
            Vh = V.reshape(T, n_head, head_dim).transpose(1, 0, 2)

            scores = Qh @ Kh.transpose(0, 2, 1) / math.sqrt(head_dim)
            mask = np.triu(np.ones((T, T), dtype=bool), 1)[None]
            scores = np.where(mask, -1e9, scores)
            attn = softmax(scores)

            attended = (attn @ Vh).transpose(1, 0, 2).reshape(T, n_embd)
            x = x + attended @ p[f"l{li}.wo"].T
            x = x + np.maximum(0, rmsnorm(x) @ p[f"l{li}.w1"].T) @ p[f"l{li}.w2"].T

        return softmax(x[-1] @ p["lm_head"].T / max(temperature, 0.01))


    def generate(p, prompt_ids, bos, i2c, n_head, block_size, max_new, temperature):
        ids = list(prompt_ids)
        for _ in range(max_new):
            _probs = next_probs(p, ids, n_head, block_size, temperature)
            nxt = int(np.random.choice(len(_probs), p=_probs))
            if nxt == bos:
                break
            ids.append(nxt)

        return "".join(i2c.get(t, "") for t in ids[len(prompt_ids):])


    return (
        adam_init,
        adam_step,
        backward,
        forward,
        generate,
        init_model,
        next_probs,
        tokenize,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Minimal LLM -- NumPy

    This notebook starts from the small `ann_numpy.py` and adds the pieces one
    at a time.

    ```text
    ann_numpy.py

        input → hidden → output

    this model

        characters → embeddings → attention → MLP → next character
    ```

    The training objective is still simple:

    ```text
    predict the next character
    ```

    With a context of 8, for example:

    ```text
    "the wor" → "l"
    "he world" → " "
    ```

    The important conceptual change is that **attention lets every position
    look at the preceding positions**. The causal mask prevents it from seeing
    the future.

    The backward pass is written explicitly with NumPy. There is no autograd.
    """)


@app.cell
def _(mo):
    default_data = """hello world
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

    data_ta = mo.ui.text_area(
        value=default_data,
        label="Training data (one document per line)",
        rows=10,
        full_width=True,
    )

    n_embd_sl = mo.ui.slider(8, 64, step=8, value=16, label="embedding size")
    n_head_sl = mo.ui.slider(1, 8, step=1, value=4, label="attention heads")
    n_layer_sl = mo.ui.slider(1, 4, step=1, value=1, label="transformer layers")
    block_sl = mo.ui.slider(8, 64, step=8, value=32, label="context / block size")

    lr_sl = mo.ui.slider(1, 50, step=1, value=10, label="learning rate (× 1e-3)")
    steps_sl = mo.ui.slider(10, 500, step=10, value=100, label="steps per click")

    prompt_in = mo.ui.text(
        value="",
        placeholder="leave blank = BOS",
        label="Prompt",
    )
    temp_sl = mo.ui.slider(1, 20, step=1, value=5, label="temperature (× 0.1)")
    maxnew_sl = mo.ui.slider(10, 200, step=10, value=50, label="new characters")

    init_btn = mo.ui.button(label="Initialize", kind="success")
    train_btn = mo.ui.button(label="Train", kind="warn")
    gen_btn = mo.ui.button(label="Generate")

    return (
        block_sl,
        data_ta,
        gen_btn,
        init_btn,
        lr_sl,
        maxnew_sl,
        n_embd_sl,
        n_head_sl,
        n_layer_sl,
        prompt_in,
        steps_sl,
        temp_sl,
        train_btn,
    )


@app.cell
def _(
    block_sl,
    data_ta,
    gen_btn,
    init_btn,
    lr_sl,
    maxnew_sl,
    mo,
    n_embd_sl,
    n_head_sl,
    n_layer_sl,
    prompt_in,
    steps_sl,
    temp_sl,
    train_btn,
):
    mo.hstack([
        mo.vstack([mo.md("### Data"), data_ta], align="start"),
        mo.vstack([
            mo.md("### Architecture"),
            n_embd_sl, n_head_sl, n_layer_sl, block_sl,
            mo.md("### Training"),
            lr_sl, steps_sl,
            mo.hstack([init_btn, train_btn]),
        ], align="start"),
        mo.vstack([
            mo.md("### Generation"),
            prompt_in, temp_sl, maxnew_sl, gen_btn,
        ], align="start"),
    ], gap=2)


@app.cell
def _(
    data_ta,
    get_st,
    init_btn,
    init_model,
    mo,
    n_embd_sl,
    n_head_sl,
    n_layer_sl,
    block_sl,
    set_st,
    tokenize,
):
    mo.stop(
        init_btn.value == 0,
        mo.callout("Click **Initialize** to build the model.", kind="info"),
    )

    _docs = [x.strip() for x in data_ta.value.splitlines() if x.strip()]
    mo.stop(
        not _docs,
        mo.callout("Enter at least one training document.", kind="danger"),
    )

    _c2i, _i2c, _bos, _vocab = tokenize(_docs)

    _n_head = n_head_sl.value
    _n_embd = (n_embd_sl.value // _n_head) * _n_head
    _n_layer = n_layer_sl.value
    _block = block_sl.value

    # Each document contributes BOS + characters + BOS, so block must be
    # large enough to contain at least one prediction target.
    if max(len(d) for d in _docs) + 1 > _block:
        _longest = max(len(d) for d in _docs)
        mo.stop(
            True,
            mo.callout(
                f"Longest document has {_longest} characters. "
                f"Increase block size to at least {_longest + 1}.",
                kind="danger",
            ),
        )

    _params = init_model(_vocab, _n_embd, _n_head, _n_layer, _block)
    _adam = adam_init(_params)
    _n_params = sum(v.size for v in _params.values())

    set_st({
        "params": _params,
        "adam": _adam,
        "step": 0,
        "losses": [],
        "tok": (_c2i, _i2c, _bos, _vocab),
        "cfg": (_n_embd, _n_head, _n_layer, _block),
    })

    mo.callout(
        f"**Ready.** vocab {_vocab} | embedding {_n_embd} | heads {_n_head} | "
        f"layers {_n_layer} | block {_block} | **{_n_params:,} parameters**",
        kind="success",
    )


@app.cell
def _(
    adam_step,
    backward,
    data_ta,
    forward,
    get_st,
    lr_sl,
    mo,
    set_st,
    steps_sl,
    train_btn,
):
    mo.stop(
        train_btn.value == 0,
        mo.callout("Click **Train** to run gradient steps.", kind="info"),
    )

    _st = get_st()
    mo.stop(
        _st["params"] is None,
        mo.callout("Initialize the model first.", kind="danger"),
    )

    _p = {k: v.copy() for k, v in _st["params"].items()}
    _adam = {k: [m.copy(), v.copy()] for k, (m, v) in _st["adam"].items()}
    _step = _st["step"]
    _losses = list(_st["losses"])

    _c2i, _i2c, _bos, _vocab = _st["tok"]
    _n_embd, _n_head, _n_layer, _block = _st["cfg"]
    _docs = [x.strip() for x in data_ta.value.splitlines() if x.strip()]

    _lr = lr_sl.value * 1e-3
    _n_steps = steps_sl.value

    for _local_step in range(_n_steps):
        _doc = _docs[(_step + _local_step) % len(_docs)]
        _tokens = [_bos] + [_c2i[c] for c in _doc] + [_bos]

        _loss, _cache = forward(_tokens, _p, _n_head)
        if _cache is None:
            continue

        _grads = backward(_p, _cache)
        _p, _adam = adam_step(_p, _grads, _adam, _step + _local_step, _lr)
        _losses.append(float(_loss))

    set_st({
        **_st,
        "params": _p,
        "adam": _adam,
        "step": _step + _n_steps,
        "losses": _losses,
    })

    mo.callout(
        f"Trained **{_n_steps}** steps | total **{_step + _n_steps}** | "
        f"loss **{_losses[-1]:.4f}**",
        kind="success",
    )


@app.cell
def _(get_st, math, mo, np, plt):
    _st = get_st()
    _losses = _st["losses"]

    if not _losses:
        mo.md("*(loss curve appears here after training)*")
    elif _st["tok"] is None:
        mo.md("*(initialize the model first)*")
    else:
        _fig, _ax = plt.subplots(figsize=(9, 3))
        _ax.plot(_losses, lw=0.7, label="loss")

        if len(_losses) >= 25:
            _ma = np.convolve(_losses, np.ones(25) / 25, mode="valid")
            _ax.plot(range(24, len(_losses)), _ma, lw=1.8, label="moving average")

        _vocab = _st["tok"][3]
        _ax.axhline(
            math.log(_vocab),
            lw=0.8,
            ls="--",
            label=f"uniform prediction = log({_vocab})",
        )

        _ax.set(
            xlabel="training step",
            ylabel="cross-entropy",
            title=f"step {_st['step']} | last {_losses[-1]:.4f} | min {min(_losses):.4f}",
        )
        _ax.legend(fontsize=8)
        _ax.grid(True, alpha=0.2)
        plt.tight_layout()
        _fig


@app.cell
def _(get_st, mo, next_probs, np, plt, prompt_in, temp_sl):
    _st = get_st()

    mo.stop(
        _st["params"] is None or _st["tok"] is None or _st["cfg"] is None,
        mo.md("*(initialize the model to inspect next-character probabilities)*"),
    )

    _c2i, _i2c, _bos, _vocab = _st["tok"]
    _n_embd, _n_head, _n_layer, _block = _st["cfg"]
    _prompt = prompt_in.value

    _unknown = sorted(set(c for c in _prompt if c not in _c2i))
    mo.stop(
        _unknown,
        mo.callout(f"Unknown character(s): {_unknown}", kind="danger"),
    )

    _ids = [_bos] + [_c2i[c] for c in _prompt] if _prompt else [_bos]
    _probs = next_probs(
        _st["params"], _ids, _n_head, _block, temp_sl.value * 0.1
    )

    _order = np.argsort(_probs)[::-1][:min(12, _vocab)]
    _labels = [repr(_i2c.get(int(i), "<BOS>")) for i in _order]

    _fig, _ax = plt.subplots(figsize=(9, 3))
    _ax.bar(_labels, _probs[_order])
    _ax.set(
        xlabel="next token",
        ylabel="probability",
        title=f"P(next character | `{_prompt or '<BOS>'}`)",
    )
    _ax.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    _fig


@app.cell
def _(
    gen_btn,
    generate,
    get_st,
    maxnew_sl,
    mo,
    prompt_in,
    temp_sl,
):
    mo.stop(
        gen_btn.value == 0,
        mo.callout("Enter a prompt and click **Generate**.", kind="info"),
    )

    _st = get_st()
    mo.stop(
        _st["params"] is None or _st["tok"] is None or _st["cfg"] is None,
        mo.callout("Initialize the model first.", kind="danger"),
    )

    _c2i, _i2c, _bos, _vocab = _st["tok"]
    _n_embd, _n_head, _n_layer, _block = _st["cfg"]
    _raw = prompt_in.value

    _unknown = sorted(set(c for c in _raw if c not in _c2i))
    mo.stop(
        _unknown,
        mo.callout(f"Unknown character(s): {_unknown}", kind="danger"),
    )

    _prompt_ids = [_bos] + [_c2i[c] for c in _raw] if _raw else [_bos]
    _generated = generate(
        _st["params"],
        _prompt_ids,
        _bos,
        _i2c,
        _n_head,
        _block,
        maxnew_sl.value,
        temp_sl.value * 0.1,
    )

    mo.vstack([
        mo.md(f"**Prompt:** `{_raw or '<BOS>'}`"),
        mo.callout(mo.md(f"## `{_generated or '<nothing>'}`"), kind="info"),
    ])


@app.cell
def _(get_st, mo):
    _st = get_st()

    if _st["params"] is None:
        mo.md("### Model anatomy\nInitialize the model to see its dimensions.")
    else:
        _n_embd, _n_head, _n_layer, _block = _st["cfg"]
        _vocab = _st["tok"][3]

        mo.md(f"""
        ### Model anatomy

        **Vocabulary:** {_vocab} characters

        **Context:** {_block} positions

        **Embedding:** {_n_embd} numbers per character

        **Attention:** {_n_head} heads × {_n_embd // _n_head} numbers per head

        **Transformer blocks:** {_n_layer}

        **Training step:** {_st["step"]}

        The information flow is:

        ```text
        character IDs
              ↓
        token embedding + position embedding
              ↓
        RMSNorm
              ↓
        causal self-attention
              ↓
        residual connection
              ↓
        RMSNorm → linear → ReLU → linear
              ↓
        residual connection
              ↓
        next-character probabilities
        ```
        """)


if __name__ == "__main__":
    app.run()
