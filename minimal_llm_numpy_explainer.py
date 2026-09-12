import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", app_title="Minimal LLM -- NumPy, step by step")


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
        "params": None,
        "adam": None,
        "step": 0,
        "losses": [],
        "tok": None,
        "cfg": None,
        "gen_ids": None,
        "gen_prompt": "",
        "generated": "",
        "last_sample": None,
        "last_update": None,
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
        return s * (
            dy - x * s * s * np.mean(dy * x, axis=-1, keepdims=True)
        )

    def tokenize(docs):
        chars = sorted(set("".join(docs)))
        bos = len(chars)
        c2i = {c: i for i, c in enumerate(chars)}
        i2c = {i: c for i, c in enumerate(chars)}
        return c2i, i2c, bos, len(chars) + 1

    def display_token(i, i2c, bos):
        if i == bos:
            return "<BOS>"
        ch = i2c.get(int(i), "?")
        if ch == " ":
            return "<space>"
        if ch == "\n":
            return "<newline>"
        if ch == "\t":
            return "<tab>"
        return ch

    def visible_text(text):
        return (
            str(text)
            .replace(" ", "<space>")
            .replace("\n", "<newline>")
            .replace("\t", "<tab>")
        )

    def init_model(vocab, n_embd, n_head, n_layer, block_size, seed=42):
        rng = np.random.default_rng(seed)

        def g(r, c):
            return rng.normal(0, 0.08, (r, c))

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
        return {
            k: [np.zeros_like(v), np.zeros_like(v)]
            for k, v in p.items()
        }

    def forward(tokens, p, n_head):
        inp = np.asarray(tokens[:-1], dtype=np.int32)
        tgt = np.asarray(tokens[1:], dtype=np.int32)

        T = len(inp)
        E = p["wte"].shape[1]
        D = E // n_head
        L = sum(k.endswith(".wq") for k in p)

        if T < 1 or T > p["wpe"].shape[0]:
            return None, None

        xp = p["wte"][inp] + p["wpe"][:T]
        x = rmsnorm(xp)
        caches = []

        for li in range(L):
            c = {"xi": x.copy()}

            xn = rmsnorm(x)
            Q = xn @ p[f"l{li}.wq"].T
            K = xn @ p[f"l{li}.wk"].T
            V = xn @ p[f"l{li}.wv"].T

            Qh = Q.reshape(T, n_head, D).transpose(1, 0, 2)
            Kh = K.reshape(T, n_head, D).transpose(1, 0, 2)
            Vh = V.reshape(T, n_head, D).transpose(1, 0, 2)

            scores = Qh @ Kh.transpose(0, 2, 1) / math.sqrt(D)
            mask = np.triu(np.ones((T, T), dtype=bool), 1)[None]
            scores = np.where(mask, -1e9, scores)
            attn = softmax(scores)

            attended = (attn @ Vh).transpose(1, 0, 2).reshape(T, E)
            x_attn = x + attended @ p[f"l{li}.wo"].T

            xnm = rmsnorm(x_attn)
            h1 = xnm @ p[f"l{li}.w1"].T
            h = np.maximum(0, h1)
            x = x_attn + h @ p[f"l{li}.w2"].T

            c.update({
                "xna": xn,
                "Qh": Qh,
                "Kh": Kh,
                "Vh": Vh,
                "attn": attn,
                "attended": attended,
                "xim": x_attn,
                "xnm": xnm,
                "h1": h1,
                "h": h,
                "xout": x.copy(),
            })
            caches.append(c)

        logits = x @ p["lm_head"].T
        probs = softmax(logits)
        loss = -np.mean(np.log(probs[np.arange(T), tgt] + 1e-12))

        return loss, {
            "inp": inp,
            "tgt": tgt,
            "T": T,
            "n_head": n_head,
            "head_dim": D,
            "xp": xp,
            "xf": x,
            "probs": probs,
            "layers": caches,
        }

    def backward(p, c):
        T, H, D = c["T"], c["n_head"], c["head_dim"]
        inp, tgt = c["inp"], c["tgt"]
        g = {k: np.zeros_like(v) for k, v in p.items()}

        dl = c["probs"].copy()
        dl[np.arange(T), tgt] -= 1
        dl /= T

        g["lm_head"] += dl.T @ c["xf"]
        dx = dl @ p["lm_head"]

        for li in reversed(range(len(c["layers"]))):
            z = c["layers"][li]

            g[f"l{li}.w2"] += dx.T @ z["h"]
            dh = dx @ p[f"l{li}.w2"]
            dh1 = dh * (z["h1"] > 0)

            g[f"l{li}.w1"] += dh1.T @ z["xnm"]
            dx = dx + rmsnorm_backward(
                dh1 @ p[f"l{li}.w1"], z["xim"]
            )

            g[f"l{li}.wo"] += dx.T @ z["attended"]
            da = dx @ p[f"l{li}.wo"]
            dah = da.reshape(T, H, D).transpose(1, 0, 2)

            dVh = z["attn"].transpose(0, 2, 1) @ dah
            datt = dah @ z["Vh"].transpose(0, 2, 1)

            ds = z["attn"] * (
                datt - (datt * z["attn"]).sum(-1, keepdims=True)
            )
            mask = np.triu(np.ones((T, T), dtype=bool), 1)[None]
            ds = np.where(mask, 0.0, ds) / math.sqrt(D)

            dQh = ds @ z["Kh"]
            dKh = ds.transpose(0, 2, 1) @ z["Qh"]

            dQ = dQh.transpose(1, 0, 2).reshape(T, -1)
            dK = dKh.transpose(1, 0, 2).reshape(T, -1)
            dV = dVh.transpose(1, 0, 2).reshape(T, -1)

            g[f"l{li}.wq"] += dQ.T @ z["xna"]
            g[f"l{li}.wk"] += dK.T @ z["xna"]
            g[f"l{li}.wv"] += dV.T @ z["xna"]

            dxna = (
                dQ @ p[f"l{li}.wq"]
                + dK @ p[f"l{li}.wk"]
                + dV @ p[f"l{li}.wv"]
            )
            dx = dx + rmsnorm_backward(dxna, z["xi"])

        dxp = rmsnorm_backward(dx, c["xp"])
        np.add.at(g["wte"], inp, dxp)
        g["wpe"][:T] += dxp
        return g

    def adam_step(
        p, g, state, step, lr,
        beta1=.85, beta2=.99, eps=1e-8
    ):
        delta_norms = {}
        grad_norms = {}

        for k in p:
            old = p[k].copy()
            m, v = state[k]

            m = beta1 * m + (1 - beta1) * g[k]
            v = beta2 * v + (1 - beta2) * g[k] ** 2

            mh = m / (1 - beta1 ** (step + 1))
            vh = v / (1 - beta2 ** (step + 1))

            p[k] -= lr * mh / (np.sqrt(vh) + eps)
            state[k] = [m, v]

            grad_norms[k] = float(np.linalg.norm(g[k]))
            delta_norms[k] = float(np.linalg.norm(p[k] - old))

        return p, state, grad_norms, delta_norms

    def explain_forward(p, ids, n_head, block_size, temperature=1.0):
        ids = list(ids[-block_size:])
        T = len(ids)
        E = p["wte"].shape[1]
        D = E // n_head
        L = sum(k.endswith(".wq") for k in p)

        x0 = p["wte"][ids] + p["wpe"][:T]
        x = rmsnorm(x0)
        layers = []

        for li in range(L):
            x_in = x.copy()
            xn = rmsnorm(x)

            Q = xn @ p[f"l{li}.wq"].T
            K = xn @ p[f"l{li}.wk"].T
            V = xn @ p[f"l{li}.wv"].T

            Qh = Q.reshape(T, n_head, D).transpose(1, 0, 2)
            Kh = K.reshape(T, n_head, D).transpose(1, 0, 2)
            Vh = V.reshape(T, n_head, D).transpose(1, 0, 2)

            scores = Qh @ Kh.transpose(0, 2, 1) / math.sqrt(D)
            mask = np.triu(np.ones((T, T), dtype=bool), 1)[None]
            scores = np.where(mask, -1e9, scores)
            attn = softmax(scores)

            attended = (attn @ Vh).transpose(1, 0, 2).reshape(T, E)
            attn_proj = attended @ p[f"l{li}.wo"].T
            x_attn = x + attn_proj

            xnm = rmsnorm(x_attn)
            h1 = xnm @ p[f"l{li}.w1"].T
            h = np.maximum(0, h1)
            mlp_proj = h @ p[f"l{li}.w2"].T
            x = x_attn + mlp_proj

            layers.append({
                "x_in": x_in,
                "xn": xn,
                "Q": Qh,
                "K": Kh,
                "V": Vh,
                "scores": scores,
                "attn": attn,
                "attended": attended,
                "attn_proj": attn_proj,
                "x_attn": x_attn,
                "xnm": xnm,
                "h1": h1,
                "h": h,
                "mlp_proj": mlp_proj,
                "x_out": x.copy(),
            })

        logits = x[-1] @ p["lm_head"].T
        probs = softmax(logits / max(temperature, .01))

        return {
            "ids": ids,
            "embedding": x0,
            "layers": layers,
            "final": x,
            "logits": logits,
            "probs": probs,
        }

    def next_probs(p, ids, n_head, block_size, temperature):
        return explain_forward(
            p, ids, n_head, block_size, temperature
        )["probs"]

    def matrix_preview(a, rows=4, cols=8, digits=3):
        a = np.asarray(a)
        if a.ndim == 1:
            return np.array2string(
                a[:cols], precision=digits, suppress_small=True
            )
        return np.array2string(
            a[:rows, :cols],
            precision=digits,
            suppress_small=True,
            max_line_width=120,
        )

    return (
        adam_init, adam_step, backward, display_token, explain_forward, forward,
        init_model, matrix_preview, next_probs, softmax, tokenize, visible_text
    )


@app.cell
def _(mo):
    mo.md(r"""# Minimal LLM -- NumPy

A small character-level Transformer where the same NumPy matrices are used for:

\[
\text{training}
\rightarrow
\text{parameter updates}
\rightarrow
\text{prompt}
\rightarrow
\text{one next token}
\rightarrow
\text{repeat}
\]

There is no autograd. The backward pass and Adam updates are written explicitly.""")


@app.cell
def _(mo):
    _data = """hello world
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

    expl_data = mo.ui.text_area(
        value=_data,
        label="Training data: one document per line",
        rows=8,
        full_width=True,
    )
    expl_n_embd = mo.ui.slider(
        8, 64, step=8, value=16, label="embedding size"
    )
    expl_n_head = mo.ui.slider(
        1, 8, value=4, label="attention heads"
    )
    expl_n_layer = mo.ui.slider(
        1, 4, value=1, label="transformer layers"
    )
    expl_block = mo.ui.slider(
        8, 64, step=8, value=32, label="context / block size"
    )
    expl_lr = mo.ui.slider(
        1, 50, value=10, label="learning rate × 10⁻³"
    )
    expl_steps = mo.ui.slider(
        10, 500, step=10, value=100, label="steps per click"
    )
    expl_prompt = mo.ui.text(
        value="the world",
        label="Prompt",
        full_width=True,
    )
    expl_temp = mo.ui.slider(
        1, 20, value=5, label="temperature × 0.1"
    )
    expl_greedy = mo.ui.checkbox(
        value=True,
        label="choose highest-probability token"
    )
    expl_layer = mo.ui.slider(
        1, 4, value=1, label="inspect layer"
    )
    expl_head = mo.ui.slider(
        1, 8, value=1, label="inspect head"
    )
    expl_pos = mo.ui.slider(
        1, 32, value=1, label="inspect token position"
    )
    # Action buttons use run_button so they execute only when clicked.
    # This is important because the action cells also update mo.state; an
    # ordinary reactive button would make those state updates look like a new click.
    expl_init = mo.ui.run_button(label="Initialize", kind="success")
    expl_train = mo.ui.run_button(label="Train", kind="warn")
    expl_start = mo.ui.run_button(label="Start from prompt", kind="success")
    expl_next = mo.ui.run_button(label="Generate one token")
    return (
        expl_block, expl_data, expl_greedy, expl_head, expl_init,
        expl_layer, expl_lr, expl_n_embd, expl_n_head, expl_n_layer,
        expl_next, expl_pos, expl_prompt, expl_start, expl_steps,
        expl_temp, expl_train
    )


@app.cell
def _(
    expl_block, expl_data, expl_greedy, expl_head, expl_init, expl_layer,
    expl_lr, expl_n_embd, expl_n_head, expl_n_layer, expl_next, expl_pos,
    expl_prompt, expl_start, expl_steps, expl_temp, expl_train, mo
):
    mo.vstack([
        mo.md("## 1. Build and train"),
        mo.callout(
            mo.md(
                "Enter one training example per line. Then click **Initialize** once. "
                "After initialization, click **Train** to run the selected number of "
                "explicit forward/backward/Adam steps. If you change the architecture "
                "or training vocabulary, click **Initialize** again before training."
            ),
            kind="info",
        ),
        expl_data,
        mo.md("### Architecture"),
        mo.hstack(
            [expl_n_embd, expl_n_head, expl_n_layer, expl_block],
            gap=1,
        ),
        mo.hstack(
            [expl_lr, expl_steps, expl_init, expl_train],
            gap=1,
        ),
        mo.md("## 2. Predict and generate"),
        mo.hstack(
            [expl_prompt, expl_temp],
            widths="equal",
            gap=1,
        ),
        mo.hstack(
            [expl_greedy, expl_start, expl_next],
            gap=1,
        ),
        mo.callout(
            mo.md(
                "**How to use this section**\n\n"
                "1. Edit **Prompt**.\n"
                "2. Click **Start from prompt** — this copies the prompt into the model's active context.\n"
                "3. Click **Generate one token** — exactly one new character is predicted and appended.\n"
                "4. Repeat step 3 to watch the text grow one character at a time.\n\n"
                "The **Current generation state** below shows the prompt field and the active context separately."
            ),
            kind="info",
        ),
        mo.md("## 3. Inspect one position"),
        mo.hstack([expl_layer, expl_head, expl_pos], gap=1),
    ])


@app.cell
def _(
    adam_init, expl_block, expl_data, expl_init, expl_n_embd, expl_n_head,
    expl_n_layer, get_st, init_model, mo, set_st, tokenize
):
    if not expl_init.value:
        _init_output = None
    else:
        _st = get_st()
        _docs = [x.strip() for x in expl_data.value.splitlines() if x.strip()]
        if not _docs:
            _init_output = mo.callout("Enter at least one training document.", kind="danger")
        else:
            _c2i, _i2c, _bos, _vocab = tokenize(_docs)
            _H = expl_n_head.value
            _E = (expl_n_embd.value // _H) * _H
            _L = expl_n_layer.value
            _B = expl_block.value
            _longest = max(len(x) for x in _docs)
            if _longest + 1 > _B:
                _init_output = mo.callout(
                    f"Longest document has {_longest} characters. Increase context to at least {_longest + 1}.",
                    kind="danger",
                )
            else:
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
                    "gen_ids": None,
                    "gen_prompt": "",
                    "generated": "",
                    "last_sample": None,
                    "last_update": None,
                })
                _init_output = mo.callout(
                    f"Ready. Vocabulary {_vocab} | embedding {_E} | heads {_H} | layers {_L} | context {_B}",
                    kind="success",
                )
    _init_output

@app.cell
def _(
    adam_step, backward, expl_data, expl_lr, expl_steps, expl_train,
    forward, get_st, mo, set_st
):
    if not expl_train.value:
        _train_output = None
    else:
        _st = get_st()
        if _st["params"] is None or _st["tok"] is None or _st["cfg"] is None:
            _train_output = mo.callout("Initialize the model first.", kind="danger")
        else:
            _p = {k: v.copy() for k, v in _st["params"].items()}
            _adam = {k: [m.copy(), v.copy()] for k, (m, v) in _st["adam"].items()}
            _step = _st["step"]
            _losses = list(_st["losses"])
            _c2i, _i2c, _bos, _vocab = _st["tok"]
            _E, _H, _L, _B = _st["cfg"]
            _docs = [x.strip() for x in expl_data.value.splitlines() if x.strip()]
            _unknown = sorted(set(c for _doc in _docs for c in _doc if c not in _c2i))
            if not _docs:
                _train_output = mo.callout("Enter at least one training document.", kind="danger")
            elif _unknown:
                _train_output = mo.callout(
                    f"Vocabulary changed: {_unknown}. Click Initialize again.",
                    kind="danger",
                )
            else:
                _last_grad = _last_delta = _last_doc = None
                for _local in range(expl_steps.value):
                    _doc = _docs[(_step + _local) % len(_docs)]
                    _tokens = [_bos] + [_c2i[c] for c in _doc] + [_bos]
                    _loss, _cache = forward(_tokens, _p, _H)
                    if _cache is not None:
                        _grads = backward(_p, _cache)
                        _p, _adam, _last_grad, _last_delta = adam_step(
                            _p, _grads, _adam, _step + _local, expl_lr.value * 1e-3
                        )
                        _losses.append(float(_loss))
                        _last_doc = _doc
                _new_step = _step + expl_steps.value
                set_st({
                    **_st,
                    "params": _p,
                    "adam": _adam,
                    "step": _new_step,
                    "losses": _losses,
                    "last_update": {
                        "grad_norms": _last_grad,
                        "delta_norms": _last_delta,
                        "document": _last_doc,
                        "lr": expl_lr.value * 1e-3,
                    },
                })
                _train_output = mo.callout(
                    f"Trained {_new_step} steps | latest loss {_losses[-1]:.4f}",
                    kind="success",
                )
    _train_output

@app.cell
def _(expl_prompt, expl_start, get_st, mo, set_st, visible_text):
    if not expl_start.value:
        _start_output = None
    else:
        _st = get_st()
        if _st["params"] is None or _st["tok"] is None or _st["cfg"] is None:
            _start_output = mo.callout("Initialize and train the model first.", kind="danger")
        else:
            _c2i, _i2c, _bos, _vocab = _st["tok"]
            _raw = str(expl_prompt.value)
            _unknown = sorted(set(c for c in _raw if c not in _c2i))
            if _unknown:
                _start_output = mo.callout(f"Unknown character(s): {_unknown}", kind="danger")
            else:
                _ids = [_bos] + [_c2i[c] for c in _raw] if _raw else [_bos]
                set_st({
                    **_st,
                    "gen_ids": _ids,
                    "gen_prompt": _raw,
                    "generated": "",
                    "last_sample": None,
                })
                _start_output = mo.callout(
                    f"Started generation from `{visible_text(_raw) or '<BOS>'}`. "
                    f"The next click will predict exactly one new character. "
                    f"Active context length: {len(_ids)} tokens.",
                    kind="success",
                )
    _start_output

@app.cell
def _(
    display_token, expl_greedy, expl_next, expl_prompt, expl_temp, get_st, mo,
    next_probs, np, set_st, visible_text
):
    if not expl_next.value:
        _next_output = None
    else:
        _st = get_st()
        if _st["params"] is None or _st["tok"] is None or _st["cfg"] is None:
            _next_output = mo.callout("Initialize the model first.", kind="danger")
        else:
            _c2i, _i2c, _bos, _vocab = _st["tok"]
            _E, _H, _L, _B = _st["cfg"]
            _ui_prompt = str(expl_prompt.value)
            # If the prompt was edited without pressing Start, safely capture it
            # and use it for this one-token generation.
            if _st["gen_ids"] is None or _st["gen_prompt"] != _ui_prompt:
                _unknown = sorted(set(c for c in _ui_prompt if c not in _c2i))
                if _unknown:
                    _next_output = mo.callout(
                        f"Unknown character(s): {_unknown}. Change the prompt or re-initialize the vocabulary.",
                        kind="danger",
                    )
                else:
                    _ids = [_bos] + [_c2i[c] for c in _ui_prompt] if _ui_prompt else [_bos]
                    _st = {**_st, "gen_ids": _ids, "gen_prompt": _ui_prompt, "generated": "", "last_sample": None}
                    _probs = next_probs(_st["params"], _st["gen_ids"], _H, _B, expl_temp.value * 0.1).copy()
                    _probs[_bos] = 0.0
                    _probs /= _probs.sum()
                    _nxt = int(np.argmax(_probs)) if expl_greedy.value else int(np.random.choice(len(_probs), p=_probs))
                    _char = _i2c[_nxt]
                    _token_label = display_token(_nxt, _i2c, _bos)
                    set_st({
                        **_st,
                        "gen_ids": list(_st["gen_ids"]) + [_nxt],
                        "generated": _char,
                        "last_sample": {"token": _char, "prob": float(_probs[_nxt]), "probs": _probs},
                    })
                    _next_output = mo.callout(
                        f"Prompt changed, so this click first captured `{visible_text(_ui_prompt) or '<BOS>'}`. "
                        f"Then it ran one Transformer forward pass and generated **{_token_label}** with probability `{_probs[_nxt]:.4f}`.",
                        kind="success",
                    )
            else:
                _input_ids = list(_st["gen_ids"])
                _input_context = _st["gen_prompt"] + _st["generated"]
                _probs = next_probs(_st["params"], _input_ids, _H, _B, expl_temp.value * 0.1).copy()
                _probs[_bos] = 0.0
                _probs /= _probs.sum()
                _nxt = int(np.argmax(_probs)) if expl_greedy.value else int(np.random.choice(len(_probs), p=_probs))
                _char = _i2c[_nxt]
                _token_label = display_token(_nxt, _i2c, _bos)
                set_st({
                    **_st,
                    "gen_ids": _input_ids + [_nxt],
                    "generated": _st["generated"] + _char,
                    "last_sample": {"token": _char, "prob": float(_probs[_nxt]), "probs": _probs},
                })
                _next_output = mo.callout(
                    f"Forward-pass input: `{visible_text(_input_context) or '<BOS>'}`\n\n"
                    f"Generated exactly one token: **{_token_label}** (id `{_nxt}`) with probability `{_probs[_nxt]:.4f}`.\n\n"
                    f"New context: `{visible_text(_input_context + _char) or '<BOS>'}`",
                    kind="success",
                )
    _next_output

@app.cell
def _(
    display_token, expl_prompt, expl_temp, get_st, mo, next_probs, np, visible_text
):
    _st = get_st()

    if (
        _st["params"] is None
        or _st["tok"] is None
        or _st["cfg"] is None
    ):
        _prediction_output = mo.md(
            "### Next-character prediction\nInitialize and train the model first."
        )
    else:
        _c2i, _i2c, _bos, _vocab = _st["tok"]
        _E, _H, _L, _B = _st["cfg"]
        _ui_prompt = str(expl_prompt.value)

        if _st["gen_ids"] is not None:
            _ids = _st["gen_ids"]
            _context = _st["gen_prompt"] + _st["generated"]
            _active = _st["gen_prompt"]
        else:
            _ids = [_bos] + [_c2i[c] for c in _ui_prompt] if _ui_prompt and not sorted(set(c for c in _ui_prompt if c not in _c2i)) else [_bos]
            _context = _ui_prompt
            _active = ""

        _unknown = sorted(set(c for c in _ui_prompt if c not in _c2i))
        if _unknown and _st["gen_ids"] is None:
            _prediction_output = mo.callout(
                f"Unknown character(s): {_unknown}", kind="danger"
            )
        else:
            _probs = next_probs(
                _st["params"], _ids, _H, _B, expl_temp.value * 0.1
            )
            _order = np.argsort(_probs)[::-1][:min(12, _vocab)]
            _items = "\n".join(
                f"`{display_token(i, _i2c, _bos)}` — **{_probs[i]:.4f}**"
                for i in _order
            )
            if _active != _ui_prompt:
                _status = (
                    "**Prompt field changed.** Click **Start from prompt** to "
                    "make it the active generation context."
                )
            else:
                _status = "**Active generation context:** the prompt is captured and ready."

            _prediction_output = mo.vstack([
                mo.md("### Current generation state"),
                mo.callout(
                    mo.md(
                        f"Prompt field: `{visible_text(_ui_prompt) or '<BOS>'}`\n\n"
                        f"Active context: `{visible_text(_context) or '<BOS>'}`\n\n"
                        f"Tokenized: `{ ' '.join(display_token(i, _i2c, _bos) for i in _ids) }`\n\n"
                        f"{_status}"
                    ),
                    kind="info" if _active == _ui_prompt else "warn",
                ),
                mo.md("### Next-character distribution"),
                mo.md(_items),
            ])

    _prediction_output


@app.cell
def _(display_token, get_st, mo, visible_text):
    _st = get_st()

    if (
        _st["params"] is None
        or _st["tok"] is None
        or _st["cfg"] is None
        or _st["gen_ids"] is None
    ):
        _generation_output = mo.vstack([
            mo.md("### Step-by-step generation"),
            mo.md("Initialize the model, then click **Start from prompt**. After that, click **Generate one token** once per character."),
        ])
    else:
        _c2i, _i2c, _bos, _vocab = _st["tok"]
        _full = _st["gen_prompt"] + _st["generated"]
        _sample = _st["last_sample"]
        _generated_ids = _st["gen_ids"][1 + len(_st["gen_prompt"]):]
        _token_stream = " ".join(
            display_token(i, _i2c, _bos) for i in _generated_ids
        )
        _generation_output = mo.vstack([
            mo.md("### Step-by-step generation"),
            mo.md(f"**Prompt captured:** `{visible_text(_st['gen_prompt']) or '<BOS>'}`"),
            mo.md(f"**Generated tokens:** `{_token_stream or '<nothing yet>'}`"),
            mo.md(f"**Current text:** `{visible_text(_full) or '<BOS>'}`"),
            mo.md(
                f"**Last generated token:** `{display_token(_st['gen_ids'][-1], _i2c, _bos)}` "
                f"with probability `{_sample['prob']:.4f}`." if _sample is not None
                else "No token has been generated yet."
            ),
            mo.md(
                "Each **Generate one token** click takes the current text, runs "
                "the full Transformer forward pass, chooses one next character, "
                "and appends it to the context."
            ),
        ])

    _generation_output


@app.cell
def _(mo):
    mo.md(r"""# Exact equations and their NumPy translation

For one sequence of \(T\) tokens and embedding size \(E\):

### 1. Token and position embeddings

\[
X_0 = W_{\mathrm{token}}[\mathrm{ids}] + W_{\mathrm{position}}[0:T]
\]

NumPy:

```python
x0 = p["wte"][ids] + p["wpe"][:T]
x = rmsnorm(x0)
```

### 2. Q, K and V

For each Transformer layer:

\[
Q = X_n W_Q^T,\qquad
K = X_n W_K^T,\qquad
V = X_n W_V^T
\]

NumPy:

```python
Q = xn @ p[f"l{li}.wq"].T
K = xn @ p[f"l{li}.wk"].T
V = xn @ p[f"l{li}.wv"].T
```

If there are \(H\) heads, \(E=H D\), and each matrix is reshaped:

\[
Q \in \mathbb{R}^{T\times E}
\rightarrow
Q_h \in \mathbb{R}^{H\times T\times D}
\]

### 3. Causal attention

\[
S = \frac{QK^T}{\sqrt D}
\]

Future positions are masked, then:

\[
A = \operatorname{softmax}(S)
\]

\[
C = AV
\]

NumPy:

```python
scores = Qh @ Kh.transpose(0, 2, 1) / math.sqrt(D)
scores = np.where(mask, -1e9, scores)
attn = softmax(scores)
attended = attn @ Vh
```

### 4. Attention output and residual connection

\[
X_{\mathrm{attn}} =
X + C W_O^T
\]

NumPy:

```python
attn_proj = attended @ p[f"l{li}.wo"].T
x_attn = x + attn_proj
```

### 5. MLP

\[
H_1 = X_n W_1^T
\]

\[
H = \max(0,H_1)
\]

\[
X_{\mathrm{out}} = X_{\mathrm{attn}} + H W_2^T
\]

NumPy:

```python
h1 = xnm @ p[f"l{li}.w1"].T
h = np.maximum(0, h1)
mlp_proj = h @ p[f"l{li}.w2"].T
x = x_attn + mlp_proj
```

### 6. Logits and next-token probabilities

For the final position \(t=T\):

\[
z = x_T W_{\mathrm{lm}}^T
\]

Temperature \(\tau\):

\[
p_i =
\frac{\exp(z_i/\tau)}
{\sum_j \exp(z_j/\tau)}
\]

NumPy:

```python
logits = x[-1] @ p["lm_head"].T
probs = softmax(logits / temperature)
```

### 7. Training loss

For the correct next token \(y_t\):

\[
L =
-\frac{1}{T}
\sum_{t=1}^{T}
\log p_{t,y_t}
\]

The backward pass computes every
\(\partial L/\partial W\) explicitly.

For example:

\[
\frac{\partial L}{\partial W_{\mathrm{lm}}}
=
(dL/dz)^T X
\]

NumPy:

```python
g["lm_head"] += dl.T @ c["xf"]
```

### 8. Adam matrix update

For every parameter matrix \(W\):

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
\]

\[
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
\]

\[
\hat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}
\]

\[
W_t =
W_{t-1}
-\eta
\frac{\hat m_t}
{\sqrt{\hat v_t}+\epsilon}
\]

The same operation is applied element by element to every
parameter matrix.""")


@app.cell
def _(
    expl_head, expl_layer, expl_pos, get_st, matrix_preview, mo
):
    _st = get_st()

    if (
        _st["params"] is None
        or _st["tok"] is None
        or _st["cfg"] is None
        or _st["gen_ids"] is None
    ):
        _trace_output = mo.md(
            "## Matrix trace for one generation step\n"
            "Start generation to inspect the actual matrices."
        )
    else:
        _c2i, _i2c, _bos, _vocab = _st["tok"]
        _E, _H, _L, _B = _st["cfg"]

        # This cell only displays cached quantities from a fresh explicit
        # forward calculation in the next cell's dependency chain.
        _trace_output = mo.md(
            "## Matrix trace\n"
            "The next section shows actual values for the current context."
        )

    _trace_output


@app.cell
def _(
    explain_forward, expl_head, expl_layer, expl_pos, expl_temp,
    get_st, matrix_preview, mo, np
):
    _st = get_st()

    if (
        _st["params"] is None
        or _st["tok"] is None
        or _st["cfg"] is None
        or _st["gen_ids"] is None
    ):
        _matrix_output = None
    else:
        _c2i, _i2c, _bos, _vocab = _st["tok"]
        _E, _H, _L, _B = _st["cfg"]

        _tr = explain_forward(
            _st["params"],
            _st["gen_ids"],
            _H,
            _B,
            expl_temp.value * 0.1,
        )

        _layer = min(expl_layer.value - 1, _L - 1)
        _head = min(expl_head.value - 1, _H - 1)
        _pos = min(expl_pos.value - 1, len(_tr["ids"]) - 1)

        _tokens = [
            "<BOS>" if i == _bos else "<space>" if _i2c.get(i) == " "
            else "<newline>" if _i2c.get(i) == "\n"
            else _i2c.get(i, "?")
            for i in _tr["ids"]
        ]
        _z = _tr["layers"][_layer]

        _row = _z["attn"][_head, _pos]
        _top = np.argsort(_row)[::-1][:min(6, len(_row))]
        _attn_text = "\n".join(
            f"{_tokens[int(i)]}: {_row[i]:.4f}" for i in _top
        )

        _q = _z["Q"][_head, _pos]
        _k = _z["K"][_head, _pos]
        _v = _z["V"][_head, _pos]

        _matrix_output = mo.vstack([
            mo.md(
                f"""## Matrix trace for the current step

Current token sequence:

`{' '.join('<space>' if t == ' ' else t for t in _tokens)}`

Inspecting layer **{_layer + 1}**, head **{_head + 1}**,
position **{_pos + 1}**.
"""
            ),
            mo.callout(
                mo.md(
                    "### Embedding matrix entering the model\n\n"
                    f"Shape: `{_tr['embedding'].shape}`\n\n"
                    "```text\n"
                    f"{matrix_preview(_tr['embedding'])}\n"
                    "```"
                ),
                kind="info",
            ),
            mo.callout(
                mo.md(
                    "### Q, K, V for the selected token\n\n"
                    f"Q = `xn @ Wq.T` → `{matrix_preview(_q)}`\n\n"
                    f"K = `xn @ Wk.T` → `{matrix_preview(_k)}`\n\n"
                    f"V = `xn @ Wv.T` → `{matrix_preview(_v)}`"
                ),
                kind="info",
            ),
            mo.callout(
                mo.md(
                    "### Attention row\n\n"
                    "The selected token distributes probability over "
                    "earlier tokens:\n\n"
                    f"{_attn_text}\n\n"
                    f"Full row: `{matrix_preview(_row)}`"
                ),
                kind="info",
            ),
            mo.callout(
                mo.md(
                    "### MLP at the selected position\n\n"
                    f"`h1 = xnm @ W1.T` shape: `{_z['h1'][_pos].shape}`\n\n"
                    f"`ReLU(h1)` first values:\n\n"
                    "```text\n"
                    f"{matrix_preview(_z['h'][_pos])}\n"
                    "```"
                ),
                kind="info",
            ),
            mo.callout(
                mo.md(
                    "### Final logits and probabilities\n\n"
                    f"Logits: `{matrix_preview(_tr['logits'])}`\n\n"
                    f"Probabilities: `{matrix_preview(_tr['probs'])}`"
                ),
                kind="info",
            ),
        ])

    _matrix_output


@app.cell
def _(
    explain_forward, expl_head, expl_layer, expl_temp, get_st, mo, plt
):
    _st = get_st()

    if (
        _st["params"] is None
        or _st["tok"] is None
        or _st["cfg"] is None
        or _st["gen_ids"] is None
    ):
        _attn_fig = None
    else:
        _c2i, _i2c, _bos, _vocab = _st["tok"]
        _E, _H, _L, _B = _st["cfg"]

        _tr = explain_forward(
            _st["params"],
            _st["gen_ids"],
            _H,
            _B,
            expl_temp.value * 0.1,
        )
        _layer = min(expl_layer.value - 1, _L - 1)
        _head = min(expl_head.value - 1, _H - 1)

        _attn = _tr["layers"][_layer]["attn"][_head]
        _tokens = [
            "<BOS>" if i == _bos else "<space>" if _i2c.get(i) == " "
            else _i2c.get(i, "?")
            for i in _tr["ids"]
        ]

        _fig, _ax = plt.subplots(
            figsize=(
                max(6, len(_tokens) * .65),
                max(5, len(_tokens) * .55),
            )
        )
        _im = _ax.imshow(_attn, vmin=0, vmax=1, aspect="auto")
        _ax.set_xticks(range(len(_tokens)), _tokens, rotation=45)
        _ax.set_yticks(range(len(_tokens)), _tokens)
        _ax.set_xlabel("K: token being read")
        _ax.set_ylabel("Q: current token")
        _ax.set_title(
            f"Layer {_layer + 1}, head {_head + 1}: causal attention"
        )
        _fig.colorbar(_im, ax=_ax, label="attention weight")
        _fig.tight_layout()
        plt.close(_fig)
        _attn_fig = _fig

    _attn_fig


@app.cell
def _(get_st, mo):
    _st = get_st()
    _u = _st["last_update"]

    if _u is None:
        _update_output = mo.md(
            "## How matrices change during training\n"
            "Train the model to inspect gradient and update magnitudes."
        )
    else:
        _keys = [
            "wte", "lm_head", "l0.wq", "l0.wk", "l0.wv",
            "l0.wo", "l0.w1", "l0.w2"
        ]
        _lines = []
        for _k in _keys:
            if _k in _u["grad_norms"]:
                _lines.append(
                    f"`{_k}`: "
                    f"||gradient|| = `{_u['grad_norms'][_k]:.5f}`, "
                    f"||Adam update|| = `{_u['delta_norms'][_k]:.5f}`"
                )

        _update_output = mo.vstack([
            mo.md("## How matrices changed in the last training step"),
            mo.md(
                f"Document used: `{_u['document']}`\n\n"
                f"Learning rate: `{_u['lr']}`"
            ),
            mo.callout(
                mo.md("\n\n".join(_lines)),
                kind="info",
            ),
            mo.md(
                "For each matrix, the code computed its full gradient "
                "array, then Adam converted that gradient into an "
                "element-wise update. The displayed norms summarize the "
                "size of those arrays; the actual update is applied to "
                "every matrix element."
            ),
        ])

    _update_output


@app.cell
def _(get_st, mo, plt):
    _st = get_st()
    _losses = _st["losses"]

    if not _losses:
        _loss_output = mo.md(
            "## Training loss\nTrain the model to see the loss trajectory."
        )
    else:
        _fig, _ax = plt.subplots(figsize=(10, 3))
        _ax.plot(_losses, lw=.8)
        _ax.set_xlabel("training step")
        _ax.set_ylabel("cross-entropy")
        _ax.set_title(f"training loss | step {_st['step']}")
        _ax.grid(True, alpha=.2)
        _fig.tight_layout()
        plt.close(_fig)
        _loss_output = _fig

    _loss_output


@app.cell
def _(get_st, mo):
    _st = get_st()

    if (
        _st["params"] is None
        or _st["tok"] is None
        or _st["cfg"] is None
    ):
        _state_output = mo.md(
            "## Model state\nInitialize the model to see parameter counts."
        )
    else:
        _n = sum(v.size for v in _st["params"].values())
        _E, _H, _L, _B = _st["cfg"]

        _state_output = mo.md(
            f"""## Model state

Parameters: **{_n:,}**

Embedding: **{_E}**

Heads: **{_H} × {_E // _H}**

Layers: **{_L}**

Context: **{_B}**

Training steps: **{_st["step"]}**"""
        )

    _state_output


if __name__ == "__main__":
    app.run()
