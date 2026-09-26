# SmolVLA (LeRobot @ `8515d45`) — VLM → action-expert interface and gradient-flow map

Scope: the *installed* implementation this harness runs, read from
`.venv/lib/python3.12/site-packages/lerobot/policies/smolvla/` (abbreviated below as
`MS` = `modeling_smolvla.py`, `SWE` = `smolvlm_with_expert.py`, `CFG` = `configuration_smolvla.py`),
instantiated with the pinned checkpoint `HuggingFaceVLA/smolvla_libero@6721902` on backbone
`HuggingFaceTB/SmolVLM2-500M-Instruct@7b375e1`. Dimensions and parameter counts were read from the
checkpoint tensors and from a CPU instantiation; gradient reach was measured with one dummy
training step (script: scratchpad `inspect_model.py`). No model code was modified and no
experiment was run.

---

## 1. Components as instantiated

| Module (state-dict prefix `model.`) | Shape / config | Params | Runtime dtype | Trainable in checkpoint regime |
| --- | --- | ---: | --- | --- |
| `vlm_with_expert.vlm.model.vision_model` (SigLIP-style) | 512×512 input, patch 16 → 1024 patches, hidden 768 | 86.43 M | bf16 | no (`freeze_vision_encoder=True`, kept in `eval()`) |
| `…vlm.model.connector` (pixel-shuffle ×4 + `Linear(12288→960)`) | 1024 patches → **64 tokens** / image | 11.80 M | bf16 | no — but **only** because `train_expert_only=True`; `freeze_vision_encoder` does *not* cover it (verified: trains under `train_expert_only=False`) |
| `…vlm.model.text_model.embed_tokens` | 49280 × 960 | 47.31 M | bf16 | no |
| `…vlm.model.text_model.layers[0..31]` (Llama block) | hidden 960, 15 heads × 64, **5 KV heads** (GQA ×3), MLP 2560, RoPE θ=1e5 | 314.63 M | bf16 | no |
| `…vlm.model.text_model.norm`, `…vlm.lm_head` | RMSNorm 960; 49280×960 | 47.31 M | bf16 | no; **never used by the policy** (dead weight, 47 M) |
| `…lm_expert.layers[0..31]` (Llama block, `AutoModel.from_config`) | hidden **480**, MLP **1280**, `q_proj 480→960`, `o_proj 960→480`, 15 heads × 64, 5 KV heads | 93.42 M | bf16 | **yes** |
| `…lm_expert.layers[odd].self_attn.{k,v}_proj` (replaced, SWE:120-134) | `Linear(320→320)` — input is the **VLM's** KV width | 3.28 M | **fp32** | **yes** |
| `…lm_expert.layers[even].self_attn.{k,v}_proj` (original) | `Linear(480→320)` | (in 93.42) | bf16 | yes |
| `…lm_expert.norm` | RMSNorm 480 | 480 | bf16 | yes |
| `state_proj` | `Linear(32→960)` | 0.03 M | fp32 | yes (`train_state_proj=True`) |
| `action_in_proj` / `action_out_proj` | `Linear(32→480)` / `Linear(480→32)` | 0.02 M each | fp32 | yes |
| `action_time_mlp_in` / `action_time_mlp_out` | `Linear(960→480)` / `Linear(480→480)` | 0.46 M / 0.23 M | fp32 | yes |
| **Total** | | **604.9 M** | | **≈ 97.5 M trainable** |

Config facts that determine the wiring (checkpoint `config.json`): `attention_mode="cross_attn"`,
`self_attn_every_n_layers=2`, `num_vlm_layers=0` (→ all 32 kept), `num_expert_layers=-1` (→ 32,
1:1 layer pairing, `SWE:401-413`), `expert_width_multiplier=0.5`, `chunk_size=50`,
`tokenizer_max_length=48`, `prefix_length=0` (no fixed prefix padding), `add_image_special_tokens=False`,
`resize_imgs_with_padding=[512,512]`, `num_steps=10` (harness overrides to 1), `n_action_steps=1`.

Why the expert is bf16 while its odd-layer k/v are fp32: `lm_expert_config` is a deepcopy of the VLM
text config *after* `from_pretrained(torch_dtype=bfloat16)`, so `AutoModel.from_config` builds bf16;
the replacement `nn.Linear`s are created afterwards with the default fp32 (SWE:125-134).

## 2. Token layout and attention masks

Prefix (VLM stream), built in `MS:637-729 embed_prefix`:

```
idx:  [0 … 63]      [64 … 127]      [128 … 128+L-1]     [128+L]
      image  (64)   image2 (64)     language (L ≤ 48)   state (1)
att:   0             0               0                    1
```

- Image tokens: `embed_image` (SWE:191-204, frozen vision + connector), scaled by √960.
  LIBERO frames are 256×256 → `resize_with_pad` upsamples to 512×512 (MS:415-461).
- Language: `embed_tokens`, scaled by √960. The LIBERO-Spatial task strings tokenize to ~20 tokens
  (+ newline added by `NewLineTaskProcessorStep`), right-padded to `longest` in the batch.
- State: 8-d → zero-padded to 32 → `state_proj` → **one 960-d token that lives inside the VLM stream**.
  Its `att_mask=1` (MS:715) means image/language tokens can never attend to it; it attends to everything
  before it. So the state token is an *input to the VLM* whose only consumer is the expert (via K/V).
- Measured prefix length for a LIBERO-Spatial prompt: 149 = 2×64 + 20 + 1.

Suffix (expert stream), `MS:731-772 embed_suffix`: 50 action tokens =
`action_time_mlp_out(SiLU(action_time_mlp_in([action_in_proj(x_t) ‖ sincos(t)])))`, all with
`att_mask=1` (MS:767) ⇒ under `make_att_2d_masks` (`mask[i,j] = cumsum[j] ≤ cumsum[i]`) the action
tokens are **causal among themselves** (token k sees prefix + actions 0..k), and **prefix tokens never
see actions**. The VLM stream is therefore action-independent, which is what makes the inference KV
cache exact.

Position ids: global `cumsum(pad)−1` over `[prefix ‖ suffix]` for joint (even) layers; in cross (odd)
layers the expert's positions are re-based to start at 0 (SWE:377-379) while the VLM keys it reads keep
their prefix positions. This asymmetry is baked into the trained weights — any interface transform
must be applied *after* RoPE to leave it intact.

## 3. The interface: one channel, two layer types

`SmolVLMWithExpertModel.forward` (SWE:415-510) runs both streams layer-by-layer. There is **no
residual mixing, no FiLM, no shared MLP, no use of the VLM's final hidden state**. The *only* place
information moves from the VLM to the expert is attention over the VLM's per-layer key/value
projections:

    K_vlm^l = k_proj_vlm^l(RMSNorm(h_vlm^l)) ,  V_vlm^l = v_proj_vlm^l(RMSNorm(h_vlm^l))      (5 heads × 64 = 320-d)

Dispatch (SWE:437-467): layer `l` is a **joint self-attention layer** if `l % 2 == 0` (or whenever
`fill_kv_cache=True`), otherwise a **cross-attention layer**.

### 3a. Even layers `l ∈ {0,2,…,30}` — `forward_attn_layer` (SWE:209-284)

```
                 VLM stream (960)                          expert stream (480)
      h_vlm^l ──RMSNorm──┬─ q_proj_vlm ─► Q_p (15×64)     h_exp^l ──RMSNorm──┬─ q_proj_exp ─► Q_s (15×64)
                         ├─ k_proj_vlm ─► K_p (5×64)                         ├─ k_proj_exp ─► K_s (5×64)
                         └─ v_proj_vlm ─► V_p (5×64)                         └─ v_proj_exp ─► V_s (5×64)
                     cat over sequence:  Q=[Q_p;Q_s]  K=[K_p;K_s]  V=[V_p;V_s]   (SWE:245-247) → RoPE (259-260)
                     ONE attention call, mask = [prefix rows: prefix only | suffix rows: prefix + causal suffix]
      out[:P] ─ o_proj_vlm ─(+res)─ MLP_vlm ─(+res)─► h_vlm^{l+1}     out[P:] ─ o_proj_exp(960→480) ─(+res)─ MLP_exp ─(+res)─► h_exp^{l+1}
```

Coupling here: expert queries `Q_s` attend directly to the **VLM's own** `K_p`/`V_p` (no adapter
between them; the expert's `q_proj` output width 960 was chosen to match the VLM head layout so a
single `eager_attention_forward` (SWE:516-561) can serve both). `V_p` rows are summed straight into
the expert's attention output.

Inference (`past_key_values` set, `fill_kv_cache=False`): `inputs_embeds=[None, suffix]`, so
`K=[K_p^cache ; K_s]`, `V=[V_p^cache ; V_s]` (SWE:276-277); the cached `K_p` is **post-RoPE**.

### 3b. Odd layers `l ∈ {1,3,…,31}` — `forward_cross_attn_layer` (SWE:286-399)

```
      prefix branch (training only; at inference read from cache, SWE:349-350):
          Q_p,K_p,V_p from VLM layer l → RoPE(K_p) → prefix self-attention → att_outputs[0]   (SWE:306-331)
      expert branch (SWE:353-394):
          Q_s = q_proj_exp(RMSNorm(h_exp^l))                       RoPE at positions 0..49
          K   = k_proj_exp^{cross}( flatten(K_p, 320) )            ← Linear(320→320), fp32, the ONLY learned adapter at the interface
          V   = v_proj_exp^{cross}( flatten(V_p, 320) )
          attention(Q_s, K, V) with mask = attention_mask[:, -50:, :P]   ← prefix columns only: NO action-to-action attention in odd layers
```

Then the same per-stream `o_proj → +res → post-LN → MLP → +res` (SWE:470-500) for both streams. The
VLM stream is still advanced in odd layers (its prefix self-attention runs), so `h_vlm^{l+1}` exists
for the next even layer.

### 3c. Head-level summary of the forward coupling F

    F = { (K_vlm^l, V_vlm^l) : l = 0..31 }   —  32 layers × [B, P, 5, 64] × 2 tensors

That set is *complete*: masking it to zero would disconnect the expert from images, language and
state entirely (the expert has no other input besides noisy actions and time).

## 4. Inference path as pinned (`num_steps=1`, `n_action_steps=1`)

1. `sample_actions` (MS:812-881): `embed_prefix` → `vlm_with_expert.forward(inputs_embeds=[prefix, None], use_cache=True, fill_kv_cache=True)`
   (MS:836). `fill_kv_cache=True` forces the self-attn path for **all 32 layers** (SWE:438-439), so the
   cache holds `{l: (RoPE(K_p^l), V_p^l)}`. The VLM stream is computed once per environment step.
2. `denoise_step` (MS:883-916), once (`dt=-1`): `embed_suffix(noise, t=1.0)` →
   `forward(inputs_embeds=[None, suffix], past_key_values=cache)` → expert final norm → fp32 →
   `action_out_proj` → `v_t`; `x_0 = x_1 − v_t`. (With `num_steps=10` the same cache serves 10 expert passes.)
3. `select_action` (MS:325-352) keeps a deque of `n_action_steps=1`, so the full VLM+expert forward
   runs **every** environment step (≈115 ms at batch 5 in the baseline). Actions are un-padded 32→7.

## 5. Training path and backward gradient flow

`SmolVLAPolicy.forward` (MS:358-413) → `VLAFlowMatching.forward` (MS:774-810):
`t ~ 0.001 + 0.999·Beta(1.5,1)`, `x_t = t·ε + (1−t)·a`, `u_t = ε − a`, one **joint** pass with
`inputs_embeds=[prefix, suffix]`, `use_cache=False` (MS:797), `v_t = action_out_proj(norm(h_exp^{32}))`,
`loss = MSE(u_t, v_t)` over the 50×7 valid entries (padding-masked). The VLM prefix stream is recomputed
every step **inside the autograd graph** — there is no `torch.no_grad()` around it even though its weights are frozen.

Backward gradient of the flow-matching loss:

```
loss ─► action_out_proj ─► lm_expert.norm ─► expert layers 31…0 (o_proj / MLP / q_proj / self K,V on even layers)
   │                                            │
   │                                            └─► attention probs & values  ─► K_vlm^l , V_vlm^l   (every layer, both kinds; via k/v_proj^{cross} on odd layers)
   │                                                                                 │
   │                                                                                 └─► k_proj_vlm^l / v_proj_vlm^l ─► RMSNorm ─► h_vlm^l ─► (o_proj/MLP/attn of VLM layers l-1 … 0)
   │                                                                                                                             │
   └─► action_in_proj / action_time_mlp_* (suffix embedding)                                                                     └─► prefix embeddings: image (vision+connector), language (embed_tokens), state (state_proj)
```

Measured on one CPU step (dummy LIBERO-shaped batch, B=2), `‖grad‖` per group:

| Group | A: checkpoint regime (`train_expert_only=True`) | B: `train_expert_only=False`, `freeze_vision_encoder=True` |
| --- | ---: | ---: |
| vision_model | frozen, 0 | frozen, 0 |
| connector | frozen, 0 | **4.06** (11.8 M trainable) |
| embed_tokens | frozen, 0 | 1.79 |
| text layers | frozen, 0 | 6.26 — grads on 305.4 M of 314.6 M (layer 31's q/o/MLP receive none: nothing consumes its output) |
| lm_head / text norm | 0 (unused) | 0 (unused) |
| expert layers (excl. cross k/v) | 5.65 | 6.65 |
| expert cross `k/v_proj` (odd layers) | 2.42 | 2.94 |
| `state_proj` | **1.44** | 2.17 |
| action_in / action_out / time_mlp_in / time_mlp_out | 0.31 / 5.24 / 1.46 / 1.93 | 0.43 / 5.97 / 2.46 / 2.82 |

Key reading of row `state_proj` in column A: **the action loss already back-propagates through all 32
frozen VLM layers** in the checkpoint's own training regime. Freezing weights does not cut the
activation gradient; the K/V edges carry it from the expert into the VLM stream and down to
`state_proj` (0.03 M params) at the cost of a full VLM backward. "Backward coupling" therefore exists
today at λ = 1 regardless of `train_expert_only`; the flag only decides whether VLM *weights* consume it.

## 6. Existing control knobs and their gaps

| Knob | Effect | Gap |
| --- | --- | --- |
| `freeze_vision_encoder` (SWE:151-154) | `vision_model` frozen + `eval()` | leaves `connector` trainable |
| `train_expert_only` (SWE:155-158) | whole `vlm` frozen + `eval()` (kept in eval by `train()` override, SWE:182-189) | parameter-level only; activation gradient still flows (§5) |
| `train_state_proj` (MS:617-619) | `state_proj` requires_grad | — |
| "freeze last VLM layer(s) + final norm" branch (SWE:159-176) | intended for DDP unused-param safety | **dead code under transformers 5.5.4**: patterns `text_model.model.layers.N.` / `text_model.model.norm.weight` never match the real names `model.text_model.layers.N.` (verified: all 32 layers trainable in mode B); only `lm_head` still matches |
| `set_requires_grad()` (both classes) | freeze-only | **cannot unfreeze**; flipping `train_expert_only` after construction and calling it leaves 0 trainable VLM params (verified). Coupling settings must be applied at construction or with an explicit `requires_grad` reset |
| PEFT defaults (MS:495-503) | LoRA on expert `q/v_proj` + the five projections | chooses *which weights* learn; does not touch the K/V channel |
| `get_optim_params` (MS:273) | returns all params | optimizer relies on `requires_grad`; nothing per-layer |

None of these knobs (a) act on the K/V channel, (b) distinguish forward feature coupling from backward
gradient coupling, or (c) resolve per layer or per prefix-token group.

## 7. Minimal modification points for independent F / B parameterization

Because §3c shows the K/V channel is the unique interface, one wrapper applied to `(K_vlm^l, V_vlm^l)`
**immediately before the expert consumes them** gives complete and independent control:

```python
def couple(k, v, layer_idx, groups=None):
    # F — forward semantic feature coupling (any of: identity, per-layer scalar α_l, per-head gate,
    #     per-prefix-token-group gate g[groups], feature dropout, mixing with a learned/null KV …)
    k, v = forward_transform(k, v, layer_idx, groups)
    # B — backward action-loss gradient coupling: forward identity, backward × λ_l
    #     λ=1 → unchanged, λ=0 → exact stop-gradient (== .detach(), and skips the VLM backward)
    k = k.detach() + lam[layer_idx] * (k - k.detach())
    v = v.detach() + lam[layer_idx] * (v - v.detach())
    return k, v
```

F and B are orthogonal by construction: `forward_transform` never sees λ, and the straight-through
term never changes forward values. Both can be fixed hyperparameters or learnable (`nn.Parameter`
α/λ; learnable λ needs a custom `autograd.Function` instead of the detach trick).

### 7.1 Where (all in `SWE`; nothing in `MS` needs to change for the channel itself)

| # | Site | What to do | Why |
| --- | --- | --- | --- |
| M1 | `forward_cross_attn_layer`, just before SWE:363 (`_key_states = key_states…`) | `key_states, value_states = couple(key_states, value_states, layer_idx)` | odd layers already separate "VLM K/V as read by the expert" (fresh at 320-326 in training, cached at 349-350 at inference). The prefix branch (328-331) keeps using the raw tensors, so the VLM stream is untouched. One insertion covers both training and inference. |
| M2 | `forward_attn_layer` training path (SWE:245-247) | split the single joint attention into (i) prefix rows over raw `[K_p;V_p]` and (ii) suffix rows over `[couple(RoPE(K_p),V_p) ; K_s,V_s]` | today one attention call serves both streams with **shared** K/V; scaling K_p in place would also alter the VLM's own self-attention and, for B, also scale the VLM-internal gradient edge. Prefix rows are masked from suffix keys, so the split is mathematically identical at `couple = id` and costs no extra FLOPs. |
| M3 | `forward_attn_layer` inference path (SWE:276-277) | apply `couple` to the cached `K_p, V_p` before the `cat` with `K_s, V_s` | cache stays raw VLM K/V; consumption-point semantics identical to M1/M2. B is irrelevant here (no_grad), only F. |
| M4 | (optional) `MS:637-729 embed_prefix` → return a `prefix_groups` int tensor (`0=image,1=image2,2=language,3=state`) and plumb it through `VLAFlowMatching.forward/sample_actions/denoise_step` (MS:797, 836, 904) into `SmolVLMWithExpertModel.forward` | needed only for **token-group-resolved** F/B (e.g. attenuate language K/V but keep vision). Prefix layout is deterministic (§2) so the group ids can be built from `img` token counts and `lang_masks`. | 

Applying F once at cache-fill time (SWE:266-270) instead of at M3 is a valid inference-only
optimisation when F is static; keep M3 as the semantic definition so training and inference agree.
Apply F **after RoPE** in both M2 and M3 (per-token/per-head scalars commute with RoPE; per-channel
transforms do not).

### 7.2 Secondary paths to pin down (so they are held fixed, not accidentally varied)

- **State token**: enters the VLM prefix via `state_proj` (MS:704) and reaches the expert only through
  F. Under λ=0 at all layers, `state_proj` stops learning (its only gradient path is B). Choice to make
  explicitly: accept a frozen `state_proj`, exempt the state column from λ (M4 groups make this a
  one-liner), or route state into the expert directly (an architecture change — out of scope here).
- **Connector / embed_tokens / vision**: parameter-level; control them with explicit `requires_grad`
  per module rather than the two flags (connector is not covered by `freeze_vision_encoder`).
- **VLM-internal gradient** (from `K_vlm^{l+1}` back through VLM layer `l`): downstream of the B edges,
  so per-layer λ already bounds it; no separate knob needed.
- **Layer 31**: only its `k/v_proj` are consumed; its `q/o/MLP` are dead for the policy (as is the
  whole `lm_head`/final norm). Irrelevant to F/B but worth knowing when counting "coupled" parameters.
- **Dtype**: `K_p/V_p` are bf16; odd-layer `k/v_proj^{cross}` are fp32 and cast their input. `couple`
  should preserve the incoming dtype so the checkpoint's numerics are reproduced at α=λ=1.

### 7.3 Implementation strategy without touching the pinned lock

- Put the override in this repo (e.g. `src/coupling.py`): subclass `SmolVLMWithExpertModel` (or
  assign the two overridden methods on the instance after `make_policy`). State-dict keys are
  unchanged, so the checkpoint loads as-is; any new α/λ parameters appear as *missing keys*, which
  `PreTrainedPolicy.from_pretrained(strict=False)` (default) tolerates and logs.
- Configure via a `coupling` block in the JSON config (per-layer `forward` / `backward` schedules,
  optional per-group masks), recorded into `metrics.json` provenance by `evaluate.py`.
- Invariants to test before any run (CPU, seconds): with α≡1, λ≡1, `sample_actions` with fixed noise
  must equal the unpatched output to bf16 tolerance; with λ≡0 and `train_expert_only=True`,
  `state_proj.grad` must be `None`/zero and no VLM activation may require grad (also verifies the
  VLM backward is actually skipped); with λ≡1 and the split M2 the gradients on the expert must match
  the unsplit implementation.
- Because `set_requires_grad()` is freeze-only, set the desired `requires_grad` pattern explicitly
  *after* constructing the policy, and never rely on flipping `train_expert_only` at runtime.

## 8. File anchors (installed revision)

`SWE`: class 72 · layer truncation 100-103 · expert config 106-116 · cross k/v replacement 120-134 ·
`set_requires_grad` 150-180 · `train` 182-189 · `embed_image` 191-204 · `forward_attn_layer` 209-284
(cat 245-247, RoPE 259-260, cache 265-277) · `forward_cross_attn_layer` 286-399 (prefix 306-331, cache
338-350, expert k/v 363-375, mask 380-382, attention 386-393) · `get_model_layers` 401-413 · `forward`
415-510 (dispatch 437-467, residual/MLP 470-500, final norm 502-509) · eager attention 516-561.

`MS`: `SmolVLAPolicy` 226 · `get_optim_params` 273 · `select_action` 325 · loss wrapper `forward` 358 ·
`prepare_images` 415 · PEFT targets 495 · `VLAFlowMatching` 541 (projections 583-591, `set_requires_grad`
617) · `embed_prefix` 637-729 (image 665, language 693, state 704, state mask 715) · `embed_suffix`
731-772 (causal action mask 767) · training `forward` 774-810 (x_t/u_t 785-786, joint pass 797, loss
808-809) · `sample_actions` 812-881 (cache fill 836, step loop 844-848) · `denoise_step` 883-916.

## 9. Implemented interface (`src/semantic_control.py`, tests in `tests/`)

The first, deliberately minimal instantiation of §7: two knobs, no gradient scaling, no gates, no
token-group routing.

| Knob | Values | Mechanism |
| --- | --- | --- |
| `semantic_layers` | `"all"` (native) / `"cross_only"` | `SemanticSmolVLMWithExpertModel.forward_attn_layer` (a `__class__` swap on the upstream instance; state-dict keys unchanged). For `cross_only`, in the joint self-attention layers the suffix (action) rows of the attention mask get their prefix columns set to `False` — both in the training joint pass and in the cached-KV inference pass. The float32-minimum fill before the softmax yields *exactly* zero probability and zero gradient on the hidden VLM K/V, so this is equivalent to M2/M3 of §7.1 for a binary on/off choice and needs no attention split. Cross-attention layers and the VLM stream (verified bitwise on the KV cache) are untouched. |
| `update_vlm` | `False` / `True` | `apply_trainability` sets `requires_grad` for every parameter from scratch. Vision encoder always frozen; `state_proj` always trainable. `True` trains text layers, token embeddings and connector, but keeps parameters with no path to the action loss frozen (`lm_head`, final text norm, last VLM layer's q/o/MLP/post-attention norm) so that "trainable ⇔ receives gradient" holds exactly. |

Presets: A = all/frozen, B = all/trainable, C = cross_only/frozen, D = cross_only/trainable
(`configs/semantic_control.json`, pinned to `lerobot/smolvla_base@c83c316` on
`SmolVLM2-500M-Video-Instruct@7b375e1`: 16 VLM layers, expert width 0.75, 450.0 M parameters).
Verified on the RTX 5090: A is bitwise identical to upstream (loss, actions, every gradient, every
attention mask); trainable parameters A/C 99,880,992, B/D 307,086,432.
