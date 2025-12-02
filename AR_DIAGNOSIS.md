# AR Generator Diagnosis Report

## Executive Summary

After comparing FractalMIDI's AR implementation with fractalgen's reference implementation, I've identified **4 critical issues** that explain why AR generates noise or sparse notes:

1. **Critical Bug in `predict()` function**: Incorrect indexing for causal prediction
2. **Position embedding weakness**: Learned embeddings vs. 2D rotary embeddings
3. **Missing training optimization**: No KV cache or proper causal masking
4. **Initialization issues**: Potentially unstable weight initialization

## Detailed Analysis

### Issue 1: Critical Bug in `predict()` Function (HIGH PRIORITY)

**Location**: `models/ar_generator.py`, line 238

**Current Implementation**:
```python
if input_pos is not None:
    middle_cond = x[:, input_pos : input_pos + 1]
else:
    middle_cond = x[:, :-1]
```

**Problem**: 
- When `input_pos=k`, this extracts `x[:, k:k+1]`, which is the hidden state at position k
- However, in causal AR, the hidden state at position k should predict token k+1
- The fractalgen implementation correctly handles this by using the output at position k to predict the NEXT token

**Fractalgen's Correct Implementation** (`fractalgen/models/ar.py`, lines 430-434):
```python
if input_pos is not None:
    middle_cond = x[:, 0]  # Single token output when using KV cache
else:
    middle_cond = x[:, :-1]  # All predictors for training
```

**Impact**: This bug causes the model to use the wrong hidden states for prediction, leading to:
- Misaligned training signals
- Poor generation quality
- Inability to learn proper causal dependencies

**Fix**: Correct the indexing logic to match causal prediction semantics

---

### Issue 2: Position Embedding Weakness (MEDIUM PRIORITY)

**FractalMIDI** (`models/ar_generator.py`, line 46):
```python
self.pos_embed = nn.Parameter(torch.zeros(1, (self.max_seq_len + 1), embed_dim))
# Initialized with: torch.nn.init.normal_(self.pos_embed, std=.02)
```

**Fractalgen** (`fractalgen/models/ar.py`, lines 333-336):
```python
# Uses 2D rotary position embedding
self.freqs_cis = precompute_freqs_cis_2d(grid_size, model_args.dim // model_args.n_head,
                                         model_args.rope_base, cls_token_num=1)
```

**Differences**:
1. **Learned vs. Rotary**: FractalMIDI uses learned positional embeddings; fractalgen uses rotary embeddings
2. **2D structure**: Fractalgen's rotary embeddings encode 2D spatial structure (h, w); FractalMIDI treats patches as 1D sequence
3. **Generalization**: Rotary embeddings generalize better to unseen sequence lengths

**Impact**:
- Learned embeddings may not capture spatial relationships as well
- Harder to extrapolate to longer sequences
- Less inductive bias for 2D structure

**Recommendation**: 
- **Option A** (Quick fix): Improve learned embedding initialization (use smaller std, add warmup)
- **Option B** (Better but complex): Implement 2D rotary embeddings like fractalgen

---

### Issue 3: Training Forward Pass Differences (MEDIUM PRIORITY)

**FractalMIDI** (`models/ar_generator.py`, lines 246-285):
```python
def forward(self, piano_rolls, cond_list):
    patches = self.patchify(piano_rolls)
    # No explicit masking or prefix handling
    cond_list_next = self.predict(patches, cond_list, input_pos=None)
    # Returns all patches for next level
    return patches_flat, cond_list_next, torch.tensor(0.0), stats
```

**Fractalgen** (`fractalgen/models/ar.py`, lines 437-453):
```python
def forward(self, imgs, cond_list):
    patches = self.patchify(imgs)
    mask = torch.ones(patches.size(0), patches.size(1)).to(patches.device)
    cond_list_next = self.predict(patches, cond_list)
    # Reshapes for next level
    return patches, cond_list_next, 0
```

**Differences**:
1. Fractalgen creates an explicit mask (though all ones for AR)
2. Both return all patches, so this is similar
3. FractalMIDI adds padding handling which is good

**Impact**: Minimal - both implementations are similar here

---

### Issue 4: Initialization and Stability (LOW-MEDIUM PRIORITY)

**FractalMIDI** (`models/ar_generator.py`, lines 106-119):
```python
def initialize_weights(self):
    torch.nn.init.normal_(self.pos_embed, std=.02)
    self.apply(self._init_weights)

def _init_weights(self, m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
```

**Fractalgen** (`fractalgen/models/ar.py`, lines 346-363):
```python
def initialize_weights(self):
    torch.nn.init.normal_(self.pos_embed_learned, std=.02)
    self.apply(self._init_weights)

def _init_weights(self, m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        # Same initialization
```

**Differences**: Very similar initialization strategies

**Impact**: Minimal - initialization looks reasonable

---

### Issue 5: Generation Loop Implementation (MEDIUM PRIORITY)

**FractalMIDI** (`models/ar_generator.py`, lines 287-381):
```python
def sample(self, ...):
    # O(N²) approach: re-process entire history at each step
    for step_idx in range(actual_seq_len):
        if step_idx == 0:
            history = torch.zeros(bsz, 0, self.patch_size**2, ...)
        else:
            history = canvas[:, :step_idx, :]
        
        # Predict next token by passing full history
        conds = self.predict(history, cond_list, input_pos=step_idx)
```

**Fractalgen** (`fractalgen/models/ar.py`, lines 455-499):
```python
def sample(self, ...):
    # Uses KV cache for O(N) generation
    self.setup_caches(max_batch_size=..., max_seq_length=num_iter)
    
    for step in range(num_iter):
        # Only process new token with KV cache
        cond_list_next = self.predict(patches, cond_list, 
                                      input_pos=torch.Tensor([step]).int())
```

**Differences**:
1. **Efficiency**: Fractalgen uses KV cache (O(N)), FractalMIDI re-processes history (O(N²))
2. **Correctness**: Both should work, but FractalMIDI's approach is slower
3. **Bug in FractalMIDI**: The `input_pos` parameter usage is incorrect (Issue #1)

**Impact**:
- Slower generation (but should still work if correct)
- The bug in `input_pos` handling makes generation incorrect

---

## Root Cause Analysis

The **primary issue** is **Issue #1**: The incorrect `predict()` function logic.

When generating:
1. At step k, we pass `input_pos=k` to predict token at position k
2. Current code: `middle_cond = x[:, input_pos : input_pos + 1]` extracts hidden state at position k
3. But this hidden state has already seen token k (due to causal attention)!
4. We should use the hidden state at position k-1 to predict token k

This misalignment causes:
- Training: Model learns wrong associations
- Generation: Model uses wrong context for prediction
- Result: Noise or sparse/random notes

**Secondary issues** (Issues #2, #3, #5) compound the problem but are not the root cause.

---

## Recommended Fix Priority

### Priority 1 (CRITICAL - Fix Immediately):
1. **Fix `predict()` function indexing** (Issue #1)
   - Correct the causal prediction logic
   - Ensure training and generation use consistent indexing

### Priority 2 (HIGH - Fix Soon):
2. **Improve position embeddings** (Issue #2)
   - Option A: Better initialization and warmup
   - Option B: Implement 2D rotary embeddings

### Priority 3 (MEDIUM - Optimize Later):
3. **Add KV cache for generation** (Issue #5)
   - Makes generation much faster
   - Not critical for correctness if Issue #1 is fixed

### Priority 4 (LOW - Monitor):
4. **Training stability improvements** (Issue #4)
   - Add gradient clipping (already done in trainer)
   - Monitor loss curves
   - Adjust learning rate if needed

---

## Testing Plan

After fixes:
1. **Unit test**: Test `predict()` function with known inputs
2. **Small model test**: Train a tiny AR model (2 layers, 64 dim) on a few samples
3. **Verify generation**: Check that generated notes are coherent
4. **Compare with MAR**: AR should be close to MAR quality (per fractalgen paper)
5. **Full training**: If small model works, retrain full model

---

## Code Changes Required

### File: `models/ar_generator.py`

1. Fix `predict()` function (lines 183-244)
2. Optionally improve position embedding initialization
3. Add diagnostic logging for attention patterns

### File: `trainer.py`

1. Add AR-specific metrics logging
2. Monitor per-level losses separately
3. Add gradient norm logging

### File: `models/velocity_loss.py`

1. Verify loss computation is correct (looks OK)
2. Add entropy logging (already done)

---

## Expected Outcomes

After fixing Issue #1:
- AR generation should produce coherent music
- Quality should be close to MAR (within 10-20% performance gap)
- Training loss should decrease smoothly
- Generated samples should have proper musical structure

If Issue #1 fix alone doesn't work:
- Proceed to Issue #2 (position embeddings)
- Check for other subtle bugs in data flow
- Verify velocity_loss layer is working correctly

---

## Conclusion

The AR generator has a **critical bug in the `predict()` function** that causes incorrect causal prediction. This is the primary reason for poor generation quality. Fixing this bug should significantly improve AR performance. Secondary improvements (position embeddings, KV cache) can further enhance quality and speed.

**Next Steps**: Implement fixes in order of priority, test incrementally, and monitor results.

