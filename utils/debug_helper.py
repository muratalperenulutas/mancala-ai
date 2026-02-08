import numpy as np
import tensorflow as tf
from utils.configs import Config

def mask_logits(logits, legal_mask):
    legal = np.array(legal_mask, dtype=logits.dtype)
    return logits * legal + (1.0 - legal) * Config.LARGE_NEG

def probs_from_logits(masked_logits):
    exp = np.exp(masked_logits - np.max(masked_logits, axis=1, keepdims=True))
    return exp / (np.sum(exp, axis=1, keepdims=True) + Config.EPS)

def entropy_from_probs(probs):
    return -np.sum(probs * np.log(probs + Config.EPS), axis=1)

def debug_single_batch(model, states, legal_mask, y_onehot=None):
    logits = model.predict(states, verbose=0)  # (B,6)
    masked_logits = mask_logits(logits, legal_mask)
    probs = probs_from_logits(masked_logits)
    ent = entropy_from_probs(probs)
    arg_before = np.argmax(logits, axis=1)
    arg_after = np.argmax(masked_logits, axis=1)
    illegal_before = 1.0 - np.mean([legal_mask[i, arg_before[i]] for i in range(len(arg_before))])
    illegal_after = 1.0 - np.mean([legal_mask[i, arg_after[i]] for i in range(len(arg_after))])
    
    loss = None
    if y_onehot is not None:
        loss = float(tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=y_onehot, 
                logits=masked_logits
            )
        ).numpy())
        
    print(f"DEBUG BATCH: loss={loss}, illegal_before={illegal_before:.3f}, illegal_after={illegal_after:.3f}, entropy_mean={ent.mean():.4f}")
    
    for i in range(min(5, logits.shape[0])):
        print(f"Sample {i}:")
        print(f"  logits: {np.round(logits[i],3)}")
        print(f"  masked: {np.round(masked_logits[i],3)}")
        print(f"  probs: {np.round(probs[i],3)}, entropy: {float(ent[i]):.4f}")
