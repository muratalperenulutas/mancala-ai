import numpy as np
import tensorflow as tf
from utils.configs import Config, Constant

def mask_logits(logits, legal_mask):
    legal = np.array(legal_mask, dtype=logits.dtype)
    return logits * legal + (1.0 - legal) * Constant.LARGE_NEG

def probs_from_logits(masked_logits):
    shifted = masked_logits - np.max(masked_logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / (np.sum(exp, axis=1, keepdims=True) + Constant.EPS)

def entropy_from_probs(probs):
    return -np.sum(probs * np.log(probs + Constant.EPS), axis=1)

def debug_single_batch(model, states, legal_mask, y_onehot=None):
    logits = model.predict(states, verbose=0)
    masked_logits = mask_logits(logits, legal_mask)
    probs = probs_from_logits(masked_logits)
    ent = entropy_from_probs(probs)
    
    arg_before = np.argmax(logits, axis=1)
    arg_after = np.argmax(masked_logits, axis=1)
    
    illegal_before = 1.0 - np.mean([legal_mask[i, arg_before[i]] for i in range(len(arg_before))])
    illegal_after = 1.0 - np.mean([legal_mask[i, arg_after[i]] for i in range(len(arg_after))])
    
    loss_str = "N/A"
    if y_onehot is not None:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_onehot, logits=masked_logits))
        loss_str = f"{float(loss.numpy()):.4f}"
        
    print(f"\n--- DEBUG BATCH ---")
    print(f"Loss: {loss_str} | Illegal (Pre/Post): {illegal_before:.3f}/{illegal_after:.3f} | Entropy: {ent.mean():.4f}")
    
    for i in range(min(3, logits.shape[0])):
        print(f"Sample {i}:")
        print(f"  Logits: {np.round(logits[i], 2)}")
        print(f"  Probs : {np.round(probs[i], 3)} (H={ent[i]:.4f})")
    print("-------------------\n")
