"""
Export the trained microGPT weights (model.pkl) to weights.json so the browser
playground can load a pre-trained model. Pickle can't be read in the browser, so
we dump the config, vocabulary, and the flat parameter list as JSON.

Run this AFTER training (i.e. after model.pkl exists):
    python export_weights.py

Author: Nishant Tamilselvan
"""
import pickle
import json

# These MUST match the constants in microgpt.py so the browser rebuilds the
# identical architecture and parameter ordering.
n_layer, n_embd, block_size, n_head = 1, 16, 16, 4

# Rebuild the vocabulary exactly as microgpt.py does (sorted unique characters).
docs = [line.strip() for line in open('input.txt', encoding='utf-8') if line.strip()]
uchars = sorted(set(''.join(docs)))
vocab_size = len(uchars) + 1  # +1 for BOS

with open('model.pkl', 'rb') as f:
    weights = pickle.load(f)  # flat list[float], same order as params in microgpt.py

out = {
    'config': {
        'n_layer': n_layer,
        'n_embd': n_embd,
        'block_size': block_size,
        'n_head': n_head,
        'vocab_size': vocab_size,
    },
    'uchars': uchars,
    'weights': weights,
}

with open('weights.json', 'w', encoding='utf-8') as f:
    json.dump(out, f)

print(f"wrote weights.json: {len(weights)} params, vocab_size {vocab_size}, {len(uchars)} chars")
