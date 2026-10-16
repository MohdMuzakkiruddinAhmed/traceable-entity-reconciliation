"""Pinned, frozen MiniLM sentence embeddings. Inputs are visible text only."""
import hashlib
from pathlib import Path
import sqlite3
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from ..io import ROOT


class FrozenEmbeddings:
    def __init__(self):
        folder = ROOT / 'data/raw/models/all-MiniLM-L6-v2'
        self.tokenizer = Tokenizer.from_file(str(folder / 'tokenizer.json'))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=0, pad_token='[PAD]')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(folder / 'model_quint8_avx2.onnx'), options, providers=['CPUExecutionProvider'])
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.cache = sqlite3.connect(ROOT / 'data/expanded_embedding_cache.sqlite')
        self.cache.execute('CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY, vector BLOB)')

    @staticmethod
    def text(record):
        # No source UID, canonical/group IDs, category IDs, dates or transaction keys.
        return ' '.join(str(record.get(k) or '') for k in ['name', 'brand', 'description'])[:2000]

    def encode(self, texts):
        unique = list(dict.fromkeys(texts))
        keys = {s: hashlib.sha256(('MiniLM-pinned-1110a243-v1|' + s).encode()).hexdigest() for s in unique}
        vectors = {}
        missing = []
        for text in unique:
            value = self.cache.execute('SELECT vector FROM cache WHERE key=?', (keys[text],)).fetchone()
            if value:
                vectors[text] = np.frombuffer(value[0], dtype=np.float32)
            else:
                missing.append(text)
        for start in range(0, len(missing), 64):
            batch = missing[start:start + 64]
            encoded = self.tokenizer.encode_batch(batch)
            mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
            feed = {'input_ids': np.array([e.ids for e in encoded], dtype=np.int64), 'attention_mask': mask,
                    'token_type_ids': np.array([e.type_ids for e in encoded], dtype=np.int64)}
            hidden = self.session.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0]
            if hidden.ndim != 3:
                raise ValueError('Expected token embeddings for explicit mean pooling')
            pooled = (hidden * mask[:, :, None]).sum(axis=1) / np.maximum(1, mask.sum(axis=1))[:, None]
            pooled /= np.maximum(1e-12, np.linalg.norm(pooled, axis=1, keepdims=True))
            for text, vector in zip(batch, pooled.astype(np.float32)):
                vectors[text] = vector
                self.cache.execute('INSERT OR REPLACE INTO cache VALUES (?,?)', (keys[text], vector.tobytes()))
            self.cache.commit()
        return np.array([vectors[text] for text in texts], dtype=np.float32)

    def similarities(self, pairs, records):
        visible = list(dict.fromkeys(u for pair in pairs for u in pair))
        encoded = self.encode([self.text(records[u]) for u in visible])
        by_uid = dict(zip(visible, encoded))
        return np.array([float(by_uid[a] @ by_uid[b]) for a, b in pairs]).reshape(-1, 1)

    def close(self):
        self.cache.close()
