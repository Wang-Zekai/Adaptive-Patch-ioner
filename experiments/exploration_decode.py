"""Batched, frozen Patch-ioner text projection, decoding, and reranking."""

from functools import lru_cache
import re

import torch


RERANK_METHODS = (
    "H_original_likelihood", "H_detail_likelihood", "H_joint_likelihood",
    "H_decoder_confidence", "H_consensus", "H_joint_consensus",
)


class ExplorationDecoder:
    def __init__(self, captioner, *, kv_cache=False, projection_batch_size=24):
        from src.decap import decap as upstream

        self.model = captioner.model
        self.device = captioner.device
        self.memory = self.model.im_proj.embs_dataset
        self.decap = self.model.decoder
        self.upstream = upstream
        self.tokenizer = self.model._get_decoder_tokenizer()
        family = (self.model.decoder_config or {}).get("decoder_family")
        self.family = family.lower() if family else None
        self.eos = upstream._get_default_eos_token_id(self.tokenizer, self.family)
        self.period = upstream._get_default_full_stop_token_id(self.tokenizer, self.family)
        self.kv_cache = kv_cache
        self.batch_size = int(projection_batch_size)
        self.normalized_memory = None
        strategy = getattr(self.model, "decoding_strategy", None) or ""
        if (self.model.embed_inversion or self.model.calculate_argmax_text or "beam" in strategy.lower()
                or getattr(self.model, "batched_decoding_config", None) or self.memory.dtype != torch.float32
                or self.decap.decoder.config.model_type != "gpt2" or self.batch_size <= 0):
            raise ValueError("Expected pinned FP32 Talk2DINO + greedy DeCap Patch-ioner")
        self.encode_text = lru_cache(maxsize=8192)(self._encode_text)

    def _rows(self, vectors):
        if (vectors.ndim != 2 or not len(vectors) or vectors.shape[1] != self.memory.shape[1]
                or not torch.isfinite(vectors).all() or not (vectors.norm(dim=-1) > 0).all()):
            raise ValueError("Expected a nonempty batch of finite nonzero region features")
        return vectors.to(device=self.device, dtype=torch.float32)

    def clear_projection_cache(self):
        """Release the optional second copy before measuring online inference."""
        self.normalized_memory = None

    @torch.inference_mode()
    def project(self, vectors, *, cached=True):
        rows = self._rows(vectors)
        prefix = []
        if cached and self.normalized_memory is None:
            self.normalized_memory = self.memory / self.memory.norm(dim=-1, keepdim=True)
        for query in rows.split(self.batch_size):
            if not cached:
                prefix.append(self.model.im_proj.project(query.clone(), normalize=self.model.normalize))
                continue
            query = query / query.norm(dim=-1, keepdim=True)
            similarities = query @ self.normalized_memory.T
            probabilities = (similarities / 0.01).softmax(dim=-1)
            embedding = probabilities @ self.memory
            if self.model.normalize:
                embedding /= embedding.norm(dim=-1, keepdim=True)
            prefix.append(embedding)
        return torch.cat(prefix, dim=0)

    def _official_decode(self, prefixes):
        return self.upstream.decoding_batched(
            self.decap, prefixes, decoding_method=self.model.decoding_method,
            tokenizer=self.tokenizer, decoder_family=self.family,
        )

    def _kv_decode(self, prefixes):
        # Preserve the official greedy vocabulary, stopping conditions, and formatting.
        lm = self.decap.decoder
        embedding = lm.get_input_embeddings()
        current = self.decap.clip_project(prefixes).unsqueeze(1)
        history = None
        finished = torch.zeros(len(prefixes), dtype=torch.bool, device=self.device)
        tokens = []
        for step in range(30):
            output = lm(inputs_embeds=current, past_key_values=history, use_cache=True)
            history = output.past_key_values
            logits = output.logits[:, -1, :]
            logits[finished] = -torch.inf
            logits[finished, self.eos] = 0
            new_token = logits.softmax(dim=-1).argmax(dim=-1)
            finished |= (new_token == self.eos) | (new_token == self.period)
            next_input = new_token.masked_fill(finished, self.eos)
            tokens.append(next_input if step == 0 else new_token)
            if finished.all():
                break
            current = embedding(next_input[:, None])
        token_lists = torch.stack(tokens, dim=1).tolist()
        decode_tokens = self.model.decoding_method
        if decode_tokens is None:
            def decode_tokens(values):
                return self.upstream._decode_token_list(self.tokenizer, values)
        return [decode_tokens(values).split("<|endoftext|>", 1)[0].replace("<|startoftext|>", "")
                for values in token_lists]

    @torch.inference_mode()
    def decode(self, vectors, *, cached=True):
        if not cached:
            return [caption for rows in self._rows(vectors).split(self.batch_size)
                    for caption in self.model.caption_tokens(rows.clone())]
        decoder = self._kv_decode if self.kv_cache else self._official_decode
        return [caption for prefixes in self.project(vectors).split(self.batch_size)
                for caption in decoder(prefixes)]

    def _encode_text(self, text):
        ids = list(self.tokenizer.encode(text)[:30])
        if ids and len(ids) < 30 and ids[-1] not in (self.period, self.eos):
            ids.append(self.eos)
        return ids

    def _likelihoods(self, prefix_vectors, captions):
        scores = []
        for index, prefix in enumerate(prefix_vectors.split(self.batch_size)):
            start = index * self.batch_size
            encoded = [self.encode_text(caption) for caption in captions[start:start + len(prefix)]]
            width = max(1, *(len(ids) for ids in encoded))
            targets = torch.tensor([ids + [self.eos] * (width - len(ids)) for ids in encoded], device=self.device)
            length = torch.tensor([len(ids) for ids in encoded], device=self.device)
            mask = torch.arange(width, device=self.device)[None] < length[:, None]
            logits = self.decap(prefix, targets[:, :-1]).logits
            selected = logits.gather(-1, targets[..., None]).squeeze(-1)
            token_scores = selected - logits.logsumexp(dim=-1)
            values = (token_scores.masked_fill(~mask, 0).sum(dim=-1) / length.clamp_min(1))
            scores.append(values.masked_fill(length == 0, -torch.inf))
        return torch.cat(scores)

    @torch.inference_mode()
    def rank(self, candidates, original_vectors, detail_vectors, *, source_vectors=None, cached=True):
        """Return chosen candidate indices using the decoder and candidate text only.

        source_vectors, when provided, is [regions, candidates, features] and
        contains the actual features that generated the respective candidates.
        """
        count = len(candidates)
        if count != len(original_vectors) or count != len(detail_vectors) or count == 0:
            raise ValueError("Need aligned nonempty candidate, original and detail batches")
        width = len(candidates[0])
        if width < 2 or any(len(pool) != width or not any(c.strip() for c in pool) for pool in candidates):
            raise ValueError("Need at least two aligned candidates, including a nonempty caption for every region")
        flat = [caption for pool in candidates for caption in pool]
        original = self.project(original_vectors, cached=cached).repeat_interleave(width, dim=0)
        detail = self.project(detail_vectors, cached=cached).repeat_interleave(width, dim=0)
        orig_score = self._likelihoods(original, flat).view(count, width)
        detail_score = self._likelihoods(detail, flat).view(count, width)
        joint = (orig_score + detail_score) / 2
        consensus = torch.tensor([lexical_consensus(pool) for pool in candidates], device=self.device)
        options = {
            "H_original_likelihood": orig_score,
            "H_detail_likelihood": detail_score,
            "H_joint_likelihood": joint,
            "H_consensus": consensus.masked_fill(~joint.isfinite(), -torch.inf),
            "H_joint_consensus": joint + 0.35 * consensus,
        }
        if source_vectors is not None:
            if source_vectors.shape != (count, width, self.memory.shape[1]):
                raise ValueError("Source features must align with each caption candidate")
            prefixes = self.project(source_vectors.flatten(0, 1), cached=cached)
            options["H_decoder_confidence"] = self._likelihoods(prefixes, flat).view(count, width)
        return {name: matrix.argmax(dim=-1).tolist() for name, matrix in options.items()}


def lexical_consensus(candidates):
    words = [set(re.findall(r"\w+", caption.lower())) for caption in candidates]
    result = []
    for i, candidate in enumerate(words):
        similarities = [len(candidate & peer) / max(1, len(candidate | peer))
                        for j, peer in enumerate(words) if j != i]
        result.append(sum(similarities) / len(similarities))
    return result
