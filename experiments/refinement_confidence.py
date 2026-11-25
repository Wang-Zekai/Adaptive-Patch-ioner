"""Greedy DeCap confidence, full-memory confidence, and selective reranking."""

import torch

from exploration_decode import ExplorationDecoder, RERANK_METHODS, lexical_consensus


TOKEN_SIGNALS = ('mean_token_logprob', 'token_entropy', 'min_token_logprob')
MEMORY_SIGNALS = ('memory_max_cosine', 'memory_margin', 'memory_entropy')
CONFIDENCE_SIGNALS = TOKEN_SIGNALS + MEMORY_SIGNALS
WEIGHTED_RANK = 'F_weighted_joint'


class RefinementDecoder(ExplorationDecoder):
    @torch.inference_mode()
    def project_with_signals(self, vectors, *, cached=True, signals=MEMORY_SIGNALS):
        """Return the usual prefix and CPU vectors of full-memory statistics."""
        signals = tuple(dict.fromkeys(signals))
        if not set(signals) <= set(MEMORY_SIGNALS):
            raise ValueError('Unknown memory confidence signal')
        if not signals:
            return self.project(vectors, cached=cached), {}

        rows = self._rows(vectors)
        normalized = self.normalized_memory if cached else None
        if normalized is None:
            normalized = self.memory / self.memory.norm(dim=-1, keepdim=True)
            if cached:
                self.normalized_memory = normalized

        chunks = []
        metrics = {signal: [] for signal in signals}
        for query in rows.split(self.batch_size):
            query = query / query.norm(dim=-1, keepdim=True)
            cosine = query @ normalized.T
            logits = cosine / 0.01
            probabilities = logits.softmax(dim=-1)
            embedding = probabilities @ self.memory
            if self.model.normalize:
                embedding /= embedding.norm(dim=-1, keepdim=True)
            chunks.append(embedding)

            if 'memory_margin' in metrics:
                nearest = cosine.topk(2, dim=-1).values
                metrics['memory_margin'].append((nearest[:, 0] - nearest[:, 1]).cpu())
                if 'memory_max_cosine' in metrics:
                    metrics['memory_max_cosine'].append(nearest[:, 0].cpu())
            elif 'memory_max_cosine' in metrics:
                metrics['memory_max_cosine'].append(cosine.amax(dim=-1).cpu())
            if 'memory_entropy' in metrics:
                entropy = logits.logsumexp(dim=-1) - (probabilities * logits).sum(dim=-1)
                metrics['memory_entropy'].append(entropy.clamp_min(0).cpu())
        return torch.cat(chunks), {name: torch.cat(values) for name, values in metrics.items()}

    @torch.inference_mode()
    def decode_prefix_with_signals(self, prefixes, *, cached=True, signals=TOKEN_SIGNALS):
        """Observe actual greedy token logits without a second decoder pass."""
        signals = tuple(dict.fromkeys(signals))
        if not set(signals) <= set(TOKEN_SIGNALS):
            raise ValueError('Unknown token confidence signal')
        decode = self._kv_decode if cached and self.kv_cache else self._official_decode
        descriptions = []
        metrics = {signal: [] for signal in signals}
        for prefix in prefixes.split(self.batch_size):
            if not metrics:
                descriptions.extend(decode(prefix))
                continue
            active = torch.ones(len(prefix), dtype=torch.bool, device=prefix.device)
            count = torch.zeros(len(prefix), device=prefix.device)
            total_logprob = torch.zeros_like(count)
            total_entropy = torch.zeros_like(count)
            min_logprob = torch.zeros_like(count)

            def observe(_module, _inputs, output):
                # Hook fires before the official decoder masks finished sequences.
                logits = output.logits[:, -1, :]
                log_normalizer = logits.logsumexp(dim=-1)
                probabilities = logits.softmax(dim=-1)
                token = probabilities.argmax(dim=-1)
                logprob = logits.gather(1, token[:, None]).squeeze(1) - log_normalizer
                count.add_(active)
                if 'mean_token_logprob' in metrics:
                    total_logprob.add_(logprob.masked_fill(~active, 0))
                if 'min_token_logprob' in metrics:
                    min_logprob.copy_(torch.minimum(min_logprob, logprob.masked_fill(~active, 0)))
                if 'token_entropy' in metrics:
                    entropy = log_normalizer - (probabilities * logits).sum(dim=-1)
                    total_entropy.add_(entropy.clamp_min(0).masked_fill(~active, 0))
                active.logical_and_((token != self.eos) & (token != self.period))

            handle = self.decap.decoder.register_forward_hook(observe)
            try:
                descriptions.extend(decode(prefix))
            finally:
                handle.remove()
            values = {'mean_token_logprob': total_logprob / count.clamp_min(1),
                      'token_entropy': total_entropy / count.clamp_min(1),
                      'min_token_logprob': min_logprob}
            for name in signals:
                metrics[name].append(values[name].cpu())
        return descriptions, {name: torch.cat(values) for name, values in metrics.items()}

    @torch.inference_mode()
    def decode_with_signals(self, vectors, *, cached=True, signals=CONFIDENCE_SIGNALS):
        """Return captions and the requested confidence tensors in input order."""
        signals = tuple(dict.fromkeys(signals))
        if not set(signals) <= set(CONFIDENCE_SIGNALS):
            raise ValueError('Unknown confidence signal')
        memory_keys = [name for name in signals if name in MEMORY_SIGNALS]
        token_keys = [name for name in signals if name in TOKEN_SIGNALS]
        prefixes, memory = self.project_with_signals(vectors, cached=cached, signals=memory_keys)
        captions, token = self.decode_prefix_with_signals(prefixes, cached=cached, signals=token_keys)
        return captions, {**memory, **token}

    @torch.inference_mode()
    def rank(self, candidates, original_vectors, detail_vectors, *, source_vectors=None, cached=True, methods=None):
        """Evaluate only requested policies; all evidence comes from model features/text."""
        if methods is None:
            methods = tuple(name for name in RERANK_METHODS
                            if name != 'H_decoder_confidence' or source_vectors is not None)
        if isinstance(methods, str):
            methods = (methods,)
        methods = tuple(dict.fromkeys(methods))
        if not methods or not set(methods) <= set((*RERANK_METHODS, WEIGHTED_RANK)):
            raise ValueError('Unknown reranker policy')
        count = len(candidates)
        if not count or count != len(original_vectors) or count != len(detail_vectors):
            raise ValueError('Candidate and visual feature counts differ')
        width = len(candidates[0])
        if width < 2 or any(len(pool) != width or not any(c.strip() for c in pool) for pool in candidates):
            raise ValueError('Need two or more candidates and one nonempty caption per region')

        active = set(methods)
        joint_needed = bool(active & {'H_joint_likelihood', 'H_joint_consensus', WEIGHTED_RANK})
        original_needed = joint_needed or 'H_original_likelihood' in active
        detail_needed = joint_needed or 'H_detail_likelihood' in active
        source_needed = bool(active & {'H_decoder_confidence', WEIGHTED_RANK})
        consensus_needed = bool(active & {'H_consensus', 'H_joint_consensus', WEIGHTED_RANK})
        flat = [caption for pool in candidates for caption in pool]
        computed = {}
        for name, feature, needed in [('original', original_vectors, original_needed),
                                      ('detail', detail_vectors, detail_needed)]:
            if needed:
                prefix = self.project(feature, cached=cached).repeat_interleave(width, dim=0)
                computed[name] = self._likelihoods(prefix, flat).view(count, width)
        if joint_needed:
            computed['joint'] = (computed['original'] + computed['detail']) / 2
        if source_needed:
            if source_vectors is None or source_vectors.shape != (count, width, self.memory.shape[1]):
                raise ValueError('Source features must align with each caption candidate')
            sources = self.project(source_vectors.flatten(0, 1), cached=cached)
            computed['source'] = self._likelihoods(sources, flat).view(count, width)
        if consensus_needed:
            computed['consensus'] = torch.tensor([lexical_consensus(pool) for pool in candidates], device=self.device)
        if 'H_joint_consensus' in active:
            computed['joint_consensus'] = computed['joint'] + 0.35 * computed['consensus']
        if WEIGHTED_RANK in active:
            computed['weighted'] = 0.7 * computed['joint'] + 0.3 * computed['source'] + 0.35 * computed['consensus']

        nonempty = torch.tensor([[bool(c.strip()) for c in pool] for pool in candidates], device=self.device)
        mapping = {'H_original_likelihood': 'original', 'H_detail_likelihood': 'detail',
                   'H_joint_likelihood': 'joint', 'H_decoder_confidence': 'source',
                   'H_consensus': 'consensus', 'H_joint_consensus': 'joint_consensus', WEIGHTED_RANK: 'weighted'}
        return {method: computed[mapping[method]].masked_fill(~nonempty, -torch.inf).argmax(dim=-1).tolist()
                for method in methods}
