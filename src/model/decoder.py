import torch
import torch.nn as nn


class BeamSearchDecoder(nn.Module):
    def __init__(self, model, target_dict):
        super().__init__()

        self.target_dict = target_dict
        self.decoder_model = model
        self.beam_width = None

    def generate_sequence(self, keys, values, src_lengths, max_len, beam_width):
        self.max_len = max_len
        self.beam_width = beam_width

        # setup beams
        self.seq_length, self.batch_size = keys.shape[:2]

        if src_lengths is None:
            src_lengths = torch.full((self.batch_size,), fill_value=self.seq_length, device=values.device)

        keys, values, src_lengths = self.initialize_beams(keys, values, src_lengths)

        for i in range(1, self.max_len - 1):
            if self.active_mask.sum() == 0:
                break

            # extend beams
            lprobs = self.extend_beams(keys, values, src_lengths, i)

            # prune beams
            self.prune_beams(lprobs, i)

        # Set the last index of every beam as eos
        self.beam_search_seq[-1] = self.target_dict.eos()

        # Return the top candidates from every beam
        top_beam_candidates = self.scores.argmax(dim=1)
        padded_sequences = self.beam_search_seq[:, torch.arange(self.batch_size), top_beam_candidates]
        sequence_lengths = self.lengths[torch.arange(self.batch_size), top_beam_candidates]

        self.attns = torch.stack(self.attns, dim=0)

        return padded_sequences, sequence_lengths, self.attns

    def initialize_beams(self, keys, values, src_lengths):
        self.beam_search_seq = torch.full((self.max_len, self.batch_size, self.beam_width),
                                          fill_value=self.target_dict.pad(),
                                          dtype=torch.long,
                                          device=values.device)
        self.lengths = torch.zeros((self.batch_size, self.beam_width), dtype=torch.long, device=values.device)

        tokens = torch.full((self.batch_size,), fill_value=self.target_dict.sos(), dtype=torch.long,
                            device=values.device)
        self.beam_search_seq[0, :, :] = self.target_dict.sos()

        logits, self.hidden_states, self.context, attn = self.decoder_model.forward(keys, values, tokens, src_lengths,
                                                                                    hidden_states=None, context=None)

        self.scores, search_results = logits.log_softmax(dim=-1).topk(self.beam_width, sorted=False)
        self.beam_search_seq[1, :, :] = search_results

        self.active_mask = ~search_results.eq(self.target_dict.eos())
        self.lengths[~self.active_mask] = 2

        top_candidates = self.scores.argmax(dim=1)
        finished_beam_mask = self.active_mask[torch.arange(self.batch_size), top_candidates].eq(0)
        self.active_mask[finished_beam_mask.nonzero().squeeze(1)] = 0

        self.hidden_states = [[self.hidden_states[i][j].repeat_interleave(self.beam_width, dim=0) for j in range(2)] for
                              i in range(2)]
        self.attns = [attn.repeat_interleave(self.beam_width, dim=0)]
        self.context = self.context.repeat_interleave(self.beam_width, dim=0)
        keys = keys.repeat_interleave(self.beam_width, dim=1)
        values = values.repeat_interleave(self.beam_width, dim=1)
        src_lengths = src_lengths.repeat_interleave(self.beam_width, dim=0)

        return keys, values, src_lengths

    def extend_beams(self, keys, values, src_lengths, step_num):
        tokens = self.beam_search_seq[step_num].flatten()
        logits, self.hidden_states, self.context, attn = self.decoder_model.forward(keys, values, tokens, src_lengths,
                                                                                    self.hidden_states, self.context)
        scores = logits.reshape(self.batch_size, self.beam_width, -1).log_softmax(dim=-1)
        self.attns.append(attn)

        return scores

    def prune_beams(self, lprobs, step_num):
        # TODO: Check how broadcasting works in torch
        extended_scores = self.scores.unsqueeze(-1) + lprobs

        # print("beam search seq now: ", self.beam_search_seq[:step_num+1])
        # exit(0)

        for batch in range(self.batch_size):
            num_active_beams = self.active_mask[batch].sum()
            if num_active_beams > 0:
                candidate_scores, tokens = extended_scores[batch].topk(num_active_beams, sorted=False)
                candidate_scores[~self.active_mask[batch]] = float('-inf')

                pruned_scores, token_ids = candidate_scores.flatten().topk(num_active_beams, sorted=False)
                pruned_token_row_id = torch.floor(torch.div(token_ids, self.beam_width).float()).long()
                pruned_token_col_id = torch.fmod(token_ids, num_active_beams)
                pruned_tokens = tokens[pruned_token_row_id, pruned_token_col_id]

                active_ids = self.active_mask[batch].nonzero().squeeze(1)
                self.scores[batch, active_ids] = pruned_scores
                self.beam_search_seq[:step_num + 1, batch, active_ids] = self.beam_search_seq[:step_num + 1, batch,
                                                                         pruned_token_row_id]
                self.beam_search_seq[step_num + 1, batch, active_ids] = pruned_tokens

        eos_mask = self.beam_search_seq[step_num + 1].eq(self.target_dict.eos())
        self.active_mask[eos_mask] = 0
        self.lengths[eos_mask] = step_num + 2

        # Finish the entire beam if top candidate is finished
        top_candidates = self.scores.argmax(dim=1)
        finished_top_beams = self.active_mask[torch.arange(self.batch_size), top_candidates].eq(0)
        self.active_mask[finished_top_beams.nonzero().squeeze(1)] = 0


class GreedyDecoder(nn.Module):
    def __init__(self, model, target_dict, train=False):
        super().__init__()

        self.decoder_model = model
        self.target_dict = target_dict

    def generate_sequence(self, keys, values, target, src_lengths=None, sample_prob=0.0):

        device = values.device

        seq_length, batch_size = keys.shape[:2]
        if src_lengths is None:
            src_lengths = torch.full((batch_size,), fill_value=seq_length, device=device)

        max_len = target.shape[0]

        # Teacher forcing probability calculation: The sample_prob determines the
        # probability of getting a one from the bernoulli distribution.
        # Eg. If sample_prob = 0.3 then there is 30% chance of getting a 1 and 70%
        # chance of getting a zero.
        sampler = torch.distributions.bernoulli.Bernoulli(torch.tensor([sample_prob]))
        sample_steps = sampler.sample([max_len])
        sample_steps[0] = 0  # We never sample the first step, it will be <SOS> for every sequence.

        seq_logits = []
        attns = []
        hidden_states = None
        context = None

        for i, sample_step in enumerate(sample_steps):
            if sample_step:
                tokens = logits.argmax(dim=-1)
            else:
                tokens = target[i, :]
            # tokens = logits.argmax(dim=-1) if sample_step else target[i, :]
            logits, hidden_states, context, attn = self.decoder_model.forward(keys, values, tokens,
                                                                              src_lengths, hidden_states, context)

            attns.append(attn)
            seq_logits.append(logits)

        seq_logits = torch.stack(seq_logits, dim=0)
        attns = torch.stack(attns, dim=0)
        return seq_logits, attns
