import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from .decoder import BeamSearchDecoder, GreedyDecoder
from .attention import Attention
from .modelutils import BeamSearch
from .locked_dropout import LockedDropout

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# BLSTM layer for pBLSTM
# Step 1. Reduce time resolution to half
# Step 2. Run through BLSTM
# Note the input should have timestep%2 == 0
class pBLSTMLayer(nn.Module):
    def __init__(self, input_feature_dims, hidden_dim):
        super(pBLSTMLayer, self).__init__()
        self.BiLSTM = nn.LSTM(input_size=input_feature_dims,
                              hidden_size=hidden_dim,
                              bidirectional=True)

    def forward(self, x, x_len):

        rnn_input, rnn_input_len = self.condense_input(x, x_len)
        padded_input = pack_padded_sequence(rnn_input, lengths=rnn_input_len, batch_first=False,
                                            enforce_sorted=False)

        # Bidirectional RNN
        output, hidden = self.BiLSTM(padded_input)
        return output, hidden

    def condense_input(self, rnn_input, input_len, batch_first=False):

        if batch_first:
            batch_size, seq_len, inp_dim = rnn_input.shape
        else:
            seq_len, batch_size, inp_dim = rnn_input.shape

        if seq_len % 2 != 0:
            rnn_input = rnn_input[:, :-1, :] if batch_first else rnn_input[:-1, :, :]
            seq_len -= 1

        if batch_first:
            rnn_input = rnn_input.reshape(batch_size, seq_len // 2, inp_dim * 2)
        else:
            rnn_input = rnn_input.transpose(0, 1)
            rnn_input = rnn_input.reshape(batch_size, seq_len // 2, inp_dim * 2)
            rnn_input = rnn_input.transpose(0, 1)

        return rnn_input, input_len // 2


# Listener is a pBLSTM stacking 3 layers to reduce time resolution 8 times
# Input shape should be [# of sample, timestep, features]
class Listener(nn.Module):
    def __init__(self, input_feature_dims, listener_hidden_dims, num_layers, key_size, value_size, dropout=0.0):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        assert self.num_layers >= 1, 'Listener should have at least 1 layer'

        self.LSTM_layers = []
        self.LockedDropout_layers = []

        # default layer
        self.LSTM_layers.append(nn.LSTM(input_feature_dims, listener_hidden_dims, bidirectional=True))
        self.LockedDropout_layers.append(LockedDropout())

        # pyramidal layers
        for i in range(self.num_layers):
            self.LSTM_layers.append(pBLSTMLayer(4 * listener_hidden_dims, listener_hidden_dims))
            self.LockedDropout_layers.append(LockedDropout())

        self.LSTM_layers = nn.ModuleList(self.LSTM_layers)
        self.LockedDropout_layers = nn.ModuleList(self.LockedDropout_layers)

        self.key_network = nn.Linear(listener_hidden_dims * 2, key_size)
        self.value_network = nn.Linear(listener_hidden_dims * 2, value_size)

    def forward(self, input_x, lengths):
        rnn_input = pack_padded_sequence(input_x, lengths, enforce_sorted=False)

        output, _ = self.LSTM_layers[0](rnn_input)
        output, _ = pad_packed_sequence(output)
        output = self.LockedDropout_layers[0](output, self.dropout)

        for pBLSTM, locked_dropout in zip(self.LSTM_layers[1:], self.LockedDropout_layers[1:]):
            output, _ = pBLSTM(output, lengths)
            output, lengths = pad_packed_sequence(output)
            output = locked_dropout(output, self.dropout)

        keys = self.key_network(output)
        values = self.value_network(output)

        return keys, values, lengths
# class pBLSTM(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super(pBLSTM, self).__init__()
#         # Double input dim to accommodate two time-steps of information
#         self.blstm = nn.LSTM(input_size=input_dim * 2, hidden_size=hidden_dim, num_layers=1, bidirectional=True)
#
#     def forward(self, x_padded, lens, batch_first=False):
#         """
#         :param x_padded: (L, N, input_dim)  padded sequence
#         :param lens: (N, )
#         :return output: packed sequence
#         :return hidden: h_t, c_t
#         """
#         reshaped_x, half_lens = self._halve_time_resolution(x_padded, lens, batch_first=batch_first)  # (L//2, N, input_dim*2), (N, )
#         rnn_inp = pack_padded_sequence(reshaped_x, lengths=half_lens, batch_first=batch_first, enforce_sorted=False)
#         output, hidden = self.blstm(rnn_inp)
#         return output, hidden
#
#     def _halve_time_resolution(self, x, lens, batch_first=False):
#         """
#         :param x: (L, N, input_dim)  padded sequence
#         :param lens: (N, )
#         """
#         if batch_first:
#             N, L, D = x.shape
#         else:
#             L, N, D = x.shape
#
#         # if L is odd, truncate to even
#         if L % 2 != 0:
#             x = x[:, :-1, :] if batch_first else x[:-1, :, :]
#             L -= 1
#
#         # compress two time-steps of information into one
#         if batch_first:
#             x = x.reshape(N, L // 2, D * 2)
#         else:
#             x = x.transpose(0, 1)
#             x = x.reshape(N, L // 2, D * 2)
#             x = x.transpose(0, 1)
#
#         lens = lens // 2
#         return x, lens
#
#
# class Listener(nn.Module):
#     '''
#     Encoder takes the utterances as inputs and returns the key and value.
#     Key and value are nothing but simple projections of the output from pBLSTM network.
#     '''
#     def __init__(self, input_dim, hidden_dim, value_size, key_size, num_pyramidal_layers=3, dropout=0.0):
#         super().__init__()
#         self.dropout = dropout
#         # Initial BLSTM layer
#         self.blstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True)
#         self.locked_drop1 = LockedDropout()
#
#         # pBLSTM layers
#         p_layers = [pBLSTM(hidden_dim * 2, hidden_dim) for _ in range(num_pyramidal_layers)]
#         pdrops = [LockedDropout() for _ in range(num_pyramidal_layers)]
#         self.pblstms = nn.ModuleList(p_layers)
#         self.pdrops = nn.ModuleList(pdrops)
#
#         # output layers
#         self.key_network = nn.Linear(hidden_dim * 2, value_size)
#         self.value_network = nn.Linear(hidden_dim * 2, key_size)
#
#     def forward(self, x, lens):
#         """Puts the input x and lengths through the pBLSTM network and returns keys, values and their reduced lengths"""
#         rnn_inp = pack_padded_sequence(x, lengths=lens, batch_first=False, enforce_sorted=False)
#
#         outputs, _ = self.blstm(rnn_inp)
#         outputs, _ = pad_packed_sequence(outputs)  # (L, N, input_dim), _
#         outputs = self.locked_drop1(outputs, self.dropout)
#
#         for pblstm, locked_drop in zip(self.pblstms, self.pdrops):
#             outputs, _ = pblstm(outputs, lens)
#             outputs, lens = pad_packed_sequence(outputs)
#             outputs = locked_drop(outputs, self.dropout)
#
#         keys = self.key_network(outputs)
#         values = self.value_network(outputs)
#
#         return keys, values, lens.to(x.device)

class Speller(nn.Module):
    def __init__(self, speller_hidden_dims, embedding_dims, vocab_size,
                 key_size, value_size, isAttended=False, mode=None, tgt_dict=None):
        super(Speller, self).__init__()

        self.target_dict = tgt_dict
        self.embedding = nn.Embedding(vocab_size, embedding_dims, padding_idx=self.target_dict.pad())
        self.lstm1 = nn.LSTMCell(input_size=embedding_dims + value_size, hidden_size=speller_hidden_dims)
        self.lstm2 = nn.LSTMCell(input_size=speller_hidden_dims, hidden_size=key_size)

        self.isAttended = isAttended
        if isAttended:
            self.attention = Attention(mode)

        self.charProb = nn.Linear(key_size + value_size, vocab_size)

        # TODO: Remove the two lines of code below if they don't work.
        # self.MAX_DECODE_LEN = max_decode_len
        # self.VOCAB_SIZE = vocab_size

    def forward(self, keys, values, tokens, src_lengths=None, hidden_states=None, context=None):
        """
        :param keys: (T, N, key_size) Output of the Encoder Key projection layer
        :param values: (T, N, value_size) Output of the Encoder Value projection layer
        :param tokens: (N, text_len) Batch input of text with text_length
        :param src_lengths:
        :param hidden_states:
        :param context:
        :return predictions: Returns the character perdiction probability
        """

        seq_len, batch_size = keys.shape[:2]

        if src_lengths is None:
            src_lengths = torch.full((batch_size,), fill_value=seq_len)

        if hidden_states is None:
            hidden_states = [None] * 2

        if context is None:
            context = torch.zeros_like(values[0, :, :], device=values.device)

        input_embed = self.embedding(tokens)
        next_states = []

        inp = torch.cat([input_embed, context], dim=1)
        next_states.append(self.lstm1(inp, hidden_states[0]))

        inp2 = next_states[-1][0]
        next_states.append(self.lstm2(inp2, hidden_states[1]))

        query_state = next_states[-1][0]
        context, attn = self.attention(query=query_state, key=keys, value=values, mask_lengths=src_lengths)
        logits = self.charProb(torch.cat([query_state, context], dim=1))

        return logits, next_states, context, attn

    # def beam_decoder(self, keys, values, src_key_lens=None, max_len=250, beam_width=3):
    #     """
    #     :param key :(S, N, key_size) Output of the Encoder Key projection layer
    #     :param values: (S, N, value_size) Output of the Encoder Value projection layer
    #     :params src_key_lens: (N, ) lengths of source sequences in the batch
    #     :params max_len: maximum sequence length to generate
    #     :params beam_width: beam width for the beam search algorithm
    #     :return padded_seqs: (max_len, N) output sequences (with <sos> and <eos> tokens)
    #     :return seq_lens: (N, ) includes <sos> and <eos> tokens
    #     :return attns: (max_len, N, S)
    #     """
    #     S, N = keys.shape[:2]
    #     if src_key_lens is None:
    #         src_key_lens = torch.full((N,), fill_value=S, device=values.device)  # default full length
    #
    #     self.pad_idx = self.target_dict.pad()
    #     self.sos_idx = self.target_dict.sos()
    #     self.eos_idx = self.target_dict.eos()
    #
    #     search = BeamSearch(max_len, self.pad_idx, self.sos_idx, self.eos_idx, N, beam_width)
    #     search.seqs = torch.full((max_len, N, beam_width), fill_value=self.pad_idx, dtype=torch.long,
    #                              device=values.device)  # (T, N, K)
    #     search.seqs[0, :, :] = self.sos_idx
    #     attns = []
    #
    #     # Generate step 0
    #     tokens = torch.full((N,), fill_value=self.sos_idx, dtype=torch.long, device=values.device)  # (N, )
    #     context = None
    #     hidden_states = None
    #     step_logits, hidden_states, context, attn = self.forward(keys, values, tokens, src_key_lens, hidden_states,
    #                                                              context)
    #     attns.append(attn.repeat_interleave(beam_width, dim=0))
    #     search.step(step_logits, 0)
    #
    #     # Broadcast N inputs from encoder into N * K inputs for subsequent steps
    #     hidden_states = [
    #         [hidden_states[0][0].repeat_interleave(beam_width, dim=0),
    #          hidden_states[0][1].repeat_interleave(beam_width, dim=0)],
    #         [hidden_states[1][0].repeat_interleave(beam_width, dim=0),
    #          hidden_states[1][1].repeat_interleave(beam_width, dim=0)]
    #     ]
    #     context = context.repeat_interleave(beam_width, dim=0)
    #     keys = keys.repeat_interleave(beam_width, dim=1)
    #     values = values.repeat_interleave(beam_width, dim=1)
    #     src_key_lens = src_key_lens.repeat_interleave(beam_width, dim=0)
    #
    #     # Generate subsequent steps
    #     for step_num in range(1, max_len - 1):  # -1 because final token must be <eos>
    #         if search.active_mask.sum() == 0:
    #             break
    #         # Step
    #         tokens = search.seqs[step_num].flatten()  # (N * K, )
    #         step_logits, hidden_states, context, attn = self.forward(keys, values, tokens, src_key_lens, hidden_states,
    #                                                                  context)
    #         attns.append(attn)
    #         lprob = search.step(step_logits, step_num)
    #     search.seqs[-1, :, :] = self.eos_idx  # ensure all seqs end with <eos> even if not predicted by model
    #
    #     # Return top candidate from each beam
    #     top_cands = torch.argmax(search.scores, dim=1)  # (N, )
    #     padded_seqs = search.seqs[:, torch.arange(search.N), top_cands]  # (T, N)
    #     seq_lens = search.lens[torch.arange(search.N), top_cands]  # (N, )
    #
    #     attns = torch.stack(attns, dim=0)
    #     return padded_seqs, seq_lens, attns

    def beam_decoder(self, keys, values, src_key_lens=None, max_len=250, beam_width=1):
        """
        :param key :(S, N, key_size) Output of the Encoder Key projection layer
        :param values: (S, N, value_size) Output of the Encoder Value projection layer
        :params src_key_lens: (N, ) lengths of source sequences in the batch
        :params max_len: maximum sequence length to generate
        :params beam_width: beam width for the beam search algorithm
        :return padded_seqs: (max_len, N) output sequences (with <sos> and <eos> tokens)
        :return seq_lens: (N, ) includes <sos> and <eos> tokens
        :return attns: (max_len, N, S)
        """

        self.pad_idx = self.target_dict.pad()
        self.sos_idx = self.target_dict.sos()
        self.eos_idx = self.target_dict.eos()

        S, N = keys.shape[:2]
        if src_key_lens is None:
            src_key_lens = torch.full((N,), fill_value=S, device=values.device)  # default full length

        search = BeamSearch(max_len, self.pad_idx, self.sos_idx, self.eos_idx, N, beam_width)
        search.seqs = torch.full((max_len, N, beam_width), fill_value=self.pad_idx, dtype=torch.long,
                                 device=values.device)  # (T, N, K)
        search.seqs[0, :, :] = self.sos_idx
        attns = []

        # Generate step 0
        tokens = torch.full((N,), fill_value=self.sos_idx, dtype=torch.long, device=values.device)  # (N, )
        context = None
        hidden_states = None
        step_logits, hidden_states, context, attn = self.forward(keys, values, tokens, src_key_lens, hidden_states,
                                                                 context)
        attns.append(attn.repeat_interleave(beam_width, dim=0))
        search.step(step_logits, 0)

        # Broadcast N inputs from encoder into N * K inputs for subsequent steps
        hidden_states = [
            [hidden_states[0][0].repeat_interleave(beam_width, dim=0),
             hidden_states[0][1].repeat_interleave(beam_width, dim=0)],
            [hidden_states[1][0].repeat_interleave(beam_width, dim=0),
             hidden_states[1][1].repeat_interleave(beam_width, dim=0)]
        ]
        context = context.repeat_interleave(beam_width, dim=0)
        keys = keys.repeat_interleave(beam_width, dim=1)
        values = values.repeat_interleave(beam_width, dim=1)
        src_key_lens = src_key_lens.repeat_interleave(beam_width, dim=0)

        # Generate subsequent steps
        for step_num in range(1, max_len - 1):  # -1 because final token must be <eos>
            if search.active_mask.sum() == 0:
                break
            # Step
            tokens = search.seqs[step_num].flatten()  # (N * K, )
            step_logits, hidden_states, context, attn = self.forward(keys, values, tokens, src_key_lens, hidden_states,
                                                                     context)
            attns.append(attn)
            search.step(step_logits, step_num)
        search.seqs[-1, :, :] = self.eos_idx  # ensure all seqs end with <eos> even if not predicted by model

        # Return top candidate from each beam
        top_cands = torch.argmax(search.scores, dim=1)  # (N, )
        padded_seqs = search.seqs[:, torch.arange(search.N), top_cands]  # (T, N)
        seq_lens = search.lens[torch.arange(search.N), top_cands]  # (N, )

        attns = torch.stack(attns, dim=0)
        return padded_seqs, seq_lens, attns

    def greedy_decoder(self, keys, values, y, src_key_lens=None, sample_prob=0.0, search_fn=lambda x: x.argmax(dim=-1)):
        '''
        :param key :(S, N, key_size) Output of the Encoder Key projection layer
        :param values: (S, N, value_size) Output of the Encoder Value projection layer
        :param y: (max_len, N) Batch input of target text EXCLUDING <eos> tokens
        :params src_key_lens: (N, ) lengths of source sequences in the batch INCLUDING <eos> tokens
        :params sample_prob: float probability of using model output as next time step input.
                             0.0 achieves teacher forcing, 1.0 achieves Greedy search
        :params search_fn: function for selecting tokens (N, ) given logits from one time step (N, V)
        :return seq_logits: (max_len, N, V)
        :return attns: (max_len, N, S)
        '''
        S, N = keys.shape[:2]
        if src_key_lens is None:
            src_key_lens = torch.full((N,), fill_value=S, device=values.device)  # default full length

        # Schedule of when to use inputs from model instead of teacher forced inputs
        max_len = y.shape[0]
        sampler = torch.distributions.bernoulli.Bernoulli(torch.tensor([sample_prob]))
        sample_schedule = sampler.sample([max_len])  # (max_len, )
        sample_schedule[0] = 0  # never sample first step (should be <sos> for every sequence in the batch)

        # Generate through time
        seq_logits = []
        attns = []
        hidden_states = None
        context = None
        for i, sample_this_step in enumerate(sample_schedule):
            if sample_this_step:
                tokens = search_fn(step_logits)  # (N, )
                assert tokens.shape == (N,), f"tokens.shape was {tokens.shape}"  # check search_fn output size
            else:
                tokens = y[i, :]  # (N, )
            step_logits, hidden_states, context, attn = self.forward(keys, values, tokens, src_key_lens, hidden_states, context)
            attns.append(attn)
            seq_logits.append(step_logits)

        seq_logits = torch.stack(seq_logits, dim=0)
        attns = torch.stack(attns, dim=0)
        return seq_logits, attns


    # def greedy_decoder(self, keys, values, target, src_lengths=None, sample_prob=0.0):
    #
    #     device = values.device
    #
    #     seq_length, batch_size = keys.shape[:2]
    #     if src_lengths is None:
    #             src_lengths = torch.full((batch_size,), fill_value=seq_length, device=device)
    #
    #     max_len = target.shape[0]
    #
    #     # Teacher forcing probability calculation: The sample_prob determines the
    #     # probability of getting a one from the bernoulli distribution.
    #     # Eg. If sample_prob = 0.3 then there is 30% chance of getting a 1 and 70%
    #     # chance of getting a zero.
    #     sampler = torch.distributions.bernoulli.Bernoulli(torch.tensor([sample_prob]))
    #     sample_steps = sampler.sample([max_len])
    #     sample_steps[0] = 0  # We never sample the first step, it will be <SOS> for every sequence.
    #
    #     seq_logits = []
    #     attns = []
    #     hidden_states = None
    #     context = None
    #
    #     for i, sample_step in enumerate(sample_steps):
    #         if sample_step:
    #             tokens = logits.argmax(dim=-1)
    #         else:
    #             tokens = target[i, :]
    #         # tokens = logits.argmax(dim=-1) if sample_step else target[i, :]
    #         logits, hidden_states, context, attn = self.forward(keys, values, tokens,
    #                                                             src_lengths, hidden_states, context)
    #
    #         attns.append(attn)
    #         seq_logits.append(logits)
    #
    #     seq_logits = torch.stack(seq_logits, dim=0)
    #     attns = torch.stack(attns, dim=0)
    #     return seq_logits, attns


class LAS(nn.Module):
    def __init__(self, configs, target_dict):
        super().__init__()
        self.listener = Listener(**configs["listener"])
        self.speller_model = Speller(tgt_dict=target_dict, **configs["speller"])
        self.greedy_decoder = GreedyDecoder(model=self.speller_model, target_dict=target_dict)
        self.beam_decoder = BeamSearchDecoder(model=self.speller_model, target_dict=target_dict)

    def forward(self, speech_input, speech_lens, transcript_inputs, sample_prob):
        keys, values, src_key_lens = self.listener.forward(speech_input, speech_lens)
        # logits, attns = self.speller_model.greedy_decoder(keys, values, transcript_inputs, src_key_lens, sample_prob)
        logits, attns = self.greedy_decoder.generate_sequence(keys, values, transcript_inputs, src_key_lens, sample_prob)
        return logits, src_key_lens, attns

    def predict(self, speech_input, speech_lens, max_len, beam_size):

        assert max_len is not None, "Specify max_decode length"
        assert beam_size is not None, "Specify beam size"

        keys, values, src_key_lens = self.listener.forward(speech_input, speech_lens)

        # m_beam_search_seq, m_lens, m_a_mask, m_scores = self.speller_model.beam_decoder(keys, values, src_key_lens, max_len, beam_size)
        #
        # beam_search_seq, lens, a_mask, scores = self.beam_decoder.generate_sequence(keys, values, src_key_lens, max_len, beam_size)

        # if not torch.equal(lens, m_lens):
        #     print("lens: ", lens)
        #     print("m_lens: ", m_lens)
        # else:
        #     print("lengths are equal")
        #
        # if not torch.equal(a_mask, m_a_mask):
        #     print("a_mask: ", a_mask)
        #     print("m_a_mask: ", m_a_mask)
        # else:
        #     print("active masks are equal")
        #
        # if not torch.equal(scores, m_scores):
        #     print("scores: ", scores)
        #     print("m_scores: ", m_scores)
        # else:
        #     print("scores are equal")
        #
        # if not torch.equal(beam_search_seq[1:], m_beam_search_seq[1:]):
        #     print("seqs: ", beam_search_seq[1:])
        #     print("seqs: ", m_beam_search_seq[1:])
        # else:
        #     print("seqs are equal")

        # m_hidden_states, m_context = self.speller_model.beam_decoder(keys, values, src_key_lens, max_len, beam_size)
        #
        # hidden_states, context = self.beam_decoder.generate_sequence(keys, values, src_key_lens, max_len, beam_size)
        #
        # for i in range(2):
        #     for j in range(2):
        #         if not torch.equal(hidden_states[i][j], m_hidden_states[i][j]):
        #             print("hidden[%d][%d] are not equal"%(i, j))
        #         else:
        #             print("hiddens are equal")
        #
        # if torch.equal(context, m_context):
        #     print("contexts are equal")
        # else:
        #     print("contexts are not equal")
        #
        # exit(0)

        padded_seq, padded_seq_length, attns = self.speller_model.beam_decoder(keys, values, src_key_lens, max_len,
                                                                               beam_size)

        # padded_seq, padded_seq_length, attns = self.beam_decoder.generate_sequence(keys, values, src_key_lens, max_len,
        #                                                                        beam_size)

        return padded_seq, padded_seq_length, src_key_lens, attns
