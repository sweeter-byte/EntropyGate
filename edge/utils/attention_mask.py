"""Attention mask utilities for Fast-V and Text-Mask operations."""

import torch


class GetAttentionMaskwithFastVandTextMask:
    def __init__(self,
                attention_mask: torch.Tensor,
                key_position: dict,
                use_fast_v: bool,
                aggregate_layer_fast_v: int,
                minumum_fast_v_tokens: int,
                use_text_mask: bool,
                aggregate_layer_text_mask: int,
                minimum_text_tokens: int,
                ):

        self._attention_mask = attention_mask
        self._curr_layer_num = 0

        self._use_fast_v = use_fast_v
        self._aggregate_layer_fast_v = aggregate_layer_fast_v

        self._use_text_mask = use_text_mask
        self._aggregate_layer_text_mask = aggregate_layer_text_mask
        self._minimum_text_tokens = minimum_text_tokens

        if self._use_fast_v or self._use_text_mask:
            self._image_start = key_position['image_start']
            self._image_token_length = key_position['image_end'] - self._image_start + 1
            self._minumum_fast_v_tokens = round((0.25) * (self._image_token_length))

        if self._use_fast_v:
            assert self._aggregate_layer_fast_v > 0
        if self._use_text_mask:
            assert self._aggregate_layer_text_mask > 0
            assert self._minimum_text_tokens > 0

    def __call__(self, all_self_attns):
        if self._use_fast_v and self._curr_layer_num == self._aggregate_layer_fast_v:
            self._update_fast_v_attention_mask(all_self_attns[-1])

        if self._use_text_mask and self._curr_layer_num == self._aggregate_layer_text_mask:
            self._update_text_attention_mask(all_self_attns[-1])

        self._curr_layer_num += 1
        return self._attention_mask

    def _update_fast_v_attention_mask(self, last_layer_attention):
        last_layer_attention_avg = torch.mean(last_layer_attention, dim=1)[0]
        last_layer_attention_avg_last_tok = last_layer_attention_avg[-1]
        kv_len = last_layer_attention_avg_last_tok.shape[0]
        mask_len = self._attention_mask.shape[1]

        if mask_len < kv_len:
            pad = self._attention_mask.new_ones(self._attention_mask.shape[0], kv_len - mask_len)
            self._attention_mask = torch.cat([self._attention_mask, pad], dim=1)
        elif mask_len > kv_len:
            self._attention_mask = self._attention_mask[:, :kv_len]

        image_end = min(self._image_start + self._image_token_length, kv_len)
        last_layer_attention_avg_last_tok_image = last_layer_attention_avg_last_tok[
            self._image_start: image_end
        ]
        actual_image_len = last_layer_attention_avg_last_tok_image.shape[0]
        k_fast_v = min(self._minumum_fast_v_tokens, actual_image_len)
        if k_fast_v == 0:
            return

        top_attention_rank_index = last_layer_attention_avg_last_tok_image.topk(k_fast_v, largest=False)
        top_attention_rank_index = top_attention_rank_index.indices + self._image_start
        top_attention_rank_index = top_attention_rank_index.clamp(0, self._attention_mask.shape[1] - 1)

        fast_v_attention_mask = torch.ones_like(self._attention_mask)
        fast_v_attention_mask[:, self._image_start:image_end] = False
        fast_v_attention_mask[:, top_attention_rank_index] = True

        self._attention_mask = fast_v_attention_mask

    def _update_text_attention_mask(self, last_layer_attention):
        last_layer_attention_avg = torch.mean(last_layer_attention, dim=1)[0]
        last_tok_attn = last_layer_attention_avg[-1]
        kv_len = last_tok_attn.shape[0]
        mask_len = self._attention_mask.shape[1]

        if mask_len < kv_len:
            pad = self._attention_mask.new_ones(self._attention_mask.shape[0], kv_len - mask_len)
            self._attention_mask = torch.cat([self._attention_mask, pad], dim=1)
        elif mask_len > kv_len:
            self._attention_mask = self._attention_mask[:, :kv_len]

        total_active = int(self._attention_mask.sum().item())
        k = int(self._minimum_text_tokens) if total_active > self._minimum_text_tokens else total_active
        if k == 0:
            return

        topk = last_tok_attn.topk(k, largest=False)
        keep_idx = topk.indices
        keep_idx = keep_idx.clamp(0, self._attention_mask.shape[1] - 1)

        text_mask = torch.zeros_like(self._attention_mask, dtype=torch.bool)
        text_mask[:, keep_idx] = True

        self._attention_mask = text_mask
