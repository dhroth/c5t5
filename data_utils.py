import torch

def mask_spans(tokenizer, input_ids, mask_probability, mean_span_length):
    DEBUG = False
    length = input_ids.numel()
    if length < 2:
        return input_ids, tokenizer.sentinels(torch.tensor([0]))

    num_noise_tokens = round(length * mask_probability)
    num_noise_tokens = min(max(num_noise_tokens, 0), length - 1)
    if num_noise_tokens == 0:
        return input_ids, tokenizer.sentinels(torch.tensor([0]))
    DEBUG and print("num_noise_tokens", num_noise_tokens)
    num_nonnoise_tokens = length - num_noise_tokens
    DEBUG and print("num_nonnoise_tokens", num_nonnoise_tokens)

    num_noise_spans = round(num_noise_tokens / mean_span_length)
    num_noise_spans = max(num_noise_spans, 1)
    DEBUG and print("num_noise_spans", num_noise_spans)

    # probability of the last token being noise should be
    # mask_probability, but right now it's 100%
    if torch.rand(1).item() < mask_probability:
        num_nonnoise_spans = num_noise_spans
    else:
        num_nonnoise_spans = num_noise_spans + 1

    def _random_segmentation(num_items, num_segments):
        ones = (torch.arange(num_items - 1) < num_segments - 1).int()
        first_in_segment = torch.cat([torch.tensor([0]).int(),
                                      ones[torch.randperm(num_items-1)]])
        segment_id = torch.cumsum(first_in_segment, dim=0)
        _, lengths = segment_id.unique_consecutive(return_counts=True)
        return lengths
    noise_span_lengths = _random_segmentation(num_noise_tokens,
                                              num_noise_spans)
    DEBUG and print("noise_span_lengths", noise_span_lengths)
    nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens,
                                                 num_nonnoise_spans)
    DEBUG and print("nonnoise_span_lengths", nonnoise_span_lengths)
    #print(noise_span_lengths.float().mean().item(), noise_span_lengths)
    #print(nonnoise_span_lengths)
    if num_nonnoise_spans > num_noise_spans:
        noise_span_lengths = torch.cat([noise_span_lengths,
                                        torch.tensor([0])])
    interleaved_span_lengths = torch.stack([
            nonnoise_span_lengths, noise_span_lengths
        ], dim=1).view(-1)
    if num_nonnoise_spans > num_noise_spans:
        interleaved_span_lengths = interleaved_span_lengths[:-1]

    DEBUG and print('interleaved', interleaved_span_lengths)
    span_starts = torch.cumsum(interleaved_span_lengths, dim=0)[:-1]
    DEBUG and print("span_starts", span_starts)
    span_start_indicator = torch.zeros(length).bool()
    span_start_indicator[span_starts] = 1
    DEBUG and print("span_start_indicator", span_start_indicator)
    span_num = torch.cumsum(span_start_indicator, dim=0)
    DEBUG and print("span_num", span_num)
    is_noise = span_num % 2 == 1
    DEBUG and print("is_noise", is_noise)

    def sentinelify(tokens, noise_mask):
        prev_token_is_noise = torch.cat([torch.tensor([0]).bool(),
                                         noise_mask[:-1]])
        first_noise_tokens = noise_mask & ~prev_token_is_noise
        subsequent_noise_tokens = noise_mask & prev_token_is_noise
        sentinels = tokenizer.sentinels(
                    torch.cumsum(first_noise_tokens, dim=0) - 1
                )
        tokens = torch.where(first_noise_tokens, sentinels, tokens)
        return tokens[~subsequent_noise_tokens]

    masked_input = sentinelify(input_ids, is_noise)
    DEBUG and print("masked_input", masked_input)
    target_ids = sentinelify(input_ids, ~is_noise)
    DEBUG and print("target_ids", target_ids)

    return masked_input, target_ids


def collapse_sentinels(tokenizer, input_ids, target_ids):
    def remove_extraneous(ids):
        # delete everything after </s>
        eos = tokenizer.eos_token_id
        pad_mask = (ids == eos).cumsum(dim=0).clamp(0, 1).bool()
        ids = ids[:ids.numel() - pad_mask.sum()]
        return ids

    input_ids = remove_extraneous(input_ids)
    target_ids = remove_extraneous(target_ids)

    num_sentinels = tokenizer._extra_ids
    all_sentinel_ids = tokenizer.sentinels(
                torch.arange(num_sentinels).to(input_ids.device)
            )
    min_sentinel_id = all_sentinel_ids.min()
    max_sentinel_id = all_sentinel_ids.max()

    def validate(ids, name="ids"):
        #mask = (min_sentinel_id <= ids) & (ids <= max_sentinel_id)
        mask = tokenizer.sentinel_mask(ids)
        sentinels = ids[mask]
        msg = "sentinels in {} are in the wrong order"
        if not torch.all(sentinels==all_sentinel_ids[:sentinels.numel()]):
            raise ValueError(msg.format(name))
        return mask

    input_sentinel_mask = validate(input_ids, "input_ids")
    target_sentinel_mask = validate(target_ids, "target_ids")

    input_span_types, input_span_lengths = \
            input_sentinel_mask.unique_consecutive(return_counts=True)
    target_span_types, target_span_lengths = \
            target_sentinel_mask.unique_consecutive(return_counts=True)

    input_sentinel_span_lengths = input_span_lengths[input_span_types]
    target_sentinel_span_lengths = target_span_lengths[target_span_types]
    if input_sentinel_span_lengths.sum() != input_span_types.sum():
        raise ValueError("consecutive sentinel tokens in input_ids")
    if target_sentinel_span_lengths.sum() != target_span_types.sum():
        raise ValueError("consecutive sentinel tokens in target_ids")

    msg = "invalid interleaving of sentinels between inputs and target"
    if input_span_types.numel() != target_span_types.numel():
        raise ValueError(msg)
    xor = torch.logical_xor(input_span_types, target_span_types)
    if xor.sum() != input_span_types.numel():
        raise ValueError(msg)

    input_repeat = input_sentinel_mask.long()
    input_repeat[input_sentinel_mask] = target_span_lengths[~target_span_types]
    input_repeat[input_repeat == 0] = 1

    target_repeat = target_sentinel_mask.long()
    target_repeat[target_sentinel_mask] = input_span_lengths[~input_span_types]
    target_repeat[target_repeat == 0] = 1

    input_repeated = input_ids.repeat_interleave(input_repeat)
    target_repeated = target_ids.repeat_interleave(target_repeat)

    #use_target = (min_sentinel_id <= input_repeated) & (input_repeated <= max_sentinel_id)
    use_target = tokenizer.sentinel_mask(input_repeated)
    collapsed = torch.where(use_target, target_repeated, input_repeated)

    return collapsed
