import warnings
warnings.filterwarnings("ignore")
import argparse
import random
import time
from typing import Dict

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
from peft import PeftModel, PeftConfig

# Dream model classes (local cache wrappers)
from model_cache.dream.model_dream import DreamModel
from model_cache.dream.configuration_dream import DreamConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_full_block_attention_mask(prompt_length: int, max_length: int, block_size: int, device=None, dtype=None):
    if dtype is None:
        dtype = torch.bfloat16
    attention_mask = torch.full((1, 1, max_length, max_length), -torch.inf, device=device, dtype=dtype)
    # Prompt attends to itself
    attention_mask[:, :, :prompt_length, :prompt_length] = 0

    remaining_length = max_length - prompt_length
    num_blocks = (remaining_length + block_size - 1) // block_size
    for b in range(num_blocks):
        block_start = prompt_length + b * block_size
        block_end = min(prompt_length + (b + 1) * block_size, max_length)
        # Current block can see the prompt
        attention_mask[:, :, block_start:block_end, :prompt_length] = 0
        # Current block can see all previous regular blocks
        for prev_b in range(b):
            prev_start = prompt_length + prev_b * block_size
            prev_end = min(prompt_length + (prev_b + 1) * block_size, max_length)
            attention_mask[:, :, block_start:block_end, prev_start:prev_end] = 0
        # Current block can see itself
        attention_mask[:, :, block_start:block_end, block_start:block_end] = 0
    return attention_mask


def extract_attention_mask(full_mask: torch.Tensor, start_pos: int, input_length: int, cache_length: int) -> torch.Tensor:
    end_pos = start_pos + input_length
    total_length = cache_length + input_length
    extracted_mask = torch.full((1, 1, input_length, total_length), -torch.inf, device=full_mask.device, dtype=full_mask.dtype)
    extracted_mask[:, :, :, :cache_length] = full_mask[:, :, start_pos:end_pos, :cache_length]
    extracted_mask[:, :, :, cache_length:] = full_mask[:, :, start_pos:end_pos, start_pos:end_pos]
    return extracted_mask


def top_p_logits(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def top_k_logits(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits: torch.Tensor, temperature: float = 0.0, top_p: float = None, top_k: int = None):
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)
    if temperature > 0:
        try:
            x0 = torch.distributions.Categorical(probs=probs).sample()
            initial_confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except Exception:
            initial_confidence, x0 = probs.max(dim=-1)
    else:
        initial_confidence, x0 = probs.max(dim=-1)
    return initial_confidence, x0


def shift_logits(logits: torch.Tensor, last_logit: torch.Tensor = None) -> torch.Tensor:
    if logits.shape[1] == 0:
        raise Exception("logits sequence length is 0")
    shifted = torch.zeros_like(logits)
    shifted[:, 1:, :] = logits[:, :-1, :]
    if last_logit is not None:
        shifted[:, 0, :] = last_logit
        return shifted
    shifted[:, 0, :] = 1.0
    return shifted


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Simple Dream block-based generation example")
    parser.add_argument('--pretrained_path', type=str, default='Dream-org/Dream-v0-Base-7B')
    parser.add_argument('--lora_path', type=str, default='SJTU-Deng-Lab/D2F_Dream_Base_7B_Lora')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--block_size', type=int, default=4)
    parser.add_argument('--block_add_threshold', type=float, default=0.5)
    parser.add_argument('--decoded_token_threshold', type=float, default=0.9)
    parser.add_argument('--skip_threshold', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if args.dtype == 'bfloat16' and torch.cuda.is_bf16_supported():
        target_dtype = torch.bfloat16
    elif args.dtype == 'float16':
        target_dtype = torch.float16
    else:
        target_dtype = torch.float32

    # Load model + LoRA
    model_config = DreamConfig.from_pretrained(args.pretrained_path, trust_remote_code=True)
    model = DreamModel.from_pretrained(
        args.pretrained_path, config=model_config, torch_dtype=target_dtype, trust_remote_code=True
    ).eval()
    model = PeftModel.from_pretrained(model, args.lora_path)
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mask_token_id = 151666
    eos_token_id = tokenizer.eos_token_id

    # Build a few-shot GSM8K prompt
    dataset = load_dataset('gsm8k', 'main')
    prompt = ''
    for i in range(5):
        prompt += 'Question: ' + dataset['train'][i]['question'] + '\nAnswer: ' + dataset['train'][i]['answer'] + '\n'
    prompt += "Question: John takes care of 10 dogs. Each dog takes .5 hours a day to walk and take care of their business. How many hours a week does he spend taking care of dogs?"

    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Truncate left if needed
    if inputs.shape[1] > args.max_length - args.max_new_tokens:
        inputs = inputs[:, -(args.max_length - args.max_new_tokens):]

    # Precompute full attention mask for maximum sequence length
    prompt_length = inputs.shape[1]
    full_attention_mask = create_full_block_attention_mask(
        prompt_length=prompt_length,
        max_length=args.max_length,
        block_size=args.block_size,
        device=device,
        dtype=target_dtype if target_dtype is not None else torch.bfloat16,
    )

    # Initialize sequence and block states
    x_t = inputs
    block_states: Dict[int, Dict] = {
        0: {
            'start_pos': 0,
            'end_pos': prompt_length,
            'mask_count': 0,
            'total_masks': prompt_length,
            'state': 'to_cache',
            'is_complete': True,
        }
    }
    past_key_values = None
    current_blocks = 0
    step = 0
    eos_detected = False
    last_logits = None

    start_time = time.time()

    while True:
        step += 1

        # Add a new block if progress threshold met and budget allows
        if len(block_states) - 1 < (args.max_new_tokens // args.block_size) and not eos_detected:
            last_block_id = max(block_states.keys())
            progress = 1.0
            if block_states[last_block_id]['total_masks'] > 0:
                progress = (block_states[last_block_id]['total_masks'] - block_states[last_block_id]['mask_count']) / block_states[last_block_id]['total_masks']
            if progress >= args.block_add_threshold:
                new_block_id = last_block_id + 1
                new_start_pos = x_t.shape[1]
                if new_start_pos + args.block_size <= args.max_length:
                    x_t = torch.cat([x_t, torch.full((1, args.block_size), mask_token_id, device=device, dtype=torch.long)], dim=1)
                    block_states[new_block_id] = {
                        'start_pos': new_start_pos,
                        'end_pos': new_start_pos + args.block_size,
                        'mask_count': args.block_size,
                        'total_masks': args.block_size,
                        'state': 'active',
                        'is_complete': False,
                    }
                    current_blocks += 1

        # Update completion states
        for block_id in sorted(block_states.keys()):
            decoded_tokens = block_states[block_id]['total_masks'] - block_states[block_id]['mask_count']
            if block_states[block_id]['total_masks'] > 0:
                decode_ratio = decoded_tokens / block_states[block_id]['total_masks']
                if decode_ratio >= args.decoded_token_threshold:
                    if (block_id + 1) in block_states:
                        block_states[block_id + 1]['is_complete'] = True

        if (x_t == mask_token_id).sum() == 0 and current_blocks == 0:
            break

        # Determine cache length from past_key_values
        cache_length = 0 if past_key_values is None else past_key_values.get_seq_length()

        # Determine blocks to cache and input segment
        blocks_to_cache = [bid for bid, state in block_states.items() if state['state'] == 'to_cache']
        update_kvcache = 0
        if blocks_to_cache:
            start_pos = block_states[min(blocks_to_cache)]['start_pos']
            end_pos = block_states[max(blocks_to_cache)]['end_pos']
            update_kvcache = end_pos - start_pos
            input_seq = x_t[:, start_pos:]
            process_start_pos = start_pos
        else:
            active_blocks = [bid for bid, state in block_states.items() if state['state'] == 'active' and state['start_pos'] >= cache_length]
            if not active_blocks:
                break
            start_pos = min(block_states[bid]['start_pos'] for bid in active_blocks)
            input_seq = x_t[:, start_pos:]
            process_start_pos = start_pos

        if input_seq.shape[1] == 0:
            break

        # Extract per-step attention mask
        attention_mask = extract_attention_mask(full_attention_mask, process_start_pos, input_seq.shape[1], cache_length)

        outputs = model(
            input_seq,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            update_kvcache=update_kvcache,
        )

        if update_kvcache > 0:
            cache_end_idx = update_kvcache - 1
            last_logits = outputs.logits[:, cache_end_idx, :].unsqueeze(1)
            past_key_values = outputs.past_key_values
            for bid in blocks_to_cache:
                block_states[bid]['state'] = 'in_cache'

        # Shift logits for AR prediction
        logits = shift_logits(outputs.logits, last_logit=last_logits)

        # Decode masked tokens within each active block
        blocks_to_deactivate = []
        for block_id, state in block_states.items():
            if state['state'] != 'active':
                continue
            block_start = state['start_pos']
            block_end = state['end_pos']
            block_mask_locs = (x_t[0, block_start:block_end] == mask_token_id).nonzero().squeeze(-1)
            if block_mask_locs.numel() == 0:
                blocks_to_deactivate.append(block_id)
                continue
            logit_offset = block_start - process_start_pos
            block_mask_logits = logits[:, logit_offset + block_mask_locs, :]

            initial_confidence, x0 = sample_tokens(
                block_mask_logits.squeeze(0),
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )

            high_conf_indices = (initial_confidence > args.skip_threshold).nonzero().squeeze(-1)
            if state['is_complete'] and high_conf_indices.numel() == 0 and block_mask_logits.numel() > 0:
                _, top_idx = torch.topk(initial_confidence, 1)
                selected_indices = top_idx
            else:
                selected_indices = high_conf_indices

            if selected_indices.numel() > 0:
                positions_to_update = block_start + block_mask_locs[selected_indices]
                x_t[0, positions_to_update] = x0[selected_indices]
                state['mask_count'] -= selected_indices.numel()
                if (x0[selected_indices] == eos_token_id).any():
                    eos_detected = True
            if state['mask_count'] == 0:
                blocks_to_deactivate.append(block_id)

        for bid in blocks_to_deactivate:
            if block_states[bid]['state'] == 'active' and all(block_states.get(i, {}).get('state') != 'active' for i in range(bid)):
                block_states[bid]['state'] = 'to_cache'
                current_blocks -= 1

        if step > 10000:
            break

    # Final decode (truncate at EOS)
    gen_ids = x_t[0, prompt_length:]
    eos_positions = (gen_ids == eos_token_id).nonzero()
    if eos_positions.numel() > 0:
        gen_ids = gen_ids[:eos_positions[0, 0] + 1]
    text = tokenizer.decode(gen_ids, skip_special_tokens=False)

    elapsed = time.time() - start_time
    print(text)
    print(f"\n[info] Generated {len(gen_ids)} tokens in {elapsed:.2f}s ({len(gen_ids)/max(elapsed,1e-6):.2f} tok/s)")


if __name__ == "__main__":
    main()


