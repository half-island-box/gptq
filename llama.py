import csv
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from gptq import *
from modelutils import *
from modelscope_utils import modelscope_model_path
from quant import *
from quant_methods import add_method_args, check_save_supported, quantize_block, resolve_method


def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(modelscope_model_path(model), torch_dtype='auto')
    model.seqlen = 2048
    return model

_LLAMA_LAYER_KWARGS = ("attention_mask", "position_ids", "position_embeddings")


def _capture_llama_layer_kwargs(cache, kwargs):
    """Capture kwargs needed by both old and new Transformers Llama layers."""
    for name in _LLAMA_LAYER_KWARGS:
        if name in kwargs:
            cache[name] = kwargs[name]


def _get_llama_layer_kwargs(cache):
    """Return captured kwargs that exist in the installed Transformers version."""
    return {name: cache[name] for name in _LLAMA_LAYER_KWARGS if name in cache}


@torch.no_grad()
def llama_sequential(model, dataloader, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            _capture_llama_layer_kwargs(cache, kwargs)
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    layer_kwargs = _get_llama_layer_kwargs(cache)

    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)

        # --true-sequential：把 block 内的层切成若干子集，逐子集量化；
        # 只有 GPTQ 系的误差补偿受益于此（AWQ 需要整块搜索缩放，内部会自动合并）
        if args.true_sequential:
            sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj']
            ]
        else:
            sequential = None

        def fwd(x, layer=layer):
            return layer(x, **layer_kwargs)[0]

        # 统一驱动：标定(可选) -> AWQ 等价变换(可选) -> 量化 -> 重算该层输出
        layer_quantizers = quantize_block(
            layer, inps, outs, fwd, args.quant_method, args,
            layer_idx=i,
            layer_kwargs=layer_kwargs,
            subsets=sequential,
        )
        for name, quantizer in layer_quantizers.items():
            quantizers['model.layers.%d.%s' % (i, name)] = quantizer

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    
    return quantizers

@torch.no_grad()
def llama_eval(model, testenc, dev):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            _capture_llama_layer_kwargs(cache, kwargs)
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    layer_kwargs = _get_llama_layer_kwargs(cache)

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)
        
        # 量化（含 RTN 基线）已在 llama_sequential 中完成，这里只做前向
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())
    model.config.use_cache = use_cache
    return ppl

def append_llama_result(args, dataset, ppl):
    """Append one evaluated Llama PPL result to a CSV file."""
    if args.wbits == 16:
        quant_method = 'FP16'
    else:
        quant_method = args.quant_method.upper()

    fieldnames = [
        'timestamp', 'model_name', 'quant_bits', 'quant_method', 'dataset', 'ppl'
    ]
    result = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'model_name': args.model,
        'quant_bits': args.wbits,
        'quant_method': quant_method,
        'dataset': dataset,
        'ppl': format(ppl, '.12g'),
    }

    output_path = Path(args.results_csv)
    output_exists = output_path.exists()
    with output_path.open('a', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        if not output_exists:
            writer.writeheader()
        writer.writerow(result)
    print(f'Saved result to {args.results_csv}')


def llama_pack3(model, quantizers):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers)
    qlayers = find_layers(model, [Quant3Linear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)
    print('Done.')
    return model


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    ) 
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval.'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--true-sequential', action='store_true',
        help='Whether to run in true sequential model.'
    )
    parser.add_argument(
        '--static-groups', action='store_true',
        help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.'
    )
    parser.add_argument(
        '--results-csv', type=str, default='results.csv',
        help='CSV file for appending model, quantization, dataset, and PPL results.'
    )
    add_method_args(parser)

    args = parser.parse_args()
    try:
        # --nearest / --quant_method 归一化：rtn / gptq / awq / awq_gptq
        args.quant_method = resolve_method(args)
        # 尽早检查 --save（3-bit 打包）与所选方法/分组是否兼容，避免量化跑完才报错
        check_save_supported(args, args.quant_method)
    except NotImplementedError as err:
        parser.error(str(err))

    model = get_llama(args.model)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    if args.wbits < 16:
        tick = time.time()
        quantizers = llama_sequential(model, dataloader, DEV)
        print(time.time() - tick)

    datasets = ['wikitext2', 'ptb', 'c4'] 
    if args.new_eval:
        datasets = ['wikitext2', 'ptb-new', 'c4-new']
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        ppl = llama_eval(model, testloader, DEV)
        append_llama_result(args, dataset, ppl)

    if args.save:
        llama_pack3(model, quantizers)
        torch.save(model.state_dict(), args.save)

