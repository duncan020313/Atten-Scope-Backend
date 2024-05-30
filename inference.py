from pprint import pprint
from typing import List
from transformer_lens import HookedTransformer, ActivationCache
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch as t
from AttenScope import AttentionPostprocessing, TensorToHeatmap

device = t.device("cpu")
previous_model = None


def get_hooked_model(model_name: str) -> HookedTransformer:
    global device
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    hooked_model = HookedTransformer.from_pretrained_no_processing(
        model_name=model_name,
        hf_model=model,
        tokenizer=tokenizer,
        device=device,
        dtype="float16",
    )
    return hooked_model


def get_htmls(prompt: str, hooked_model: HookedTransformer) -> List[str]:
    assert isinstance(prompt, str)
    assert isinstance(hooked_model, HookedTransformer)

    global previous_model

    prompt = prompt.replace("\r\n", "\n")
    print("Prompt:", prompt)
    if previous_model != hooked_model:
        if previous_model is not None:
            previous_model = previous_model.to("cpu")
        previous_model = hooked_model
        hooked_model = hooked_model.to("cuda")
    toks = hooked_model.to_tokens(prompt)
    str_toks = hooked_model.to_str_tokens(toks)
    _, cache = hooked_model.run_with_cache(toks)
    cache = cache.to("cpu")
    print("Str tokens:", str_toks)

    htmls = []
    num_of_layer = hooked_model.cfg.n_layers
    for layer in range(num_of_layer):
        attention = cache["pattern", layer].squeeze()
        value = cache["v", layer].squeeze().transpose(0, 1)
        z = cache["z", layer].squeeze().transpose(0, 1)
        post_processed_attention = [
            AttentionPostprocessing.get_effective_attention(a, z)
            for a, z in zip(attention, z)
        ]
        post_processed_attention = [
            AttentionPostprocessing.apply_value_norm_to_attention(a, v).numpy()
            for a, v in zip(post_processed_attention, value)
        ]

        labels = [f"Head {head}" for head in range(attention.size(0))]
        html = TensorToHeatmap.generate_heatmap_html(
            post_processed_attention, labels, str_toks
        )
        htmls.append(html)
    return htmls
