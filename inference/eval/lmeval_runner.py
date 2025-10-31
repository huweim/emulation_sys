# lmeval_runner.py
import torch
import torch.nn as nn
from tqdm import tqdm

def run_lm_eval(args, model, tokenizer, tasks: str, batch_size: int, num_fewshot: int, limit: int | None):
    """
    tasks e.g.: "arc_easy,hellaswag,winogrande"
    """
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    from lm_eval.utils import make_table

    if tasks in ['wikitext', 'c4', 'ptb']:
    # https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/llama.py#L206
        from ..utils.dataload_utils import get_loaders
        model.seqlen = 2048
        _, testenc = get_loaders(tasks, model=args.model_path, seqlen=model.seqlen)
        
        testenc = testenc.input_ids.to(model.device)
        nsamples = testenc.numel() // model.seqlen
        # nsamples = 10
        model = model.eval()
        nlls = []
        for i in tqdm(range(nsamples), desc="evaluating..."):
            batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
                model.device
            )
            with torch.no_grad():
                lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = testenc[
                :, (i * model.seqlen) : ((i + 1) * model.seqlen)
            ][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * model.seqlen
            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
        print(f'PPL on {tasks}: {ppl.item()}')
        return {"results": {tasks: {"ppl": float(ppl)}}}

    else:
        hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)

        task_list = [t.strip() for t in tasks.split(",") if t.strip()]
        print(f"\n[LM-Eval] Running tasks: {task_list}  (fewshot={num_fewshot}, batch_size={batch_size}, limit={limit})")

        results = evaluator.simple_evaluate(
            model=hflm,
            tasks=task_list,
            batch_size=batch_size,
            num_fewshot=num_fewshot,
            limit=limit,               # Can be None: full eval; or set an int for quick run
        )

        print(make_table(results))
        # Also return structured results (convenient for saving to JSON)
        return results