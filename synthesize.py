from ast import Str
import re
import argparse
from string import punctuation

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from dataset import TextDataset
from text import text_to_sequence

import os
import sys
import time



def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def preprocess_mandarin(text, preprocess_config):
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    phones = []
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")

    phones = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def synthesize(model, args, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        batch = to_device(batch, device)
        if args.jit:
            model = torch.jit.trace(model, batch[2:], check_trace=False, strict=False)
            model = torch.jit.freeze(model)
            print("---- Use trace model")
        with torch.no_grad():
            # Forward
            total_time = 0.0
            total_sample = 0
            if args.profile:
                prof_act = [torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]
                with torch.profiler.profile(
                    activities=prof_act,
                    record_shapes=True,
                    schedule=torch.profiler.schedule(
                        wait=int(args.num_iter/2),
                        warmup=2,
                        active=1,
                    ),
                    on_trace_ready=trace_handler,
                ) as p:
                    for i in range(args.num_iter):
                        elapsed = time.time()
                        output = model(*(batch[2:]), p_control=pitch_control, e_control=energy_control, d_control=duration_control)
                        if torch.cuda.is_available(): torch.cuda.synchronize()
                        p.step()
                        elapsed = time.time() - elapsed
                        print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                        if i >= args.num_warmup:
                            total_time += elapsed
                            total_sample += args.batch_size
            else:
                for i in range(args.num_iter):
                    elapsed = time.time()
                    output = model(*(batch[2:]), p_control=pitch_control, e_control=energy_control, d_control=duration_control)
                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    elapsed = time.time() - elapsed
                    print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                    if i >= args.num_warmup:
                        total_time += elapsed
                        total_sample += args.batch_size

            print("\n", "-"*20, "Summary", "-"*20)
            latency = total_time / total_sample * 1000
            throughput = total_sample / total_time
            print("inference Latency: {} ms".format(latency))
            print("inference Throughput: {} samples/s".format(throughput))
            # synth_samples(
            #     batch,
            #     output,
            #     vocoder,
            #     model_config,
            #     preprocess_config,
            #     train_config["path"]["result_path"],
            # )
        break

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total")
    print(output)
    import pathlib
    timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
    if not os.path.exists(timeline_dir):
        try:
            os.makedirs(timeline_dir)
        except:
            pass
    timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + '-' + \
                'FastSpeech2-' + str(p.step_num) + '-' + str(os.getpid()) + '.json'
    p.export_chrome_trace(timeline_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--precision', default="float32", type=str, help='precision')
    parser.add_argument('--channels_last', default=1, type=int, help='Use NHWC or not')
    parser.add_argument('--ipex', action='store_true', default=False, help='enable ipex')
    parser.add_argument('--jit', action='store_true', default=False, help='enable JIT')
    parser.add_argument('--device', default="cpu", type=str, help='cpu, cuda or xpu')
    parser.add_argument('--profile', action='store_true', default=False, help='collect timeline')
    parser.add_argument('--num_iter', default=1, type=int, help='test iterations')
    parser.add_argument('--num_warmup', default=0, type=int, help='test warmup')
    parser.add_argument('--ckpt_dir', default=None, type=str, help='path to ckpt')
    parser.add_argument("--compile", action='store_true', default=False,
                    help="enable torch.compile")
    parser.add_argument("--backend", type=str, default='inductor',
                    help="enable torch.compile backend")
    parser.add_argument("--triton_cpu", action='store_true', default=False,
                    help="enable triton_cpu")
    args = parser.parse_args()
    
    device = torch.device("cuda" if args.device == 'cuda' and torch.cuda.is_available() else "cpu")

    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None
    if args.triton_cpu:
        print("run with triton cpu backend")
        import torch._inductor.config
        torch._inductor.config.cpu_backend="triton"
    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    if args.ckpt_dir is not None:
        train_config["path"]["ckpt_path"] = args.ckpt_dir
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
        print("---- Use NHWC model")
    if args.ipex:
        import intel_extension_for_pytorch as ipex
        if args.precision == "bfloat16":
            model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True)
        else:
            model = ipex.optimize(model, dtype=torch.float32, inplace=True)
        print("---- Use ipex model")
    if args.compile:
        model = torch.compile(model, backend=args.backend, options={"freezing": True})

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=dataset.collate_fn,
        )
    if args.mode == "single":
        ids = raw_texts = [args.text[:100]]
        speakers = np.array([args.speaker_id])
        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(args.text, preprocess_config)])
        elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
            texts = np.array([preprocess_mandarin(args.text, preprocess_config)])
        text_lens = np.array([len(texts[0])])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

    control_values = args.pitch_control, args.energy_control, args.duration_control
    if args.precision == "bfloat16":
        print("---- Use cpu AMP bfloat16")
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=True, dtype=torch.bfloat16):
            synthesize(model, args, configs, vocoder, batchs, control_values)
    elif args.precision == "float16":
        print("---- Use cuda AMP float16")
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=True, dtype=torch.half):
            synthesize(model, args, configs, vocoder, batchs, control_values)
    else:
        synthesize(model, args, configs, vocoder, batchs, control_values)
