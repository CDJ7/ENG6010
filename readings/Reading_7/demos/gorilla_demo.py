"""
gorilla_demo.py - Article 1: Gorilla (Patil et al., 2023).

Gorilla teaches an LLM to write *correct* API calls. Three pieces of the
paper are reimplemented here so they can be run and inspected:

1. An API database in APIBench's JSON style (`API_DB`).
   APIBench has 1,645 entries scraped from TorchHub, TensorFlow Hub and
   HuggingFace. This file holds a small hand-built subset (15 entries) so the
   demo stays readable; it is NOT the real benchmark.
2. A BM25 document retriever (`BM25`), one of the two retrievers the paper
   tests. In "retrieval mode" the top-1 API document is appended to the
   user's prompt: "<prompt> Use this API documentation for reference: <doc>".
3. The paper's evaluation metric: AST sub-tree matching (`ast_verdict`).
   A generated call is parsed into a Python AST and compared with every API
   in the database on the arguments that identify the API:
     * matches the reference API for the task  -> "correct"
     * matches some other real API             -> "error" (wrong API)
     * matches no API in the database          -> "hallucination"
       (the paper: "invoking an entirely imagined tool")

There is no LLM here. Where the paper's model would generate an answer, the
demo uses fixed example answers (`Query.candidates`) and a stand-in for a
retriever-aware model that copies the API call from the retrieved document,
which is the behaviour Gorilla's retriever-aware training teaches.
"""

import ast
import math
import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 1. A small API database in APIBench's format
# ---------------------------------------------------------------------------
API_DB = [
    dict(id="densenet121", domain="Classification", api_provider="TorchHub",
         api_call="torch.hub.load('pytorch/vision', 'densenet121', pretrained=True)",
         functionality="image classification densenet", params_m=8.0, top1=74.4,
         description="DenseNet-121 classifies an image into 1000 ImageNet classes."),
    dict(id="resnet50", domain="Classification", api_provider="TorchHub",
         api_call="torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)",
         functionality="image classification resnet", params_m=25.6, top1=76.1,
         description="ResNet-50 residual network that classifies images into ImageNet classes."),
    dict(id="mobilenet_v2", domain="Classification", api_provider="TorchHub",
         api_call="torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=True)",
         functionality="image classification small mobile efficient", params_m=3.5, top1=71.88,
         description="MobileNet V2 is a small, efficient image classifier for mobile devices."),
    dict(id="resnext101_wsl", domain="Classification", api_provider="TorchHub",
         api_call="torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl')",
         functionality="image classification high accuracy large", params_m=194.0, top1=84.2,
         description="ResNeXt-101 32x16d weakly supervised: very accurate large image classifier."),
    dict(id="fcn_resnet50", domain="Semantic Segmentation", api_provider="TorchHub",
         api_call="torch.hub.load('pytorch/vision', 'fcn_resnet50', pretrained=True)",
         functionality="semantic segmentation separate foreground background pixels",
         description="FCN with a ResNet-50 backbone labels every pixel, e.g. to remove an image background."),
    dict(id="silero_stt", domain="Speech-to-Text", api_provider="TorchHub",
         api_call="torch.hub.load('snakers4/silero-models', 'silero_stt', language='en')",
         functionality="speech to text transcribe spoken audio recording",
         description="Silero speech-to-text model transcribes spoken language in an audio recording."),
    dict(id="tacotron2", domain="Text-to-Speech", api_provider="TorchHub",
         api_call="torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2')",
         functionality="text to speech synthesis voice",
         description="Tacotron 2 turns written text into a natural sounding spoken voice."),
    dict(id="hybridnets", domain="Object Detection", api_provider="TorchHub",
         api_call="torch.hub.load('datvuthanh/hybridnets', 'hybridnets', pretrained=True)",
         functionality="traffic object detection drivable area lane detection cars pedestrians",
         description="HybridNets detects traffic objects such as cars and pedestrians and segments lanes."),
    dict(id="yolov5", domain="Object Detection", api_provider="TorchHub",
         api_call="torch.hub.load('ultralytics/yolov5', 'yolov5s')",
         functionality="object detection bounding boxes animals people general",
         description="YOLOv5 small detects and localises common objects and animals in an image."),
    dict(id="use", domain="text-embedding", api_provider="TensorFlow Hub",
         api_call="hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')",
         functionality="text embedding sentence similarity",
         description="Universal Sentence Encoder turns sentences into vectors to compare their meaning."),
    dict(id="tf_mobilenet", domain="image-classification", api_provider="TensorFlow Hub",
         api_call="hub.KerasLayer('https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/5')",
         functionality="image classification keras layer",
         description="MobileNet V2 image classifier as a Keras layer."),
    dict(id="whisper", domain="Audio Automatic Speech Recognition", api_provider="HuggingFace",
         api_call="pipeline('automatic-speech-recognition', model='openai/whisper-base')",
         functionality="speech recognition transcribe audio",
         description="Whisper base model transcribes speech audio into text."),
    dict(id="sd21", domain="Multimodal Text-to-Image", api_provider="HuggingFace",
         api_call="StableDiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-2-1')",
         functionality="text to image generation picture from description",
         description="Stable Diffusion 2.1 generates an image from a text description."),
    dict(id="dpt", domain="Computer Vision Depth Estimation", api_provider="HuggingFace",
         api_call="pipeline('depth-estimation', model='Intel/dpt-large')",
         functionality="depth estimation distance from camera",
         description="DPT-Large estimates the depth of every pixel in an image."),
    dict(id="bart_cnn", domain="Natural Language Processing Summarization", api_provider="HuggingFace",
         api_call="pipeline('summarization', model='facebook/bart-large-cnn')",
         functionality="summarization shorten long text article",
         description="BART large CNN writes a short summary of a long article."),
]

# The paper's Fig. 6 example: the documentation is updated after training
# (FCN backbone upgraded from ResNet-50 to ResNet-101).
DOC_UPDATE = dict(id="fcn_resnet50", new=dict(
    api_call="torch.hub.load('pytorch/vision', 'fcn_resnet101', pretrained=True)",
    description="FCN with a ResNet-101 backbone labels every pixel, e.g. to remove an image background.",
    functionality="semantic segmentation separate foreground background pixels upgraded"))


def database(updated=False):
    """The API database, optionally after the test-time documentation change."""
    db = [dict(d) for d in API_DB]
    if updated:
        for d in db:
            if d["id"] == DOC_UPDATE["id"]:
                d.update(DOC_UPDATE["new"])
    return db


# ---------------------------------------------------------------------------
# 2. BM25 retriever (Robertson & Zaragoza; standard Okapi BM25)
# ---------------------------------------------------------------------------
STOP = set("a an the to of in and or for with from my me i using use help find api that this is be "
           "it on at by as want would like can into using".split())


def tokenize(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


class BM25:
    """Okapi BM25 over the API documents (each API is one document, as in the paper)."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.toks = [tokenize(" ".join([d["domain"], d["functionality"], d["description"]])) for d in docs]
        self.avgdl = sum(len(t) for t in self.toks) / len(self.toks)
        n = len(docs)
        vocab = {w for t in self.toks for w in t}
        self.idf = {w: math.log(1 + (n - df + 0.5) / (df + 0.5))
                    for w in vocab for df in [sum(w in t for t in self.toks)]}

    def scores(self, query):
        q = tokenize(query)
        out = []
        for toks in self.toks:
            s = 0.0
            for w in q:
                f = toks.count(w)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(toks) / self.avgdl))
            out.append(s)
        return out

    def contributions(self, query):
        """Per query word, per document: how much that word adds to the BM25
        score (the sum over words is scores()). Used for the word heatmap."""
        q = list(dict.fromkeys(tokenize(query)))
        out = {}
        for w in q:
            row = []
            for toks in self.toks:
                f = toks.count(w)
                row.append(self.idf.get(w, 0.0) * f * (self.k1 + 1) /
                           (f + self.k1 * (1 - self.b + self.b * len(toks) / self.avgdl)) if f else 0.0)
            out[w] = row
        return out

    def top(self, query, k=1):
        s = self.scores(query)
        order = sorted(range(len(s)), key=lambda i: -s[i])
        return [(self.docs[i], s[i]) for i in order[:k]]


def retrieval_prompt(user_prompt, doc):
    """Exactly how the paper builds the prompt in retrieval mode."""
    return f"{user_prompt} Use this API documentation for reference: {doc['api_call']}"


# ---------------------------------------------------------------------------
# 3. AST sub-tree matching (the paper's metric, Sec. 3.3)
# ---------------------------------------------------------------------------
# Which arguments identify an API (paper, Sec. 8.3.1): torch.hub.load ->
# repo_or_dir and model; hub.load / hub.KerasLayer -> handle; HuggingFace ->
# pretrained_model_name_or_path, except pipeline() which is matched on task.
KEY_ARGS = {
    "torch.hub.load": ["repo_or_dir", "model"],
    "hub.load": ["handle"],
    "hub.KerasLayer": ["handle"],
    "pipeline": ["task", "model"],
    "from_pretrained": ["pretrained_model_name_or_path"],
}


def _dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def _norm(v):
    # 'pytorch/vision:v0.10.0' and 'pytorch/vision' are the same repository
    return re.sub(r":v[0-9][\w.]*$", "", v) if isinstance(v, str) else v


def parse_call(code):
    """Parse one API call into (function name, {argument: value}) or None."""
    try:
        tree = ast.parse(code.strip())
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _dotted(node.func) or ""
            short = "from_pretrained" if name.endswith("from_pretrained") else name
            keys = KEY_ARGS.get(short, [])
            args = {}
            for i, a in enumerate(node.args):
                if isinstance(a, ast.Constant):
                    args[keys[i] if i < len(keys) else f"arg{i}"] = _norm(a.value)
            for kw in node.keywords:
                if isinstance(kw.value, ast.Constant):
                    args[kw.arg] = _norm(kw.value.value)
            return name, short, args
    return None


def matches(candidate, reference):
    """Is the candidate call a sub-tree match of the reference API?

    Returns (bool, list of (node label, matched?)) so the tree can be drawn.
    """
    c, r = parse_call(candidate), parse_call(reference)
    if not c or not r:
        return False, []
    name_ok = c[0] == r[0] or (c[1] == r[1] == "from_pretrained")
    nodes = [(c[0], name_ok)]
    ok = name_ok
    for key in KEY_ARGS.get(r[1], []):
        if key == "model" and r[1] == "pipeline" and "model" not in r[2]:
            continue
        cv, rv = c[2].get(key), r[2].get(key)
        hit = cv is not None and cv == rv
        nodes.append((f"{key}: {cv}", hit))
        ok = ok and hit
    # optional arguments (e.g. pretrained=True) are not checked, as in the paper
    for k, v in c[2].items():
        if k not in KEY_ARGS.get(r[1], []):
            nodes.append((f"{k}: {v}  (optional, not checked)", None))
    return ok, nodes


def ast_verdict(candidate, reference_id, db):
    """Classify a generated call: 'correct', 'error' or 'hallucination'."""
    ref = next(d for d in db if d["id"] == reference_id)
    ok, nodes = matches(candidate, ref["api_call"])
    if ok:
        return "correct", ref, nodes
    for d in db:
        if matches(candidate, d["api_call"])[0]:
            return "error", d, matches(candidate, d["api_call"])[1]
    return "hallucination", None, nodes


# ---------------------------------------------------------------------------
# Example prompts (paraphrasing the paper's figures) and candidate answers
# ---------------------------------------------------------------------------
@dataclass
class Query:
    prompt: str
    reference: str                       # id of the correct API in API_DB
    candidates: list = field(default_factory=list)   # (label, code)
    note: str = ""


QUERIES = [
    Query("Help me find an API to convert the spoken language in a recorded audio to text using Torch Hub.",
          "silero_stt",
          [("Answer A (invents a model)", "torch.hub.load('snakers4/silero-models', 'asr', source='local')"),
           ("Answer B (wrong library)", "torchaudio.pipelines.WAV2VEC2_ASR_PIPELINE('audio.wav')"),
           ("Answer C (correct)", "torch.hub.load('snakers4/silero-models', 'silero_stt', language='en')")],
          note="Paper Fig. 1: GPT-4 invented a model and Claude picked the wrong library."),
    Query("I want to automatically remove the background from an input image.",
          "fcn_resnet50",
          [("Answer A (correct)", "torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)"),
           ("Answer B (real API, wrong task)", "torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)"),
           ("Answer C (made-up repo)", "torch.hub.load('background-removal/models', 'remove_bg')")],
          note="Paper Fig. 6: the example used for test-time documentation changes."),
    Query("I am an engineer at Uber and I need to classify pedestrians, cars, etc. from an image of the scene.",
          "hybridnets",
          [("Answer A (correct)", "torch.hub.load('datvuthanh/hybridnets', 'hybridnets', pretrained=True)"),
           ("Answer B (real API, wrong task)", "torch.hub.load('pytorch/vision', 'densenet121', pretrained=True)"),
           ("Answer C (made-up model name)", "torch.hub.load('datvuthanh/hybridnets', 'hybridnet_v2')")],
          note="Paper Fig. 8: an example from the dataset with a retrieved reference API."),
    Query("Invoke an image classification model that uses less than 10M parameters, "
          "but maintains an ImageNet accuracy of at least 70%.",
          "mobilenet_v2",
          [("Answer A (meets both constraints)", "torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=True)"),
           ("Answer B (too big: 25.6M params)", "torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)"),
           ("Answer C (made-up model)", "torch.hub.load('pytorch/vision', 'tinynet_70', pretrained=True)")],
          note="Paper Sec. 3.2: the constraint example. The model must reason about numbers, not just topic."),
    Query("I would like to generate an image of cats dancing from a text description.",
          "sd21",
          [("Answer A (correct)", "StableDiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-2-1')"),
           ("Answer B (placeholder path)", "StableDiffusionPipeline.from_pretrained('your_model_name')"),
           ("Answer C (real API, wrong task)", "pipeline('depth-estimation', model='Intel/dpt-large')")],
          note="Paper Fig. 3 and Fig. 9: GPT-4 sometimes assumes a local path like 'your_model_name'."),
    Query("Read this paragraph out loud with a natural voice.",
          "tacotron2",
          [("Answer A (correct)", "torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2')"),
           ("Answer B (real API, wrong task)", "torch.hub.load('snakers4/silero-models', 'silero_stt', language='en')"),
           ("Answer C (made-up repo)", "torch.hub.load('openai/voices', 'tts_natural')")]),
]


# Parts for the interactive "API builder": combine a function, a repository
# and a model name, then let the AST matcher judge the result.
BUILDER = {
    "function": ["torch.hub.load", "hub.load", "pipeline"],
    "repo": ["pytorch/vision", "snakers4/silero-models", "ultralytics/yolov5", "NVIDIA/DeepLearningExamples:torchhub",
             "facebook/models", "automatic-speech-recognition", "summarization", "https://tfhub.dev/google/universal-sentence-encoder/4"],
    "model": ["mobilenet_v2", "resnet50", "fcn_resnet50", "silero_stt", "yolov5s", "nvidia_tacotron2",
              "asr", "openai/whisper-base", "facebook/bart-large-cnn", "(none)"],
}


def build_call(function, repo, model):
    """Assemble the code string for the builder's current choice."""
    if function == "hub.load":
        return f"hub.load('{repo}')"
    if function == "pipeline":
        return f"pipeline('{repo}')" if model == "(none)" else f"pipeline('{repo}', model='{model}')"
    return f"torch.hub.load('{repo}')" if model == "(none)" else f"torch.hub.load('{repo}', '{model}')"


def which_api(code, db):
    """Return the database entry the call matches, or None (a hallucination)."""
    for d in db:
        if matches(code, d["api_call"])[0]:
            return d
    return None


def retriever_aware_model(user_prompt, db):
    """Stand-in for a retriever-aware fine-tuned model (paper Sec. 3.2).

    Gorilla is trained with the retrieved documentation in its prompt, so it
    learns to read the API from that documentation. This stand-in does just
    that: retrieve the top-1 document with BM25 and copy its API call.
    """
    doc, score = BM25(db).top(user_prompt, 1)[0]
    return doc["api_call"], doc, score


def frozen_model(reference_id):
    """Stand-in for a model fine-tuned WITHOUT retrieval: it memorised the API
    as it was at training time, so it cannot see later documentation changes."""
    return next(d for d in API_DB if d["id"] == reference_id)["api_call"]


if __name__ == "__main__":
    db = database()
    bm25 = BM25(db)
    for q in QUERIES:
        top, score = bm25.top(q.prompt, 1)[0]
        print(f"\n{q.prompt}\n  BM25 top-1: {top['id']} (score {score:.2f})"
              f" -> {'right' if top['id'] == q.reference else 'WRONG'} document")
        for label, code in q.candidates:
            v, matched, _ = ast_verdict(code, q.reference, db)
            print(f"  {label:34s} -> {v}" + (f" (matches {matched['id']})" if v == "error" else ""))
