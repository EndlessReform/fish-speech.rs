# Learned chunk detection

## Using the model

TODO

## Methodology

Here's the polished version with some engaging examples and minor fixes:

Like most TTS models, Fish Speech generates audio in two stages:
- A LLaMA-style autoregressive transformer backbone generates sequences of 21.535hz **semantic codebooks**
- A non-autoregressive HiFi-GAN vocoder upscales an *entire sequence* of codebooks to 44.1khz WAV output

Semantic codes are generated *streaming* (like ChatGPT), but the actual output audio is *blocking* until the vocoder has enough tokens to generate the full sequence. Without any optimization, you'd have to wait until your whole text is finished to get any output! You can imagine that, if you're waiting for the *first* audio to come out of the system in a streaming response, you'd be waiting a while. If you're having a conversation with an AI, for example, this is unacceptable.

To solve this, we *chunk* our tokens into coherent segments to send to the vocoder, where there are natural gaps in speech. Let's take an example input:
> "Well, although I must confess that the quantum chromodynamics of stellar evolution has always fascinated me beyond measure, particularly in the context of neutron star formation and subsequent pulsar behavior, I find myself equally drawn to the more practical applications in modern particle accelerators, where we can actually test these theories in controlled environments, don't you think? Of course, that's just my opinion."

- If we split our audio on *sentences*, each sentence sounds coherent, but we have to wait for that entire 47-word first sentence to finish generating.
- Couldn't we just send semantic tokens one-at-a-time, or in fixed windows of 20 or 30? This has fast output, but now the vocoder has to process speech out-of-context: "...quantum chromo-" *pause* "-dynamics of" *pause* "stellar evo-". When we stitch these generations together, we get *edge artifacts* where the pieces don't quite fit.
- Couldn't we split on commas or parentheses? Good idea, but *which* commas? "Well (short pause) although I must confess (barely noticeable pause) that the quantum chromodynamics (actual significant pause) of stellar evolution..."

All these linguistic features, however, are missing the point. We just want to split on *silence*, which is a feature of the *audio*. Clearly the semantic model itself already knows when it's generating silence! The model has learned natural speech patterns like:
- Pausing after nested thoughts in parentheses: "The results (as shown in Figure 3) [pause] demonstrate..."
- Different pause lengths for different punctuation: "First,[short] Second,[medium] and Finally[long]..."
- Language-specific pausing: English "Hello... [pause] World" vs Japanese "こんにちは[no pause]世界"

So, just before we generate our codebooks, we can extract whether we're generating silence directly from the model, with a single linear layer.
- We get a representative sample of speakers by generating 1000 speaker prompts for English, Japanese, and Chinese using the [Emilia dataset](https://huggingface.co/datasets/amphion/Emilia-Dataset).
- We then sample 500 more speaker prompts for each language (about 10,000 semantic frames of output), then vocode them and save both the audio output and the hidden states at layer 24 (just before the model predicts whether it's the end of output or not).
- We train a simple linear probe with sigmoid loss on the hidden states to predict whether the audio is quiet or not during that frame.
- Just to be safe and to avoid cutting up output on minor hitches, we train our probe to NOT fire when the audio is ending anyway, and wait for 3 frames of silence (~140ms) before chunking the audio.

Since the model already uses its hidden states to predict features, it's easy to extract simple features like this. After training on just 30 minutes of audio, our strategy has the following accuracy on our test set for detecting multi-frame silences (excluding end-of-audio):
```
True Positive Spans: 792
False Negative Spans: 68
False Positive Spans: 128
Precision: 0.8609
Recall: 0.9209
F1: 0.8899
```

Using this strategy, we get fast streaming TTS with no edge artifacts, comparable latency to human response times (~150-250ms to first audio), and natural-sounding pauses. The probe is tiny (1024->1 linear layer) and downloads automatically from HF, adding basically zero overhead to the model.
If you're running a server and really want to wait for full sentences... well, there's a flag for that. But you probably don't.

Note that this is still audio chunking - the text generation itself is always streaming, so you'll see tokens appear before you hear them. Just like a real person thinking before they speak, except with better pronunciation and no awkward coughs.

## Training your own

First, install [uv](https://github.com/astral-sh/uv) for Python.

Then, in this folder:

```
uv add ipykernel
uv run --with jupyter jupyter lab
```

Follow the notebooks!
