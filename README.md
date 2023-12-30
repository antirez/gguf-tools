# GGUF tools

This is a work in progress library to manipulate GGUF files.
While the library aims to be useful, one of the main goals is to provide
an accessible code base that as a side effect documents the GGUF
files used by the awesome [llama.cpp](https://github.com/ggerganov/llama.cpp) project.

The program **gguf-tools** use the library to implement both useful and
useless stuff, to show the library usage in the real world. For now
the utility implements the following subcommands:

### gguf-tools show file.gguf

shows detailed info about the GGUF file. This will include all the key-value pairs, including arrays, and detailed tensors informations. Tensor offsets will be relative to the start *of the file* (so they are actually absolute offsets), not the start of the data section like in the GGUF format.

### gguf-tools compare file1.gguf file2.gguf

For each matching tensor (same name and parameters count) compute the average weights difference. This is useful to see if a model is a finetune of another model, how much it was finetuned, which layers were frozen while finetuning and so forth. Note that becasue of quantization, even tensors that are functionally equivalent may have some small average difference.

Example output:

```
./gguf-tools compare mistral-7b-instruct-v0.2.Q8_0.gguf \
                     solar-10.7b-instruct-v1.0-uncensored.Q8_0.gguf
[token_embd.weight]: avg weights difference: 44.539944%
[blk.0.attn_q.weight]: avg weights difference: 48.717736%
[blk.0.attn_k.weight]: avg weights difference: 56.201885%
[blk.0.attn_v.weight]: avg weights difference: 47.087249%
[blk.0.attn_output.weight]: avg weights difference: 47.663048%
[blk.0.ffn_gate.weight]: avg weights difference: 37.508761%
[blk.0.ffn_up.weight]: avg weights difference: 39.061584%
[blk.0.ffn_down.weight]: avg weights difference: 39.632648%
...
```

### gguf-tools inspect-tensor file.gguf tensor.name [count]

Show all (if count is not specified, otherwise only the first _count_) weights values of the specified tensor. This is useful for low level stuff, like checking if quantization is working as expected, see the introduced error, model fingerprinting and so forth.

### gguf-tools split-mixtral 65230776370407150546470161412165 mixtral.gguf out.gguf

Extracts a 7B model `out.gguf` from Mixtral 7B MoE using the specified MoE ID for each layer (there are 32 digits in the sequence 652...).

Note that split-mixtral is quite useless as models obtained in this way will not perform any useful action. This is just an experiment and a non trivial task to show how to use the library. Likely it will be removed soon, once I have more interesting and useful examples to show, like models merging.

## Specification documents

* [Official GGUF specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md), where the file layout and meta-data is described.
* [Quantization formats](https://github.com/ggerganov/ggml/blob/master/src/ggml-quants.h) used in quantized GGUF models.
