# GGUF tools

This is a work in progress library to manipulate GGUF files.
The program 'gguf-tools' use the library to implement both useful and
useless stuff, to show the library usage.

    gguf-tools show file.gguf

shows detailed info about the GGUF file. This will include all the key-value pairs, including arrays, and detailed tensors informations. Tensor offsets will be relative to the start *of the file*, not the start of the data section like in the GGUF format (absolute file offsets are more useful and simpler to use).

    gguf-tools split-mixtral 65230776370407150546470161412165 mixtral.gguf out.gguf

Extracts a 7B model `out.gguf` from Mixtral 7B MoE using the specified MoE ID for each layer (there are 32 digits in the sequence 652...).

Note that split-mixtral is quite useless as models obtained in this way will not perform any useful action. This is just an experiment and a non trivial task to show how to use the library. Likely it will be removed soon, once I have more interesting and useful examples to show.

## Specification documents

* [Official GGUF specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md), where the file layout and meta-data is described.
* [Quantization formats](https://github.com/ggerganov/ggml/blob/master/src/ggml-quants.h) used in quantized GGUF models.
