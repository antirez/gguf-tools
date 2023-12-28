# GGUF tools

This is a work in progress library to manipulate GGUF files.
The program 'gguf-tools' use the library to implement both useful and
useless stuff, to show the library usage.

    gguf-tools show file.gguf

shows detailed info about the GGUF file. This will include all the key-value pairs, including arrays, and detailed tensors informations. Tensor offsets will be relative to the start *of the file* (so they are actually absolute offsets), not the start of the data section like in the GGUF format.

    gguf-tools compare file1.gguf file2.gguf

For each matching tensor (same name and parameters count) compute the average weights difference. This is useful to see if a model is a finetune of another model, how much it was finetuned, which layers were frozen while finetuning and so forth. Note that becasue of quantization, even tensors that are functionally equivalent may have some small average difference.

    gguf-tools inspect-tensor file.gguf tensor.name [count]

Show all (if count is not specified, otherwise only the first _count_) weights values of the specified tensor. This is useful for low level stuff, like checking if quantization is working as expected, see the introduced error, model fingerprinting and so forth.

    gguf-tools split-mixtral 65230776370407150546470161412165 mixtral.gguf out.gguf

Extracts a 7B model `out.gguf` from Mixtral 7B MoE using the specified MoE ID for each layer (there are 32 digits in the sequence 652...).

Note that split-mixtral is quite useless as models obtained in this way will not perform any useful action. This is just an experiment and a non trivial task to show how to use the library. Likely it will be removed soon, once I have more interesting and useful examples to show, like models merging.

## Specification documents

* [Official GGUF specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md), where the file layout and meta-data is described.
* [Quantization formats](https://github.com/ggerganov/ggml/blob/master/src/ggml-quants.h) used in quantized GGUF models.
