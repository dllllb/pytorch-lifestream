# Inference

Simple inference example are in `demo/coles-emb.ipynb`. 
It uses pretrained `ptls.trames.coles.CoLESModule`. It's also possible `ptls.frames.supervised.SequenceToTarget`
with any pretrained. Both modules call `forward` method of internal model.
`forward` returns tensor with embeddings, scores or `PaddedBatch`.

In addition to the model output, we need to know which user this output is assigned to.
In simple case we get it from data. We iterate over data twice: first to get sequential features from the model,
second to get user ids. We should also avoid of data shuffle and keep item order
to get correct match between model output and ids.

More complicated inference way are in `demo/extended_inference.ipynb`.
`ptls.frames.inference_module.InferenceModule` used.

`InferenceModule.forward` accept `PaddedPatch` input with any types of features.
Sequential are used to get model output. Other are passed to forward output.
In other words `InferenceModule` works with nearly raw data and update it with model prediction.
Usually we don't need sequential features in output, they will be dropped with `InferenceModule.drop_seq_features=True`.
Output transformed to `pandas.DataFrame` with `InferenceModule.pandas_output=True`.

You can't use `InferenceModule` for train due to output format.

`InferenceModule` can be used with any pretrained models.
