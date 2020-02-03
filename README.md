# arabic-did
Arabic Dialect Identification Using Different DL Models

This repo is concerned with sentence level Arabic dialect identification using deep learning models. The included models are VDCNN, AWD-LSTM, & BERT using Pytorch. The hyperparameters have been automatically optimized using <a href="https://github.com/ray-project/ray/tree/master/python/ray/tune" target="_blank">**Tune**</a>. An attempt of explaining the model's output was done using <a href="https://github.com/limetext/lime" target="_blank">**LIME**</a>. Experiments have been logged using Comet.ml and can be viewed <a href="https://www.comet.ml/iahmedmaher/arabic-did-paper/view/new" target="_blank">**here**</a>. You can also view the main experiments from the links in the tables below.

## Results 
### Shami Corpus
The following table summarized the performance of each architecture on the <a href="https://github.com/GU-CLASP/shami-corpus/tree/master/Data" target="_blank">**Shami Corpus**</a>. The reported numbers are the best macro-average performance achieved using automatic HPO (HyperOpt) except for the pretrained BERT. The input token mode was treated as a hyperparameter in that search. It is to be noted that the training and validation set are shuffled each HyperOpt iteration to prevent it from overfitting on the validation set. You can view details about the experiment that achieved the reported numbers by visiting the link associated with each model. This achieves state-of-the-art results on that dataset.

| Model | Input Token Mode | Validation Dataset Perf | Test Dataset Perf | Experiment Link |
| --- | --- | --- | --- | --- |
| VDCNN | char | 90.219 % | 89.247 % | <a href="https://www.comet.ml/iahmedmaher/arabic-did-paper/369aec9af3184d7f92c289ae4847a925?experiment-tab=metrics"> Link </a> |
| AWD-LSTM | char | 89.784% | 89.403 % | <a href="https://www.comet.ml/iahmedmaher/arabic-did-paper/929d7a485c1949499cf7fb6090b3f8e3?experiment-tab=metrics"> Link </a> |
| BERT | subword | 81.823% | 80.715 % | <a href="https://www.comet.ml/iahmedmaher/arabic-did-paper/f441f9dda76b4782b43bb979944ff851?experiment-tab=metrics"> Link </a> |
| BERT (Pretrained) | subword | 81.336% | 80.474 % | <a href="https://www.comet.ml/iahmedmaher/arabic-did-paper/c71b479d192340f58218a413ea321d9d?experiment-tab=metrics"> Link </a> |

### Arabic Online Commentary Dataset
The following table includes the performance results of applying the best model, AWD-LSTM, in the previous table along with its hyperparameters on the <a href="https://www.aclweb.org/anthology/P11-2007/"> AOC dataset </a> with the same train\dev\test splits done <a href="https://github.com/UBC-NLP/aoc_id/tree/master/data">here</a>; no form of manual or automatic optimization was done on AOC except for training the parameters of the AWD-LSTM model. This achieves state-of-the-art results on that dataset. Each result is linked with a Comet.ml experiment.

| Model | Dev Micro Avg | Dev Macro Avg | Test Micro Avg | Test Macro Avg |
| --- | --- | --- | --- | --- |
| AWD_LSTM | <a href="https://www.comet.ml/iahmedmaher/arabic-did-paper/a9f8af25f53f437685c678fd251c2f4a?experiment-tab=metrics"> 84.741%</a> | <a href="https://www.comet.ml/iahmedmaher/arabic-did-paper/a9f8af25f53f437685c678fd251c2f4a?experiment-tab=metrics">79.743%</a> | <a href="https://www.comet.ml/iahmedmaher/arabic-did-paper/c8676329737743119b5a73a5078c53f6?experiment-tab=metrics">83.380%</a> | <a href="https://www.comet.ml/iahmedmaher/arabic-did-paper/c8676329737743119b5a73a5078c53f6?experiment-tab=metrics">75.721%</a> |
