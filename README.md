

# The repository provides the source code for the paper "Combining Textual Features for the Detection of Hateful and Offensive Language" submitted to [HASOC 2021](https://hasocfire.github.io/hasoc/2021/call_for_participation.html) English Subtask 1A.


### Publication

[Arxiv](https://arxiv.org/pdf/2112.04803.pdf)


### Installation (requires >=Python 3.6 )

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

```bash
download the 'resources.zip' file here: https://drive.google.com/file/d/1X88cMrLVpAcJd5Z4Gg6MfTLclIuGF-d6/view?usp=sharing
extract the content of 'resources.zip'
```

### Training and Evaluation on HASOC datasets (2019, 2020, 2021)

Execute the following command to train and evaluate the model. The evaluation results are saved under the folder 'results'.

```python
python main.py -c config.json
```

### Optimizing Hyperparameters

The "config.json" file contains hyperparameters that can be changed to train different variants of the model.

```json
{
  "base_dir": "",
  "batch_size": 64,
  "epochs": 20,
  "epoch_patience": 5,
  "bert_model_dir": "resources/hatebert",
  "monitor": "loss",
  "tweet_text_seq_len": 80,
  "tweet_text_char_len": 128,
  "char_size": 29,
  "max_learning_rate": 0.001,
  "end_learning_rate": 0.0000001,
  "rnn_type": "lstm",
  "rnn_layer_size": 200,
  "text_models": ["char_emb", "bert", "hate_words"],
  "normalize_text": true,
  "dataset_year": "2021",
  "optimizer": "adam",
  "text_use_attention": false,
  "oversample": true,
  "feature_normalization_layer_size": 512,
  "min_feature_normalization_layer_size": 64
}
```

**bert_model_dir**

```json
"bert_model_dir": "resources/hatebert"
     OR
"bert_model_dir": "resources/bert-base"
```

**dataset_year**

```json
"dataset_year": "2019"
	OR
"dataset_year": "2020"
	OR
"dataset_year": "2021"
```

**text_models**

```json
"text_models": ["hate_words"]
	OR
"text_models": ["bert"]
	OR
"text_models": ["char_emb"]
	OR
"text_models": ["char_emb", "bert", "hate_words"]
```

**rnn_type**

```json
"rnn_type": "lstm"
	OR
"rnn_type": "gru"
	OR
"rnn_type": "bi-gru"
```