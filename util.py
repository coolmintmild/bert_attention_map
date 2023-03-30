import torch
from transformers import BertTokenizer, BertModel, AlbertTokenizer, AlbertModel
import numpy as np
import seaborn as sns


def load_model(architecture, version, cuda=False):
    if architecture == 'bert':
        # Load pre-trained model tokenizer (vocabulary)
        tokenizer = BertTokenizer.from_pretrained(version, do_lower_case = True)
        # Load pre-trained model (weights)
        model = BertModel.from_pretrained(version, output_attentions=True)
    if architecture == 'albert':
        # Load pre-trained model tokenizer (vocabulary)
        tokenizer = AlbertTokenizer.from_pretrained(version)
        # Load pre-trained model (weights)
        model = AlbertModel.from_pretrained(version, output_attentions=True)
    numLayers = model.config.to_dict()['num_hidden_layers']
    numHeads = model.config.to_dict()['num_attention_heads']
    if cuda :
        model.to('cuda')
    return model, tokenizer, numLayers, numHeads

def format_attention(attention):
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                             "output_attentions=True when initializing your model.")
        squeezed.append(layer_attention.squeeze(0))
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)

def get_attentions(model, tokenizer, row, numLayers, numHeads, cuda=False):
    From = row.From
    To = tokenizer.tokenize(row.To)
    maskedSentence = row.Sentence.replace(From, '[MASK]')

    inputs = tokenizer.encode_plus(maskedSentence, return_tensors='pt', add_special_tokens=True)
    input_ids = inputs['input_ids']
    if cuda:
        attention = model(input_ids.to('cuda'))[-1]
    else:
        attention = model(input_ids)[-1]
    # attention 구조 : [layer][head]left_word][right_word]
    input_id_list = input_ids[0].tolist() # Batch index 0
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)
    attn = format_attention(attention)
    attn_data = {
            'attn': attn.tolist(),
            'left_text': tokens,
            'right_text': tokens
    }
    From_idx = tokens.index('[MASK]')
    To_idx = [tokens.index(t) for t in To if t in tokens]

    attentions = np.zeros((numLayers, numHeads))
    for Layer in range(numLayers):
        for Head in range(numHeads):
            targets = attn_data['attn'][Layer][Head][From_idx][To_idx[0]:To_idx[-1]+1]
            attentions[Layer, Head] = sum(targets)

    return attentions

def get_heatmaps(target_attentions, numHeads):
    ax = sns.heatmap(target_attentions, vmin=0, vmax=1, cmap="Reds")
    xticks = np.arange(numHeads)
    if numHeads > 32:
        xticks = np.arange(0, numHeads+1, 4)
        ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, ha='center')
    fig = ax.get_figure()

    return fig