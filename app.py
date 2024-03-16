import streamlit as st
import torch
import torch.nn as nn
from functools import partial

# some config
LSTM_INPUT_SIZE = 128
LSTM_HIDDEN = 128
LSTM_LAYER = 2

# Model
class CpGPredictor(torch.nn.Module):
    ''' Simple model that uses a LSTM to count the number of CpGs in a sequence '''
    def __init__(self):
        super(CpGPredictor, self).__init__()
        # TODO complete model, you are free to add whatever layers you need here
        # We do need a lstm and a classifier layer here but you are free to implement them in your way
        self.lstm = nn.LSTM(LSTM_INPUT_SIZE, LSTM_HIDDEN, LSTM_LAYER, batch_first=True)

        # Have a hidden layer to map non-linearities!
        self.hidden = nn.Linear(LSTM_HIDDEN, LSTM_HIDDEN)

        # Final classification layer
        self.classifier = nn.Linear(LSTM_HIDDEN, 1)

    def forward(self, x):
        # TODO complete forward function
        lstm_out, _ = self.lstm(x)
        # print(lstm_out.shape) # batch_size x LSTM_HIDDEN
        # This might not make sense as our tensor is 2D for eg: (32, 128)
        # TODO: Can we make it to 3D tensor and model! Think more on pros and cons! 
        # last_output = lstm_out[:, -1, :]
        output = lstm_out

        output = self.hidden(output)
        logits = self.classifier(output)
        # print(logits.shape) # batch_size x 1
        # print(logits)
        return logits.squeeze(1) # removing 1

# Load the trained model
model = CpGPredictor()
model.load_state_dict(torch.load('trial_model_part1.pkl'))
model.eval()

alphabet = 'NACGT'
dna2int = { a: i for a, i in zip(alphabet, range(6))}
int2dna = { i: a for a, i in zip(alphabet, range(6))}
dna2int.update({"pad": 0})
int2dna.update({0: "<pad>"})

intseq_to_dnaseq = partial(map, int2dna.get)
dnaseq_to_intseq = partial(map, dna2int.get)

# Define function to predict CpG sites
def predict_cpg(sequence):
    max_len = 128
    # print(sequence)
    encoded_seq = list(dnaseq_to_intseq(sequence))
    encoded_seq = encoded_seq[:max_len] + [0] * (max_len - len(encoded_seq))
    encoded_seq_t = torch.tensor(encoded_seq, dtype=torch.float32).unsqueeze(0)
    # print(encoded_seq_t)
    # print(encoded_seq_t.shape)
    with torch.no_grad():
        pred = model(encoded_seq_t)  
    # Not rounding it to nearest integer!
    return pred.item()
    # return int(pred.item())

# Streamlit app
st.title('CpG Site Predictor')

# Input box
sequence_input = st.text_input('Enter DNA sequence (e.g., "NACGT")')

# Prediction
if sequence_input:
    prediction = predict_cpg(sequence_input)
    st.write(f'Predicted CpG sites: {prediction}')