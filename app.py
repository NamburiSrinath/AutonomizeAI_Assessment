import streamlit as st
import torch
import torch.nn as nn
from functools import partial

# some config
LSTM_INPUT_SIZE = 6
LSTM_HIDDEN = 256
LSTM_LAYER = 2

def count_CG(string):
    count = 0
    for i in range(len(string) - 1):
        if string[i:i+2] == "CG":
            count += 1
    return count
# Model
class CpGPredictor(torch.nn.Module):
    ''' Simple model that uses a LSTM to count the number of CpGs in a sequence '''
    def __init__(self):
        super(CpGPredictor, self).__init__()
        # TODO complete model, you are free to add whatever layers you need here
        # We do need a lstm and a classifier layer here but you are free to implement them in your way
        self.lstm = nn.LSTM(LSTM_INPUT_SIZE, LSTM_HIDDEN, LSTM_LAYER, batch_first=True,
                            dropout=0.2, bidirectional=True)

        self.dropout = nn.Dropout(0.2)
        # Have a hidden layer to map non-linearities!
        # NOTE: If it's BiLSTM -> Need to have *2 !! 
        self.hidden1 = nn.Linear(LSTM_HIDDEN*2, LSTM_HIDDEN//2)
        self.relu = nn.ReLU()
        self.hidden2 = nn.Linear(LSTM_HIDDEN//2, LSTM_HIDDEN//4)
        # Final classification layer
        self.classifier = nn.Linear(LSTM_HIDDEN//4, 1)

    def forward(self, x):
        # TODO complete forward function
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = last_output

        output = self.hidden1(output)
        output = self.relu(output)
        output = self.dropout(output)

        output = self.hidden2(output)
        output = self.relu(output)
        output = self.dropout(output)
        
        logits = self.classifier(output)
        return logits.squeeze(1) # removing 1

# Load the trained model
model = CpGPredictor()
model.load_state_dict(torch.load('best_model_part2.pkl', map_location='cpu'))
model.eval()

alphabet = 'NACGT'
dna2int = { a: i for a, i in zip(alphabet, range(1, 6))}
int2dna = { i: a for a, i in zip(alphabet, range(1, 6))}
dna2int.update({"pad": 0})
int2dna.update({0: "<pad>"})

intseq_to_dnaseq = partial(map, int2dna.get)
dnaseq_to_intseq = partial(map, dna2int.get)

# Define function to predict CpG sites
def predict_cpg(sequence):
    max_len = 128
    encoded_seq = list(dnaseq_to_intseq(sequence))
    encoded_seq = encoded_seq[:max_len] + [0] * (max_len - len(encoded_seq))
    encoded_seq_t = torch.nn.functional.one_hot(torch.tensor(encoded_seq), num_classes=len(alphabet)+1).float()
    encoded_seq_t = encoded_seq_t.unsqueeze(0)
    
    print(encoded_seq_t.shape)
    with torch.no_grad():
        pred = model(encoded_seq_t)  
    # Not rounding it to nearest integer!
    return pred.item()

# Streamlit app
st.title('CpG Site Predictor')

# Input box
sequence_input = st.text_input('Enter DNA sequence (e.g., "NACGT")')

# Prediction
if sequence_input:
    prediction = predict_cpg(sequence_input)
    st.write(f'Length of sequence before padding: {len(sequence_input)}')
    st.write(f'Predicted CpG sites: {prediction}')
    st.write(f'Actual number of CpG sites: {count_CG(sequence_input)}')