# CpG Detector (work in progress)

Predict the number of "CG" in the given string

**Idea 1 (Done)** 
- Formulate it as a regression problem, regress the total number of "CG" in the string
- *Initial Result:* The R2 is low, accuracy (rounded the predictions) turned out to be ~20% and the predictions are mostly 5 (which is close to mean of the distribution. So, the model has induced to learn the mean of the distribution

**Potential Idea to experiment with**
- Formulate it as classification problem where the string will be divided into a window of 2, and each window, predict whether it is "CG" or not. Add the final predictions.
- Standard Binary Cross Entropy problem! Need to preprocess the dataset and dataloaders to fit this problem!
- Metrics will be accuracy, precision, recall and F1 score (standard binary classification problem)

For more details and clear documentation, refer **Report** - https://docs.google.com/document/d/1uF8eOnVi4QrO3cDNYDd1OFeRmaW3PqNt8gjZWluS1IE/edit?usp=sharing

### For streamlit app
```
streamlit run app.py --server.headless=true
```

**Note:** Running in remote desktop is giving both Network and External URLs but returning "connection request"
