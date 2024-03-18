# CpG Detector (work in progress)

Predict the number of "CG" in the given string

**Idea** 
- Formulate it as a regression problem, regress the total number of "CG" in the string
- *Result:* The R2 is around 0.99. The accuracy (rounded the predictions to nearest integer is also 0.99.
- *Discussion:* The model predicts reasonably well on a lot of scenarios (when there are less CGs and more CGs in a normal sequence).
![2](https://github.com/NamburiSrinath/AutonomizeAI_Assessment/assets/40389487/b08ed80d-0078-405f-b925-c073e0a34e75)

But for sequences where there are only CG's (such as below screenshot), the model fails at long sequences. This is out-of-distribution scenario, so it is expected.
![1](https://github.com/NamburiSrinath/AutonomizeAI_Assessment/assets/40389487/837862e8-7763-4f78-9af9-fb34c2040ee8)

For more details and clear documentation, refer 

**Report** - https://docs.google.com/document/d/1uF8eOnVi4QrO3cDNYDd1OFeRmaW3PqNt8gjZWluS1IE/edit?usp=sharing

### For streamlit app
```
streamlit run app.py --server.headless=true
```
Run the above command to get the external URL in which we can enter the string and get the output number of "CG"s.

**Potential Idea to experiment with (refer document for more information)**
- Formulate it as classification problem where the string will be divided into a window of 2, and each window, predict whether it is "CG" or not. Add the final predictions.
- Standard Binary Cross Entropy problem! Need to preprocess the dataset and dataloaders to fit this problem!
- Metrics will be accuracy, precision, recall and F1 score (standard binary classification problem)
