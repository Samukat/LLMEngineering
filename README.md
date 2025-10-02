# About our project
Testing a methodology the paper "Refusal in Language Models Is Mediated by a Single Direction" on the new GPT-OSS 20B and Qwen3-8B models.

Tech used: Juptyer Notebooks run on GPUs rented via RunPod.

# Results:
Our project was succesful in identifying vectors which could be ablated from the activations to inhibit refusal and significantly increase the rate at which the model offered harmful information.

When given harmful prompts, our **altered gpt-oss model** had a refusal rate of 0%. It gave unsafe responses to 60% of harmful requests. 

![gpt_oss 20b results](Images/gpt-oss%20results%20graph.png)

When given harmful prompts, our **altered Qwen 3 model** had a refusal rate of 0%. It gave unsafe responses to 90% of harmful requests. 

![Qwen3 8b Results](Images/Qwen3%20Results%20Graph.png)

# Guide to this repo:
Our work is split across several Jupyter Notebooks:

## 1. Computing Candidate Refusal Vectors
In this notebook, we record the model's activations on harmful and harmless generations. 
We then take the difference of these activations at many different token positions and layers to find a set of candidate refusal vectors. These vectors are stored in the data folder of this repo.

## 2. Generating Outputs with Candidate Refusal Vectors  
This notebook implements the experimental setup from *“Refusal in Language Models is Mediated by a Single Direction.”*  
It tests candidate refusal vectors through batched ablation experiments, recording their behaviour when given both  harmful and harmless prompts.  
The resulting data frame provides a structured basis for identifying the best single direction for mediating refusal.

## 3. Safety Scoring
Here, we use the Llama Guard 2 model to measure whether model responses contain harmful material. The evaluation is stored in a new column in the dataframe of outputs. This dataframe is stored as a .csv file in the data folder of this repo.

## 4. Vector Selection & Evaluation
Here, we use a series of graphs to find the ideal refusal vector to ablate in the intervention. We then graph the impact that ablating this vector has on model safety & refusal.

# Limitations and Next Steps:
We hope to re-run these experiments with a higher sample size and longer outputs (particularly for the Qwen model). We will then capture our findings in a research paper. When completed, this paper will be added to the repo. 
