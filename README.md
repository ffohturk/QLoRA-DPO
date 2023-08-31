In this project we combine the computational efficiency of QLoRA with the DPO algorithm. 

Many implementations exist, but this is mine. 

Workflow:

1) Finetune model on RLHF type dataset to put it in the correct format. The dataset for finetuning could for instance consist of prompt together with chosen response. 
2) To prepare the DPO training we can concatenate prompt together with 'chosen' or 'preferred' response.
3) The reference model can be obtained by disabling the peft adapters. 
4) Run a forward pass to generate the required log probabilities of the chosen and rejected response, done in one forward pass.
5) Construct DPO loss and do the backward pass.
