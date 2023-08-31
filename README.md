# QLoRA meets DPO

We combine the computational efficiency of QLoRA with the DPO algorithm. 

Many implementations exist, but this is mine. 

Workflow:

1) Finetune model on RLHF type dataset to put it in the correct format. This can be done with the standard QLoRA code after which you merge the model with the adapter weights. The dataset for finetuning could for instance consist of prompt together with chosen response. I have included the one I used, split in two sets (labelled 0 and 1). I used 0 for finetuning and 1 for DPO. 
2) To prepare the DPO training we can concatenate prompt together with 'chosen' or 'preferred' response.
3) The reference model can be obtained by disabling the peft adapters. 
4) Run a forward pass to generate the required log probabilities of the chosen and rejected response, done in one forward pass.
5) Construct DPO loss and do the backward pass.
