In this project we try to combine the computational efficiency of QLoRA with the effeciency of the DPO algorithm. 

Workflow:

1) Finetune model on RLHF type dataset to put it in the correct format. Can use prompt together with 'chosen' or 'preferred' response.
2) Prepare data so that you have a chosen and rejected 
3) Generate data with a given prompt in both the trainable model and reference model. This gives some log probs and you want to increase the log probs of the tokens of the preferred response and decrease those of the rejected response.
4) Construct DPO loss and do the training.
