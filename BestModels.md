

## BERT Finetune
20 Test students
There 9 ties with Equivalent final val_auprc. Picked this based on vibes. 
- test20_lr5e-07_alpha0.01_beta0.01*

15 Test students
Best final val auroc with lr below 0.0001. No evidence that early stopping might help
- finetune /cluster/tufts/hugheslab/kheuto01/sensemaking/bertfinetune_test/test15_lr1e-05_alpha0.0001_beta0.1/

- bow  /cluster/tufts/hugheslab/kheuto01/sensemaking/bow/20250317_new_embed/test_15/test15_lr0.01_wd0.001_bsN/
- frozen /cluster/tufts/hugheslab/kheuto01/sensemaking/bertfrozen/20250310_new_embed/test_15/test15_lr0.001_wd0.001_bsN/
