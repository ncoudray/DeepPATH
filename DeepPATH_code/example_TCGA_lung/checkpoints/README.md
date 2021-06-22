The checkpoints different runs using 2168 WSI (Lung) from the TCGA are accessible at [https://genome.med.nyu.edu/public/tsirigoslab/DeepLearning/LungCancer_LUADLUSC_Mutations/]. If you use them, please cite the following paper:

(Nicolas Coudray, Paolo Santiago Ocampo, Theodore Sakellaropoulos, Navneet Narula, Matija Snuderl, David Fenyö, Andre L. Moreira, Narges Razavian, Aristotelis Tsirigos. Classification and mutation prediction from non–small cell lung cancer histopathology images using deep learning. Nature Medicine, 2018; DOI: 10.1038/s41591-018-0177-5)
[https://www.nature.com/articles/s41591-018-0177-5]


`run1a_3D_classifier` was run using batch size of 400; checkpoints at 69k
(AUC of validation for Normal/LUAD/LUSC: 0.9997/0.970/0.967; for test set: 0.991/0.949/0.942)

`run1b_10way_MutationClassifier`: softmax classifier of gene mutations (selecting LUAD slides using checkpoints above)
(Valid and test AUCs: EGFR (0.832/0.777); FAT1 (0.741/0.653); FAT4 (0.672/0.673); KEAP1 (0.864/0.699); KRAS (0.768/0.753); LRP1B (0.758/0.733); NF1 (0.738/0.619); SETBP1 (0.832/0.606); STK11 (0.837/0.828); TP53 (0.831/0.734))

`run2a_3D_classifier` was run with a batch size of 100 for 400k iterations (validation AUC peaking around 108k iterations). 
(AUC of validation for Normal/LUAD/LUSC: 0.9998/0.977/0.977; for test set: 0.990/9.968/0.953)





