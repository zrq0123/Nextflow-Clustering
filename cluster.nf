params.python_path='/home/heyx/anaconda3/envs/torch_38/bin/python'
params.py='/home/hyx/nf'
params.input_h5py=''
params.input_txt_data=''
params.input_txt_label=''
params.File_Type='Plasschaert'
params.Gene_Sel=0
params.Gene_Calsim=0
params.Est_Clu=0
params.Clustering=0
params.Validation=0
params.output_file=''

process File_Type {
	debug true
	tag 'CAKE'

	input:
	val x

	output:
	stdout 

	script:
    """
    #! $params.python_path
    import argparse
    import sys
    import os
    dir = os.path.abspath(os.path.join(os.getcwd(),"../../.."))
    sys.path.append(dir)
    from utils import *
    from train_KD import * 
    import torch
    import _pickle as cPickle


    parser = argparse.ArgumentParser()
    config = yaml_config_hook(f"/home/heyx/nextflow/CAKE/config/config_${params.File_Type}.yaml")

    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    set_seed(args.seed)
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(args.data_type)
    """
}

process Gene_Sel {
	debug true
	tag 'CAKE'

	input:
	val x

	output:
	stdout 

	script:
    """
    #! $params.python_path
    import argparse
    import sys
    import os
    dir = os.path.abspath(os.path.join(os.getcwd(),"../../.."))
    sys.path.append(dir)
    from utils import *
    from train_KD import * 
    import torch
    import _pickle as cPickle
    parser = argparse.ArgumentParser()
    config = yaml_config_hook(f"/home/heyx/nextflow/CAKE/config/config_${params.File_Type}.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    print(args.num_genes)
    """
}
process Cal_Sim {
	debug true
	tag 'CAKE'

	input:
	val x

	output:
	stdout 

	script:
    """
    #! $params.python_path
    import argparse
    import sys
    import os
    dir = os.path.abspath(os.path.join(os.getcwd(),"../../.."))
    sys.path.append(dir)
    from utils import *
    from train_KD import * 
    import torch
    import _pickle as cPickle
    parser = argparse.ArgumentParser()
    config = yaml_config_hook(f"/home/heyx/nextflow/CAKE/config/config_${params.File_Type}.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    print(args.n)
    """
}
process Est_Num {
	debug true
	tag 'CAKE'

	input:
	val x

	output:
	stdout 

	script:
    """
    #! $params.python_path
    import argparse
    import sys
    import os
    dir = os.path.abspath(os.path.join(os.getcwd(),"../../.."))
    sys.path.append(dir)
    from utils import *
    from train_KD import * 
    import torch
    import _pickle as cPickle

    parser = argparse.ArgumentParser()
    config = yaml_config_hook(f"/home/heyx/nextflow/CAKE/config/config_${params.File_Type}.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    print(args.classnum)
    """
}
process Clustering {
	debug true
	tag 'CAKE'

	input:
	val x

	output:
	stdout 

	script:
    """
    #! $params.python_path
    import argparse
    import sys
    import os
    dir = os.path.abspath(os.path.join(os.getcwd(),"../../.."))
    sys.path.append(dir)
    from train_KD import * 
    import _pickle as cPickle
    import numpy as np
    import prettytable as pt
    from loaders import *
    from moco import *
    from read_data import *
    from utils import *
    from evaluation import *
    from pretrain import *
    from cluster import *
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    import torch
    from torch import nn
    from torch.utils.data import DataLoader


    parser = argparse.ArgumentParser()
    config = yaml_config_hook(f"/home/heyx/nextflow/CAKE/config/config_${params.File_Type}.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    set_seed(args.seed)
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    # 1. pretrain moco 
    #print("---------- Step 1: Pretrain MoCo ----------")
    model, start_time1 = pretrain(args, device=device)

    # 2. get pseudo label 
    #print('---------- Step 2: Get Pseudo Labels ----------')
    adata_embedding, adata, Y, leiden_pred, _, val_loader = get_pseudo_label(args,
                                                                             model, 
                                                                             device=device)
    #end_time1 = time.time()
    #plot(adata_embedding, 
    #     Y, 
    #     args.dataset_name, 
    #     epoch=args.epochs, 
    #     seed=args.seed,
    #     dir_path_name="pictures")

    # 3. train teacher model
    # 3.1 data prepare
    #print('---------- Step 3: Train & Evaluate Teacher Model ----------')
    adata, adata_embedding = get_anchor(adata, 
                                        adata_embedding, 
                                        pseudo_label='leiden',
                                        k=30, 
                                        percent=0.5)
    train_adata = adata[adata.obs.leiden_density_status == 'low', :].copy()
    test_adata = adata[adata.obs.leiden_density_status == 'high', :].copy()

    pseudo_labels = np.array(list(map(int, train_adata.obs['leiden'].values)))
    #print(f"extracted_nmi: {normalized_mutual_info_score(train_adata.obs['Group'].values, pseudo_labels):.4f}")
    #print(f"extracted_ari : {adjusted_rand_score(train_adata.obs['Group'].values, pseudo_labels):.4f}")

    train_dataset = CellDatasetPseudoLabel(train_adata, 
                                           pseudo_label='leiden', 
                                           oversample_flag=True)
    test_dataset = CellDatasetPseudoLabel(test_adata,
                                          pseudo_label='leiden', 
                                          oversample_flag=False)

    #print(f"teacher train dataset: {len(train_dataset)}")
    #print(f"teacher test dataset: {len(test_dataset)}")

    # 3.2 build KD model
    in_features = adata.shape[1]
    teacher = Encoder(in_features=in_features,
                      num_cluster=len(np.unique(leiden_pred)),
                      latent_features=args.latent_feature,
                      device=device,
                      p=args.p)
 
    student = Encoder(in_features=in_features,
                      num_cluster=len(np.unique(leiden_pred)),
                      latent_features=args.latent_feature,
                      device=device,
                      p=args.p)

    # 3.3 loader pretrained weight for teacher model
    for name, param in teacher.named_parameters():
        if name not in ["fc.weight", "fc.bias"]:
            param.requires_grad = False
        
    teacher.fc.weight.data.normal_(mean=0.0, std=0.01)
    teacher.fc.bias.data.zero_()

    epoch = args.epochs
    model_path = os.path.join(args.model_path, f"seed_{args.seed}")
    model_fp = os.path.join(model_path, f"checkpoint_{epoch}.tar")
    state_dict = torch.load(model_fp, map_location="cuda:1")['net']

    for k in list(state_dict.keys()):
        if k.startswith("encoder_k.") and not k.startswith("encoder_k.fc"):
            state_dict[k[len("encoder_k."):]] = state_dict[k]
    
        del state_dict[k]
    
    #print(list(state_dict.keys()))
    msg = teacher.load_state_dict(state_dict, strict=False)
    #print(set(msg.missing_keys))

    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers)
    teacher_criterion = nn.CrossEntropyLoss()
    parameters = list(filter(lambda p : p.requires_grad, teacher.parameters()))

    #assert len(parameters) == 2

    teacher_optimizer = torch.optim.Adam(parameters, 
                                         lr=args.learning_rate, 
                                         weight_decay=0.0)
    val_teacher_loader = DataLoader(test_dataset, 
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)
    
    # 4. train teacher model
    start_time2 = time.time()
    epochs = 100
    best_acc1 = 0.0
    for epoch in range(args.start_epoch, epochs+1):
        loss_epoch = train_teacher(train_loader, 
                                   teacher, 
                                   teacher_criterion, 
                                   teacher_optimizer, 
                                   device, 
                                   epochs)
        
        # evaluate on validation set
        # acc1 = validate_teacher(val_teacher_loader, teacher, teacher_criterion, device)
        # remember best acc@1 and save checkpoint
        # best_acc1 = max(acc1, best_acc1)
        
        #print(f"Epoch [{epoch}/{epochs}]\t Loss: {loss_epoch}")
        #print('-' * 60)
        
    # print(f"Best Accuracy: {best_acc1}")

    # 4.2 evaluation tearcher model performance
    teacher_pred = get_prediction(teacher, device, val_loader)
    teacher_pred = np.array(teacher_pred, dtype=np.int32)
    adata_embedding.obs['teacher_prediction'] = teacher_pred
    adata_embedding.obs['teacher_prediction'] = adata_embedding.obs['teacher_prediction'].astype('category')
    
    teacher_ari = adjusted_rand_score(Y, teacher_pred)
    teacher_nmi = normalized_mutual_info_score(Y, teacher_pred)

    leiden_ari = adjusted_rand_score(Y, leiden_pred)
    leiden_nmi = normalized_mutual_info_score(Y, leiden_pred)

    # 5. train distiller
    #print('---------- Step 4: Train & Evaluate Distiller ----------')
    distiller_loss = DistillerLoss(alpha=args.kd_alpha, 
                                   temperature=args.kd_temperature)
    distiller_optimizer = torch.optim.Adam(student.parameters(), 
                                           lr=args.learning_rate, 
                                           weight_decay=0.0)

    # freeze parameters in teacher model
    for name, param in teacher.named_parameters():
        param.requires_grad = False
        
    epochs = 50
    for epoch in range(args.start_epoch, epochs+1):
        loss_epoch = train_distiller(train_loader, 
                                     student, 
                                     teacher, 
                                     distiller_loss, 
                                     distiller_optimizer, 
                                     device)

        #print(f"Epoch [{epoch}/{epochs}]\t Loss: {loss_epoch}")
        #print('-' * 60)
    end_time2 = time.time()

    # 5.2 evalation student model
    student_pred = get_prediction(student, device, val_loader)
    student_pred = np.array(student_pred, dtype=np.int32)
    adata_embedding.obs['student_prediction'] = student_pred
    adata_embedding.obs['student_prediction'] = adata_embedding.obs['student_prediction'].astype("category")

    write_file = open('/home/heyx/nextflow/CAKE/temp/Y.pkl','wb')
    cPickle.dump(Y, write_file)
    write_file.close()

    np.savetxt("/home/heyx/nextflow/CAKE/output/${params.File_Type}.csv", Y, delimiter=',')

    write_file = open('/home/heyx/nextflow/CAKE/temp/student_pred.pkl','wb')
    cPickle.dump(student_pred, write_file)
    write_file.close()

    write_file = open('/home/heyx/nextflow/CAKE/temp/adata_embedding.pkl','wb')
    cPickle.dump(adata_embedding, write_file)
    write_file.close()


    """
}
process Visualization {
	debug true
	tag 'CAKE'

	input:
	val x

	output:
	stdout 

	script:
"""
#! $params.python_path
#import argparse
import sys
import os
dir = os.path.abspath(os.path.join(os.getcwd(),"../../.."))
sys.path.append(dir)
from utils import *
from train_KD import * 
from evaluation import *
import torch
import _pickle as cPickle
import argparse


parser = argparse.ArgumentParser()
config = yaml_config_hook(f"/home/heyx/nextflow/CAKE/config/config_${params.File_Type}.yaml")

for k, v in config.items():
    parser.add_argument(f"--{k}", default=v, type=type(v))

args = parser.parse_args()

#if not os.path.exists(args.model_path):
    #os.makedirs(args.model_path)

read_file = open('/home/heyx/nextflow/CAKE/temp/Y.pkl','rb')
Y=cPickle.load(read_file)
read_file.close()

read_file = open('/home/heyx/nextflow/CAKE/temp/student_pred.pkl','rb')
student_pred=cPickle.load(read_file)
read_file.close()

read_file = open('/home/heyx/nextflow/CAKE/temp/adata_embedding.pkl','rb')
adata_embedding=cPickle.load(read_file)
read_file.close()

# visualize result
plot(adata_embedding, 
        Y, 
        args.dataset_name, 
        args.epochs, 
        seed=args.seed, 
        colors=['student_prediction', 'annotation'],
        titles=['Student Prediction', 'Cell Type'],
        dir_path_name="/home/heyx/nextflow/CAKE/output")

plot(adata_embedding, 
        Y, 
        args.dataset_name, 
        args.epochs, 
        seed=args.seed, 
        colors=['teacher_prediction', 'student_prediction'],
        titles=['Teacher Prediction', 'Student Prediction'],
        dir_path_name="/home/heyx/nextflow/CAKE/output")

"""
}

process Validation {
	debug true
	tag 'CAKE'

	input:
	val x

	output:
	stdout 

	script:
"""
#! $params.python_path
import argparse
import sys
import os
dir = os.path.abspath(os.path.join(os.getcwd(),"../../.."))
sys.path.append(dir)
from utils import *
from train_KD import * 
from evaluation import *
import torch
import _pickle as cPickle
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


parser = argparse.ArgumentParser()
config = yaml_config_hook(f"/home/heyx/nextflow/CAKE/config/config_${params.File_Type}.yaml")

for k, v in config.items():
    parser.add_argument(f"--{k}", default=v, type=type(v))

args = parser.parse_args()

#if not os.path.exists(args.model_path):
    #os.makedirs(args.model_path)

read_file = open('/home/heyx/nextflow/CAKE/temp/Y.pkl','rb')
Y=cPickle.load(read_file)
read_file.close()

read_file = open('/home/heyx/nextflow/CAKE/temp/student_pred.pkl','rb')
student_pred=cPickle.load(read_file)
read_file.close()

read_file = open('/home/heyx/nextflow/CAKE/temp/adata_embedding.pkl','rb')
adata_embedding=cPickle.load(read_file)
read_file.close()

student_ari = adjusted_rand_score(Y, student_pred)
student_nmi = normalized_mutual_info_score(Y, student_pred)

print("NMI:",student_nmi)
print("ARI:",student_ari)

"""
}


workflow {
    wk = Channel.of(1)
    File_Type(wk)|Gene_Sel|Cal_Sim|Est_Num|Clustering|Visualization|Validation   
}