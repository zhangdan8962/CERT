import torch
from transformers import BertTokenizer,BertForSequenceClassification
import torch.nn as nn
import csv
import pickle
import os
from pycocotools.coco import COCO
# load model
state_dict = torch.load('./checkpoints/moco.p')
# for para in model:
#     print(para,"\t",model[para].size())

# create model
net = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=80,
        output_attentions=False,
        output_hidden_states=False,
      )

# for para in net.state_dict():
#     print(para,"\t",net.state_dict()[para].size())

# fc_features = net.classifier.in_features
# net.classifier = nn.Linear(fc_features,2)

# load parameters
#net.load_state_dict(state_dict)

# for para in net.state_dict():
#     print(para,"\t",net.state_dict()[para].size())

# read train.csv




# predict doc embeddings
input_ids = []
attention_masks = []

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

dataDir = '/home/dzhang4'
dataType = 'val2017'
instances_annFile = os.path.join(dataDir, 'coco/annotations/annotations/instances_{}.json'.format(dataType))
coco = COCO(instances_annFile)

    # initialize COCO API for caption annotations
captions_annFile = os.path.join(dataDir, 'coco/annotations/annotations/captions_{}.json'.format(dataType))
coco_caps = COCO(captions_annFile)
ids = list(coco.anns.keys())

img_ids = []
for index in ids:
    img_id = coco.anns[index]['image_id']
    if img_id not in img_ids:
        img_ids.append(img_id)
print("total samples:"+str(len(img_ids)))
train_label = []
outputs = []
net.eval()
for index in img_ids:
        #img_id = coco.anns[index]['image_id']
        ann = coco.getAnnIds(imgIds=index)

        cat_id = coco.anns[ann[0]]['category_id']
        train_label.append(int(cat_id))
        annIds = coco_caps.getAnnIds(imgIds=img_id)
        anns = coco_caps.loadAnns(annIds)
        sentence1 = anns[0]['caption']
        
        pos_dict1 = tokenizer.encode_plus(sentence1, add_special_tokens=True, max_length=64,
                                             pad_to_max_length=True,
                                             return_attention_mask=True, return_tensors='pt')
        
        input_id = pos_dict1['input_ids']
        mask = pos_dict1['attention_mask']
        with torch.no_grad():
            output = net(input_id,mask)

        #print(output[0].shape)
        outputs.append(output[0])

outputs = torch.cat(outputs,0)
train_label = torch.tensor(train_label)

print("output size:"+ str(outputs.size()))
print("label siez:"+ str(train_label.size()))
print("write files")
# save train_data
with open("train_data_f.pkl","wb") as file:
    pickle.dump(outputs,file)

# save train_label
with open("ttain_label_f.pkl","wb") as f:
    pickle.dump(train_label,f)
