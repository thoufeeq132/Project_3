import time
import argparse
import numpy as np
import pandas as pd

import torch
from build_graph import AnomalyMotifAugmentedNet, BaseGraph, MotifAugmentedNet
from loss import HOGATloss

import dgl
from hogat import HOGAT


def evaluate(topk_indexs, true_indexs):
    recalled_num = 0
    for index in topk_indexs:
        if index in true_indexs:
            recalled_num += 1
    precision = recalled_num / len(topk_indexs)
    recall = recalled_num / len(true_indexs)
    return precision, recall
    



def main(args):
    if args.gpu < 1:
        print("False")
        cuda = False
    else:
        cuda = True
        


    anomaly_motif_data = AnomalyMotifAugmentedNet("cora")

    print("Motify nodes number: ", anomaly_motif_data.num_nodes)
    print("Motify edges number: ", anomaly_motif_data.num_edges)

    adj = anomaly_motif_data.adj.to_dense()
    features = anomaly_motif_data.features

    print("features shape: {}, adj shape: {}".format(features.shape, adj.shape))  

    feature_dim = features.shape[1]
    model = HOGAT(feature_dim, nhid=args.dim, alpha = 0.1,
                 dropout=args.dropout)
    device = "cpu"
    if cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        device = "cuda"

    loss_fcn = HOGATloss()
    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    dur = []
    for epoch in range(args.epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        rec_adj, rec_features = model(features, adj)           
        loss = loss_fcn(rec_adj, rec_features, features, adj)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        if epoch >= 3:
            dur.append(time.time() - t0)

        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} |"
              "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                            anomaly_motif_data.num_edges / np.mean(dur) / 1000))


    print("start caculating anomaly score......")

    topk_list = [40,50,60,70,100,200,300,400]


    nodes_num = anomaly_motif_data.num_nodes - anomaly_motif_data.num_motifs

    model.eval()
    rec_adj, rec_features = model(features, adj)
    structure_anomaly_score = torch.sum((rec_adj - adj)**2, dim = 1)
    attribute_anomaly_score = torch.sum((rec_features - features)**2, dim = 1)

    san_score = structure_anomaly_score[:nodes_num]
    san_true = anomaly_motif_data.structure_anomaly_nodes
    sam_score = structure_anomaly_score[nodes_num:]
    sam_true = anomaly_motif_data.structure_anomaly_motifs

    aan_score = attribute_anomaly_score[:nodes_num]
    aan_true = anomaly_motif_data.attribute_anomaly_nodes
    aam_score = attribute_anomaly_score[nodes_num:]
    aam_true = anomaly_motif_data.attribute_anomaly_motifs
    
    for k in topk_list:
        print("top k: {}".format(k))
        san_topk_indexs = torch.topk(san_score, k).indices
        san_precision, san_recall = evaluate(san_topk_indexs, san_true)
        print("san precision: {:.4f}, san recall: {:.4f}".format(san_precision, san_recall))
        sam_topk_indexs = torch.topk(sam_score, k).indices
        sam_topk_indexs += torch.ones(len(sam_topk_indexs), dtype=torch.long, device=device)*nodes_num
        sam_precision, sam_recall = evaluate(sam_topk_indexs, sam_true)
        print("sam precision: {:.4f}, sam recall: {:.4f}".format(sam_precision, sam_recall))

        aan_topk_indexs = torch.topk(aan_score, k).indices
        aan_precision, aan_recall = evaluate(aan_topk_indexs, aan_true)
        print("aan precision: {:.4f}, aan recall: {:.4f}".format(aan_precision, aan_recall))

        aam_topk_indexs = torch.topk(aam_score, k).indices
        aam_topk_indexs += torch.ones(len(aam_topk_indexs), dtype = torch.long,device=device)*nodes_num
        aam_precision, aam_recall = evaluate(aam_topk_indexs, aam_true)
        print("aam precision: {:.4f}, aam recall: {:.4f}".format(aam_precision, aam_recall))

import pandas as pd

# Create dictionary with the data
data = {
    'top k': [40, 50, 60, 70, 100, 200, 300, 400],
    'san precision': [0.0750, 0.0600, 0.0500, 0.0429, 0.0300, 0.0200, 0.0133, 0.0100],
    'san recall': [0.1875, 0.1875, 0.1875, 0.1875, 0.1875, 0.2500, 0.2500, 0.2500],
    'sam precision': [0.0750, 0.0800, 0.0667, 0.0714, 0.0600, 0.0650, 0.0633, 0.0600],
    'sam recall': [0.0625, 0.0833, 0.0833, 0.1042, 0.1250, 0.2708, 0.3958, 0.5000],
    'aan precision': [0.0500, 0.0400, 0.0333, 0.0429, 0.0400, 0.0400, 0.0433, 0.0425],
    'aan recall': [0.0139, 0.0139, 0.0139, 0.0208, 0.0278, 0.0556, 0.0903, 0.1181],
    'aam precision': [0.0250, 0.0400, 0.0333, 0.0286, 0.0300, 0.0250, 0.0200, 0.0275],
    'aam recall': [0.0208, 0.0417, 0.0417, 0.0417, 0.0625, 0.1042, 0.1250, 0.2292]
}

# Create DataFrame from dictionary
df = pd.DataFrame(data)

# Display the DataFrame
print(df)


import matplotlib.pyplot as plt

# Data
top_k = [40, 50, 60, 70, 100, 200, 300, 400]
san_precision = [0.0750, 0.0600, 0.0500, 0.0429, 0.0300, 0.0200, 0.0133, 0.0100]
san_recall = [0.1875, 0.1875, 0.1875, 0.1875, 0.1875, 0.2500, 0.2500, 0.2500]
sam_precision = [0.0750, 0.0800, 0.0667, 0.0714, 0.0600, 0.0650, 0.0633, 0.0600]
sam_recall = [0.0625, 0.0833, 0.0833, 0.1042, 0.1250, 0.2708, 0.3958, 0.5000]
aan_precision = [0.0500, 0.0400, 0.0333, 0.0429, 0.0400, 0.0400, 0.0433, 0.0425]
aan_recall = [0.0139, 0.0139, 0.0139, 0.0208, 0.0278, 0.0556, 0.0903, 0.1181]
aam_precision = [0.0250, 0.0400, 0.0333, 0.0286, 0.0300, 0.0250, 0.0200, 0.0275]
aam_recall = [0.0208, 0.0417, 0.0417, 0.0417, 0.0625, 0.1042, 0.1250, 0.2292]

# Plotting the bar chart
plt.figure(figsize=(12, 6))
plt.bar(top_k, san_precision, width=5, label='SAN Precision')
plt.bar(top_k, san_recall, width=5, label='SAN Recall')
plt.bar(top_k, sam_precision, width=5, label='SAM Precision')
plt.bar(top_k, sam_recall, width=5, label='SAM Recall')
plt.bar(top_k, aan_precision, width=5, label='AAN Precision')
plt.bar(top_k, aan_recall, width=5, label='AAN Recall')
plt.bar(top_k, aam_precision, width=5, label='AAM Precision')
plt.bar(top_k, aam_recall, width=5, label='AAM Recall')

plt.xlabel('Top K')
plt.ylabel('Precision/Recall')
plt.title('Precision and Recall for Different Top K Values')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt

# Data
top_k = [40, 50, 60, 70, 100, 200, 300, 400]
san_pre = [0.0750, 0.0600, 0.0500, 0.0429, 0.0300, 0.0200, 0.0133, 0.0100]
san_recall = [0.1875, 0.1875, 0.1875, 0.1875, 0.1875, 0.2500, 0.2500, 0.2500]
sam_pre = [0.0750, 0.0800, 0.0667, 0.0714, 0.0600, 0.0650, 0.0633, 0.0600]
sam_recall = [0.0625, 0.0833, 0.0833, 0.1042, 0.1250, 0.2708, 0.3958, 0.5000]
aan_pre = [0.0500, 0.0400, 0.0333, 0.0429, 0.0400, 0.0400, 0.0433, 0.0425]
aan_recall = [0.0139, 0.0139, 0.0139, 0.0208, 0.0278, 0.0556, 0.0903, 0.1181]
aam_pre = [0.0250, 0.0400, 0.0333, 0.0286, 0.0300, 0.0250, 0.0200, 0.0275]
aam_recall = [0.0208, 0.0417, 0.0417, 0.0417, 0.0625, 0.1042, 0.1250, 0.2292]

# Create figure and axes
fig, ax = plt.subplots()

# Plot lines
ax.plot(top_k, san_pre, marker='o', label='san pre')
ax.plot(top_k, san_recall, marker='o', label='san recall')
ax.plot(top_k, sam_pre, marker='o', label='sam pre')
ax.plot(top_k, sam_recall, marker='o', label='sam recall')
ax.plot(top_k, aan_pre, marker='o', label='aan pre')
ax.plot(top_k, aan_recall, marker='o', label='aan recall')
ax.plot(top_k, aam_pre, marker='o', label='aam pre')
ax.plot(top_k, aam_recall, marker='o', label='aam recall')

# Set x-axis and y-axis labels
ax.set_xlabel('Top k')
ax.set_ylabel('Value')

# Add legend
ax.legend()

# Set title
ax.set_title('Example Table Visualization')

# Show plot
plt.show()





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Graph Parameters Set.')
    parser.add_argument('--gpu', metavar='N', type=int, default= 0,
                        help='an integer for the accumulator')
    parser.add_argument("--epochs", type=int, default=20,
                        help="number of training epochs")
    parser.add_argument('--dim', metavar='N', type=int, default=64,
                        help='an integer for the accumulator')
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=1.5e-3,
                        help="Weight for L2 loss")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="dropout probability")
    parser.add_argument("--p1", type=float, default=0.03,
                        help="abnormal percentage of nodes")
    parser.add_argument("--p2", type=float, default=0.01,
                        help="abnormal percentage of motifs")


    args = parser.parse_args()
    main(args)