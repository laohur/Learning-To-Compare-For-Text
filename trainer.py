import torch
from torch import nn
from time import time
from torch.autograd import Variable
import numpy as np

from Util import few_data, device
from task_generator import ClassifyTask, get_data_loader


def train(feature_encoder, relation_network, train_data, config):
    feature_encoder.train()
    relation_network.train()
    task = ClassifyTask(train_data, config["CLASS_NUM"], config["SAMPLE_NUM_PER_CLASS"], config["BATCH_NUM_PER_CLASS"])
    sample_dataloader = get_data_loader(task, config, num_per_class=config["SAMPLE_NUM_PER_CLASS"], split="train",
                                        shuffle=False)
    batch_dataloader = get_data_loader(task, config, num_per_class=config["BATCH_NUM_PER_CLASS"], split="test",
                                       shuffle=True)

    # sample datas
    samples, sample_labels = sample_dataloader.__iter__().next()
    batches, batch_labels = batch_dataloader.__iter__().next()

    # calculate features
    sample_features0 = feature_encoder(Variable(samples)).to(device)  # 25*128
    sample_features1 = sample_features0.view(config["CLASS_NUM"], config["SAMPLE_NUM_PER_CLASS"], -1)
    sample_features = torch.sum(sample_features1, 1).squeeze(1)
    batch_features = feature_encoder(Variable(batches)).to(device)  # 75*300

    # calculate relations
    sample_features_ext = sample_features.unsqueeze(0).repeat(config["BATCH_NUM_PER_CLASS"] * config["CLASS_NUM"], 1, 1)
    batch_features_ext = batch_features.unsqueeze(0).repeat(config["CLASS_NUM"], 1, 1)
    batch_features_ext = torch.transpose(batch_features_ext, 0, 1)

    relations = relation_network(sample_features_ext, batch_features_ext).view(-1, config["CLASS_NUM"])

    mse = nn.MSELoss().to(device)
    batch_labels = batch_labels.long()
    one_hot_labels = torch.zeros(config["BATCH_NUM_PER_CLASS"] * config["CLASS_NUM"], config["CLASS_NUM"])
    one_hot_labels = one_hot_labels.scatter_(1, batch_labels.view(-1, 1), 1)
    one_hot_labels = Variable(one_hot_labels).to(device)

    loss = mse(relations, one_hot_labels)
    feature_encoder.zero_grad()
    relation_network.zero_grad()
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(feature_encoder.parameters(), 0.5)
    # torch.nn.utils.clip_grad_norm_(relation_network.parameters(), 0.5)

    return loss.item()


def valid(feature_encoder, relation_network, test_data, config):
    # test
    feature_encoder.eval()
    relation_network.eval()

    total_rewards = 0

    for i in range(config["TEST_EPISODE"]):  # 训练测试 集合数量不同
        task = ClassifyTask(test_data, config["CLASS_NUM"], config["SAMPLE_NUM_PER_CLASS"],
                            config["BATCH_NUM_PER_CLASS"])
        sample_dataloader = get_data_loader(task, config, num_per_class=config["SAMPLE_NUM_PER_CLASS"], split="train", shuffle=False)
        test_dataloader = get_data_loader(task, config, num_per_class=config["SAMPLE_NUM_PER_CLASS"], split="test", shuffle=True)

        sample_images, sample_labels = sample_dataloader.__iter__().next()
        test_images, test_labels = test_dataloader.__iter__().next()

        # calculate features
        sample_features = feature_encoder(Variable(sample_images).to(device))  # 5x28->   #5*50
        sample_features = sample_features.view(config["CLASS_NUM"], config["SAMPLE_NUM_PER_CLASS"], -1)
        sample_features = torch.sum(sample_features, 1).squeeze(1)
        test_features = feature_encoder(Variable(test_images).to(device))  # 20x64

        sample_features_ext = sample_features.unsqueeze(0).repeat(config["SAMPLE_NUM_PER_CLASS"] * config["CLASS_NUM"],
                                                                  1, 1)
        test_features_ext = test_features.unsqueeze(0).repeat(config["CLASS_NUM"], 1, 1)
        test_features_ext = torch.transpose(test_features_ext, 0, 1)

        relations = relation_network(sample_features_ext, test_features_ext)  # 25
        relations = relations.view(-1, config["CLASS_NUM"])  # 5*5

        _, predict_labels = torch.max(relations.data, 1)
        predict_labels = predict_labels.cpu()
        rewards = [1 if int(predict_labels[j]) == int(test_labels[j]) else 0 for j in
                   range(config["CLASS_NUM"] * config["SAMPLE_NUM_PER_CLASS"])]

        total_rewards += np.sum(rewards)

    test_accuracy = total_rewards / 1.0 / config["CLASS_NUM"] / config["SAMPLE_NUM_PER_CLASS"] / config["TEST_EPISODE"]
    print("test accuracy:", test_accuracy)

    return test_accuracy


if __name__ == "__main__":
    t0 = time()
    print("耗时", time() - t0)
