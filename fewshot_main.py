import torch
from models import RelationNetwork
from time import time
from torch.optim.lr_scheduler import StepLR
from StructuredSelfAttention import StructuredSelfAttention
from task_generator import omniglot_character_folders
from trainer import train, valid

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
    train_data, test_data, word2index, labels=omniglot_character_folders()
    config = {
        "CLASS_NUM":5,
        "SAMPLE_NUM_PER_CLASS":5,
        "BATCH_NUM_PER_CLASS":15,
        "EPISODE":1000000, #1000000
        "TEST_EPISODE":1000, #1000
        "LEARNING_RATE":0.0001, #0.01
        "FEATURE_DIM":300,
        "RELATION_DIM":8,
        "max_len":12,
        "emb_dim": 300,
        "lstm_hid_dim": 64,
        "d_a": 64,
        "r": 16,
        "max_len": 10,
        "n_classes": 5,
        "num_layers": 1,
        "dropout": 0.1,
        "type": 1,
        "use_pretrained_embeddings": True,
        "word2index":word2index,
        "vocab_size": len(word2index)
    }
    feature_encoder = StructuredSelfAttention(config).to(device)
    relation_network = RelationNetwork(config["FEATURE_DIM"], config["RELATION_DIM"]).to(device)

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=config["LEARNING_RATE"])
    feature_encoder_scheduler = StepLR(feature_encoder_optim, step_size=100000, gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=config["LEARNING_RATE"])
    relation_network_scheduler = StepLR(relation_network_optim, step_size=100000, gamma=0.5)

    print("开始训练")
    t0 = time()
    last_accuracy = 0.0

    for episode in range(config["EPISODE"]):
        feature_encoder.train()
        relation_network.train()
        feature_encoder_scheduler.step(episode)
        relation_network_scheduler.step(episode)

        loss = train(feature_encoder, relation_network, train_data, config)

        feature_encoder_optim.step()
        relation_network_optim.step()

        if (episode + 1) % 100 == 0:
            print("episode:", episode + 1, "loss", loss, "耗时", time() - t0)
            t0 = time()

        if (episode + 1) % (5*config["TEST_EPISODE"]) == 0:
            test_accuracy = valid(feature_encoder, relation_network, test_data, config)
            t0 = time()
    print("直接词向量")
    print("完成")


if __name__ == "__main__":
    t0 = time()
    main()
    print("耗时", time() - t0)
