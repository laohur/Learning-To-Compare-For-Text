import os
import torch
from time import time
from Comparator import RelationNetwork
from torch.optim.lr_scheduler import StepLR
from Encoder import StructuredSelfAttention
from task_generator import get_data
from trainer import train, valid

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main():
    train_data, test_data, word2index, labels = get_data()
    config = {
        "CLASS_NUM": 5,
        "SAMPLE_NUM_PER_CLASS": 5,
        "BATCH_NUM_PER_CLASS": 15,
        "EPISODE": 10000,  # 1000000
        "TEST_EPISODE": 100,  # 1000
        "LEARNING_RATE": 0.0001,  # 0.01
        "FEATURE_DIM": 256,  # lstm_hid_dim *2
        "RELATION_DIM": 8,
        "use_bert": False,
        "max_len": 30,
        "emb_dim": 300,
        "lstm_hid_dim": 128,
        "d_a": 64,
        "r": 1,
        "n_classes": 5,
        "num_layers": 1,
        "dropout": 0.1,
        "type": 1,
        "use_pretrained_embeddings": True,
        "word2index": word2index,
        "vocab_size": len(word2index)
    }
    feature_encoder = StructuredSelfAttention(config).to(device)
    relation_network = RelationNetwork(2 * config["FEATURE_DIM"], config["RELATION_DIM"]).to(device)

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=config["LEARNING_RATE"], weight_decay=1e-4)
    feature_encoder_scheduler = StepLR(feature_encoder_optim, step_size=100000, gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=config["LEARNING_RATE"], weight_decay=1e-4)
    relation_network_scheduler = StepLR(relation_network_optim, step_size=100000, gamma=0.5)
    print("开始训练")
    best_accuracy=0
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
        t0=time()
        if (episode + 1) % (5 * config["TEST_EPISODE"]) == 0:
            test_accuracy = valid(feature_encoder, relation_network, test_data, config)
            if test_accuracy>best_accuracy:
                best_accuracy=test_accuracy
                if not os.path.exists('save/'):
                    os.mkdir('save/')
                torch.save(feature_encoder.state_dict(),"save/encoder.pkl")
                torch.save(relation_network.state_dict(),"save/relation.pkl")

    return best_accuracy

if __name__ == "__main__":
    t0 = time()
    accuracy=  main()
    print(f"精度{accuracy},耗时{time() - t0}")
