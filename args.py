import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--modeldir', type=str, default='models')
    parser.add_argument('--datadir', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='your_dataset')
    parser.add_argument('--log_file_name', type=str, default=None)
    parser.add_argument('--init_seed', type=int, default=42)
    parser.add_argument('--download_data', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_parties', type=int, default=1)
    parser.add_argument('--sample_fraction', type=float, default=1.0)
    parser.add_argument('--comm_round', type=int, default=1)
    parser.add_argument('--test_imb', type=int, default=0)
    parser.add_argument('--load_model_file', type=str, default=None)
    parser.add_argument('--load_model_round', type=int, default=0)
    parser.add_argument('--server_momentum', type=int, default=0)
    parser.add_argument('--alg', type=str, default='fedavg')

    args = parser.parse_args()
    return args
