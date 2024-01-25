import argparse

def train_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs of training')
    parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default="./dataset/train", help='root directory of the dataset')
    parser.add_argument('--whole_dir', type=str, default="./dataset/evaluation_6mm/parts", help='root directory of 6mm dataset')
    parser.add_argument('--pretrained_root', type=str, default="./pre_trained/netG_A2B_pretrained.pth", help='root directory of the pre-trained model')
    parser.add_argument('--pretrained', type=bool, default=False, help='whether use pre-trained model')
    parser.add_argument('--p_val', type=float, default=0.3, help='proportion of validation samples')



    parser.add_argument('--log_dir_train', type=str, default='./logs/train/', help='log path')
    parser.add_argument('--log_dir_test', type=str, default='./logs/test/', help='log path')
    parser.add_argument('--tensorboard_dir', type=str, default='./logs/tensorboard/', help='log path')
    parser.add_argument('--tensorboard', action='store_true', help='use tensorboard')
    parser.add_argument('--log', type=str, default='0', help='log file index')
    parser.add_argument('--check_dir', type=str, default='./checkpoint', help='checkpoint path')


    parser.add_argument('--B2A', type=bool, default=False, help='save netB2A')
    # parser.add_argument('--scheduler', action='store_false', help='save netB2A')
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=30, help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--sizeA', type=int, default=256, help='size of the data crop (squared assumed)')
    parser.add_argument('--sizeB', type=int, default=256, help='size of the data crop (squared assumed)')
    parser.add_argument('--n_layers', type=int, default=5, help='layer of the discriminators')
    parser.add_argument('--dis_size', type=int, default=6, help='output size of discriminators')
    parser.add_argument('--n_downsampling', type=int, default=2, help='num of downsample')

    parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
    parser.add_argument('--input_D_nc', type=int, default=1, help='number of channels of patchGAN')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=2, help='number of cpu threads to use during batch generation')
    parser.add_argument('--dis_iter', type=int, default=1, help='number of iterations for training discriminators')
    parser.add_argument('--hfb', type=float, default=0.8, help='hfb weights')

    parser.add_argument('--seed', type=int, default=3074, help='Seed')
    parser.add_argument('--beta1', type=float, default=10.0, help='beta 1 ABA')
    parser.add_argument('--beta2', type=float, default=10.0, help='beta 2 BAB')
    parser.add_argument('--beta3', type=float, default=5.0, help='beta 3 idtA')
    parser.add_argument('--beta4', type=float, default=5.0, help='beta 4 idtB')
    parser.add_argument('--beta5', type=float, default=1.0, help='beta 5 ganhr')
    parser.add_argument('--beta6', type=float, default=1.0, help='beta 6 ganlr')
    parser.add_argument('--beta7', type=float, default=1.0, help='beta 6 ffl')

    parser.add_argument('--l2h_hb', type=int, default=11, help='HF bandwidth for low2high resolution')
    parser.add_argument('--l2h_lb', type=int, default=9, help='LF bandwidth for low2high resolution')
    parser.add_argument('--h2l_hb', type=int, default=5, help='HF bandwidth for high2low resolution')
    parser.add_argument('--h2l_lb', type=int, default=13, help='LF bandwidth for high2low resolution')
    parser.add_argument('--loss_weight', type=float, default=10.0, help='loss_weight')
    parser.add_argument('--loss_alpha', type=float, default=1.0, help='loss_weight')

    parser.add_argument('--ffl_h', action='store_true', help='visualize high ffl')
    parser.add_argument('--ffl_l', action='store_true', help='visualize low ffl')

    opt = parser.parse_args()
    print(opt)

    return opt

