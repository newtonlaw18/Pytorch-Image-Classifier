import argparse
import data_directory
import neural_network_model
import model_checkpoint

parser = argparse.ArgumentParser()

parser.add_argument('--data_directory', help = 'specify data directory folder', default='./flowers')
parser.add_argument('--arch', type = str, help = 'set CNN architecture', default='vgg16')
parser.add_argument('--save_dir', help = 'set model checkpoint name', default='checkpoint.pth')
parser.add_argument('--learning_rate', type=float, help = 'set model learning rate', default=0.001)
parser.add_argument('--hidden_input', type=int, help = 'set hidden units', default=600)
parser.add_argument('--epochs', type=int, help = 'set model epoch count', default=5)
parser.add_argument('--gpu', help = 'set gpu to use gpu computation', default="gpu")

args = parser.parse_args()
print(args)

data_dir = args.data_directory
nn_architecture = args.arch
hidden_units = args.hidden_input
nn_learning_rate = args.learning_rate
epoch_count = args.epochs

# get datasets
train_datasets, train_loader, test_loader, valid_loader = data_directory.get_datasets(data_dir)

# start to create nn model first
model, optimizer, criterion = neural_network_model.create_nn_model(
    nn_architecture, hidden_units, nn_learning_rate, args.gpu)

# initiate nn model training
neural_network_model.train_nn_model(model, optimizer, criterion, train_loader, 
                        valid_loader, epoch_count, args.gpu)

# initiate nn model save checkpoint function
model_checkpoint.save_nn_model_checkpoint(model, args, optimizer, train_datasets)