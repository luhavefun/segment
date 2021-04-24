from denoise_net import Net
import torch
import argparse
from torch.utils.data import DataLoader
from data_loader import MyDataset
import matplotlib as plt
from torch.nn import MSELoss

parser = argparse.ArgumentParser(description='denoise ')
parser.add_argument('--cuda', action='store_true', help='Choose device to use cpu cuda:0')
parser.add_argument('--train_file', action='store', type=str,
                        default='train.txxt', help='Root path of image')
parser.add_argument('--test_file', action='store', type=str,
                        default='test.txt', help='Root path of image')
parser.add_argument('--batch_size', action='store', type=int,
                        default=8, help='number of data in a batch')
parser.add_argument('--lr', action='store', type=float,
                        default=0.0001, help='initial learning rate')
parser.add_argument('--epochs', action='store', type=int,
                        default=200, help='train rounds over training set')

def train(opts):
    # device = torch.device('cpu') if not torch.cuda.is_available or opts.cpu else torch.device('cuda')
    device = torch.device("cuda")
    print(device)
    # load dataset
    
    dataset_train = MyDataset('train.txt')
    dataset_test = MyDataset('test.txt')
    # define training and validation data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=opts.batch_size, shuffle=True, num_workers=1)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=opts.batch_size, shuffle=False, num_workers=1)


    model = Net()
    # model = nn.DataParallel(model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=opts.lr,betas=(0.9,0.99))
    # optimizer = torch.optim.Adamax(model.parameters(),lr=opts.lr,betas=(0.9,0.999),eps=1e-8,weight_decay=0.1)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                                step_size=50,
    #                                                gamma=0.1)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[50,100,150,200,250,300],gamma=0.1)
    weights = torch.FloatTensor([6,2,5,1]).to(device)
    loss_fct = MSELoss()   #
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    # writer = SummaryWriter(log_dir='')
    for epoch in range(opts.epochs):
        train_batch_num = 0
        train_loss = 0.0
        model.train()
        counts=0
        for seq, label in data_loader_train:
            seq = seq.to(device)
            label = label.to(device)
            seq = seq.unsqueeze(1)
            optimizer.zero_grad()
            pred = model(seq)
            loss = loss_fct(pred, label.view(-1))
            loss.backward()
            optimizer.step()
            train_batch_num += 1
            train_loss += loss.item()
            predict = pred.argmax(dim=1,keepdims=True)
            counts += predict.cpu().eq(label.cpu().view_as(predict)).sum().item()
        avg_acc = counts * 1.0 / len(data_loader_train.dataset)
        train_loss_list.append(train_loss / len(data_loader_train.dataset))
        train_acc_list.append(avg_acc)
        # writer.add_graph(model, seq)
        # write csv file
        train_loss_dataframe = pd.DataFrame(data=train_loss_list)
        train_acc_dataframe = pd.DataFrame(data=train_acc_list)
        train_loss_dataframe.to_csv('./output_results/train_loss.csv',index=False)
        train_acc_dataframe.to_csv('./output_results/train_accuracy.csv',index=False)

        model.eval()
        # for name,layer in model._modules.items():
        #     # view feature map
        #     seq_1 = seq.transpose(0,1)
        #     seq_grid = vutils.make_grid(seq_1,normalize=True,scale_each=True)
        #     writer.add_image(f'{name}_feature_maps',seq_grid,global_step=0)

        test_y = []
        test_y_pred = []
        counts=0
        test_loss = 0
        test_batch_num = 0
        outs = []
        labels = []
        with torch.no_grad():
            for test_seq, test_label in data_loader_test:
                test_seq = test_seq.to(device)
                test_label = test_label.to(device)
                test_seq = test_seq.unsqueeze(1)
                t_pred = model(test_seq)
                outs.append(t_pred.cpu())
                labels.append(test_label.cpu())
                # accuracy
                loss = loss_fct(t_pred, test_label.view(-1))
                test_loss += loss.item()
                test_batch_num += 1
                test_y += list(test_label.data.cpu().numpy().flatten())
                test_y_pred += list(t_pred.data.cpu().numpy().flatten())
                predict = t_pred.argmax(dim=1,keepdims=True)
                counts += predict.cpu().eq(test_label.cpu().view_as(predict)).sum().item()

        outs = torch.cat(outs,dim=0)
        labels = torch.cat(labels).reshape(-1)
        avg_acc = counts * 1.0 / len(data_loader_test.dataset)
        test_acc_list.append(avg_acc)
        test_loss_list.append(test_loss / len(data_loader_test.dataset))
        print('epoch: %d, train loss: %.4f, test loss: %.4f,test accuracy: %.4f' %
              (epoch, train_loss / train_batch_num, test_loss/ test_batch_num,avg_acc))
        # writer.add_scalar('scalar/train_loss', train_loss / train_batch_num, epoch)
        # writer.add_scalar('scalar/test_loss', test_loss / test_batch_num, epoch)
        # write csv file
        test_loss_dataframe = pd.DataFrame(data = test_loss_list)
        test_acc_dataframe = pd.DataFrame(data = test_acc_list)
        test_loss_dataframe.to_csv('./output_results/test_loss.csv',index=False)
        test_acc_dataframe.to_csv('./output_results/test_accuracy.csv',index=False)
    # writer.close()

    draw_test_info(test_loss_list, test_acc_list)
    draw_train_info(train_loss_list, train_acc_list)
    draw_roc_confusion(outs, labels)



if __name__ == "__main__":
    opts = parser.parse_args() # Namespace object
    train(opts)

