import torch
import time
import math
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR
#from model_blocks.model import SwinTransformer3D
import matplotlib.pyplot as plt
#from models_part.new_model import uniformer_base,uniformer_small
from LMTformer_model.LMTformer import LMTformer
plt.switch_backend('agg')
import logging

# save model_blocks
def save_model(model):
    torch.save(model.state_dict(),
               f'model_dict_vst.pt'
               )
    # torch.save(model, f'Bestmodle')


def initiate(train_loader, test_loader,device, args):
    # random seed
    torch.manual_seed(1234)
    

    model = LMTformer(image_size=(24,128,128),patches=(2,4,4), dim=32, num_heads=4, num_layers=12, dropout_rate=0.1,device=device).to(device)
    #model.apply(model._init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)  # optimizer-Adam
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)  # scheduler-MutiStepLR
    logging.info('optimizer : '+ str(optimizer))
    logging.info('scheduler : '+ str(scheduler))
    criterion_MSE = torch.nn.MSELoss()
    criterion_MAE = torch.nn.L1Loss()

    """
    #############################################################
    main function
    #############################################################
    """
    best_valid = 1e8
    Final_best_Trainloss = 0
    best_MAE = 1e8

    epoch_data = []
    RMSE_data = []
    MAE_data = []

    for epoch in range(1, args.Epochs + 1):

        start = time.time()
        # ------------------train----------------------
        train_loss = train(train_loader, model, optimizer, criterion_MSE, scheduler, device, epoch, args)
        # ------------------test---------------------
        test_MSE, test_MAE = evaluate(test_loader, model, criterion_MSE, criterion_MAE, device)
        test_RMSE = math.sqrt(test_MSE)
        end = time.time()
        duration = end - start
        print(
                'Epoch {:3d} | Time {:5.2f} sec |Train Loss {:5.2f} | Test RMSE {:5.2f} | Test MAE {:5.2f}'.
                format(epoch, duration, train_loss, test_RMSE, test_MAE))
        logging.info('Epoch {:3d} | Time {:5.2f} sec |Train Loss {:5.2f} | Test RMSE {:5.2f} | Test MAE {:5.2f}'.
                format(epoch, duration, train_loss, test_RMSE, test_MAE))
        # -------------------更新模型------------------------------
        if test_RMSE < best_valid:
            # print(f"*****Saved New model_blocks base RMSE *****")
            save_model(model)
            best_valid = test_RMSE
            best_MAE = test_MAE
            Final_best_Trainloss = train_loss

        else:
            pass
        epoch_data.append(epoch)
        RMSE_data.append(test_RMSE)
        MAE_data.append(test_MAE)
    # plot RMSE graph
    plt.plot(epoch_data, RMSE_data, linewidth=1, label='RMSE')
    plt.plot(epoch_data, MAE_data, linewidth=1, label='MAE')
    plt.legend()
    plt.savefig(args.imgload)
    #plt.show()
    logging.info("-" * 87)
    logging.info('Best get : Train  {:5.4f} | Test RMSE   {:5.4f}| Test MAE {:5.4f}'.
        format(Final_best_Trainloss, best_valid, best_MAE))
    print("-" * 87)
    print(
        'Best get : Train  {:5.4f} | Test RMSE   {:5.4f}| Test MAE {:5.4f}'.
        format(Final_best_Trainloss, best_valid, best_MAE))



    # --------------------------------  train  ------------------------------
    # 1.train model
def train(train_loader, model, optimizer, criterion, scheduler, device, epoch, args):
    epoch_loss = 0  #  the loss of each epoch
    model.train()
    proc_loss, proc_size = 0, 0

    # loop = tqdm(enumerate(train_loader), total=len(train_loader))
    for i_batch, (data, target) in enumerate(train_loader):  # in loop:
        data, target = data.to(device), target.to(device)  # target.shape([batch_size])
        optimizer.zero_grad()  # grad initialization
        batch_size = data.size(0)

        output = model(data)
        output = torch.squeeze(output)  # output.shape([batch_size],[1])-->([batch_size])
        loss = criterion(output, target.float())
        loss = torch.sqrt(loss)  # RMSE
        combined_loss = loss
        combined_loss.backward()
        optimizer.step()
        scheduler.step()
        # loop.set_description(f'Train: Epoch [{epoch}/{args.Epochs}]')


        # proc_loss += loss.item() * batch_size
        # proc_size += batch_size
        epoch_loss += combined_loss.item() * batch_size

    epoch_loss = epoch_loss / len(train_loader.dataset)
    return epoch_loss  # one epoch loss


# 2.test
def evaluate(test_loader, model, criterion_MSE, criterion_MAE, device):
    model.eval()
    loader = test_loader
    MSE_loss = 0.0
    MAE_loss = 0.0

    with torch.no_grad():
        # dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for i_batch, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)

            preds = model(data)
            preds = torch.squeeze(preds)
            MSE = criterion_MSE(preds, target)
            MAE = criterion_MAE(preds, target)
            # loop.set_description(f'Evaluate: Epoch [{epoch}/{Epochs}]')
            MSE_loss += MSE.item() * batch_size
            MAE_loss += MAE.item() * batch_size

    avg_MSE_loss = MSE_loss / (len(test_loader.dataset))
    avg_MAE_loss = MAE_loss / (len(test_loader.dataset))

    return avg_MSE_loss, avg_MAE_loss
