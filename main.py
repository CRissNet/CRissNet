import torch
import torch.nn as nn
import os
from utils.parser import args
from utils import logger, Trainer, Tester
from utils import init_device, init_model, FakeLR, WarmUpCosineAnnealingLR, painting
from dataset import Cost2100DataLoader


def main():
    logger.info('=> PyTorch Version: {}'.format(torch.__version__))
    model_path = '/home/psdz/Downloads/WBH/WBH3'  # /home/t1/Downloads/test-wbh/cqnetplus
    path = os.path.join(model_path, f"model{args.condition}/{args.condition}.pth")
    # Environment initialization
    path1 = os.path.join(model_path, f"model{args.condition}/{args.condition}.png")
    path2 = os.path.join(model_path, f"model{args.condition}/train{args.condition}.txt")
    path3 = os.path.join(model_path, f"model{args.condition}/test{args.condition}.txt")
    device, pin_memory = init_device(args.seed, args.cpu, args.gpu, args.cpu_affinity)

    # Create the data loader
    train_loader, val_loader, test_loader, inputimage = Cost2100DataLoader(
        root=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        scenario=args.scenario)()

    # Define model
    model = init_model(args)
    model.to(device)

    # Define loss function
    criterion = nn.MSELoss().to(device)

    # Inference mode
    if args.evaluate:
        Tester(model, device, criterion)(test_loader)
        return

    # Define optimizer and scheduler
    lr_init = 1e-3 if args.scheduler == 'const' else 3e-3
    optimizer = torch.optim.Adam(model.parameters(), lr_init)
    if args.scheduler == 'const':
        scheduler = FakeLR(optimizer=optimizer)
    else:
        scheduler = WarmUpCosineAnnealingLR(optimizer=optimizer,
                                            T_max=1500 * len(train_loader),
                                            T_warmup=30 * len(train_loader),
                                            eta_min=5e-5)

    # Define the training pipeline
    trainer = Trainer(model=model,
                      device=device,
                      optimizer=optimizer,
                      criterion=criterion,
                      scheduler=scheduler)

    # Start training
    train_loss, test_loss, best_nmse = trainer.loop(args.epochs, train_loader, val_loader, test_loader, path)
    model.load_state_dict(torch.load(path))
    # Final testing
    best_nmse = 0
    loss, nmse, best_nmse = Tester(model, device, criterion)(test_loader, path, best_nmse)
    print(f"\n=! Final test loss: {loss:.3e}"
          f"\n         test NMSE: {nmse:.3e}\n"
          f"\n         compress: {args.n1:.3e}\n")

    a = painting(epoch=args.epochs, listA=train_loss, listB=test_loss, model=model, path=path1)
    a.paint()
    fileObject1 = open(path2, 'w')
    for ip in train_loss:
        fileObject1.write(str(ip))
        fileObject1.write('\n')
    fileObject1.close()
    fileObject2 = open(path3, 'w')
    for ip in test_loss:
        fileObject2.write(str(ip))
        fileObject2.write('\n')
    fileObject2.close()


if __name__ == "__main__":
    main()