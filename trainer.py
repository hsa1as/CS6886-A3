import torch
import torch.optim as optim
import torch.nn as nn
import wandb

class Trainer:
    def __init__(self, model, loaders, device="cuda"):
        self.device = device
        self.model = model.to(device).to(memory_format=torch.channels_last)
        self.model = torch.compile(self.model)   # accelerate training/eval
        self.trainloader, self.testloader = loaders
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1,
                                   momentum=0.9, weight_decay=4e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        self.scaler = torch.amp.GradScaler('cuda')

    def train_one_epoch(self, epoch):
        self.model.train()
        total, correct, loss_sum = 0, 0, 0
        for x,y in self.trainloader:
            x,y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                out = self.model(x)
                loss = self.criterion(out, y)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            loss_sum += loss.item()*y.size(0)
            _,pred = out.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
        acc = 100.*correct/total
        print(f"Epoch {epoch}: Train Loss {loss_sum/total:.4f} Acc {acc:.2f}%")
        wandb.log({
            "epoch": epoch,
            "train/loss": loss_sum/total,
            "train/acc": acc,
            "lr": self.scheduler.get_last_lr()[0]
        })

    def test(self, epoch):
        self.model.eval()
        total, correct, loss_sum = 0, 0, 0
        with torch.no_grad():
            for x,y in self.testloader:
                x,y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    out = self.model(x)
                    loss = self.criterion(out, y)
                loss_sum += loss.item()*y.size(0)
                _,pred = out.max(1)
                total += y.size(0)
                correct += pred.eq(y).sum().item()
        acc = 100.*correct/total
        print(f"Epoch {epoch}: Test Loss {loss_sum/total:.4f} Acc {acc:.2f}%")
        wandb.log({
            "epoch": epoch,
            "val/loss": loss_sum/total,
            "val/acc": acc
        })
        return acc

    def fit(self, epochs=20, save_path="mobilenetv2_cifar10.pth"):
        best_acc = 0
        for epoch in range(1, epochs+1):
            self.train_one_epoch(epoch)
            acc = self.test(epoch)
            self.scheduler.step()
            if acc > best_acc:
                best_acc = acc
                torch.save(self.model.state_dict(), save_path)
                print(f"  Saved model at epoch {epoch} with acc {acc:.2f}%")
        wandb.log({"best/acc": best_acc})
        return best_acc