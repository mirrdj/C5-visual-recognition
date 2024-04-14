import torch


class Validator:
    def validation(self, model, dataloader_val, criterion, params, device):
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in dataloader_val:
                imgs, labels = imgs.to(device), labels.to(device)

                outputs =  model(imgs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(dataloader_val)
        val_accuracy = correct / total
        return val_loss, val_accuracy


class SiameseValidator(Validator):
    def validation(self, model, dataloader_val, criterion, params, device):
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs1, imgs2, targets in dataloader_val:
                imgs1, imgs2, targets = imgs1.to(device), imgs2.to(device), targets.to(device)

                outputs1, outputs2 = model(imgs1, imgs2)
                loss = criterion(outputs1, outputs2, targets)
                
                val_loss += loss.item()
                _, predicted1 = torch.max(outputs1, 1)
                _, predicted2 = torch.max(outputs2, 1)
                
                total += targets.size(0)
                correct += ((predicted1 == predicted2) == targets).sum().item()

        val_loss /= len(dataloader_val)
        val_accuracy = correct / total
        return val_loss, val_accuracy
    

class TripletValidator(Validator):
    def validation(self, model, dataloader_val, criterion, params, device):
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for anchor, pos, neg in dataloader_val:
                anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)

                outputs1, outputs2, outputs3 = model(anchor, pos, neg)
                loss = criterion(outputs1, outputs2, outputs3)
                
                val_loss += loss.item()
                # _, predicted1 = torch.max(outputs1, 1)
                # _, predicted2 = torch.max(outputs2, 1)
                # _, predicted3 = torch.max(outputs3, 1)

                total += neg.size(0)
                # TODO: Implement correct accuracy according to
                # https://stackoverflow.com/a/47625727
                # correct += ((predicted1 == predicted2) == neg).sum().item()

        val_loss /= len(dataloader_val)
        # val_accuracy = correct / total
        return val_loss, 0
