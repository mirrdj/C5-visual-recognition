from utils import get_indices
from tqdm import tqdm

class TripletTrainer():
    def train(self, model, dataloader_train, criterion, optimizer, params, device, miner=False):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        if params['text_encoder'] == 'FASTTEXT':
            vocab = model.get_vocab()

        for anchor, pos, neg in tqdm(dataloader_train):
            
            if params['task'] == 'B':
                if params['text_encoder'] == 'FASTTEXT':
                    tensor_list = []
                    for caption in anchor:
                        tensor_list.append(get_indices(vocab, caption))

                    anchor, pos, neg = [tensor.to(device) for tensor in tensor_list], pos.to(device), neg.to(device)
                
                elif params['text_encoder'] == 'BERT':
                    pos, neg = pos.to(device), neg.to(device)
            
            elif params['task'] == 'A':
                
                if params['text_encoder'] == 'FASTTEXT':
                    tensor_list_pos = []
                    tensor_list_neg = []
                    
                    for pos_caption in pos:
                        tensor_list_pos.append(get_indices(vocab, pos_caption))
                    
                    for neg_caption in neg:
                        tensor_list_neg.append(get_indices(vocab, neg_caption))

                    anchor, pos, neg = anchor.to(device), [tensor.to(device) for tensor in tensor_list_pos], [tensor.to(device) for tensor in tensor_list_neg]

                elif params['text_encoder'] == 'BERT':
                    anchor = anchor.to(device)


            # zero the parameter gradients
            optimizer.zero_grad()

            outputs1, outputs2, outputs3 = model(anchor, pos, neg)

            loss = criterion(outputs1, outputs2, outputs3)

            loss.backward()

            optimizer.step()

            print(f"Loss is {loss}")

            # Compute training accuracy and loss
            train_loss += loss.item() * outputs1.size(0)
            # train_loss += loss.item() * anchor.size(0)
            total += outputs3.size(0)

            # TODO: Implement correct accuracy according to
            # https://stackoverflow.com/a/47625727

        train_loss /= len(dataloader_train.dataset)
        train_accuracy = correct / total
        return train_loss, 0
