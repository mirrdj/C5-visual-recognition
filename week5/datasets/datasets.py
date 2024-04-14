from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import time
# from utils import read_txt


# Implement 3 datasets:
# - returns pairs from the same class but with negative pairs being different colors
# - returns pairs from different classes but different colors
# - returns pairs from different classes and same color + same class different colors

# - returns pairs from the same class but with negative pairs being different colors
class TripletCOCO_B(Dataset):
    def __init__(self, data_dir=None, gen_dict=None, transform=None, mode='train'):
        #Key is the name of file and value is the caption
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.gen_dict = gen_dict

        self.images = [] # Path to images
        self.labels = [] # Label according to the strategy used in fine-tuning
        self.captions = {}
        # self.classes = set(['bus', 'cat', 'cupcake', 'dog', 'house', 't-shirt', 'table'])
        # self.colors = set(['black', 'red', 'blue', 'white', 'purple', 'orange', 'yellow'])

        self.classes_idx = {}
        self.colors_idx = {}

        if mode == 'train':
            for i, (img_name, caption) in enumerate(gen_dict.items()):
                
                # Get class, color and id from name
                cls, color, id = img_name.rstrip(".png").split("_")

                img_path = os.path.join(data_dir, cls, color, id+'.png')

                if cls not in self.classes_idx:
                    self.classes_idx[cls] = {}
                if color not in self.classes_idx[cls]:
                    self.classes_idx[cls][color] = []
                
                if color not in self.colors_idx:
                     self.colors_idx[color] = {}
                if cls not in self.colors_idx[color]:
                    self.colors_idx[color][cls] = []
                
                self.classes_idx[cls][color].append(i)
                self.colors_idx[color][cls].append(i)
                
                self.images.append(img_path)

                self.labels.append((cls, color))
                self.captions[i] = [caption.strip(), cls, color, img_path]

                

        # self.imgs_set = set(self.image_idxs.keys()) #Get all the image IDs

        # # Create fixed triplets for testing
        # elif mode == 'test':
        #     random_state = np.random.RandomState(29)

        #     triplets = [[i,
        #                  random_state.choice(self.label_to_indices[self.labels[i].item()]),
        #                  random_state.choice(self.label_to_indices[
        #                                          np.random.choice(
        #                                              list(self.labels_set - set([self.labels[i].item()]))
        #                                          )
        #                                      ])
        #                  ]
        #                 for i in range(len(self.images))]

            # self.triplests = triplets

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        if self.mode == 'train':

            anchor_caption = self.captions[idx][0] #Get caption for that idx

            positive_img = self.captions[idx][3] #Get the image for that caption
            
            #APPROACH 1:
            #Get any image of any other color but same class
            negative_color = np.random.choice(list(set(self.classes_idx[self.captions[idx][1]]) - set([self.captions[idx][2]])))
            class_data = self.classes_idx.get(self.captions[idx][1]) #Get subdictionary with the colors for that class
            negative_idx = np.random.choice(class_data[negative_color])
            negative_img = self.captions[negative_idx][3]  #Get the negative image path

            # #APPROACH 2:
            # negative_class = np.random.choice(list(self.classes - set([self.captions[idx][1]])))
            # class_data = self.colors_idx.get(self.captions[idx][2]) #Get subdictionary with the classes for that color
            # negative_idx = np.random.choice(class_data[negative_class])
            # negative_img = self.captions[negative_idx][3]  #Get the negative image path

            # #Approach 3:
            # # print(f'Class: {self.captions[idx][1]} | {set(self.colors_idx[self.captions[idx][2]])}')
            # # print(f'Color: {self.captions[idx][2]} | {set(self.classes_idx[self.captions[idx][1]])}')
            # negative_class = np.random.choice(list(set(self.colors_idx[self.captions[idx][2]]) - set([self.captions[idx][1]])))
            # negative_color = np.random.choice(list(set(self.classes_idx[self.captions[idx][1]]) - set([self.captions[idx][2]])))
            # negative_idx = np.random.choice(self.classes_idx[negative_class][negative_color])
            # negative_img = self.captions[negative_idx][3]  #Get the negative image path


        else:  # Retrieve from the predefined triplets
            img1 = self.images[self.triplests[idx][0]]
            img2 = self.images[self.triplests[idx][1]]
            img3 = self.images[self.triplests[idx][2]]

        #Load images
        positive_img = Image.open(os.path.join(self.data_dir, positive_img)).convert("RGB")
        negative_img = Image.open(os.path.join(self.data_dir, negative_img)).convert("RGB")

        #Apply transforms
        if self.transform:
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        # Return anchor, pos, neg
        return anchor_caption, positive_img, negative_img
    


class TripletCOCO_B_full(Dataset):
    def __init__(self, captions, data_dir_COCO=None, data_dir_GEN = None, gen_dict=None, transform=None, mode='train', num_cap=10):
        #Key is the name of file and value is the caption
        self.data_dir_COCO = data_dir_COCO
        self.data_dir_GEN = data_dir_GEN
        self.transform = transform
        self.mode = mode
        self.gen_dict = gen_dict

        self.images = [] # Path to images
        self.labels = [] # Label according to the strategy used in fine-tuning
        self.captions = {}
        self.image_idxs = {}

        self.image_id_name = {}

        image_annotations = captions['annotations']
        images_name_id = captions['images']
        index=0
        # self.classes = set(['bus', 'cat', 'cupcake', 'dog', 'house', 't-shirt', 'table'])
        # self.colors = set(['black', 'red', 'blue', 'white', 'purple', 'orange', 'yellow'])

        self.classes_idx = {}
        self.colors_idx = {}

        if mode == 'train':
            
            #Create a dictionary where the keys are the image_ids and the value is the file_name for that image_id
            for im in images_name_id:
                self.image_id_name[im['id']] = os.path.join(self.data_dir_COCO, im['file_name'])

            #Iterate through all the annotations
            for an in image_annotations:
                
                if an['image_id'] not in self.image_idxs:    
                    self.image_idxs[an['image_id']] = []

                if len(self.image_idxs[an['image_id']]) < num_cap: #This controls the maximum number of captions that we want
                    #Save, for every annotation, its image_id, its object ID and its caption (Each image has 5 captions, but each one is saved in 5 diff. indexes)
                    self.captions[index] = [an['image_id'], an['id'], an['caption'], '', '', '']
                    
                    #Store which indeces correspond to each image
                    self.image_idxs[an['image_id']].append(index)

                    #Increment the index value
                    index+=1

            max_id = 581921
            for i, (img_name, caption) in enumerate(gen_dict.items()):
                
                max_id += 1

                if max_id not in self.image_idxs:    
                    self.image_idxs[max_id] = []

                # Get class, color and id from name
                cls, color, id = img_name.rstrip(".png").split("_")

                img_path = os.path.join(data_dir_GEN, cls, color, id+'.png')
                
                self.captions[index] = [max_id, 1, caption.strip(), cls, color, img_path]
                self.image_id_name[max_id] = img_path
                #Store which indeces correspond to each image
                self.image_idxs[max_id].append(index)
                
                if cls not in self.classes_idx:
                    self.classes_idx[cls] = {}
                if color not in self.classes_idx[cls]:
                    self.classes_idx[cls][color] = []
                
                if color not in self.colors_idx:
                     self.colors_idx[color] = {}
                if cls not in self.colors_idx[color]:
                    self.colors_idx[color][cls] = []
                
                self.classes_idx[cls][color].append(index)
                self.colors_idx[color][cls].append(index)
                
                # self.images.append(img_path)

                # self.labels.append((cls, color))
                # self.captions[i] = [caption.strip(), cls, color, img_path]
                
                index+=1
                

        self.imgs_set = set(self.image_idxs.keys()) #Get all the image IDs

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        if self.mode == 'train':

            anchor_caption = self.captions[idx][2] #Get caption for that idx

            if len(self.captions[idx][5])==0:
                positive_img = self.image_id_name[self.captions[idx][0]] #Get image path
                #Get any other image 
                negative_label = np.random.choice(list(self.imgs_set - set([self.captions[idx][0]])))
                negative_idx = np.random.choice(self.image_idxs[negative_label])
                negative_img = self.image_id_name[self.captions[negative_idx][0]] #Get path for negative image
            
            else:
                positive_img = self.captions[idx][5] #Get the image for that caption

                # APPROACH 0:
                #Get any other image 
                negative_label = np.random.choice(list(self.imgs_set - set([self.captions[idx][0]])))
                negative_idx = np.random.choice(self.image_idxs[negative_label])
                negative_img = self.image_id_name[self.captions[negative_idx][0]] #Get path for negative image

                # # APPROACH 1:
                # #Get any image of any other color but same class
                # negative_color = np.random.choice(list(set(self.classes_idx[self.captions[idx][3]]) - set([self.captions[idx][4]])))
                # class_data = self.classes_idx.get(self.captions[idx][3]) #Get subdictionary with the colors for that class
                # negative_idx = np.random.choice(class_data[negative_color])
                # negative_img = self.captions[negative_idx][5]  #Get the negative image path

                # #APPROACH 2:
                # negative_class = np.random.choice(list(set(self.colors_idx[self.captions[idx][4]]) - set([self.captions[idx][3]])))
                # class_data = self.colors_idx.get(self.captions[idx][4]) #Get subdictionary with the classes for that color
                # negative_idx = np.random.choice(class_data[negative_class])
                # negative_img = self.captions[negative_idx][5]  #Get the negative image path

                # #Approach 3:
                # # print(f'Class: {self.captions[idx][1]} | {set(self.colors_idx[self.captions[idx][2]])}')
                # # print(f'Color: {self.captions[idx][2]} | {set(self.classes_idx[self.captions[idx][1]])}')
                # negative_class = np.random.choice(list(set(self.colors_idx[self.captions[idx][4]]) - set([self.captions[idx][3]])))
                # negative_color = np.random.choice(list(set(self.classes_idx[self.captions[idx][3]]) - set([self.captions[idx][4]])))
                # negative_idx = np.random.choice(self.classes_idx[negative_class][negative_color])
                # negative_img = self.captions[negative_idx][5]  #Get the negative image path


        else:  # Retrieve from the predefined triplets
            img1 = self.images[self.triplests[idx][0]]
            img2 = self.images[self.triplests[idx][1]]
            img3 = self.images[self.triplests[idx][2]]

        
        # print(anchor_caption, positive_img, negative_img)

        #Load images
        positive_img = Image.open(positive_img).convert("RGB")
        negative_img = Image.open(negative_img).convert("RGB")

        #Apply transforms
        if self.transform:
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        # print(anchor_caption, positive_img, negative_img)

        # Return anchor, pos, neg
        return anchor_caption, positive_img, negative_img
    
