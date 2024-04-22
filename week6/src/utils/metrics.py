def compute_ap(predictions, gt, num_img_class_gt):
    
  precision = []
  recall = []
  AP = 0
  num = 0

  for i, prediction in enumerate(predictions):

      if prediction == gt:
          num +=1
          AP += (num/(i+1))
      
      precision.append(num/(i+1))
      recall.append(num/num_img_class_gt)
          
  AP = AP/num
#   print(precision)
  # print(f'P@1 of class {gt}: {precision[0]}')

  return AP, precision, recall

def compute_ap_COCO(database_label, query_label):
  
  #database_label is a list of lists of the labels of each image on the database
  #query_label is a list of labels of the 

  precision = []
  # recall = []
  AP = 0
  num = 0

  for i, element in enumerate(database_label):
      
      for value in element:
        if value in query_label:
            num +=1
            AP += (num/(i+1))
            break

      # if prediction == query_label:
      #     num +=1
      #     AP += (num/(i+1))
      
      precision.append(num/(i+1))
      # recall.append(num/num_img_class_gt)
          
  AP = AP/num
#   print(precision)
#   print(f'P@1 of class {gt}: {precision[0]}')

  return AP, precision
