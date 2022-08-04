# import torch, cv2, copy

# folder_weights_path = 'Data/'
# weights_path = 'Data/best.pt'
# model = torch.hub._load_local(folder_weights_path, 'custom', weights_path)
# path = 'D:/Minh/Projects/TutorAIv4/Data/AllImages/img881.jpg'
# img = cv2.imread(path)

# def draw_rects(img, results):
#     for res in results:
#         x, y, w, h = res[0], res[1], res[2], res[3]
#         conf = round(res[4], 2)
#         x1 = int(x-w/2)
#         y1 = int(y-h/2)
#         x2 = int(x+w/2)
#         y2 = int(y+h/2)
#         img = cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
#         img = cv2.putText(img, str(conf), (x1+100, y1), cv2.FONT_HERSHEY_SIMPLEX, 
#                    2, (255, 0, 0), 2, cv2.LINE_AA)
#     return img


# results = model(img)
# results = results.pandas().xywh[0] # predictions
# results = results[results['confidence']>0.5]
# results = results.values.tolist()
# img = draw_rects(img, results)
# img = cv2.resize(img, (800, 600))
# cv2.imshow('img', img)
# cv2.waitKey(0)


