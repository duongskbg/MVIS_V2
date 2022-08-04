from collections import Counter
import numpy as np

IMG_DIR_TRAIN = 'Data/Images/Train'
LABEL_DIR_TRAIN = 'Data/Labels/Train'

"""" convert boxes in yolo format (i.e. cell relative ratio) to whole image ratio """
def cellboxes_to_boxes(cellBoxes, C, S):
    batchSize = cellBoxes.shape[0]
    bboxes = cellBoxes[..., (C + 1) : (C + 5)]
    cellIndices = np.arange(S).repeat(batchSize, S, 1).unsqueeze(-1)
    cellIndices = cellIndices.to('cuda')
    x = 1 / S * (bboxes[..., :1] + cellIndices)
    y = 1 / S * (bboxes[..., 1:2] + cellIndices.permute(0, 2, 1, 3))
    w_h = 1 / S * bboxes[..., 2:4]
    convertedBboxes = np.cat((x, y, w_h), dim = -1)
    predictedClass = cellBoxes[..., :C ].argmax(-1).unsqueeze(-1)
    confidence = cellBoxes[..., C : (C + 1)]
    convertedPreds = np.cat((predictedClass, confidence, convertedBboxes), dim = -1)
    return convertedPreds # [ batch, S, S, 6]

def cellboxes_to_list(boxes, C, S):
    convertedBoxes = cellboxes_to_boxes(boxes, C, S).reshape(boxes.shape[0], S*S, -1)
    convertedBoxes[..., 0] = convertedBoxes[..., 0].long()
    batchSize = boxes.shape[0]
    allBboxes = []
    for exIdx in range(batchSize):
        bboxes = []
        for bboxIdx in range(S*S):
            bboxes.append([x.item() for x in convertedBoxes[exIdx, bboxIdx, :]])
        allBboxes.append(bboxes)
    return allBboxes

def non_max_suppression(boxes, iouThres, thres):
    boxes = [box for box in boxes if box[1] > thres]
    boxes = sorted(boxes, key = lambda x : x[1], reverse = True)
    boxesAfterNms = []
    while boxes: 
        chosenBox = boxes.pop(0)
        boxes = [box for box in boxes if box[0] != chosenBox[0]
                  or intersection_over_union(chosenBox[2:], box[2:]) < iouThres ]
        boxesAfterNms.append(chosenBox)
    return boxesAfterNms

def mean_average_precision(predBoxes, trueBoxes, iouThres, numClasses):
    # predBoxes or trueBoxes: each with 7 components [trainIdx, class, prob, x, y, w, h]
    averagePrecisions = []
    for c in range(numClasses):
        detections, groundTruths = [], []
        for detection in predBoxes:
            if detection[1] == c:
                detections.append(detection)
        for trueBox in trueBoxes:
            if trueBox[1] == c:
                groundTruths.append(trueBox)
        amountBoxes = Counter([gt[0] for gt in groundTruths])
        # convert into { 0: tensor[0, 0, 0], 1: tensor[0, 0, 0, 0, 0] }
        for key, val in amountBoxes.items():
            amountBoxes[key] = np.zeros(val)
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = np.zeros((len(detections)))
        FP = np.zeros((len(detections)))
        totalTrueBoxes = len(groundTruths)
        if totalTrueBoxes == 0:
            continue
        
        for detectionIdx, detection in enumerate(detections):
            groundTruthImg = [box for box in groundTruths if box[0] == detection[0]]
            numGts = len(groundTruthImg)
            bestIou = 0
            for idx, gt in enumerate(groundTruthImg):
                iou = intersection_over_union(detection[3:], gt[3:])
                if iou > bestIou:
                    bestIou = iou
                    bestGtIdx = idx
            if bestIou > iouThres:
                if amountBoxes[detection[0]][bestGtIdx] == 0:
                    TP[detectionIdx] = 1
                    amountBoxes[detection[0]][bestGtIdx] = 1
                else:
                    FP[detectionIdx] = 1
            else:
                FP[detectionIdx] = 1
        TP_cumsum, FP_cumsum = np.cumsum(TP, dim = 0), np.cumsum(FP, dim = 0)
        recalls = TP_cumsum / (totalTrueBoxes + 1e-6)
        precisions = np.divide(TP_cumsum, TP_cumsum + FP_cumsum + 1e-6)
        precisions = np.cat([1], precisions)
        recalls = np.cat([0], recalls)
        averagePrecisions.append(np.trapz(precisions, recalls))
    return sum(averagePrecisions) / len(averagePrecisions)

"""" get bounding boxes from all data (all batches), in terms of list """
def get_bboxes(loader, model, iouThres, thres, device = 'cuda'):
    allPredBoxes, allTrueBoxes = [], []
    model.eval()
    trainIdx = 0
    for batchIdx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        output = model(x)
        batchSize = x.shape[0]
        gridSize = output.shape[1]
        numClasses = output.shape[-1] - 5
        predBoxes = cellboxes_to_list(output, C = numClasses, S = gridSize)
        trueBoxes = cellboxes_to_list(y, C = numClasses, S = gridSize)
        for idx in range(batchSize):
            nmsBoxes = non_max_suppression(predBoxes[idx], iouThres = iouThres, thres = thres)
            for nmsBox in nmsBoxes:
                allPredBoxes.append([trainIdx] + nmsBox)
            for box in trueBoxes[idx]:
                if box[1] > thres:
                    allTrueBoxes.append([trainIdx] + box)
            trainIdx += 1
    model.train()
    return allPredBoxes, allTrueBoxes

def intersection_over_union(boxesPreds, boxesLabels):
    box1_x1 = boxesPreds[..., 0:1] - boxesPreds[..., 2:3] / 2
    box1_y1 = boxesPreds[..., 1:2] - boxesPreds[..., 3:4] / 2
    box1_x2 = boxesPreds[..., 0:1] + boxesPreds[..., 2:3] / 2
    box1_y2 = boxesPreds[..., 1:2] + boxesPreds[..., 3:4] / 2
    box2_x1 = boxesLabels[..., 0:1] - boxesLabels[..., 2:3] / 2
    box2_y1 = boxesLabels[..., 1:2] - boxesLabels[..., 3:4] / 2
    box2_x2 = boxesLabels[..., 0:1] + boxesLabels[..., 2:3] / 2
    box2_y2 = boxesLabels[..., 1:2] + boxesLabels[..., 3:4] / 2
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1Area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2Area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    return intersection / (box1Area + box2Area - intersection + 1e-6)
