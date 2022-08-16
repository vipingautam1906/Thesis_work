import cv2
from detection import Detection
from threading import Thread
import imutils
import numpy as np
import os
import argparse
from imutils.video import VideoStream
import torch
import torchvision.ops.boxes as bops
from Alignedreid_demo import Aligned_Reid_class

Aligned_Ried = Aligned_Reid_class()
od = Detection()
classes = od.load_class_names()

ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', type=str,
                help='path to input video file')
ap.add_argument('-t', '--tracker', type=str, default='csrt', help='OpenCV object track type')
args = vars(ap.parse_args())

tracker = cv2.TrackerCSRT_create()
init_bb = None
frame_count = 0
fps = None
result_dict = {}

object_left_fov = {'status': False}

if not args.get('video', False):
    print("Starting video stream..")
    video_stream = VideoStream(src=0).start()

else:
    video_stream = cv2.VideoCapture((args['video']))

# video_stream = cv2.VideoCapture('e1_3.mp4')
frame_width = int(video_stream.get(3))
frame_height = int(video_stream.get(4))
fcc = cv2.VideoWriter_fourcc(*'XVID')
size = (frame_width, frame_height)
result = cv2.VideoWriter('result.avi', fcc, 15, (800, 400))


def preprocessing(box, current_frame):
    x_shape, y_shape = current_frame.shape[1], current_frame.shape[0]
    x1, y1, x2, y2 = int(box[0] * x_shape), int(box[1] * y_shape), int(box[2] * x_shape), int(box[3] * y_shape)
    return [x1, y1, x2, y2]


def crop_image(box, frame, index, dir_path,b_boxes, temp):
    features_reid_score = []
    Aligned_Ried = Aligned_Reid_class()
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    x1, y1, x2, y2 = int(box[0] * x_shape), int(box[1] * y_shape), int(box[2] * x_shape), int(box[3] * y_shape)
    b_box = [x1, y1, x2, y2]
    image = frame[y1:y2, x1:x2]
    image_name = 'image_' + str(index) + '.jpg'
    img_path = dir_path + image_name
    cv2.imwrite(img_path, image)
    for img_name in os.listdir(temp):
        if img_name == 'suspect.jpg':
            distance = Aligned_Ried.compute_distance(os.path.join(temp, img_name), img_path)
            status = False
            result_list = [status, b_box, distance]
            result_dict[index] = [status, b_box, distance]

    for i,b in enumerate(b_boxes):
        x1, y1, x2, y2 = int(b[0] * x_shape), int(b[1] * y_shape), int(b[2] * x_shape), int(b[3] * y_shape)
        b = [x1, y1, x2, y2]
        # print('bounding box : ', i ,int(iou_check(b, b_box)*100), b, result_dict[index])
        if iou_check(b, b_box) > 0 and result_dict[index][2]*100 < 50:
            image = frame[y1:y2, x1:x2]
            image_name = str((iou_check(b, b_box)*100)) + str(i) + '.jpg'
            img_path = dir_path + image_name
            cv2.imwrite(img_path, image)
            for img_name in os.listdir(temp):
                if img_name != 'suspect.jpg':
                    distance = Aligned_Ried.compute_distance(os.path.join(temp, img_name), img_path)
                    features_reid_score.append(distance)
    if len(features_reid_score) != 0:
        arr = np.array(features_reid_score)
        result_list.append(np.min(arr))

    if len(result_list) == 4:
        join_reid_score = result_list[2]*result_list[3]
    else:
        join_reid_score = result_list[2]
    result_dict[index] = result_list
    if join_reid_score*100 < 30:
        result_dict[index][0] = True
    print('results list', result_list, 'disc', result_dict[index])

def iou_check(yolo_bbox, csrt_box):
    csrt_x, csrt_y, csrt_w, csrt_h = csrt_box
    yolo_x1, yolo_y1, yolo_x2, yolo_y2 = yolo_bbox
    box1 = torch.tensor([[yolo_x1, yolo_y1, yolo_x2, yolo_y2]], dtype=torch.float)
    box2 = torch.tensor([[csrt_x, csrt_y, csrt_x + csrt_w, csrt_y + csrt_h]], dtype=torch.float)
    IOU = bops.box_iou(box1, box2)
    return IOU


def reiden(b_box, frame,dir_path,suspect_image):
    Aligned_Ried = Aligned_Reid_class()
    image = frame[b_box[1]:b_box[3] , b_box[0]:b_box[2]]
    image_name = 'image_new' + '.jpg'  # image_new.jpg
    img_path = dir_path + '/' + image_name
    cv2.imwrite(img_path, image)
    status, distance = Aligned_Ried.compute_distance(suspect_image, img_path)
    return status, distance


def check_availability(iou_vector, count):
    flag = True
    for item in iou_vector:
        if item != 0:
            flag = False
            break
    if flag is True:
        count += 1
        return flag, count
    return flag, count

flag = False
no_suspect_count = 0
is_suspect_identified = False
is_suspect_data_available = False
input_image = 'C:/Users/vipin/PycharmProjects/MTechProjectCS561/CSRT_Yolo/AlignedReID/suspect.jpg'
path = 'C:/Users/vipin/PycharmProjects/MTechProjectCS561/CSRT_Yolo/AlignedReID/images1/'
suspect_data = {}
temp = 'C:/Users/vipin/PycharmProjects/MTechProjectCS561/CSRT_Yolo/AlignedReID/suspect_data/'

if __name__ == '__main__':
    while True:
        ret , current_frame = video_stream.read()
        if frame_count % 1 == 0:
            #current_frame = imutils.resize(current_frame, width=600)
            current_frame = cv2.resize(current_frame, (800, 400))
            x_shape, y_shape = current_frame.shape[1], current_frame.shape[0]

            if current_frame is None:
                break
            label, cord = od.detect(current_frame)
            b_boxes = cord
            num_of_objects = len(cord)
            Threads = [None] * num_of_objects
            iou_vector = [None] * num_of_objects

            is_suspect_data_available = os.path.isfile(os.path.join(temp, 'suspect.jpg'))

            if is_suspect_identified is False and is_suspect_data_available is True:
                if frame_count % 10 == 0:
                    for i in range(num_of_objects):
                        Threads[i] = Thread(target=crop_image, args=(b_boxes[i], current_frame, i, path, b_boxes, temp))
                        Threads[i].start()

                    for i in range(num_of_objects):
                        Threads[i].join()

                    for i in range(num_of_objects):
                        if result_dict[i][0] is True:
                            init_bb = result_dict[i][1]
                            init_bb[2] = init_bb[2] - init_bb[0]
                            init_bb[3] = init_bb[3] - init_bb[1]
                            tracker.init(current_frame, init_bb)
                            is_suspect_identified = True

                    for i in range(num_of_objects):
                        Threads.pop(-1)

            if init_bb is None:
                for i in range(len(b_boxes)):
                    cord = preprocessing(b_boxes[i], current_frame)
                    label_index = int(label[i])
                    confidence = int(b_boxes[i][4]*100)
                    txt = str(classes[label_index]) + "  " + str(confidence) + "%"
                    person = cv2.rectangle(current_frame, (cord[0], cord[1]), (cord[2], cord[3]), (0, 252, 124), 1)
                    cv2.putText(person, txt, (cord[0], cord[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 252, 124), 1)

            if init_bb is not None:
                (success, csrt_box) = tracker.update(current_frame)
                if success:
                    # if flag is True and frame_count % 2 == 0:
                    #     for i in range(len(b_boxes)):
                    #         cord = preprocessing(b_boxes[i], current_frame)
                    #         IOU = iou_check(cord, csrt_box)
                    #         if 5 < IOU * 100 < 20:
                    #             status, distance = reiden(cord, current_frame, path, input_image)
                    #             if status is True:
                    #                 init_bb = cord
                    #                 init_bb[2] = init_bb[2] - init_bb[0]
                    #                 init_bb[3] = init_bb[3] - init_bb[1]
                    #                 tracker.init(current_frame, init_bb)
                    #                 (success, csrt_box) = tracker.update(current_frame)
                    #                 flag = False

                    for i, item in enumerate(b_boxes):
                        cord = preprocessing(b_boxes[i], current_frame)
                        IOU = iou_check(cord, csrt_box)
                        iou_vector[i] = int(IOU*100)
                        cv2.rectangle(current_frame, (cord[0], cord[1]), (cord[2], cord[3]), (0, 252, 124), 1)
                        #print(csrt_box, (cord[0], cord[1]), (cord[2], cord[3]))
                        if IOU * 100 > 50:
                            suspect_cord = cord
                            suspect_cord[2] = suspect_cord[2] - suspect_cord[0]
                            suspect_cord[3] = suspect_cord[3] - suspect_cord[1]
                            tracker.init(current_frame, suspect_cord)
                            (success, csrt_box) = tracker.update(current_frame)
                            no_suspect_count = 0

                    # print('Current IOU:', iou_vector)
                    x, y, w, h = [int(x) for x in csrt_box]
                    suspect = cv2.rectangle(current_frame, (x, y), (x + w, y + h), (60, 20, 255), 1)
                    cv2.putText(suspect, 'Suspect', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 20, 255), 1)

                    # for i in range(len(b_boxes)):
                    #     cord = preprocessing(b_boxes[i], current_frame)
                    #     IOU = iou_check(cord, csrt_box)
                    #     if 5 < IOU * 100 < 50:
                    #         flag = True
            is_roi_not_present , no_suspect_count = check_availability(iou_vector,no_suspect_count)
            if is_roi_not_present is True and no_suspect_count > 2:
                print(is_roi_not_present, no_suspect_count, iou_vector)
                init_bb = None
                is_suspect_identified = False
                success = False

            cv2.imshow('video', current_frame)
            result.write(current_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                # init_bb = cv2.selectROI('video', current_frame, fromCenter=False, showCrosshair=True)
                ROIs = cv2.selectROIs('video', current_frame, fromCenter=False, showCrosshair=True)
                for index, roi in enumerate(ROIs):
                    if index == 0:
                        init_bb = roi
                        x1, y1, x2, y2 = init_bb
                        x2 = x2 + x1
                        y2 = y2 + y1
                        suspect = current_frame[y1:y2, x1:x2]
                        cv2.imwrite(os.path.join(temp, 'suspect.jpg'), suspect)
                        # cv2.imwrite('suspect.jpg', suspect)
                        tracker.init(current_frame, init_bb)
                        is_suspect_identified = True
                    else:
                        x1, y1, x2, y2 = roi
                        x2 = x2 + x1
                        y2 = y2 + y1
                        img = current_frame[y1:y2, x1:x2]
                        img_name = 'feature_' + str(index) + '.jpg'
                        cv2.imwrite(os.path.join(temp, img_name), img)
                        # cv2.imwrite(img_name, img)

            elif key == ord('e'):
                break

        frame_count = frame_count + 1
    result.release()
    cv2.destroyAllWindows()