import cv2
import numpy as np


def load_yolo():
    net = cv2.dnn.readNet("yolo_files/yolov3.weights", "yolo_files/yolov3.cfg")
    classes = []
    with open("yolo_files/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers


def detect_objects(img, net, outputLayers):			
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return outputs

def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(confidence))
                class_ids.append(class_id)
    return boxes, confs, class_ids

def draw_labels(boxes, confs, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            label_color = (255, 0, 0)  # 
            font_scale = 1  
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y - 10), font, font_scale, label_color, 2)
    return img


def image_detection(video_path, net, classes, output_layers, colors, output_video_path):
    cap = cv2.VideoCapture(video_path)

    # Récupérer les propriétés du flux vidéo
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Créer un objet VideoWriter pour enregistrer la vidéo annotée
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Changer le codec en mp4v
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        outputs = detect_objects(frame, net, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        frame = draw_labels(boxes, confs, class_ids, classes, frame)

        # Écrire le frame annoté dans le fichier vidéo
        out.write(frame)

    # Libérer les ressources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    net, classes, output_layers = load_yolo()
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    video_path = "input_video/Chien.mp4"
    output_video_path = "output_video/annotated_video.mp4"  # Changer l'extension en mp4
    image_detection(video_path, net, classes, output_layers, colors, output_video_path)

if __name__ == '__main__':
    main()
