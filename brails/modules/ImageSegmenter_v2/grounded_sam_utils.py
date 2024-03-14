import cv2
import numpy as np
import pandas as pd
import supervision as sv
import os
import glob
import pickle

import torch
import torchvision

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

def verify_and_download_models(download_url, filepath):
    if(not os.path.isfile(filepath)):
        model_path = filepath
        print('Loading default segmentation model file to the pretrained folder...')
        torch.hub.download_url_to_file(download_url,
                                        model_path, progress=False)

#build GroundingDINO and SAM
def build_models(device = "cuda:0"):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'DEVICE FOUND: {DEVICE}')
    if(not os.path.exists('tmp/models')): os.makedirs('tmp/models')
    # GroundingDINO config and checkpoint
    GROUNDING_DINO_CONFIG_PATH = "brails/modules/ImageSegmenter_v2/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT_PATH = "tmp/models/groundingdino_swint_ogc.pth"
    GROUNDING_DINO_CHECKPOINT_URL = 'https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth'
    # Segment-Anything checkpoint
    SAM_ENCODER_VERSION = "vit_h"
    SAM_CHECKPOINT_PATH = "tmp/models/sam_vit_h_4b8939.pth"
    SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    
    verify_and_download_models(GROUNDING_DINO_CHECKPOINT_URL, GROUNDING_DINO_CHECKPOINT_PATH)
    verify_and_download_models(SAM_CHECKPOINT_URL, SAM_CHECKPOINT_PATH)


    # Building GroundingDINO inference model
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, device = device)
    # Building SAM Model and SAM Predictor
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)
    return grounding_dino_model, sam_predictor

# Prompting SAM with detected boxes
def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


def run_on_one_image(
    img_source, output_dir, grounding_dino_model, sam_predictor, CLASSES,
    BOX_THRESHOLD = 0.35, TEXT_THRESHOLD = 0.25, NMS_THRESHOLD = 0.8
    ):

    #SOURCE_IMAGE_PATH="/nfs/turbo/coe-stellayu/brianwang/testData/nFloors/merged_data/Ann Arbor, MI/1_30.png"
    SOURCE_IMAGE_PATH = img_source
    img_name = SOURCE_IMAGE_PATH.split("/")[-1][:-4]

    # load image
    image = cv2.imread(SOURCE_IMAGE_PATH)

    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=BOX_THRESHOLD
    )

    # annotate image with detections
    box_annotator = sv.BoxAnnotator(text_scale = 0.3, text_padding = 1, thickness = 2)
    labels = [
        f"{CLASSES[class_id]}({confidence:0.2f})" 
        for _, _, confidence, class_id, _, _
        in detections]
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

    # NMS post process
    output_str = f"{img_name} NMS: Before = {len(detections.xyxy)} boxes, "
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy), 
        torch.from_numpy(detections.confidence), 
        NMS_THRESHOLD
    ).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]
    output_dict = {"coord":detections.xyxy, "confidence":detections.confidence, "class":detections.class_id}

    output_str += f"After = {len(detections.xyxy)} boxes"

    # convert detections to masks
    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )

    # annotate image with detections
    mask_annotator = sv.MaskAnnotator()

    with open(os.path.join(output_dir, f'{img_name}_mask_data'), "wb") as fp:   #Pickling
        pickle.dump(detections.mask,fp)   
    mask = mask_annotator.annotate(scene=np.zeros(image.shape).astype(image.dtype).astype(np.uint8), detections=detections)
    cv2.imwrite(os.path.join(output_dir, f"{img_name}_mask.png"), mask)
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    cv2.imwrite(os.path.join(output_dir, f"{img_name}_mask_overlap.png"), annotated_image)
    return annotated_image

    