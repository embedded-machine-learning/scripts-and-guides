

def save_detections_to_csv(output_object, filename, img_width, img_height, combined_data):
    """
    Save to detections file.

    Format from object:
    {'xmin': 442, 'xmax': 621, 'ymin': 91, 'ymax': 273, 'class_id': 0, 'confidence': 0.9272235631942749}

    filename,
    str(w),
    str(h),

    """

    label = output_object['class_id']
    xmin = float(output_object['xmin'])/img_width
    ymin = float(output_object['ymin'])/img_height
    xmax = float(output_object['xmax'])/img_width
    ymax = float(output_object['ymax'])/img_height
    confidence = output_object['confidence']
    combined_data.append(
        [
            filename,
            str(img_width),
            str(img_height),
            str(label),
            str(xmin),
            str(ymin),
            str(xmax),
            str(ymax),
            str(confidence),
        ]
    )
