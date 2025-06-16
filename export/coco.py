"""
Export script to generate coco data format based on our json
"""
filename_pattern = re.compile(r"_(\d{8})_(\d{6})_(\d{2})h(\d{2})m(\d{2})s(\d{3})_(\d{5})_")

def get_frame_data(annotation:Annotation, view:View):
    view_id = view.view_id
    person_id = annotation.person.person_id
    x1, y1, x2, y2 = annotation.twod_views.all()[view_id].x1, annotation.twod_views.all()[view_id].y1, annotation.twod_views.all()[view_id].x2, annotation.twod_views.all()[view_id].y2
    xw, yw, zw = annotation.Xw, annotation.Yw, annotation.Zw
    frame_id = annotation.frame.frame_id
    # if [x1, x2, y1, y2] == [-1, -1, -1, -1]:
    #     return None
    # view = annotation.twod_views.all()[0].view.view_id
    return view.view_id, frame_id, person_id, [x1, y1, x2, y2], [xw, yw, zw]


def get_time(filename:str, pattern) -> str:
    # Extract start datetime and image time offset
    # match = re.search(r"_(\d{8})_(\d{6})_(\d{2})h(\d{2})m(\d{2})s(\d{3})_(\d{5})
    match = pattern.search(filename)
    if match:
        date_str, time_str, h, m, s, ms, image_id = match.groups()
        start_dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")

        # Parse current offset time
        current_time = timedelta(
            hours=int(h),
            minutes=int(m),
            seconds=int(s),
            milliseconds=int(ms)
        )

        # Reference time is 12:00:00.000
        ref_time = timedelta(hours=12)

        # Actual offset = current_time - ref_time
        offset = current_time - ref_time

        # Final image timestamp
        image_dt = start_dt + offset

        return image_dt, image_id
    else:
        print("Filename format invalid.")

def extract_image_info(imageurl:str, url:str, image_id:int) -> dict:
    """
    returns image info dictionary for coco dataset
    """
    imagepath = Path(imageurl)
    imagename = imagepath.name

    with Image.open("path/to/image.jpg") as img:
        width, height = img.size
    time, image_id = get_time(filename=imagename)
    imageinfo = {
                "license": 1,
                "file_name": imagename,
                "coco_url": f"{url}/{imagename}",
                "height": height,
                "width": width,
                "date_captured": time,
                "id": image_id
                }
    
    return imageinfo

def get_annotations(worker:Worker, dataset:Dataset, view:View) -> dict:
    """
    Generate COCO compatible annotations from dataset for a given worker
    """
    annotations = Annotation.objects.filter(person__dataset=dataset, person__worker=worker)
    view_data = Annotation2DView.objects.filter(annotation__in=annotations, view = view).values('annotation__person__person_id', 'x1', 'y1', 'x2', 'y2', 'annotation__Xw', 'annotation__Yw', 'annotation__Zw', 'annotation__frame__frame_id')
    annotations = [
            {"id": i, 
            "image_id": view_data_entry['annotation__frame__frame_id'], 
            "category_id": view_data_entry['annotation__person__person_id'], 
            "bbox": [view_data_entry['x1'], view_data_entry['y1'], view_data_entry['x2'], view_data_entry['y2']]
            } for i, view_data_entry in enumerate(view_data)
        ]
    
    return annotations

def get_categories(worker:Worker, dataset:Dataset, view:View) -> dict:
    """
    Generate COCO compatible categories from dataset for a given worker,
    each category corresponds to a unique person id
    """
    annotations = Annotation.objects.filter(person__dataset=dataset, person__worker=worker)
    view_data = Annotation2DView.objects.filter(annotation__in=annotations, view = view).values('annotation__person__person_id')
    categories = [
        {
            "supercategory": "person",
            "id": view_data_entry['annotation__person__person_id'],
            "name": f"person_{view_data_entry['annotation__person__person_id']}"
        } for view_data_entry in view_data
        ]
    
    return categories

def get_images(worker:Worker, dataset:Dataset, view:View) -> dict:
    """
    Generate COCO compatible image infos from dataset for a given worker
    """
    annotations = Annotation.objects.filter(person__dataset=dataset, person__worker=worker)
    view_data = Annotation2DView.objects.filter(annotation__in=annotations, view = view).values('annotation__person__person_id')


def as_coco(data:Dataset, worker:Worker, filepath:str) -> dict:
    output_path = filepath
    coco = {
                "info": {
                    "description": "SCOUT 2025 Dataset",
                    "url": "http://scoutdataseturl",
                    "version": "1.0",
                    "year": 2025,
                    "contributor": "EPFL CVLab",
                    "date_created": "2025"
                },
                  "images": [
                  ],
                    "annotations": [
                        {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [1, 1, 1, 1]
                        }
                ],
                "categories": [{
                "supercategory": "person",
                "id": 1,
                "name": "person"
                },
                ],
        "licenses": [{"url":None, "id": 1, "name": None}]
    }


def main():



if __name__=="__main__":
    main()
