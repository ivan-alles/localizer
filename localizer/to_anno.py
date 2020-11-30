"""
Converts a dataset file to anno format. See https://github.com/urobots-io/anno.
"""

import copy
import json
import os
import sys


ANNO_TEMPLATE = {
    "definitions": {
        "files_root_dir": "",
        "marker_types": {
            "bounding_box": {
                "categories": [
                    {
                        "color": "#00ff00",
                        "id": 0,
                        "name": "object 0"
                    },
                    {
                        "color": "#0000ff",
                        "id": 1,
                        "name": "object 1"
                    }
                ],
                "description": " ",
                "rendering_script": [
                    "p.SetBaseTransform(true, true)",
                    "p.SetDefaultPen()",
                    "p.DrawRect(-1, -1, 2, 2)"
                ],
                "value_type": "oriented_rect"
            },
            "origin": {
                "categories": [
                    {
                        "color": "#00ff00",
                        "id": 0,
                        "name": "object 0"
                    },
                    {
                        "color": "#0000ff",
                        "id": 1,
                        "name": "object 1"
                    }
                ],
                "description": " ",
                "line_width": -5,
                "rendering_script": [
                    "p.SetBaseTransform(true, true)",
                    "p.SetDefaultPen()",
                    "p.DrawEllipse(-10, -10, 20, 20)",
                    "p.SetPen(255, 0, 0, -2)",
                    "p.DrawLine(0, 0, 50, 0)",
                    "p.DrawLine(45, 5, 50, 0)",
                    "p.DrawLine(45, -5, 50, 0)",
                    "p.SetPen(0, 255, 0, -2)",
                    "p.DrawLine(0, 0, 0, 50)",
                    "p.DrawLine(5, 45, 0, 50)",
                    "p.DrawLine(-5, 45, 0, 50)",
                    ""
                ],
                "value_type": "oriented_point"
            }
        },
        "user_data": {
        }
    },
    "files": []
}


def convert(input_file):
    with open(input_file, 'r') as f:
        dataset = json.load(f)

    anno = copy.deepcopy(ANNO_TEMPLATE)

    for image in dataset:
        file = {
            'name': image['image'],
            'markers': []
        }
        for obj in image['objects']:
            file['markers'].append({
                'type': 'origin',
                'category': obj['category'],
                'value': f"{obj['origin']['x']} {obj['origin']['y']} 100 100 {obj['origin']['angle']}",
            })
            file['markers'].append({
                'type': 'bounding_box',
                'category': obj['category'],
                'value': f"{obj['bounding_box']['x']} {obj['bounding_box']['y']} "
                         f"{obj['bounding_box']['size_x']} {obj['bounding_box']['size_y']} "
                         f"{obj['bounding_box']['angle']}",
            })

        anno['files'].append(file)

    anno_path = os.path.splitext(input_file)[0] + '.anno'
    with open(anno_path, 'w') as f:
        json.dump(anno, f, indent=1)


if len(sys.argv) != 2:
    print('Usage: python to_anno.py {DATASET.json}')
    sys.exit(-1)

convert(sys.argv[1])
