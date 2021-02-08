"""
Converts a dataset file to anno format. See https://github.com/urobots-io/anno.
"""

import copy
import json
import os
import random
import sys


ANNO_TEMPLATE = {
    "definitions": {
        "files_root_dir": "",
        "marker_types": {
            "origin": {
                "categories": [],
                "description": " ",
                "line_width": -5,
                "rendering_script": [
                    "p.SetBaseTransform(true, true)",
                    "p.SetDefaultPen()",
                    "let object_size = 88",
                    "p.DrawEllipse(-object_size / 2, -object_size / 2, object_size, object_size)",
                    "let arrow_size = object_size / 4",
                    "p.DrawLine(0, 0, arrow_size, 0)",
                    "p.DrawLine(arrow_size * 0.95, arrow_size * 0.05, arrow_size, 0)",
                    "p.DrawLine(arrow_size * 0.95, -arrow_size * 0.05, arrow_size, 0)",
                    "p.DrawLine(0, 0, 0, arrow_size)",
                    "p.DrawLine(arrow_size * 0.05, arrow_size * 0.95, 0, arrow_size)",
                    "p.DrawLine(-arrow_size * 0.05, arrow_size * 0.95, 0, arrow_size)"
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

    categories = set()

    for image in dataset:
        file = {
            'name': image['image'],
            'markers': []
        }
        for obj in image['objects']:
            categories.add(obj['category'])
            file['markers'].append({
                'type': 'origin',
                'category': obj['category'],
                'value': f"{obj['origin']['x']} {obj['origin']['y']} {obj['origin']['angle']}",
            })
        anno['files'].append(file)

    random.seed(1)
    for i, category in enumerate(categories):
        color = f'#{random.randrange(100, 255):02x}{random.randrange(100, 255):02x}{random.randrange(100, 255):02x}'
        category_data = {'color': color,
                         'id': i,
                         'name': f'object {i}'
                         }
        anno['definitions']['marker_types']['origin']['categories'].append(category_data)

    anno_path = os.path.splitext(input_file)[0] + '.anno'
    with open(anno_path, 'w') as f:
        json.dump(anno, f, indent=1)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python to_anno.py {DATASET.json}')
        sys.exit(-1)

    convert(sys.argv[1])
