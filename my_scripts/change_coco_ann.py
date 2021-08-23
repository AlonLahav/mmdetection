import json


def skip_n_frames(in_json, out_json, stride, offset=20):
    with open(in_json) as f:
        data = json.load(f)

    new_img = []
    for i, im in enumerate(data['images']):
        if (i + offset) % stride == 0:
            new_img.append(im)
    data['images'] = new_img

    with open(out_json, 'w') as json_file:
        json.dump(data, json_file)

if __name__ == '__main__':
    skip_n_frames(r"C:\datasets\my_records\maayan_room_1\annotations\instances_default.json",
                  r"C:\datasets\my_records\maayan_room_1\annotations\instances_default_stride_offset.json",
                  40)