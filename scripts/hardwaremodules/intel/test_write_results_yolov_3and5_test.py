import unittest

import helpers_yolov3and5 as inferrer


class MyTestCase(unittest.TestCase):
    def test_detections_result_writer(self):
        output_object = {'xmin': 442, 'xmax': 621, 'ymin': 91, 'ymax': 273, 'class_id': 0, 'confidence': 0.9272235631942749}
        filename = "Image1.jpg"
        img_width = 640
        img_height = 360

        combined_data = []

        inferrer.save_detections_to_csv(output_object, filename, img_width, img_height, combined_data)

        output_object = {'xmin': 441, 'xmax': 621, 'ymin': 93, 'ymax': 274, 'class_id': 0,
                         'confidence': 0.923434343}

        inferrer.save_detections_to_csv(output_object, filename, img_width, img_height, combined_data)

        print(combined_data)


        self.assertTrue(float(combined_data[0][4]), float(output_object['xmin'])/img_width)  # add assertion here


if __name__ == '__main__':
    unittest.main()
