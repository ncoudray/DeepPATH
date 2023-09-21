# From https://github.com/One-sixth/imagescope_xml_utils/blob/master/imagescope_xml_utils.py
'''
MIT License

Copyright (c) 2020 onesixth

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
## module load condaenvs/gpu/deeppath_env_conda3_520
## conda install -c anaconda lxml 

import lxml.etree as etree
import numpy as np
from typing import Tuple


TYPE_CONTOUR =  0   # 原始格式：[pt1_xy, pt2_xy, pt3_xy, ...]
TYPE_BOX =      1   # 原始格式：[pt_lt, pt_rt, pt_rb, pt_rl]
TYPE_ELLIPSE =  2   # 原始格式：[[x1, y1], [x2, y2]]
TYPE_ARROW =    3   # 原始格式：[hear_xy, tail_xy]

def color_int_to_tuple(color_int):
    '''
    将RGB颜色元组转换为颜色整数
    :param color_int:
    :return:
    '''
    color_str = hex(color_int)[2:]
    assert len(color_str) <= 6, 'Found unknow color!'
    pad_count = 6 - len(color_str)
    color_str = ''.join(['0'] * pad_count) + color_str
    b, g, r = int(color_str[0:2], 16), int(color_str[2:4], 16), int(color_str[4:6], 16)
    return r, g, b


def color_tuple_to_int(color_tuple):
    '''
    将RGB颜色元组转换为颜色整数
    :param color_tuple:
    :return:
    '''
    assert len(color_tuple) == 3, 'Found unknow color tuple!'
    r, g, b = color_tuple
    color_int = r + (g << 8) + (b << 16)
    return color_int

class ImageScopeXmlReader:
    def __init__(self, file=None, keep_arrow_tail=False, use_box_y1x1y2x2=True):
        '''

        :param file:            读取文件路径
        :param keep_arrow_tail: 读取箭头标签时是否保留箭头的尾部
        :param use_box_y1x1y2x2:读取方盒标签时是否使用y1x1y2x2坐标，若设为False则使用[左上，右上，右下，左下]坐标
        '''
        self.keep_arrow_tail = keep_arrow_tail
        self.use_box_y1x1y2x2 = use_box_y1x1y2x2
        self.contour_color_regs = {}
        self.box_color_regs = {}
        self.arrow_color_regs = {}
        self.ellipse_color_regs = {}
        if file is not None:
            self.read(file)

    def read(self, file):
        tree = etree.parse(file)
        for ann in tree.findall('./Annotation'):
            color_int = int(ann.attrib['LineColor'])
            color_tuple = color_int_to_tuple(color_int)
            for region in ann.findall('./Regions/Region'):
                reg_type = int(region.attrib['Type'])

                if reg_type == TYPE_ARROW:
                    # 读取箭头标签
                    self.arrow_color_regs.setdefault(color_tuple, [])
                    arrow_head_tail_points = []
                    for vertex in region.findall('./Vertices/Vertex'):
                        x = int(float(vertex.attrib['X']))
                        y = int(float(vertex.attrib['Y']))
                        arrow_head_tail_points.append((y, x))
                    arrow_points = np.asarray(arrow_head_tail_points, np.int)
                    if not self.keep_arrow_tail:
                        arrow_points = arrow_points[0]
                    self.arrow_color_regs[color_tuple].append(arrow_points)

                elif reg_type == TYPE_BOX:
                    # 读取盒状标签
                    self.box_color_regs.setdefault(color_tuple, [])
                    box_points = []
                    for vertex in region.findall('./Vertices/Vertex'):
                        x = int(float(vertex.attrib['X']))
                        y = int(float(vertex.attrib['Y']))
                        box_points.append((y, x))
                    box_points = np.asarray(box_points, np.int)
                    if self.use_box_y1x1y2x2:
                        y1, x1 = box_points[0]
                        y2, x2 = box_points[2]
                        box_points = np.array([y1, x1, y2, x2])
                    self.box_color_regs[color_tuple].append(box_points)

                elif reg_type == TYPE_CONTOUR:
                    # 读取轮廓标签
                    self.contour_color_regs.setdefault(color_tuple, [])
                    contours = []
                    for vertex in region.findall('./Vertices/Vertex'):
                        x = int(float(vertex.attrib['X']))
                        y = int(float(vertex.attrib['Y']))
                        contours.append((y, x))
                    contours = np.asarray(contours, np.int)
                    self.contour_color_regs[color_tuple].append(contours)

                elif reg_type == TYPE_ELLIPSE:
                    # 读取椭圆标签
                    self.ellipse_color_regs.setdefault(color_tuple, [])
                    ellipse = []
                    for vertex in region.findall('./Vertices/Vertex'):
                        x = int(float(vertex.attrib['X']))
                        y = int(float(vertex.attrib['Y']))
                        ellipse.append((y, x))
                    ellipse = np.asarray(ellipse, np.int)
                    self.ellipse_color_regs[color_tuple].append(ellipse)

                else:
                    print('Unknow type {}. Will be skip.'.format(reg_type))

    def get_contours(self):
        contours, colors = [], []
        for color in self.contour_color_regs:
            contours.extend(self.contour_color_regs[color])
            colors.extend([color]*len(self.contour_color_regs[color]))
        return contours, colors

    def get_boxes(self):
        boxes, colors = [], []
        for color in self.box_color_regs:
            boxes.extend(self.box_color_regs[color])
            colors.extend([color]*len(self.box_color_regs[color]))
        return boxes, colors

    def get_arrows(self):
        arrows, colors = [], []
        for color in self.arrow_color_regs:
            arrows.extend(self.arrow_color_regs[color])
            colors.extend([color]*len(self.arrow_color_regs[color]))
        return arrows, colors

    def get_ellipses(self):
        ellipses, colors = [], []
        for color in self.ellipse_color_regs:
            ellipses.extend(self.ellipse_color_regs[color])
            colors.extend([color]*len(self.ellipse_color_regs[color]))
        return ellipses, colors

class ImageScopeXmlWriter:

    def __init__(self, contour_default_is_closure=True, allow_box_y1x1y2x2=True, auto_add_arrow_tail=True):
        '''
        :param contour_default_is_closure:  默认输入的轮廓是否是闭合的
        :param allow_box_y1x1y2x2:          是否允许方框坐标为 y1x1y2x2，若设为False，则需要手动保证方框输入坐标为 [左上，右上，右下，左下] 格式坐标
        :param auto_add_arrow_tail:         是否自动给只有箭头没有箭尾自动增加箭尾，如果设为False则需要手动保证箭头标签同时有箭头和箭尾
        '''
        self.contour_default_is_closure = contour_default_is_closure
        self.allow_box_y1x1y2x2 = allow_box_y1x1y2x2
        self.auto_add_arrow_tail = auto_add_arrow_tail
        # 每个类别的存储处，存储方式：(颜色元组) -> [(数据), (数据), ...]
        self.contour_color_regs = {}
        self.box_color_regs = {}
        self.arrow_color_regs = {}
        self.ellipse_color_regs = {}

    def add_contours(self, contours, colors, is_closures=None):
        assert is_closures is None or len(is_closures) == len(contours)
        assert len(contours) == len(colors)

        if is_closures is None:
            is_closures = [self.contour_default_is_closure] * len(contours)

        color_set = set(colors)
        for c in color_set:
            assert isinstance(c, Tuple) and len(c) == 3
            if c not in self.contour_color_regs:
                self.contour_color_regs[c] = []

        for con, color, clos in zip(contours, colors, is_closures):
            assert isinstance(con, np.ndarray)
            assert con.ndim == 2 and con.shape[1] == 2
            if clos and np.any(con[0] != con[-1]):
                con = np.resize(con, [con.shape[0]+1, con.shape[1]])
                con[-1] = con[0]
            self.contour_color_regs[color].append(con)

    def add_boxes(self, boxes, colors):
        assert len(boxes) == len(colors)

        color_set = set(colors)
        for c in color_set:
            assert isinstance(c, Tuple) and len(c) == 3
            if c not in self.box_color_regs:
                self.box_color_regs[c] = []

        for box, color in zip(boxes, colors):
            assert isinstance(box, np.ndarray)
            if self.allow_box_y1x1y2x2 and box.shape == (4,):
                y1, x1, y2, x2 = box
                box = [[y1, x1], [y1, x2], [y2, x2], [y2, x1]]
                box = np.array(box)
            assert box.shape == (4, 2)
            self.box_color_regs[color].append(box)

    def add_arrows(self, arrows, colors, auto_tail=True):
        assert len(arrows) == len(colors)

        color_set = set(colors)
        for c in color_set:
            assert isinstance(c, Tuple) and len(c) == 3
            if c not in self.arrow_color_regs:
                self.arrow_color_regs[c] = []

        for arrow, color in zip(arrows, colors):
            assert isinstance(arrow, np.ndarray)
            if auto_tail and arrow.shape == (2,):
                arrow = np.resize(arrow.reshape([1, 2]), [2, 2])
                arrow[1] = arrow[0] + 100
            assert arrow.shape == (2, 2)
            self.arrow_color_regs[color].append(arrow)

    def add_ellipses(self, ellipses, colors):
        assert len(ellipses) == len(colors)

        color_set = set(colors)
        for c in color_set:
            assert isinstance(c, Tuple) and len(c) == 3
            if c not in self.ellipse_color_regs:
                self.ellipse_color_regs[c] = []

        for ellipse, color in zip(ellipses, colors):
            assert isinstance(ellipse, np.ndarray)
            assert ellipse.shape == (2, 2)
            self.ellipse_color_regs[color].append(ellipse)

    def write(self, file):
        Annotations = etree.Element('Annotations', {'MicronsPerPixel': '0'})
        ann_id = 0
        for color_regs, type_id in zip([self.contour_color_regs, self.box_color_regs, self.arrow_color_regs, self.ellipse_color_regs],
                                       [TYPE_CONTOUR, TYPE_BOX, TYPE_ARROW, TYPE_ELLIPSE]):
            for color in color_regs.keys():
                ann_id += 1
                LineColor = str(color_tuple_to_int(color))
                Annotation = etree.SubElement(Annotations, 'Annotation',
                                              {'Id': str(ann_id), 'Name': '', 'ReadOnly': '0', 'NameReadOnly': '0',
                                               'LineColorReadOnly': '0', 'Incremental': '0', 'Type': '4',
                                               'LineColor': LineColor, 'Visible': '1', 'Selected': '0',
                                               'MarkupImagePath': '', 'MacroName': ''})

                Attributes = etree.SubElement(Annotation, 'Attributes')
                etree.SubElement(Attributes, 'Attribute', {'Name': '', 'Id': '0', 'Value': ''})
                Regions = etree.SubElement(Annotation, 'Regions')
                RegionAttributeHeaders = etree.SubElement(Regions, 'RegionAttributeHeaders')
                etree.SubElement(RegionAttributeHeaders, 'AttributeHeader',
                                 {'Id': "9999", 'Name': 'Region', 'ColumnWidth': '-1'})
                etree.SubElement(RegionAttributeHeaders, 'AttributeHeader',
                                 {'Id': "9997", 'Name': 'Length', 'ColumnWidth': '-1'})
                etree.SubElement(RegionAttributeHeaders, 'AttributeHeader',
                                 {'Id': "9996", 'Name': 'Area', 'ColumnWidth': '-1'})
                etree.SubElement(RegionAttributeHeaders, 'AttributeHeader',
                                 {'Id': "9998", 'Name': 'Text', 'ColumnWidth': '-1'})

                for contour_id, contour in enumerate(color_regs[color]):
                    Region = etree.SubElement(Regions, 'Region',
                                              {'Id': str(contour_id), 'Type': str(type_id), 'Zoom': '1', 'Selected': '0',
                                               'ImageLocation': '', 'ImageFocus': '-1', 'Length': '0', 'Area': '0',
                                               'LengthMicrons': '0', 'AreaMicrons': '0', 'Text': '', 'NegativeROA': '0',
                                               'InputRegionId': '0', 'Analyze': '1', 'DisplayId': str(contour_id)})
                    etree.SubElement(Region, 'Attributes')
                    Vertices = etree.SubElement(Region, 'Vertices')
                    for v_yx in contour:
                        etree.SubElement(Vertices, 'Vertex', {'X': str(v_yx[1]), 'Y': str(v_yx[0]), 'Z': '0'})

                etree.SubElement(Annotation, 'Plots')

        doc = etree.ElementTree(Annotations)
        doc.write(open(file, "wb"), pretty_print=True)


if __name__ == '__main__':
    print('Testing')
    reader = ImageScopeXmlReader("test.xml", keep_arrow_tail=False, use_box_y1x1y2x2=True)
    arrows, arrow_colors = reader.get_arrows()
    boxes, box_colors = reader.get_boxes()
    contours, contour_colors = reader.get_contours()
    ellipses, ellipse_colors = reader.get_ellipses()

    writer = ImageScopeXmlWriter()
    writer.add_arrows(arrows, arrow_colors)
    writer.add_boxes(boxes, box_colors)
    writer.add_contours(contours, contour_colors)
    writer.add_ellipses(ellipses, ellipse_colors)
    writer.write('test2.xml')
