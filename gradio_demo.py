import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from matplotlib import font_manager
from ultralytics import YOLO

ROOT_PATH = sys.path[0]  # 项目根目录

# --------------------- 字体库 ---------------------
SimSun_path = f"{ROOT_PATH}/fonts/SimSun.ttf"  # 宋体文件路径
TimesNesRoman_path = f"{ROOT_PATH}/fonts/TimesNewRoman.ttf"  # 新罗马字体文件路径
# 宋体
SimSun = font_manager.FontProperties(fname=SimSun_path, size=12)
# 新罗马字体
TimesNesRoman = font_manager.FontProperties(fname=TimesNesRoman_path, size=12)

import yaml
from PIL import Image, ImageDraw, ImageFont

# from util.fonts_opt import is_fonts

ROOT_PATH = sys.path[0]  # 根目录

# Gradio version
GYD_VERSION = "Gradio Lesion-Cells DET v1.0"

# 文件后缀
suffix_list = [".csv", ".yaml"]

# 字体大小
FONTSIZE = 25

# 目标尺寸
obj_style = ["small", "medium", "large"]

# title = "Multi-granularity Lesion Cells Object Detection based on deep neural network"
# description = "<center><h3>Description: This is a WebUI interface demo, Maintained by G1 JIANG SHUFAN</h3></center>"

GYD_TITLE = """
<p align='center'><a href='https://github.com/Tsumugii24'>
<img src='https://cdn.jsdelivr.net/gh/Tsumugii24/Typora-images@main/images/2023%2F11%2F12%2F2ce6ad153e2e862d5017864fc5087e59-image-20231112230354573-56a688.png' alt='Simple Icons' ></a>
<center><h1>Multi-granularity Lesion Cells Object Detection based on deep neural network</h1></center>
<center><h3>Description: This is a WebUI interface demo, Maintained by G1 JIANG SHUFAN</h3></center>
</p>
"""

GYD_SUB_TITLE = """
  Here is My GitHub: https://github.com/Tsumugii24  😊
"""

EXAMPLES_DET = [
    ["./img_examples/test/moderate0.BMP", "detr_based", "cpu", 640, 0.6,
     0.5, 10, "all range"],
    ["./img_examples/test/normal_co0.BMP", "vit_based", "cpu", 640, 0.5,
     0.5, 20, "all range"],
    ["./img_examples/test/1280_1920_1.jpg", "yolov8_based", "cpu", 640, 0.6, 0.5, 15,
     "all range"],
    ["./img_examples/test/normal_inter1.BMP", "yolov5_based", "cpu", 640, 0.5,
     0.5, 30, "all range"],
    ["./img_examples/test/1920_1280_1.jpg", "cnn_se", "cpu", 640, 0.6, 0.5, 20,
     "all range"],
    ["./img_examples/test/severe2.BMP", "detr_based", "cpu", 640, 0.5,
     0.5, 20, "all range"]
]


def parse_args(known=False):
    parser = argparse.ArgumentParser(description=GYD_VERSION)
    parser.add_argument("--model_name", "-mn", default="detr_based", type=str, help="model name")
    parser.add_argument(
        "--model_cfg",
        "-mc",
        default="./model_config/model_name_cells.yaml",
        type=str,
        help="model config",
    )
    parser.add_argument(
        "--cls_name",
        "-cls",
        default="./cls_name/cls_name_cells_en.yaml",
        type=str,
        help="cls name",
    )
    parser.add_argument(
        "--nms_conf",
        "-conf",
        default=0.5,
        type=float,
        help="model NMS confidence threshold",
    )
    parser.add_argument("--nms_iou", "-iou", default=0.5, type=float, help="model NMS IoU threshold")
    parser.add_argument("--inference_size", "-isz", default=640, type=int, help="model inference size")
    parser.add_argument("--max_detnum", "-mdn", default=50, type=float, help="model max det num")
    parser.add_argument("--slider_step", "-ss", default=0.05, type=float, help="slider step")
    parser.add_argument(
        "--is_login",
        "-isl",
        action="store_true",
        default=False,
        help="is login",
    )
    parser.add_argument('--usr_pwd',
                        "-up",
                        nargs='+',
                        type=str,
                        default=["admin", "admin"],
                        help="user & password for login")
    parser.add_argument(
        "--is_share",
        "-is",
        action="store_true",
        default=False,
        help="is login",
    )
    parser.add_argument("--server_port", "-sp", default=7860, type=int, help="server port")

    args = parser.parse_known_args()[0] if known else parser.parse_args()
    return args


# yaml文件解析
def yaml_parse(file_path):
    return yaml.safe_load(open(file_path, encoding="utf-8").read())


# yaml csv 文件解析
def yaml_csv(file_path, file_tag):
    file_suffix = Path(file_path).suffix
    if file_suffix == suffix_list[0]:
        # 模型名称
        file_names = [i[0] for i in list(csv.reader(open(file_path)))]  # csv版
    elif file_suffix == suffix_list[1]:
        # 模型名称
        file_names = yaml_parse(file_path).get(file_tag)  # yaml版
    else:
        print(f"The format of {file_path} is incorrect!")
        sys.exit()

    return file_names


# 检查网络连接
def check_online():
    # reference: https://github.com/ultralytics/yolov5/blob/master/utils/general.py
    # check internet connectivity
    import socket
    try:
        socket.create_connection(("1.1.1.1", 443), 5)  # check host accessibility
        return True
    except OSError:
        return False


# 标签和边界框颜色设置
def color_set(cls_num):
    color_list = []
    for i in range(cls_num):
        color = tuple(np.random.choice(range(256), size=3))
        color_list.append(color)

    return color_list


# 随机生成浅色系或者深色系
def random_color(cls_num, is_light=True):
    color_list = []
    for i in range(cls_num):
        color = (
            random.randint(0, 127) + int(is_light) * 128,
            random.randint(0, 127) + int(is_light) * 128,
            random.randint(0, 127) + int(is_light) * 128,
        )
        color_list.append(color)

    return color_list


# 检测绘制
def pil_draw(img, score_l, bbox_l, cls_l, cls_index_l, textFont, color_list):
    img_pil = ImageDraw.Draw(img)
    id = 0

    for score, (xmin, ymin, xmax, ymax), label, cls_index in zip(score_l, bbox_l, cls_l, cls_index_l):
        img_pil.rectangle([xmin, ymin, xmax, ymax], fill=None, outline=color_list[cls_index], width=2)  # 边界框
        countdown_msg = f"{label} {score:.2f}"
        # text_w, text_h = textFont.getsize(countdown_msg)  # 标签尺寸 pillow 9.5.0
        # left, top, left + width, top + height
        # 标签尺寸 pillow 10.0.0
        text_xmin, text_ymin, text_xmax, text_ymax = textFont.getbbox(countdown_msg)
        # 标签背景
        img_pil.rectangle(
            # (xmin, ymin, xmin + text_w, ymin + text_h), # pillow 9.5.0
            (xmin, ymin, xmin + text_xmax - text_xmin, ymin + text_ymax - text_ymin),  # pillow 10.0.0
            fill=color_list[cls_index],
            outline=color_list[cls_index],
        )

        # 标签
        img_pil.multiline_text(
            (xmin, ymin),
            countdown_msg,
            fill=(0, 0, 0),
            font=textFont,
            align="center",
        )

        id += 1

    return img


# 绘制多边形
def polygon_drawing(img_mask, canvas, color_seg):
    # ------- RGB转BGR -------
    color_seg = list(color_seg)
    color_seg[0], color_seg[2] = color_seg[2], color_seg[0]
    color_seg = tuple(color_seg)
    # 定义多边形的顶点
    pts = np.array(img_mask, dtype=np.int32)

    # 多边形绘制
    cv2.drawContours(canvas, [pts], -1, color_seg, thickness=-1)


# 输出分割结果
def seg_output(img_path, seg_mask_list, color_list, cls_list):
    img = cv2.imread(img_path)
    img_c = img.copy()

    # w, h = img.shape[1], img.shape[0]

    # 获取分割坐标
    for seg_mask, cls_index in zip(seg_mask_list, cls_list):
        img_mask = []
        for i in range(len(seg_mask)):
            # img_mask.append([seg_mask[i][0] * w, seg_mask[i][1] * h])
            img_mask.append([seg_mask[i][0], seg_mask[i][1]])

        polygon_drawing(img_mask, img_c, color_list[int(cls_index)])  # 绘制分割图形

    img_mask_merge = cv2.addWeighted(img, 0.3, img_c, 0.7, 0)  # 合并图像

    return img_mask_merge


# 目标检测和图像分割模型加载
def model_loading(img_path, device_opt, conf, iou, infer_size, max_det, yolo_model="yolov8n.pt"):
    model = YOLO(yolo_model)

    results = model(source=img_path, device=device_opt, imgsz=infer_size, conf=conf, iou=iou, max_det=max_det)
    results = list(results)[0]
    return results


# YOLOv8图片检测函数
def yolo_det_img(img_path, model_name, device_opt, infer_size, conf, iou, max_det, obj_size):
    global model, model_name_tmp, device_tmp

    s_obj, m_obj, l_obj = 0, 0, 0

    area_obj_all = []  # 目标面积

    score_det_stat = []  # 置信度统计
    bbox_det_stat = []  # 边界框统计
    cls_det_stat = []  # 类别数量统计
    cls_index_det_stat = []  # 1

    # 模型加载
    predict_results = model_loading(img_path, device_opt, conf, iou, infer_size, max_det, yolo_model=f"{model_name}.pt")
    # 检测参数
    xyxy_list = predict_results.boxes.xyxy.cpu().numpy().tolist()
    conf_list = predict_results.boxes.conf.cpu().numpy().tolist()
    cls_list = predict_results.boxes.cls.cpu().numpy().tolist()

    # 颜色列表
    color_list = random_color(len(model_cls_name_cp), True)

    # 图像分割
    if (model_name[-3:] == "seg"):
        # masks_list = predict_results.masks.xyn
        masks_list = predict_results.masks.xy
        img_mask_merge = seg_output(img_path, masks_list, color_list, cls_list)
        img = Image.fromarray(cv2.cvtColor(img_mask_merge, cv2.COLOR_BGRA2RGBA))
    else:
        img = Image.open(img_path)

    # 判断检测对象是否为空
    if (xyxy_list != []):

        # ---------------- 加载字体 ----------------
        yaml_index = cls_name.index(".yaml")
        cls_name_lang = cls_name[yaml_index - 2:yaml_index]

        if cls_name_lang == "zh":
            # Chinese
            textFont = ImageFont.truetype(str(f"{ROOT_PATH}/fonts/SimSun.ttf"), size=FONTSIZE)
        elif cls_name_lang == "en":
            # English
            textFont = ImageFont.truetype(str(f"{ROOT_PATH}/fonts/TimesNewRoman.ttf"), size=FONTSIZE)
        else:
            # others
            textFont = ImageFont.truetype(str(f"{ROOT_PATH}/fonts/malgun.ttf"), size=FONTSIZE)

        for i in range(len(xyxy_list)):

            # ------------ 边框坐标 ------------
            x0 = int(xyxy_list[i][0])
            y0 = int(xyxy_list[i][1])
            x1 = int(xyxy_list[i][2])
            y1 = int(xyxy_list[i][3])

            # ---------- 加入目标尺寸 ----------
            w_obj = x1 - x0
            h_obj = y1 - y0
            area_obj = w_obj * h_obj  # 目标尺寸

            if (obj_size == "small" and area_obj > 0 and area_obj <= 32 ** 2):
                obj_cls_index = int(cls_list[i])  # 类别索引
                cls_index_det_stat.append(obj_cls_index)

                obj_cls = model_cls_name_cp[obj_cls_index]  # 类别
                cls_det_stat.append(obj_cls)

                bbox_det_stat.append((x0, y0, x1, y1))

                conf = float(conf_list[i])  # 置信度
                score_det_stat.append(conf)

                area_obj_all.append(area_obj)
            elif (obj_size == "medium" and area_obj > 32 ** 2 and area_obj <= 96 ** 2):
                obj_cls_index = int(cls_list[i])  # 类别索引
                cls_index_det_stat.append(obj_cls_index)

                obj_cls = model_cls_name_cp[obj_cls_index]  # 类别
                cls_det_stat.append(obj_cls)

                bbox_det_stat.append((x0, y0, x1, y1))

                conf = float(conf_list[i])  # 置信度
                score_det_stat.append(conf)

                area_obj_all.append(area_obj)
            elif (obj_size == "large" and area_obj > 96 ** 2):
                obj_cls_index = int(cls_list[i])  # 类别索引
                cls_index_det_stat.append(obj_cls_index)

                obj_cls = model_cls_name_cp[obj_cls_index]  # 类别
                cls_det_stat.append(obj_cls)

                bbox_det_stat.append((x0, y0, x1, y1))

                conf = float(conf_list[i])  # 置信度
                score_det_stat.append(conf)

                area_obj_all.append(area_obj)
            elif (obj_size == "all range"):
                obj_cls_index = int(cls_list[i])  # 类别索引
                cls_index_det_stat.append(obj_cls_index)

                obj_cls = model_cls_name_cp[obj_cls_index]  # 类别
                cls_det_stat.append(obj_cls)

                bbox_det_stat.append((x0, y0, x1, y1))

                conf = float(conf_list[i])  # 置信度
                score_det_stat.append(conf)

                area_obj_all.append(area_obj)

        det_img = pil_draw(img, score_det_stat, bbox_det_stat, cls_det_stat, cls_index_det_stat, textFont, color_list)

        # -------------- 目标尺寸计算 --------------
        for i in range(len(area_obj_all)):
            if (0 < area_obj_all[i] <= 32 ** 2):
                s_obj = s_obj + 1
            elif (32 ** 2 < area_obj_all[i] <= 96 ** 2):
                m_obj = m_obj + 1
            elif (area_obj_all[i] > 96 ** 2):
                l_obj = l_obj + 1

        sml_obj_total = s_obj + m_obj + l_obj
        objSize_dict = {}
        objSize_dict = {obj_style[i]: [s_obj, m_obj, l_obj][i] / sml_obj_total for i in range(3)}

        # ------------ 类别统计 ------------
        clsRatio_dict = {}
        clsDet_dict = Counter(cls_det_stat)
        clsDet_dict_sum = sum(clsDet_dict.values())
        for k, v in clsDet_dict.items():
            clsRatio_dict[k] = v / clsDet_dict_sum

        gr.Info("Inference Success!")
        return det_img, objSize_dict, clsRatio_dict
    else:
        raise gr.Error("Failed! This model cannot detect anything from this image, Please try another one.")


def main(args):
    gr.close_all()

    global model_cls_name_cp, cls_name

    nms_conf = args.nms_conf
    nms_iou = args.nms_iou
    model_name = args.model_name
    model_cfg = args.model_cfg
    cls_name = args.cls_name
    inference_size = args.inference_size
    max_detnum = args.max_detnum
    slider_step = args.slider_step

    # is_fonts(f"{ROOT_PATH}/fonts")  # 检查字体文件

    model_names = yaml_csv(model_cfg, "model_names")  # 模型名称
    model_cls_name = yaml_csv(cls_name, "model_cls_name")  # 类别名称

    model_cls_name_cp = model_cls_name.copy()  # 类别名称

    custom_theme = gr.themes.Soft(primary_hue="slate", secondary_hue="sky").set(
        button_secondary_background_fill="*neutral_100",
        button_secondary_background_fill_hover="*neutral_200")
    custom_css = '''#disp_image {
        text-align: center; /* Horizontally center the content */
    }'''

    # ------------ Gradio Blocks ------------
    with gr.Blocks(theme=custom_theme, css=custom_css) as gyd:
        with gr.Row():
            gr.Markdown(GYD_TITLE)
        with gr.Row():
            gr.Markdown(GYD_SUB_TITLE)
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.TabItem("Object Detection"):
                        with gr.Row():
                            inputs_img = gr.Image(image_mode="RGB", type="filepath", label="original image")
                        with gr.Row():
                            # device_opt = gr.Radio(choices=["cpu", "0", "1", "2", "3"], value="cpu", label="device")
                            device_opt = gr.Radio(choices=["cpu", "gpu 0", "gpu 1", "gpu 2", "gpu 3"], value="cpu",
                                                  label="device")
                        with gr.Row():
                            inputs_model = gr.Dropdown(choices=model_names, value=model_name, type="value",
                                                       label="model")
                        with gr.Row():
                            inputs_size = gr.Slider(320, 1600, step=1, value=inference_size, label="inference size")
                            max_det = gr.Slider(1, 1000, step=1, value=max_detnum, label="max bbox number")
                        with gr.Row():
                            input_conf = gr.Slider(0, 1, step=slider_step, value=nms_conf, label="confidence threshold")
                            inputs_iou = gr.Slider(0, 1, step=slider_step, value=nms_iou, label="IoU threshold")
                        with gr.Row():
                            obj_size = gr.Radio(choices=["all range", "small", "medium", "large"], value="all range",
                                                label="cell size(relative)")
                        with gr.Row():
                            gr.ClearButton(inputs_img, value="clear")
                            det_btn_img = gr.Button(value='submit', variant="primary")
                        with gr.Row():
                            gr.Examples(examples=EXAMPLES_DET,
                                        fn=yolo_det_img,
                                        inputs=[inputs_img, inputs_model, device_opt, inputs_size, input_conf,
                                                inputs_iou, max_det, obj_size],
                                        # outputs=[outputs_img, outputs_objSize, outputs_clsSize],
                                        cache_examples=False)

            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.TabItem("Object Detection"):
                        with gr.Row():
                            outputs_img = gr.Image(type="pil", label="detection results")
                        with gr.Row():
                            outputs_objSize = gr.Label(label="Percentage Statistics of cells size(relative)")
                        with gr.Row():
                            outputs_clsSize = gr.Label(label="Percentage Statistics of cells lesion degree")

        det_btn_img.click(fn=yolo_det_img,
                          inputs=[
                              inputs_img, inputs_model, device_opt, inputs_size, input_conf, inputs_iou, max_det,
                              obj_size],
                          outputs=[outputs_img, outputs_objSize, outputs_clsSize])

    return gyd


if __name__ == "__main__":
    args = parse_args()
    gyd = main(args)
    is_share = args.is_share

    gyd.queue().launch(
        inbrowser=True,  # 自动打开默认浏览器
        share=is_share,  # 项目共享，其他设备可以访问
        favicon_path="favicons/logo.ico",  # 网页图标
        show_error=True,  # 在浏览器控制台中显示错误信息
        quiet=True,  # 禁止大多数打印语句
    )
