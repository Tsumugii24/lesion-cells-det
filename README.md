![image-20231112230354573](https://cdn.jsdelivr.net/gh/Tsumugii24/Typora-images@main/images/2023%2F11%2F12%2F2ce6ad153e2e862d5017864fc5087e59-image-20231112230354573-56a688.png)

</div>

### <div align="center">Description</div>

**Lesion-Cells DET is able to detect Multi-granularity Lesion Cells by using several popular machine learning methods.**

</div>

### <div align="center">Demonstration</div>

![image-20231113140322207](https://cdn.jsdelivr.net/gh/Tsumugii24/Typora-images@main/images/2023%2F11%2F13%2F05f52e14ba9baaf83ef8f0607e0e79c7-image-20231113140322207-865f05.png)

</div>

### <div align="center">Quick Start</div>

<details open>
    <summary><h4>Install</h4></summary>

I strongly recommend you to use the Conda Environment. Both Anaconda and miniconda is OK!

Run the following command in the CLI 😊

```bash
$ git clone https://github.com/Tsumugii24/lesion-cells-det
$ cd lesion-cells-det
$ pip install -r requirements.txt
```

Then download the weights files that have already been trained properly.

</details>



<details open>
	<summary><h4>Run</h4></summary>

```bash
$ python gradio_demo.py
```



<details open>
	<summary><h4>Datasets</h4></summary>



<details open>
	<summary><h4>Training</h4></summary>


Example models of the project are trained on different methods, ranging from Convolutional Neutral Network to Vision Transformer.

| Model Name   |          Training Device           |                  Open Source                   | Average AP |
| ------------ | :--------------------------------: | :--------------------------------------------: | :--------: |
| yolov5_based | NVIDIA GeForce RTX 4090, 24563.5MB |   https://github.com/ultralytics/yolov5.git    |   0.761    |
| yolov8_based | NVIDIA GeForce RTX 4090, 24563.5MB | https://github.com/ultralytics/ultralytics.git |   0.810    |
| vit_based    | NVIDIA GeForce RTX 4090, 24563.5MB |      https://github.com/hustvl/YOLOS.git       |   0.834    |
| detr_based   | NVIDIA GeForce RTX 4090, 24563.5MB |    https://github.com/lyuwenyu/RT-DETR.git     |   0.859    |

Here is some the training results

![7dc7be0a3b11b0e248eaa4f7dea04013-image-20231112163636267-4d13c7.png](https://github.com/Tsumugii24/Typora-images/blob/main/images/2023/11/12/7dc7be0a3b11b0e248eaa4f7dea04013-image-20231112163636267-4d13c7.png?raw=true)



![image-20231112181735806](https://cdn.jsdelivr.net/gh/Tsumugii24/Typora-images@main/images/2023%2F11%2F12%2F5557c5be766788a439b018a1bef97bae-image-20231112181735806-aa3457.png)





<details open>
	<summary><h4>Train custom models</h4></summary>



<details open>
	<summary><h3>ToDo</h3></summary>


- [ ] train more accurate model for detection

- [ ] add professor Lio's brief introduction
- [ ] add a gif demonstration in the Demostration Section
- [ ] change the large .pt model with google derive link
- [ ] add the dataset's google drive link, so that everyone can train their own custom models



<details open>
	<summary><h3>References</h3></summary>


Jocher, G., Chaurasia, A., & Qiu, J. (2023). YOLO by Ultralytics (Version 8.0.0) [Computer software]. https://github.com/ultralytics/ultralytics

[Home - Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)

Jocher, G. (2020). YOLOv5 by Ultralytics (Version 7.0) [Computer software]. https://doi.org/10.5281/zenodo.3908559



[GitHub - hustvl/YOLOS: [NeurIPS 2021\] You Only Look at One Sequence](https://github.com/hustvl/YOLOS)

[GitHub - ViTAE-Transformer/ViTDet: Unofficial implementation for [ECCV'22\] "Exploring Plain Vision Transformer Backbones for Object Detection"](https://github.com/ViTAE-Transformer/ViTDet)

Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., & J'egou, H. (2020). Training data-efficient image transformers & distillation through attention. *International Conference on Machine Learning*.

Fang, Y., Liao, B., Wang, X., Fang, J., Qi, J., Wu, R., Niu, J., & Liu, W. (2021). You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection. *Neural Information Processing Systems*.

[YOLOS (huggingface.co)](https://huggingface.co/docs/transformers/main/en/model_doc/yolos)



Lv, W., Xu, S., Zhao, Y., Wang, G., Wei, J., Cui, C., Du, Y., Dang, Q., & Liu, Y. (2023). DETRs Beat YOLOs on Real-time Object Detection. *ArXiv, abs/2304.08069*.

[GitHub - facebookresearch/detr: End-to-End Object Detection with Transformers](https://github.com/facebookresearch/detr)

[PaddleDetection/configs/rtdetr at develop · PaddlePaddle/PaddleDetection · GitHub](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/rtdetr)

[GitHub - lyuwenyu/RT-DETR: Official RT-DETR (RTDETR paddle pytorch), Real-Time DEtection TRansformer, DETRs Beat YOLOs on Real-time Object Detection. 🔥 🔥 🔥](https://github.com/lyuwenyu/RT-DETR)



J. Hu, L. Shen and G. Sun, "Squeeze-and-Excitation Networks," 2018 IEEE/CVF Conference on  Computer Vision and Pattern Recognition, Salt Lake City, UT, USA, 2018, pp. 7132-7141, doi:  10.1109/CVPR.2018.00745.

Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S. (2020). End-to-End  Object Detection with Transformers. ArXiv, abs/2005.12872.

Beal, J., Kim, E., Tzeng, E., Park, D., Zhai, A., & Kislyuk, D. (2020). Toward Transformer-Based  Object Detection. ArXiv, abs/2012.09958.

Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021). Swin Transformer:  Hierarchical Vision Transformer using Shifted Windows. 2021 IEEE/CVF International Conference  on Computer Vision (ICCV), 9992-10002.

Zong, Z., Song, G., & Liu, Y. (2022). DETRs with Collaborative Hybrid Assignments  Training. ArXiv, abs/2211.12860.