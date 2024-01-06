![image-20231112230354573](https://cdn.jsdelivr.net/gh/Tsumugii24/Typora-images@main/images/2023%2F11%2F12%2F2ce6ad153e2e862d5017864fc5087e59-image-20231112230354573-56a688.png)

</div>

### <div align="center"><h2>Description</h2></div>

**Lesion-Cells DET** stands for Multi-granularity **Lesion Cells DETection**.

The project employs both CNN-based and Transformer-based neural networks for Object Detection.

The system excels at detecting 7 types of cells with varying granularity in images. Additionally, it provides statistical information on the relative sizes and lesion degree distribution ratios of the identified cells.



</div>

### <div align="center"><h2>Acknowledgements</h2></div>

***I would like to express my sincere gratitude to Professor Lio for his invaluable guidance in Office Hour and supports throughout the development of this project. Professor's expertise and insightful feedback played a crucial role in shaping the direction of the project.***

![image-20231114020123496](https://cdn.jsdelivr.net/gh/Tsumugii24/Typora-images@main/images/2023%2F11%2F14%2Fd9da5305f8dd9b7273c6cafd96a87643-image-20231114020123496-402c3d.png)

</div>

### <div align="center"><h2>Demonstration</h2></div>

You can easily and directly experience the project demo online on HuggingFace now and compare the effects of different neural network models on cells object detection.

Click here for Online Experience 👉 [Lesion-Cells DET - a Hugging Face Space by Tsumugii](https://huggingface.co/spaces/Tsumugii/lesion-cells-det)

![4a7077aee8660255dd7e08ca4cdd3680-demo-daa408.gif](https://github.com/Tsumugii24/Typora-images/blob/main/images/2023/11/14/4a7077aee8660255dd7e08ca4cdd3680-demo-daa408.gif?raw=true)





</div>

### <div align="center"><h2>ToDo</h2></div>

- [x] ~~Change the large weights files with Google Drive sharing link~~
- [x] ~~Add Professor Lio's brief introduction~~
- [x] ~~Add a .gif demonstration instead of a static image~~
- [x] ~~Deploy the project demo on HuggingFace~~
- [ ] Train models that have better performance
- [ ] Upload part of the datasets, so that everyone can train their own customized models





</div>

### <div align="center"><h2>Quick Start</h2></div>

<details open>
    <summary><h4>Installation</h4></summary>

​    *I strongly recommend you to use **conda** as the environment. Both Anaconda and miniconda is OK!*


1. Create a virtual **conda** environment for the demo 😆

```bash
$ conda create -n demo python==3.8
$ conda activate demo
```

2. Install essential **requirements** by run the following command in the CLI 😊

```bash
$ git clone https://github.com/Tsumugii24/lesion-cells-det
$ cd lesion-cells-det
$ pip install -r requirements.txt
```

3. Download the **weights** files that have already been trained properly	

   (Recommended) run the script  `download_trained_model.py` to automatically download weight files 🤗

```bash
$ python download_trained_model.py
```

   (Optional) if there is something wrong with your internet connection, you can also try to download manually from 🤗

   $\Rightarrow$ Google Drive  https://drive.google.com/drive/folders/1-H4nN8viLdH6nniuiGO-_wJDENDf-BkL?usp=sharing

![image-20231114005824778](https://cdn.jsdelivr.net/gh/Tsumugii24/Typora-images@main/images/2023%2F11%2F14%2Fb5fd2a2773cff7b112c2b3328e172bd3-image-20231114005824778-df9e54.png)

   $\Rightarrow$ Hugging Face Model Card  https://huggingface.co/Tsumugii/lesion-cells-det/tree/main

![image-20240106191732142](https://cdn.jsdelivr.net/gh/Tsumugii24/Typora-images@main/images/2024%2F01%2F06%2F78c42a6194428efa79ed499f9401e823-image-20240106191732142-b3de11.png)

   Choose one of the ways above to download your preferred models and remember to put them under the `models` directory 😉

</details>

<details open>
	<summary><h4>Run</h4></summary>

```bash
$ python gradio_demo.py
```

Now, if everything is OK, your default browser will open automatically, and Gradio is running on local URL:  http://127.0.0.1:7860

</details>

<details open>
	<summary><h4>Datasets</h4></summary>

The original datasets origins from **Kaggle**, **iFLYTEK AI algorithm competition** and **other open source** sources.

Anyway, we annotated an object detection dataset of more than **2000** cells for a total of **7** categories.

| class number | class name          |
| :----------- | :------------------ |
| 0            | normal_columnar     |
| 1            | normal_intermediate |
| 2            | normal_superficiel  |
| 3            | carcinoma_in_situ   |
| 4            | light_dysplastic    |
| 5            | moderate_dysplastic |
| 6            | severe_dysplastic   |

We decided to share about half of them, which should be an adequate number for further researches and studies.

</details>

<details open>
	<summary><h4>Train customized models</h4></summary>


You can train your own customized model as long as it can work properly.

</details>



</div>

### <div align="center"><h2>Training</h2></div>

<details open>
	<summary><h4>example weights</h4></summary>

Example models of the project are trained with different methods, ranging from **Convolutional Neutral Network** to **Vision Transformer**.

| Model Name   |          Training Device           |      Open Source Repository for Reference      | Average AP |
| ------------ | :--------------------------------: | :--------------------------------------------: | :--------: |
| yolov5_based | NVIDIA GeForce RTX 4090, 24563.5MB |   https://github.com/ultralytics/yolov5.git    |   0.721    |
| yolov8_based | NVIDIA GeForce RTX 4090, 24563.5MB | https://github.com/ultralytics/ultralytics.git |   0.810    |
| vit_based    | NVIDIA GeForce RTX 4090, 24563.5MB |      https://github.com/hustvl/YOLOS.git       |   0.834    |
| detr_based   | NVIDIA GeForce RTX 4090, 24563.5MB |    https://github.com/lyuwenyu/RT-DETR.git     |   0.859    |

</details>

<details open>
	<summary><h4>architecture baselines</h4></summary>

- #### **YOLO**

![yolo](https://cdn.jsdelivr.net/gh/Tsumugii24/Typora-images@main/images/2023%2F11%2F14%2F9601a0adf0a87a68fa21b5710abbc597-yolo-99d8b6.jpeg)



- #### **Vision Transformer**

![image-20231114014357197](https://cdn.jsdelivr.net/gh/Tsumugii24/Typora-images@main/images/2023%2F11%2F14%2Fb6e85205bbb0557f332685178afe18ae-image-20231114014357197-149db2.png)

- #### **DEtection TRansformer**

![image-20231114014411513](https://cdn.jsdelivr.net/gh/Tsumugii24/Typora-images@main/images/2023%2F11%2F14%2F9076db8eefda2096dedf6a3bb81e483c-image-20231114014411513-95398d.png)

</details>



</div>

### <div align="center"><h2>References</h2></div>



1. Jocher, G., Chaurasia, A., & Qiu, J. (2023). YOLO by Ultralytics (Version 8.0.0) [Computer software]. https://github.com/ultralytics/ultralytics

2. [Home - Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)

3. Jocher, G. (2020). YOLOv5 by Ultralytics (Version 7.0) [Computer software]. https://doi.org/10.5281/zenodo.3908559

   

4. [GitHub - hustvl/YOLOS: [NeurIPS 2021\] You Only Look at One Sequence](https://github.com/hustvl/YOLOS)

5. [GitHub - ViTAE-Transformer/ViTDet: Unofficial implementation for [ECCV'22\] "Exploring Plain Vision Transformer Backbones for Object Detection"](https://github.com/ViTAE-Transformer/ViTDet)

6. Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., & J'egou, H. (2020). Training data-efficient image transformers & distillation through attention. *International Conference on Machine Learning*.

7. Fang, Y., Liao, B., Wang, X., Fang, J., Qi, J., Wu, R., Niu, J., & Liu, W. (2021). You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection. *Neural Information Processing Systems*.

8. [YOLOS (huggingface.co)](https://huggingface.co/docs/transformers/main/en/model_doc/yolos)

   

9. Lv, W., Xu, S., Zhao, Y., Wang, G., Wei, J., Cui, C., Du, Y., Dang, Q., & Liu, Y. (2023). DETRs Beat YOLOs on Real-time Object Detection. *ArXiv, abs/2304.08069*.

10. [GitHub - facebookresearch/detr: End-to-End Object Detection with Transformers](https://github.com/facebookresearch/detr)

11. [PaddleDetection/configs/rtdetr at develop · PaddlePaddle/PaddleDetection · GitHub](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/rtdetr)

12. [GitHub - lyuwenyu/RT-DETR: Official RT-DETR (RTDETR paddle pytorch), Real-Time DEtection TRansformer, DETRs Beat YOLOs on Real-time Object Detection. 🔥 🔥 🔥](https://github.com/lyuwenyu/RT-DETR)

    

13. J. Hu, L. Shen and G. Sun, "Squeeze-and-Excitation Networks," 2018 IEEE/CVF Conference on  Computer Vision and Pattern Recognition, Salt Lake City, UT, USA, 2018, pp. 7132-7141, doi:  10.1109/CVPR.2018.00745.

14. Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S. (2020). End-to-End  Object Detection with Transformers. ArXiv, abs/2005.12872.

15. Beal, J., Kim, E., Tzeng, E., Park, D., Zhai, A., & Kislyuk, D. (2020). Toward Transformer-Based  Object Detection. ArXiv, abs/2012.09958.

16. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021). Swin Transformer:  Hierarchical Vision Transformer using Shifted Windows. 2021 IEEE/CVF International Conference  on Computer Vision (ICCV), 9992-10002.

17. Zong, Z., Song, G., & Liu, Y. (2022). DETRs with Collaborative Hybrid Assignments  Training. ArXiv, abs/2211.12860.



</div>

### <div align="center"><h2>Contact</h2></div>

*Feel free to contact me through GitHub issues or directly send me a mail if you have any questions about the project.* 👻



<div align="center"><h4>Here is my email address 👉 jsf002016@gmail.com</h4></div>

​	  											

