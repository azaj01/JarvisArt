<div align="center">
  <img src="assets/logo.png" alt="JarvisArt Icon" width="100"/>

  # JarvisArt: Liberating Human Artistic Creativity via an Intelligent Photo Retouching Agent
  <!-- **JarvisArt: Liberating Human Artistic Creativity via an Intelligent Photo Retouching Agent** -->
  <a href="https://arxiv.org/pdf/2506.17612"><img src="https://img.shields.io/badge/arXiv-2506.17612-b31b1b.svg" alt="Paper"></a>
  <a href="https://huggingface.co/papers/2506.17612"><img src="https://img.shields.io/badge/🤗-Daily%20Papers-ffbd00.svg" alt="Huggingface Daily Papers"></a>
  <a href="https://jarvisart.vercel.app/"><img src="https://img.shields.io/badge/Project%20Page-Visit-blue" alt="Project Page"></a>
  <a href="https://www.youtube.com/watch?v=Ol28DQj8wV8"><img src="https://img.shields.io/badge/YouTube-Watch-red" alt="YouTube"></a>
  <a href="https://www.bilibili.com/video/BV1Sd3nzREvP/?spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=3939804dc1d27869e194605ae46329ec"><img src="https://img.shields.io/badge/BiliBili-哔哩哔哩-FF69B4" alt="BiliBili"></a>
  <a href="https://x.com/ling_yunlong/status/1940010865627103419"><img src="https://img.shields.io/twitter/follow/LYL1015?style=social" alt="Twitter Follow"></a>
  <a href="https://github.com/LYL1015/JarvisArt"><img src="https://img.shields.io/github/stars/LYL1015/JarvisArt?style=social" alt="GitHub Stars"></a>
  </div>

<div align="center">
  <p>
    <a href="https://lyl1015.github.io/">Yunlong Lin</a><sup>1*</sup>, 
    <a href="https://github.com/iendi">Zixu Lin</a><sup>1*</sup>, 
    <a href="https://github.com/kunjie-lin">Kunjie Lin</a><sup>1*</sup>, 
    <a href="https://noyii.github.io/">Jinbin Bai</a><sup>5</sup>, 
    <a href="https://paulpanwang.github.io/">Panwang Pan</a><sup>4</sup>, 
    <a href="https://chenxinli001.github.io/">Chenxin Li</a><sup>3</sup>, 
    <a href="https://haoyuchen.com/">Haoyu Chen</a><sup>2</sup>, 
    <a href="https://zhongdao.github.io/">Zhongdao Wang</a><sup>6</sup>, 
    <a href="https://scholar.google.com/citations?user=k5hVBfMAAAAJ&hl=zh-CN">Xinghao Ding</a><sup>1†</sup>,
    <a href="https://fenglinglwb.github.io/">Wenbo Li</a><sup>3♣</sup>,
    <a href="https://yanshuicheng.info/">Shuicheng Yan</a><sup>5†</sup> 
  </p>
</div>

<div align="center">
  <p>
    <sup>1</sup>Xiamen University, <sup>2</sup>The Hong Kong University of Science and Technology (Guangzhou), <sup>3</sup> The Chinese University of Hong Kong, <sup>4</sup>Bytedance, <sup>5</sup>National University of Singapore, <sup>6</sup>Tsinghua University
  </p>
  <!-- <sup>*</sup>Equal Contributions <sup>♣</sup>Project Leader <sup>†</sup>Corresponding Author -->
  <!-- <p>Accepted by CVPR 2025</p> -->
</div>

---

## ⚠️ Security Warning

**IMPORTANT: This is the ONLY official JarvisArt repository!**

We have identified **fake repositories** claiming to be JarvisArt that may contain **malware, viruses, or malicious code**. Please be extremely cautious and only use this official repository.

**Known fake/malicious repositories:**
- ❌ `https://github.com/joelp0/JarvisArt` - **FAKE & POTENTIALLY DANGEROUS**
- ❌ Any other repositories not from our official organization

---

## 📮 Updates

- **[Coming Soon]** 🚀 Gradio demo and Hugging Face demo will be released first.
- **[Coming Soon]** 🎯 Training and inference code will be released.
- **[2025.06]** 📄 Paper is now available on arXiv.
- **[2025.06]** 🌐 Project page is live.

---

## 📝 Overview

<div align="center">
  <img src="assets/teaser.jpg" alt="JarvisArt Teaser" width="800"/>
  <br>
  <em>JarvisArt workflow and results showcase</em>
</div>

JarvisArt is a multi-modal large language model (MLLM)-driven agent for intelligent photo retouching. It is designed to liberate human creativity by understanding user intent, mimicking the reasoning of professional artists, and coordinating over 200 tools in Adobe Lightroom. JarvisArt utilizes a novel two-stage training framework, starting with Chain-of-Thought supervised fine-tuning for foundational reasoning, followed by Group Relative Policy Optimization for Retouching (GRPO-R) to enhance its decision-making and tool proficiency. Supported by the newly created MMArt dataset (55K samples) and MMArt-Bench, JarvisArt demonstrates superior performance, outperforming GPT-4o with a 60% improvement in pixel-level metrics for content fidelity while maintaining comparable instruction-following capabilities.

---

## 🎬 Demo Videos

<!-- <div align="center">
  <video width="800" controls>
    <source src="assets/demo.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <p>JarvisArt Demo Video: Showcasing intelligent photo retouching capabilities</p>
</div> -->

<!-- <div align="center">
  <img src="assets/demo1.gif" alt="JarvisArt Demo" width="800px">
  <p>JarvisArt Interactive Retouching Demonstration</p>
</div>

<div align="center">
  <img src="assets/demo2.gif" alt="JarvisArt Demo" width="800px">
  <p>JarvisArt Multimodal Instruction Understanding and Execution</p>
</div> -->
Global Retouching Case
<div align="center">
  <img src="assets/global_demo1.gif" alt="JarvisArt Demo" width="800px">
  <p></p>
</div>

Local Retouching Case
<div align="center">
  <img src="assets/local_demo1.gif" alt="JarvisArt Demo" width="800px">
  <p>JarvisArt supports multi-granularity retouching goals, ranging from scene-level adjustments to region-specific refinements. Users can perform intuitive, free-form edits through natural inputs such as text prompts and bounding boxes</p>
</div>

## 🎪 Checklist

- [x] Create repo and project page
- [x] Release preview inference code and gradio demo
- [ ] Release training code 
- [ ] Release Hugging Face demo

---

## 💻 Getting Started
For gradio demo running, please follow:
- [Gradio Demo](./docs/readme_demo.md)

## 📚 Citation

If you find JarvisArt useful in your research, please consider citing:

```bibtex
@article{jarvisart2025,
title={JarvisArt: Liberating Human Artistic Creativity via an Intelligent Photo Retouching Agent}, 
      author={Yunlong Lin and Zixu Lin and Kunjie Lin and Jinbin Bai and Panwang Pan and Chenxin Li and Haoyu Chen and Zhongdao Wang and Xinghao Ding and Wenbo Li and Shuicheng Yan},
      year={2025},
      journal={arXiv preprint arXiv:2506.17612}
}
```

---


## 📧 Contact

For any questions or inquiries, please reach out to us:

- **Yunlong Lin**: linyl@stu.xmu.edu.cn
- **Zixu Lin**: a860620266@gmail.com
- **Kunjie Lin**: linkunjie@stu.xmu.edu.cn  

---

## 🙏 Acknowledgements

We would like to express our gratitude to [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory.git) and [gradio_image_annotator](https://github.com/edgarGracia/gradio_image_annotator.git) for their valuable open-source contributions which have provided important technical references for our work.

## 🌤️ Discussion Group

If you have any questions during the trial, running or deployment, feel free to join our WeChat group discussion! If you have any ideas or suggestions for the project, you are also welcome to join our WeChat group discussion!

<div align="center">
  <img src="assets/wechat_group.jpg" alt="WeChat Group" width="300px">
  <p>Scan QR code to join WeChat group discussion</p>
</div>


---

<p align="center">
  <a href="https://star-history.com/#LYL1015/JarvisArt&Date">
    <img src="https://api.star-history.com/svg?repos=LYL1015/JarvisArt&type=Date" alt="Star History Chart">
  </a>
</p>

<div align="center">
  <sub>🎨 Liberating Human Artistic Creativity, One Photo at a Time 🎨</sub>
</div>

