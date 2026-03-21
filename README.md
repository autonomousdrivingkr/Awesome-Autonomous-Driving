# Awesome Autonomous Driving (2026 Refresh)

A curated and refreshed survey-style reading list for autonomous driving research, updated for **2026**.
This version modernizes the original repository by keeping classic references while adding the major research directions that became central after 2019: **BEV perception, multimodal fusion, end-to-end driving, foundation models / VLMs / LLMs, occupancy prediction, motion forecasting, world models, and closed-loop planning benchmarks**.

Maintainers - [Daehyun Ji](https://github.com/captainzone/) (Samsung Electronics), Dongwook Lee (Samsung Electronics), [Seho Shin](https://github.com/insightque/) (SAIT), AI Center Members in Samsung Electronics

[AutonomousDriving KR](https://www.facebook.com/groups/AutonomousDrivingKR/)

I am looking for a maintainer! Let me know (**captainzone@gmail.com**) if interested.

## Contributing
Please feel free to open [pull requests](https://github.com/autonomousdrivingkr/Awesome-Autonomous-Driving) to add papers, codebases, datasets, benchmarks, and courses.

---

## Table of Contents
- [Papers](#papers)
  - [Overall Surveys](#overall-surveys)
  - [Foundation Models, VLMs, LLMs, and World Models](#foundation-models-vlms-llms-and-world-models)
  - [Classification / Representation Learning](#classification--representation-learning)
  - [2D Object Detection](#2d-object-detection)
  - [3D Object Detection and BEV Perception](#3d-object-detection-and-bev-perception)
  - [Object Tracking](#object-tracking)
  - [Semantic Segmentation](#semantic-segmentation)
  - [Depth Estimation](#depth-estimation)
  - [Occupancy Prediction and Scene Representation](#occupancy-prediction-and-scene-representation)
  - [Localization and Mapping](#localization-and-mapping)
  - [Visual Odometry](#visual-odometry)
  - [Lane Detection and HD Map Learning](#lane-detection-and-hd-map-learning)
  - [Motion Forecasting and Behavior Prediction](#motion-forecasting-and-behavior-prediction)
  - [Decision Making](#decision-making)
  - [Planning](#planning)
  - [Control](#control)
  - [End-to-End Driving](#end-to-end-driving)
  - [Reinforcement Learning in Autonomous Driving](#reinforcement-learning-in-autonomous-driving)
- [Datasets and Benchmarks](#datasets-and-benchmarks)
- [Courses](#courses)
- [Books](#books)
- [Videos](#videos)
- [Software](#software)
  - [ROS and Autonomous Driving Stacks](#ros-and-autonomous-driving-stacks)
  - [Frameworks and Toolboxes](#frameworks-and-toolboxes)
  - [Simulation and Evaluation](#simulation-and-evaluation)
- [Conference and Workshop Channels](#conference-and-workshop-channels)
- [Maintenance Notes](#maintenance-notes)

## Papers

### Overall Surveys
- **Self-Driving Cars: A Survey** [[Paper](https://arxiv.org/abs/1901.04407)]
  - Claudine Badue, Rânik Guidolini, Raphael Vivacqua Carneiro, Pedro Azevedo, Vinicius Brito Cardoso, Avelino Forechi, Luan Ferreira Reis Jesus, Rodrigo Ferreira Berriel, Thiago Meireles Paixão, Filipe Mutz, Thiago Oliveira-Santos, Alberto Ferreira De Souza
- **Planning and Decision-Making for Autonomous Vehicles** [[Paper](https://www.annualreviews.org/doi/10.1146/annurev-control-060117-105157)]
  - Wilko Schwarting, Javier Alonso-Mora, Daniela Rus
- **A Survey of Motion Planning and Control Techniques for Self-Driving Urban Vehicles** [[Paper](https://arxiv.org/abs/1604.07446)]
  - Brian Paden, Michal Čáp, Sze Zheng Yong, Dmitry Yershov, Emilio Frazzoli
- **A Survey for Foundation Models in Autonomous Driving** [[Paper](https://arxiv.org/abs/2402.01105)]
  - Haoxiang Gao, Zhongruo Wang, Yaqian Li, Kaiwen Long, Ming Yang, Yiqing Shen
- **A Survey of World Models for Autonomous Driving** [[Paper](https://arxiv.org/abs/2501.11260)]
  - Tuo Feng, Wenguan Wang, Yi Yang
- **Foundation Models in Autonomous Driving: A Survey on Scenario Generation and Scenario Analysis** [[Paper](https://arxiv.org/abs/2506.11526)]
  - Mingyang Zhang, Haotian Wang, Yiduo Wang, et al.

### Foundation Models, VLMs, LLMs, and World Models
- **DriveLM: Driving with Graph Visual Question Answering** [[Paper](https://arxiv.org/abs/2312.14150)] [[Code](https://github.com/OpenDriveLab/DriveLM)]
  - Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, Hanxue Zhang, Chengen Xie, Jens Beißwenger, Ping Luo, Andreas Geiger, Hongyang Li
- **Planning-Oriented Autonomous Driving** [[Paper](https://arxiv.org/abs/2212.10156)] [[Code](https://github.com/OpenDriveLab/UniAD)]
  - Tianyuan Hu, Li Chen, et al.
- **TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving** [[Paper](https://arxiv.org/abs/2205.15997)] [[Code](https://github.com/autonomousvision/transfuser)]
  - Kashyap Chitta, Aditya Prakash, Bernhard Jaeger, Zehao Yu, Katrin Renz, Andreas Geiger
- **GAIA-1: A Generative World Model for Autonomous Driving** [[Paper](https://arxiv.org/abs/2309.17080)]
  - Wayve
- **DriveTransformer / DriveGPT-style driving-language works**
  - This area is moving quickly; keep an eye on VLM- and MLLM-based driving papers from CVPR, ICCV, ECCV, CoRL, and NeurIPS AD workshops.
- **World Models for Autonomous Driving: An Initial Survey** [[Paper](https://arxiv.org/abs/2403.02622)]
  - Chenhan Jiang, et al.

### Classification / Representation Learning
- **ImageNet Classification with Deep Convolutional Neural Networks** [[Paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)]
  - Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
- **Very Deep Convolutional Networks for Large-Scale Image Recognition** [[Paper](https://arxiv.org/abs/1409.1556)]
  - Karen Simonyan, Andrew Zisserman
- **Going Deeper with Convolutions** [[Paper](https://arxiv.org/abs/1409.4842)]
  - Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich
- **Deep Residual Learning for Image Recognition** [[Paper](https://arxiv.org/abs/1512.03385)]
  - Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- **Densely Connected Convolutional Networks** [[Paper](https://arxiv.org/abs/1608.06993)]
  - Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
- **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale** [[Paper](https://arxiv.org/abs/2010.11929)]
  - Alexey Dosovitskiy, et al.
- **A ConvNet for the 2020s** [[Paper](https://arxiv.org/abs/2201.03545)]
  - Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie

### 2D Object Detection
- **Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation** [[Paper](https://arxiv.org/abs/1311.2524)]
  - Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik
- **Fast R-CNN** [[Paper](https://arxiv.org/abs/1504.08083)]
  - Ross Girshick
- **Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks** [[Paper](https://arxiv.org/abs/1506.01497)]
  - Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun
- **You Only Look Once: Unified, Real-Time Object Detection** [[Paper](https://arxiv.org/abs/1506.02640)]
  - Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi
- **SSD: Single Shot MultiBox Detector** [[Paper](https://arxiv.org/abs/1512.02325)]
  - Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg
- **End-to-End Object Detection with Transformers** [[Paper](https://arxiv.org/abs/2005.12872)]
  - Nicolas Carion, et al.
- **DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection** [[Paper](https://arxiv.org/abs/2203.03605)]
  - Hao Zhang, et al.

### 3D Object Detection and BEV Perception
- **VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection** [[Paper](https://arxiv.org/abs/1711.06396)]
  - Yin Zhou, Oncel Tuzel
- **PointPillars: Fast Encoders for Object Detection from Point Clouds** [[Paper](https://arxiv.org/abs/1812.05784)]
  - Alex H. Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, Oscar Beijbom
- **SECOND: Sparsely Embedded Convolutional Detection** [[Paper](https://arxiv.org/abs/1806.12356)]
  - Yan Yan, Yuxing Mao, Bo Li
- **CenterPoint: Tracking Objects as Points** [[Paper](https://arxiv.org/abs/2006.11275)]
  - Tianwei Yin, Xingyi Zhou, Philipp Krähenbühl
- **DETR3D: 3D Object Detection from Multi-View Images via 3D-to-2D Queries** [[Paper](https://arxiv.org/abs/2110.06922)]
  - Yue Wang, et al.
- **PETR: Position Embedding Transformation for Multi-View 3D Object Detection** [[Paper](https://arxiv.org/abs/2203.05625)]
  - Yilun Liu, et al.
- **BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers** [[Paper](https://arxiv.org/abs/2203.17270)] [[Code](https://github.com/fundamentalvision/BEVFormer)]
  - Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chonghao Sima, Tong Lu, Qiao Yu, Jifeng Dai
- **BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation** [[Paper](https://arxiv.org/abs/2205.13542)] [[Code](https://github.com/mit-han-lab/bevfusion)]
  - Zhijian Liu, Haotian Tang, Alexander Amini, Hanrui Wang, Song Han
- **Occupancy and BEV methods from 2023-2025**
  - See also occupancy and end-to-end sections below, because the field increasingly merges 3D detection, map perception, forecasting, and planning.

### Object Tracking
- **Simple Online and Realtime Tracking** [[Paper](https://arxiv.org/abs/1602.00763)]
  - Alex Bewley, et al.
- **Simple Online and Realtime Tracking with a Deep Association Metric** [[Paper](https://arxiv.org/abs/1703.07402)]
  - Nicolai Wojke, Alex Bewley, Dietrich Paulus
- **AB3DMOT: A Baseline for 3D Multi-Object Tracking and New Evaluation Metrics** [[Paper](https://arxiv.org/abs/2008.08063)] [[Code](https://github.com/xinshuoweng/AB3DMOT)]
  - Xinshuo Weng, Jianren Wang, David Held, Kris Kitani
- **CenterTrack: Tracking Objects as Points** [[Paper](https://arxiv.org/abs/2004.01177)]
  - Xingyi Zhou, Vladlen Koltun, Philipp Krähenbühl

### Semantic Segmentation
- **Fully Convolutional Networks for Semantic Segmentation** [[Paper](https://arxiv.org/abs/1411.4038)]
  - Jonathan Long, Evan Shelhamer, Trevor Darrell
- **Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs** [[Paper](https://arxiv.org/abs/1412.7062)]
  - Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, Alan L. Yuille
- **DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs** [[Paper](https://arxiv.org/abs/1606.00915)]
  - Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, Alan L. Yuille
- **Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation** [[Paper](https://arxiv.org/abs/1802.02611)]
  - Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam
- **Pyramid Scene Parsing Network** [[Paper](https://arxiv.org/abs/1612.01105)]
  - Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, Jiaya Jia
- **SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers** [[Paper](https://arxiv.org/abs/2105.15203)]
  - Enze Xie, et al.
- **Masked-attention Mask Transformer for Universal Image Segmentation** [[Paper](https://arxiv.org/abs/2112.01527)]
  - Bowen Cheng, et al.

### Depth Estimation
- **Unsupervised Monocular Depth Estimation with Left-Right Consistency** [[Paper](https://arxiv.org/abs/1609.03677)] [[Code](https://github.com/mrharicot/monodepth)]
  - Clement Godard, Oisin Mac Aodha, Gabriel J. Brostow
- **Digging into Self-Supervised Monocular Depth Estimation** [[Paper](https://arxiv.org/abs/1806.01260)]
  - Clément Godard, Oisin Mac Aodha, Michael Firman, Gabriel J. Brostow
- **PackNet-SfM: 3D Packing for Self-Supervised Monocular Depth Estimation** [[Paper](https://arxiv.org/abs/1905.02693)]
  - Vitor Guizilini, et al.
- **DORN: Deep Ordinal Regression Network for Monocular Depth Estimation** [[Paper](https://arxiv.org/abs/1806.02446)]
  - Huan Fu, Mingming Gong, Chaohui Wang, Kayhan Batmanghelich, Dacheng Tao
- **Depth Anything** [[Paper](https://arxiv.org/abs/2401.10891)] [[Code](https://github.com/LiheYoung/Depth-Anything)]
  - Lihe Yang, et al.
- **Depth Anything V2** [[Paper](https://arxiv.org/abs/2406.09414)] [[Code](https://github.com/DepthAnything/Depth-Anything-V2)]
  - Lihe Yang, et al.

### Occupancy Prediction and Scene Representation
- **SurroundOcc: Multi-Camera 3D Occupancy Prediction for Autonomous Driving** [[Paper](https://arxiv.org/abs/2303.09551)] [[Code](https://github.com/weiyithu/SurroundOcc)]
  - Yi Wei, Linqing Zhao, Wenzhao Zheng, Zheng Zhu, Jie Zhou, Jiwen Lu
- **Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark for Autonomous Driving** [[Paper](https://arxiv.org/abs/2304.14365)]
  - Yiming Ge, et al.
- **TPVFormer: Tri-Perspective View for Vision-Based 3D Semantic Occupancy Prediction** [[Paper](https://arxiv.org/abs/2302.07817)]
  - Yiming Huang, et al.
- **Occupancy Network / occupancy-based scene modeling papers (2023-2025)**
  - This is now a core subfield linking perception, forecasting, and world modeling.

### Localization and Mapping
- **Visual SLAM Algorithms: A Survey from 2010 to 2016** [[Paper](https://jivp-eurasipjournals.springeropen.com/articles/10.1186/s41074-017-0027-2)]
  - Takafumi Taketomi, Hideaki Uchiyama, Sei Ikeda
- **Past, Present, and Future of Simultaneous Localization and Mapping: Toward the Robust-Perception Age** [[Paper](https://arxiv.org/abs/1606.05830)]
  - César Cadena, Luca Carlone, Henry Carrillo, Yasir Latif, Davide Scaramuzza, Jose Neira, Ian Reid, John J. Leonard
- **LOAM: Lidar Odometry and Mapping in Real-time** [[Paper](https://www.ri.cmu.edu/publications/loam-lidar-odometry-and-mapping-in-real-time/)]
  - Ji Zhang, Sanjiv Singh
- **LIO-SAM: Tightly-coupled Lidar Inertial Odometry via Smoothing and Mapping** [[Paper](https://arxiv.org/abs/2007.00258)] [[Code](https://github.com/TixiaoShan/LIO-SAM)]
  - Tianyue Shan, Brendan Englot, et al.
- **ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial, and Multi-Map SLAM** [[Paper](https://arxiv.org/abs/2007.11898)] [[Code](https://github.com/UZ-SLAMLab/ORB_SLAM3)]
  - Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M. M. Montiel, Juan D. Tardós
- **FAST-LIO2: Fast Direct LiDAR-Inertial Odometry** [[Paper](https://arxiv.org/abs/2107.06829)] [[Code](https://github.com/hku-mars/FAST_LIO)]
  - Wei Xu, et al.

### Visual Odometry
- **Review of Visual Odometry: Types, Approaches, Challenges, and Applications** [[Paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5084145/)]
  - Mohammad O. A. Aqel, Mohammad H. Marhaban, M. Iqbal Saripan, Napsiah Bt. Ismail
- **ORB-SLAM: A Versatile and Accurate Monocular SLAM System** [[Paper](https://arxiv.org/abs/1502.00956)]
  - Raúl Mur-Artal, J. M. M. Montiel, Juan D. Tardós
- **DF-VO: What Should Be Learnt for Visual Odometry?** [[Paper](https://arxiv.org/abs/1905.08772)]
  - Zhaoyang Lv, et al.
- **DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras** [[Paper](https://arxiv.org/abs/2108.10869)] [[Code](https://github.com/princeton-vl/DROID-SLAM)]
  - Zachary Teed, Jia Deng

### Lane Detection and HD Map Learning
- **Towards End-to-End Lane Detection: An Instance Segmentation Approach** [[Paper](https://arxiv.org/abs/1802.05591)]
  - Davy Neven, Bert De Brabandere, Stamatios Georgoulis, Marc Proesmans, Luc Van Gool
- **Ultra Fast Structure-Aware Deep Lane Detection** [[Paper](https://arxiv.org/abs/2004.11757)]
  - Zequn Qin, et al.
- **LaneATT: Robust Multi-Lane Detection from Stereo or Monocular Input** [[Paper](https://arxiv.org/abs/2010.12035)] [[Code](https://github.com/lucastabelini/LaneATT)]
  - Lucas Tabelini, Rodrigo Berriel, et al.
- **CLRNet: Cross Layer Refinement Network for Lane Detection** [[Paper](https://arxiv.org/abs/2203.10350)] [[Code](https://github.com/Turoad/CLRNet)]
  - Tianheng Cheng, et al.
- **MapTR: Structured Modeling and Learning for Online Vectorized HD Map Construction** [[Paper](https://arxiv.org/abs/2208.14437)] [[Code](https://github.com/hustvl/MapTR)]
  - Bencheng Liao, et al.
- **StreamMapNet: Streaming Mapping Network for Vectorized Online HD Map Construction** [[Paper](https://arxiv.org/abs/2308.12570)]
  - Jiahao He, et al.

### Motion Forecasting and Behavior Prediction
- **VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation** [[Paper](https://arxiv.org/abs/2005.04259)]
  - Li Liang, et al.
- **LaneGCN: Motion Forecasting with Lane Graph Convolutions** [[Paper](https://arxiv.org/abs/2007.13732)]
  - Ming Liang, Bin Yang, Rui Hu, Yun Chen, Raquel Urtasun
- **MTR: Motion Transformer for Motion Prediction** [[Paper](https://arxiv.org/abs/2209.13508)]
  - Shaoshuai Shi, et al.
- **Wayformer: Motion Forecasting via Simple and Efficient Attention Networks** [[Paper](https://arxiv.org/abs/2207.05844)]
  - Yixiao Wei, et al.
- **Scene Transformer: A Unified Architecture for Predicting Multiple Agent Trajectories** [[Paper](https://arxiv.org/abs/2106.08417)]
  - Junru Gu, et al.

### Decision Making
- **Planning and Decision-Making for Autonomous Vehicles** [[Paper](https://www.annualreviews.org/doi/10.1146/annurev-control-060117-105157)]
  - Wilko Schwarting, Javier Alonso-Mora, Daniela Rus
- **Perception, Planning, Control, and Coordination for Autonomous Vehicles** [[Paper](https://www.mdpi.com/2075-1702/5/1/6)]
  - R. K. Satzoda, Mohan M. Trivedi
- **A Behavioral Planning Framework for Autonomous Driving** [[Paper](https://ieeexplore.ieee.org/document/6856582)]
  - Junqing Wei, Jarrod M. Snider, Tianyu Gu, John M. Dolan, Bakhtiar Litkouhi
- **Towards a Functional System Architecture for Automated Vehicles** [[Paper](https://arxiv.org/abs/1703.08557)]
  - Simon Ulbrich, Andreas Reschka, Jens Rieken, Susanne Ernst, Gerrit Bagschik, Frank Dierkes, Marcus Nolte, Markus Maurer

### Planning
- **Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame** [[Paper](https://ieeexplore.ieee.org/document/5509799)]
  - Moritz Werling, Julius Ziegler, Sören Kammel, Sebastian Thrun
- **Path Planning for Autonomous Vehicles in Unknown Semi-Structured Environments** [[Paper](https://journals.sagepub.com/doi/10.1177/0278364909359210)]
  - Dmitri Dolgov, et al.
- **Trajectory Planning for Bertha — A Local, Continuous Method** [[Paper](https://ieeexplore.ieee.org/document/6856581)]
  - Julius Ziegler, Philipp Bender, Thao Dang, Christoph Stiller
- **Real-Time Motion Planning Methods for Autonomous On-Road Driving: State-of-the-Art and Future Research Directions** [[Paper](https://www.sciencedirect.com/science/article/pii/S0968090X15003447)]
  - Christos Katrakazas, Mohammed Quddus, Wen-Hua Chen, Lipika Deka
- **A Review of Motion Planning Techniques for Automated Vehicles** [[Paper](https://ieeexplore.ieee.org/document/7339478)]
  - David González, Joshué Pérez, Vicente Milanés, Fawzi Nashashibi
- **Towards Learning-Based Planning: The nuPlan Benchmark for Real-World Autonomous Driving** [[Paper](https://arxiv.org/abs/2403.04133)]
  - Holger Caesar, et al.

### Control
- **Stanley: The Robot that Won the DARPA Grand Challenge** [[Paper](http://robots.stanford.edu/papers/thrun.stanley05.pdf)]
  - Sebastian Thrun, et al.
- **Automatic Steering Methods for Autonomous Automobile Path Tracking** [[Paper](https://www.ri.cmu.edu/publications/automatic-steering-methods-for-autonomous-automobile-path-tracking/)]
  - Jarrod M. Snider
- **A Survey of Motion Planning and Control Techniques for Self-Driving Urban Vehicles** [[Paper](https://arxiv.org/abs/1604.07446)]
  - Brian Paden, Michal Čáp, Sze Zheng Yong, Dmitry Yershov, Emilio Frazzoli

### End-to-End Driving
- **Learning by Cheating** [[Paper](https://arxiv.org/abs/1912.12294)] [[Code](https://github.com/dotchen/LearningByCheating)]
  - Dian Chen, Brady Zhou, Vladlen Koltun, Philipp Krähenbühl
- **TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving** [[Paper](https://arxiv.org/abs/2205.15997)] [[Code](https://github.com/autonomousvision/transfuser)]
  - Kashyap Chitta, Aditya Prakash, Bernhard Jaeger, Zehao Yu, Katrin Renz, Andreas Geiger
- **NEAT: Neural Attention Fields for End-to-End Autonomous Driving** [[Paper](https://arxiv.org/abs/2109.04456)] [[Code](https://github.com/autonomousvision/neat)]
  - Bernhard Jaeger, et al.
- **TCP: Trajectory-guided Control Prediction for End-to-End Autonomous Driving** [[Paper](https://arxiv.org/abs/2206.08129)] [[Code](https://github.com/OpenDriveLab/TCP)]
  - Haotian Tang, et al.
- **Planning-Oriented Autonomous Driving** [[Paper](https://arxiv.org/abs/2212.10156)] [[Code](https://github.com/OpenDriveLab/UniAD)]
  - Tianyuan Hu, Li Chen, et al.
- **DriveLM: Driving with Graph Visual Question Answering** [[Paper](https://arxiv.org/abs/2312.14150)] [[Code](https://github.com/OpenDriveLab/DriveLM)]
  - Chonghao Sima, Katrin Renz, Kashyap Chitta, Li Chen, Hanxue Zhang, Chengen Xie, Jens Beißwenger, Ping Luo, Andreas Geiger, Hongyang Li

### Reinforcement Learning in Autonomous Driving
- **Playing for Data: Ground Truth from Computer Games** [[Paper](https://arxiv.org/abs/1608.02192)]
  - Stephan R. Richter, Vibhav Vineet, Stefan Roth, Vladlen Koltun
- **Deep Reinforcement Learning for Autonomous Driving: A Survey** [[Paper](https://arxiv.org/abs/2002.00444)]
  - Kissan Tiwari, Bikash K. Dey, et al.
- **Benchmarking Reinforcement Learning for Autonomous Driving in CARLA**
  - Search terms: RL + CARLA + CoRL / NeurIPS / ICRA / IV for the latest policy-learning papers.

## Datasets and Benchmarks
- **KITTI Vision Benchmark Suite** [[Website](https://www.cvlibs.net/datasets/kitti/)]
  - Classical benchmark for stereo, optical flow, visual odometry, 3D object detection, and tracking.
- **Cityscapes** [[Website](https://www.cityscapes-dataset.com/)]
  - Urban scene understanding benchmark with fine semantic annotations.
- **Mapillary Vistas** [[Website](https://www.mapillary.com/dataset/vistas)]
  - Large-scale, geographically diverse street-scene parsing dataset.
- **BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning** [[Paper](https://arxiv.org/abs/1805.04687)] [[Website](https://www.vis.xyz/bdd100k/)]
  - Large-scale multi-task driving dataset.
- **Waymo Open Dataset** [[Website](https://waymo.com/open/)] [[About](https://waymo.com/open/about/)]
  - Large-scale perception, motion, scenario generation, and end-to-end driving benchmark ecosystem.
- **nuScenes** [[Website](https://www.nuscenes.org/nuscenes)]
  - Multi-sensor dataset for detection, tracking, segmentation, prediction, and map-related tasks.
- **nuPlan** [[Website](https://www.nuplan.org/)] [[Paper](https://arxiv.org/abs/2403.04133)]
  - Closed-loop planning benchmark with simulation and scenario-based evaluation.
- **Argoverse 2: Next Generation Datasets for Self-Driving Perception and Forecasting** [[Website](https://www.argoverse.org/av2.html)] [[Paper](https://proceedings.neurips.cc/paper_files/paper/2021/file/5d2c2c2f3f7f3ef2c4b8f8d9429d80f7-Paper-round2.pdf)]
  - Covers sensor, lidar, motion forecasting, map-change detection, and related tasks.
- **Waymo Open Motion Dataset / Waymax ecosystem** [[Waymax](https://waymo.com/research/waymax/)] [[Docs](https://waymo-research.github.io/waymax/docs/)]
  - Useful for behavior prediction, sim agents, scenario generation, and closed-loop evaluation.
- **ApolloScape** [[Website](http://apolloscape.auto/)]
  - Includes scene parsing, car instance, lane segmentation, self-localization, and trajectory tasks.
- **SYNTHIA** [[Website](http://synthia-dataset.net/)]
  - Synthetic dataset for semantic segmentation and related perception tasks.
- **Oxford RobotCar Dataset** [[Website](https://robotcar-dataset.robots.ox.ac.uk/)]
  - Long-term autonomy dataset across weather, season, and lighting changes.
- **Oxford Radar RobotCar Dataset** [[Website](https://oxford-robotics-institute.github.io/radar-robotcar-dataset/)]
  - Adds radar and odometry for robust localization and adverse-condition research.
- **KAIST Urban Dataset / MulRan-style Korean localization datasets**
  - Keep Korean-road and Korean-traffic-specific resources in this section where possible.

## Courses
- **CS231n: Convolutional Neural Networks for Visual Recognition** [[Website](https://cs231n.stanford.edu/)]
- **Self-Driving Cars Specialization (University of Toronto / Coursera)** [[Website](https://www.coursera.org/specializations/self-driving-cars)]
- **Introduction to Self-Driving Cars** [[Website](https://www.coursera.org/learn/intro-self-driving-cars)]
- **Practical Deep Learning for Coders** [[Website](https://course.fast.ai/)]
- **Probabilistic Robotics and SLAM related graduate lectures**
  - Search with: SLAM / visual localization / motion planning / multi-agent forecasting lecture series.

## Books
- **Deep Learning** — Ian Goodfellow, Yoshua Bengio, Aaron Courville [[Book](https://www.deeplearningbook.org/)]
- **Probabilistic Robotics** — Sebastian Thrun, Wolfram Burgard, Dieter Fox [[Book](https://mitpress.mit.edu/9780262201629/probabilistic-robotics/)]
- **Planning Algorithms** — Steven M. LaValle [[Book](http://lavalle.pl/planning/)]
- **Principles of Robot Motion: Theory, Algorithms, and Implementations** — Howie Choset, et al. [[Book](https://mitpress.mit.edu/9780262033275/principles-of-robot-motion/)]
- **Computer Vision: Algorithms and Applications** — Richard Szeliski [[Book](https://szeliski.org/Book/)]

## Videos
- **Computer Vision Foundation (CVF) Open Access / YouTube** [[Channel](https://www.youtube.com/@ComputerVisionFoundation/playlists)]
- **ROSCon** [[Channel](https://www.youtube.com/@roscon)]
- **Autonomous Driving talks from Waymo Research / NVIDIA / Motional / CARLA Summit**
- **Classical deep learning lectures**
  - Andrew Ng, Geoffrey Hinton, Yann LeCun, Yoshua Bengio

## Software

### ROS and Autonomous Driving Stacks
- **ROS 2 Documentation** [[Website](https://docs.ros.org/)]
- **ROS Home** [[Website](https://www.ros.org/)]
- **Autoware** [[Website](https://autoware.org/)] [[Docs](https://autowarefoundation.github.io/autoware-documentation/main/home/)] [[GitHub](https://github.com/autowarefoundation/autoware)]
  - Open-source autonomous driving software stack built on ROS.

### Frameworks and Toolboxes
- **PyTorch** [[Website](https://pytorch.org/)]
- **TensorFlow** [[Website](https://www.tensorflow.org/)]
- **JAX** [[Website](https://jax.readthedocs.io/en/latest/)]
- **MMDetection3D** [[Docs](https://mmdetection3d.readthedocs.io/en/latest/)] [[GitHub](https://github.com/open-mmlab/mmdetection3d)]
- **OpenPCDet** [[GitHub](https://github.com/open-mmlab/OpenPCDet)]
- **Detectron2** [[GitHub](https://github.com/facebookresearch/detectron2)]
- **Nerfstudio** [[Website](https://www.nerf.studio/)]
  - Increasingly useful for 3D scene reconstruction and radiance-field-based research.

### Simulation and Evaluation
- **CARLA Simulator** [[Website](https://carla.org/)] [[Docs](https://carla.readthedocs.io/)] [[GitHub](https://github.com/carla-simulator/carla)]
- **Waymax** [[Website](https://waymo.com/research/waymax/)] [[Docs](https://waymo-research.github.io/waymax/docs/)] [[GitHub](https://github.com/waymo-research/waymax)]
- **nuPlan Devkit** [[GitHub](https://github.com/motional/nuplan-devkit)]
- **CommonRoad** [[Website](https://commonroad.in.tum.de/)]
- **NVIDIA Isaac Sim** [[Website](https://developer.nvidia.com/isaac/sim)]

## Conference and Workshop Channels
- **CVPR** [[Website](https://cvpr.thecvf.com/)]
- **ICCV** [[Website](https://iccv.thecvf.com/)]
- **ECCV** [[Website](https://eccv.ecva.net/)]
- **NeurIPS** [[Website](https://neurips.cc/)]
- **ICRA** [[Website](https://www.ieee-icra.org/)]
- **IROS** [[Website](https://www.iros.org/)]
- **IEEE Intelligent Vehicles Symposium (IV)** [[Website](https://ieee-iv.org/)]
- **CoRL (Conference on Robot Learning)** [[Website](https://www.corl.org/)]
- **Workshop keywords to watch**
  - autonomous driving, embodied AI, world models, behavior prediction, foundation models, simulation, safety validation

## Maintenance Notes
- Prefer **full paper titles** over abbreviations when adding new entries.
- Prefer **official project pages**, **official code repositories**, and **arXiv / OpenAccess** links.
- Mark deprecated toolchains clearly (for example, **Torch7**, **Theano**, **Caffe2**) instead of deleting history.
- For 2026+, the most active update zones are:
  - foundation models / VLM / LLM driving
  - world models and scenario generation
  - occupancy and BEV scene representation
  - end-to-end driving
  - motion forecasting and sim agents
  - planning benchmarks and closed-loop evaluation

---

## Suggested Next Cleanup for This Repository
- Add tags such as `Classic`, `Recommended`, `2024+`, `Code`, `Benchmark`, and `Survey`.
- Split the README into `papers.md`, `datasets.md`, `software.md`, and `courses.md` if it becomes too long.
- Add a small section for **Korean-road / Korean-traffic-light / Korean-map** resources.
- Add benchmark tables for **3D detection**, **forecasting**, **planning**, and **end-to-end driving**.
