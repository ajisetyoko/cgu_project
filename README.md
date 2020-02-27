## **CGU_Project**
This is a repositories for laboratories project on SIMSLAB NTUST Taiwan. This repositories use for detect an human action based on human skeleton. The algorithms used in this repositories are  the modified and the combination of [1] and [2]

#### Prerequisites
The dependencies of this model is the combination of [1] and [2]. You can directly use our dependencies in *enviroment.yml*


#### Program
- **Coordinate Extraction**
  - Use to make skeleton detection more accurate and faster by giving the open-pose[1] smaller resolution. This is possible if the video frame is too big compare to the object. And the object is not moving.
  - `python coordinate_maker.py video.mp4`
  - the rectangle coordinate shows in terminal which will use manually in open-pose
  - <img src="info/coordinate_maker.png" width="500">

- **Open-pose[1]**
  - The open-pose model is modified model from original implementation using tensor-flow framework by ildoonet
  - Modified list mode:
    - Dual Camera / Video Source detection
    - Mini-Stabilizer
    - Modified Json Output Sequence

#### DEMO

#### Reference
* [1] CMU-OpenPose : Deep Pose Estimation implemented using Tensor-flow with Custom Architectures for fast inference. https://github.com/ildoonet/tf-pose-estimation
* [2] ST-GCN :
